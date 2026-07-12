"""Reliability nodes: router, rewrite, chat, clarify, answer, verifier.

All model access goes through ``state["model"]`` / ``state["judge"]`` so nodes are
pure and unit-testable with fakes.
"""

from __future__ import annotations

from typing import Any

from linki.core.jsonutil import extract_json
from linki.core.telemetry import current_telemetry, invoke_model
from linki.core.trace import emit_event
from linki.graph.evidence import build_citations, number_evidence, render_evidence
from linki.graph.prompts import (
    ANSWER_PROMPT,
    CHAT_PROMPT,
    PLANNER_PROMPT,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    VERIFIER_PROMPT,
)
from linki.graph.state import LinkiGraphState, SubQuery
from linki.routing.policy import PATH_BUDGETS, decide_policy, legacy_policy
from linki.routing.risk import answer_risk, retrieval_risk
from linki.retrieval.evidence_pack import build_evidence_pack, estimate_tokens
from linki.tools.registry import render_tool_descriptions


def _text(response: Any) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _history_block(state: LinkiGraphState) -> str:
    return state.get("session_context") or "（无 / none）"


def _invoke(
    state: LinkiGraphState,
    model: Any,
    messages: Any,
    *,
    node: str,
    version: str,
    reserve_after: int = 0,
) -> Any:
    return invoke_model(
        model,
        messages,
        node=node,
        prompt_version=version,
        policy_path=state.get("policy_path", "legacy"),
        reserve_after=reserve_after,
    )


# ————————————————————————————— local policy router —————————————————————————

def policy_node(state: LinkiGraphState) -> dict[str, Any]:
    settings = state["settings"]
    mode = state.get("execution_mode") or getattr(settings, "execution_mode", "auto")
    policy_args = {
        "mode": mode,
        "session_context": state.get("session_context", ""),
        "kb_count": len(getattr(settings, "knowledge_bases", []) or [None]),
        "deadline_ms": (
            state.get("deadline_ms")
            if state.get("deadline_ms") is not None
            else getattr(settings, "default_deadline_ms", None)
        ),
    }
    shadow = None
    if not getattr(settings, "adaptive_enabled", False) and mode == "auto":
        shadow = decide_policy(state["question"], **policy_args)
        decision = legacy_policy()
    else:
        decision = decide_policy(state["question"], **policy_args)
    payload = decision.as_dict()
    telemetry = current_telemetry()
    if telemetry:
        telemetry.configure(payload)
    emit_event({
        "node": "policy", "type": "policy_decision", "policy_path": decision.path,
        "mode": decision.mode, "reason": decision.reason,
        "signals": list(decision.signals), "budget": payload["budget"],
        "shadow_policy_path": shadow.path if shadow else None,
    })
    result = {
        "policy": payload,
        "policy_path": decision.path,
        "route_reason": decision.reason,
    }
    if shadow is not None:
        result["shadow_policy"] = shadow.as_dict()
    return result


def policy_route(state: LinkiGraphState) -> str:
    path = state.get("policy_path", "legacy")
    if path == "p0":
        return "p0_local" if (state.get("policy") or {}).get("local_response") else "p0_model"
    if path == "p3" and (state.get("policy") or {}).get("needs_rewrite"):
        return "p3_rewrite"
    return path


def local_responder_node(state: LinkiGraphState) -> dict[str, Any]:
    response = (state.get("policy") or {}).get("local_response") or "How can I help?"
    return {
        "route": "chat",
        "chat_response": response,
        "final_answer": response,
        "verified": True,
        "citations": [],
    }


def direct_plan_node(state: LinkiGraphState) -> dict[str, Any]:
    """P1's zero-model planner: one intent, one selected knowledge base."""
    return {
        "sub_queries": [
            SubQuery(
                id="q1",
                query=state.get("rewritten_query") or state["question"],
                target_kb=_preferred_tool(state),
                reason="P1 direct retrieval",
            )
        ]
    }


# ————————————————————————————————— router —————————————————————————————————

def router_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    raw = _text(
        _invoke(
            state,
            state["model"],
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=f"History:\n{_history_block(state)}\n\nLatest input: {state['question']}"
                ),
            ],
            node="router",
            version="router.v2",
        )
    )
    decision = extract_json(raw, fallback={"route": "retrieve"})
    route = decision.get("route")
    if route not in {"chat", "retrieve", "clarify"}:
        route = "retrieve"  # on any doubt, prefer retrieve
    return {
        "route": route,
        "route_reason": decision.get("reason", ""),
        "clarify_question": decision.get("clarify_question", ""),
    }


def router_route(state: LinkiGraphState) -> str:
    return state.get("route", "retrieve")


# ———————————————————————————————— chat / clarify ————————————————————————————

def chat_responder_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = _text(
        _invoke(
            state,
            state["model"],
            [
                SystemMessage(content=CHAT_PROMPT),
                HumanMessage(content=state["question"]),
            ],
            node="chat_responder",
            version="chat.v1",
        )
    )
    return {"chat_response": reply, "final_answer": reply}


def clarify_node(state: LinkiGraphState) -> dict[str, Any]:
    """Phase 2: surface the clarifying question as the turn's reply; the REPL asks
    the user, who answers on the next turn. (Phase 3 upgrades this to a LangGraph
    ``interrupt`` that pauses and resumes in place.)"""
    question = state.get("clarify_question") or "Could you clarify what you mean?"
    return {"final_answer": question, "route": "clarify"}


# ————————————————————————————————— rewrite —————————————————————————————————

def rewrite_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    # First turn with no history and an already-clear question: pass through to
    # avoid over-rewriting and distorting intent.
    if not (state.get("session_context") or "").strip():
        return {"rewritten_query": state["question"]}

    rewritten = _text(
        _invoke(
            state,
            state["model"],
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=f"History:\n{_history_block(state)}\n\nLatest question: {state['question']}"
                ),
            ],
            node="rewrite",
            version="rewrite.v1",
            reserve_after=3 if state.get("policy_path") == "p3" else 0,
        )
    ).strip()
    return {"rewritten_query": rewritten or state["question"]}


# ————————————————————————————————— planner —————————————————————————————————

def _preferred_tool(state: LinkiGraphState) -> str:
    return state.get("target_kb") or state["settings"].default_kb.tool_name


def _tool_descriptions(state: LinkiGraphState) -> str:
    explicit = state.get("target_kb")
    if explicit:
        kb = state["settings"].kb(explicit)
        if kb is not None:
            return f"- {kb.tool_name}: 检索「{kb.title}」。{kb.usage_hint} (selected by caller)"
        return (
            f"- {explicit}: Selected runtime topic. Use this caller-selected "
            "knowledge base for every sub-query."
        )
    return render_tool_descriptions(state["settings"])


def _resolve_tool(state: LinkiGraphState, target: str | None) -> str:
    if state.get("target_kb"):
        return state["target_kb"]
    preferred = _preferred_tool(state)
    if target:
        kb = state["settings"].kb(target)
        if kb is not None:
            return kb.tool_name
        if target == preferred:
            return target
    return preferred


def _fallback_plan(state: LinkiGraphState, reason: str = "planner fallback") -> list[SubQuery]:
    query = state.get("rewritten_query") or state["question"]
    issues = state.get("verify_issues") or []
    if issues:
        query = "; ".join(
            str(issue.get("fix_instruction") or issue.get("problem") or issue.get("claim") or "")
            for issue in issues
        ).strip() or query
    return [
        SubQuery(
            id="q1",
            query=query,
            target_kb=_preferred_tool(state),
            reason=reason,
        )
    ]


def _normalize_sub_queries(state: LinkiGraphState, raw_items: Any) -> list[SubQuery]:
    if not isinstance(raw_items, list):
        return []

    out: list[SubQuery] = []
    for i, item in enumerate(raw_items[:4], start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        out.append(
            SubQuery(
                id=str(item.get("id") or f"q{i}"),
                query=query,
                target_kb=_resolve_tool(state, item.get("target_kb")),
                reason=str(item.get("reason") or ""),
            )
        )
    return out


def planner_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    preferred = _preferred_tool(state)
    issues = state.get("verify_issues") or []
    issues_block = "\n".join(
        f"- claim={i.get('claim', '')}; problem={i.get('problem', '')}; "
        f"fix={i.get('fix_instruction', '')}"
        for i in issues
        if isinstance(i, dict)
    ) or "(none)"
    path = state.get("policy_path", "legacy")
    reserve = 2 if path == "p3" else (1 if path == "p2" else 0)
    raw = _text(
        _invoke(
            state,
            state["model"],
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n"
                        f"Rewritten query: {state.get('rewritten_query') or state['question']}\n"
                        f"Preferred/default tool: {preferred}\n\n"
                        f"Available tools:\n{_tool_descriptions(state)}\n\n"
                        f"Verifier issues:\n{issues_block}"
                    )
                ),
            ],
            node="planner",
            version="planner.v2",
            reserve_after=reserve,
        )
    )
    plan = extract_json(raw, fallback={"sub_queries": _fallback_plan(state)})
    sub_queries = _normalize_sub_queries(state, plan.get("sub_queries"))
    if not sub_queries:
        sub_queries = _fallback_plan(state, "planner returned no valid sub-queries")
    out: dict[str, Any] = {"sub_queries": sub_queries}
    if state.get("policy_escalated"):
        out["policy_replanned"] = True
    return out


# ————————————————————————————————— answer ——————————————————————————————————

def answer_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    evidence = state.get("packed_evidence") or state.get("evidence") or []
    numbered, mapping = number_evidence(evidence)
    gaps = state.get("gaps") or []
    gaps_block = "\n".join(f"- {g}" for g in gaps) if gaps else "（无 / none）"
    memory_block = state.get("memory_context") or "（无 / none）"

    reply = _text(
        _invoke(
            state,
            state["model"],
            [
                SystemMessage(content=ANSWER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"<evidence>\n{render_evidence(numbered)}\n</evidence>\n"
                        f"<gaps>\n{gaps_block}\n</gaps>\n"
                        f"<memory>\n{memory_block}\n</memory>"
                    )
                ),
            ],
            node="answer",
            version="answer.v2",
            reserve_after=1 if state.get("policy_path") == "p3" else 0,
        )
    )
    return {"answer": reply, "citations": build_citations(reply, mapping)}


# ————————————————————————— deterministic risk gates ———————————————————————

def retrieval_gate_node(state: LinkiGraphState) -> dict[str, Any]:
    verdict = retrieval_risk(
        state.get("evidence") or [],
        state.get("gaps") or [],
        state["settings"],
    )
    path = state.get("policy_path", "legacy")
    out: dict[str, Any] = {"retrieval_risk": verdict.as_dict()}
    # A low-confidence short path may only upgrade.  It can never silently
    # generate under the original P1/P2 confidence assumption.
    if path in {"p1", "p2"} and verdict.risky and not state.get("policy_escalated"):
        policy = dict(state.get("policy") or {})
        old_budget = policy.get("budget") or {}
        policy.update({
            "path": "p3",
            "reason": f"upgraded after retrieval risk: {', '.join(verdict.reasons)}",
            "signals": list(policy.get("signals") or []) + list(verdict.reasons),
            "budget": {
                "max_model_calls": PATH_BUDGETS["p3"].max_model_calls,
                "max_input_tokens": PATH_BUDGETS["p3"].max_input_tokens,
                "max_rounds": PATH_BUDGETS["p3"].max_rounds,
                "evidence_tokens": PATH_BUDGETS["p3"].evidence_tokens,
                "deadline_ms": min(
                    int(old_budget.get("deadline_ms", PATH_BUDGETS["p3"].deadline_ms)),
                    PATH_BUDGETS["p3"].deadline_ms,
                ),
            },
        })
        telemetry = current_telemetry()
        if telemetry:
            telemetry.configure(policy)
        emit_event({
            "node": "risk_gate", "type": "policy_upgrade",
            "from": path, "to": "p3", "reasons": list(verdict.reasons),
        })
        out.update({
            "policy": policy,
            "policy_path": "p3",
            "policy_escalated": True,
        })
    return out


def retrieval_gate_route(state: LinkiGraphState) -> str:
    if (
        state.get("policy_escalated")
        and not state.get("policy_replanned")
        and state.get("policy_path") == "p3"
    ):
        # The first gate after an upgrade replans once. Later P3 retrievals
        # continue to answer even when the gap remains, so there is no loop.
        risk = state.get("retrieval_risk") or {}
        if risk.get("risky") and not state.get("answer"):
            return "planner"
    return "answer"


def evidence_pack_node(state: LinkiGraphState) -> dict[str, Any]:
    raw_evidence = state.get("evidence") or []
    if state.get("policy_path") == "legacy":
        # Freeze the historical full-parent behavior as a fair ablation slot.
        policy_budget = max(1, sum(estimate_tokens(item.get("text", "")) for item in raw_evidence))
    else:
        policy_budget = ((state.get("policy") or {}).get("budget") or {}).get("evidence_tokens")
    if policy_budget is None:
        path = state.get("policy_path", "p2")
        setting_name = {
            "p1": "evidence_budget_fast",
            "p2": "evidence_budget_balanced",
            "p3": "evidence_budget_deep",
        }.get(path, "evidence_budget_deep")
        policy_budget = getattr(state["settings"], setting_name, 4000)
    pack = build_evidence_pack(
        raw_evidence,
        token_budget=int(policy_budget),
    )
    payload = pack.as_dict()
    units = [dict(unit) for unit in pack.units]
    emit_event({
        "node": "evidence_pack", "type": "evidence_pack",
        "evidence_pack_id": pack.evidence_pack_id,
        "token_budget": pack.token_budget, "token_count": pack.token_count,
        "units": len(units),
    })
    return {
        "packed_evidence": units,
        "evidence_pack": payload,
        "evidence_pack_id": pack.evidence_pack_id,
    }


def answer_risk_node(state: LinkiGraphState) -> dict[str, Any]:
    path = state.get("policy_path", "legacy")
    verdict = answer_risk(
        state.get("answer", ""),
        state.get("citations") or [],
        state.get("packed_evidence") or state.get("evidence") or [],
    )
    if path == "legacy":
        required = True
    elif path == "p3":
        required = True
    elif path == "p2":
        required = verdict.risky
    else:
        required = False
    emit_event({
        "node": "risk_gate", "type": "answer_risk",
        "policy_path": path, "risky": verdict.risky,
        "reasons": list(verdict.reasons), "verify_required": required,
    })
    return {
        "answer_risk": verdict.as_dict(),
        "verify_required": required,
        # A deterministic pass is a valid P0/P1/P2 terminal check. P3 and
        # legacy paths overwrite this with the judge verdict.
        "verified": not verdict.risky and not required,
        "verify_issues": [
            {"claim": "deterministic answer validation", "problem": reason}
            for reason in verdict.reasons
        ],
    }


def answer_risk_route(state: LinkiGraphState) -> str:
    if state.get("verify_required"):
        return "verifier"
    if (state.get("answer_risk") or {}).get("risky"):
        return "final_with_warning"
    return "final"


# ———————————————————————————————— verifier ——————————————————————————————————

def verifier_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    judge = state.get("judge") or state["model"]
    evidence = state.get("packed_evidence") or state.get("evidence") or []
    numbered, _ = number_evidence(evidence)

    raw = _text(
        _invoke(
            state,
            judge,
            [
                SystemMessage(content=VERIFIER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"Answer:\n{state.get('answer', '')}\n\n"
                        f"<evidence>\n{render_evidence(numbered)}\n</evidence>"
                    )
                ),
            ],
            node="verifier",
            version="verifier.v2",
        )
    )
    fallback_issue = {
        "claim": "answer verification",
        "problem": "verifier-error",
        "fix_instruction": "The verifier returned an invalid verdict; do not treat the answer as verified.",
    }
    verdict = extract_json(
        raw,
        fallback={
            "passed": False,
            "issues": [fallback_issue],
            "coverage": "unknown",
            "summary": "invalid verifier response",
            "verification_error": True,
        },
    )
    return {
        "verified": bool(verdict.get("passed")),
        "verify_issues": verdict.get("issues") or [],
        "verification_error": bool(verdict.get("verification_error")),
        "attempts": state.get("attempts", 0) + 1,
    }


def verifier_route(state: LinkiGraphState) -> str:
    if state.get("verified"):
        return "final"
    if state.get("verification_error"):
        return "final_with_warning"
    settings = state["settings"]
    if state.get("attempts", 0) >= getattr(settings, "max_attempts", 2):
        return "final_with_warning"
    path = state.get("policy_path", "legacy")
    if path in {"p1", "p2"}:
        return "final_with_warning"
    if path == "p3":
        # Reflow only for a concrete support defect and only when the remaining
        # budget can still pay for plan + answer + verification.
        support_problem = any(
            any(word in str(issue.get("problem", "")).lower() for word in ("support", "missing", "coverage"))
            for issue in (state.get("verify_issues") or [])
            if isinstance(issue, dict)
        )
        remaining = current_telemetry().remaining_calls() if current_telemetry() else None
        if not support_problem or (remaining is not None and remaining < 3):
            return "final_with_warning"
    return "planner"  # reflow: plan supplemental retrieval from verifier issues


# ————————————————————————————————— finals ——————————————————————————————————

def final_node(state: LinkiGraphState) -> dict[str, Any]:
    return {"final_answer": state.get("answer", "")}


def final_with_warning_node(state: LinkiGraphState) -> dict[str, Any]:
    issues = state.get("verify_issues") or []
    lines = "\n".join(f"- {i.get('claim', '')}: {i.get('problem', '')}" for i in issues)
    warning = "⚠️ 以下回答未完全通过验证 / This answer did not fully pass verification:"
    body = state.get("answer", "")
    tail = f"\n\n{warning}\n{lines}" if lines else f"\n\n{warning}"
    return {"final_answer": f"{body}{tail}"}
