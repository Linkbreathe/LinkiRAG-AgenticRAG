"""Reliability nodes: router, rewrite, chat, clarify, answer, verifier.

All model access goes through ``state["model"]`` / ``state["judge"]`` so nodes are
pure and unit-testable with fakes.
"""

from __future__ import annotations

from typing import Any

from linki.core.jsonutil import extract_json
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
from linki.tools.registry import render_tool_descriptions


def _text(response: Any) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _history_block(state: LinkiGraphState) -> str:
    return state.get("session_context") or "（无 / none）"


# ————————————————————————————————— router —————————————————————————————————

def router_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    raw = _text(
        state["model"].invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=f"History:\n{_history_block(state)}\n\nLatest input: {state['question']}"
                ),
            ]
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
        state["model"].invoke(
            [
                SystemMessage(content=CHAT_PROMPT),
                HumanMessage(content=state["question"]),
            ]
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
        state["model"].invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=f"History:\n{_history_block(state)}\n\nLatest question: {state['question']}"
                ),
            ]
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
    raw = _text(
        state["model"].invoke(
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
            ]
        )
    )
    plan = extract_json(raw, fallback={"sub_queries": _fallback_plan(state)})
    sub_queries = _normalize_sub_queries(state, plan.get("sub_queries"))
    if not sub_queries:
        sub_queries = _fallback_plan(state, "planner returned no valid sub-queries")
    return {"sub_queries": sub_queries}


# ————————————————————————————————— answer ——————————————————————————————————

def answer_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    evidence = state.get("evidence") or []
    numbered, mapping = number_evidence(evidence)
    gaps = state.get("gaps") or []
    gaps_block = "\n".join(f"- {g}" for g in gaps) if gaps else "（无 / none）"

    reply = _text(
        state["model"].invoke(
            [
                SystemMessage(content=ANSWER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"<evidence>\n{render_evidence(numbered)}\n</evidence>\n"
                        f"<gaps>\n{gaps_block}\n</gaps>"
                    )
                ),
            ]
        )
    )
    return {"answer": reply, "citations": build_citations(reply, mapping)}


# ———————————————————————————————— verifier ——————————————————————————————————

def verifier_node(state: LinkiGraphState) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    judge = state.get("judge") or state["model"]
    evidence = state.get("evidence") or []
    numbered, _ = number_evidence(evidence)

    raw = _text(
        judge.invoke(
            [
                SystemMessage(content=VERIFIER_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"Answer:\n{state.get('answer', '')}\n\n"
                        f"<evidence>\n{render_evidence(numbered)}\n</evidence>"
                    )
                ),
            ]
        )
    )
    verdict = extract_json(raw, fallback={"passed": True, "issues": [], "coverage": "", "summary": "verifier fallback"})
    return {
        "verified": bool(verdict.get("passed")),
        "verify_issues": verdict.get("issues") or [],
        "attempts": state.get("attempts", 0) + 1,
    }


def verifier_route(state: LinkiGraphState) -> str:
    if state.get("verified"):
        return "final"
    settings = state["settings"]
    if state.get("attempts", 0) >= getattr(settings, "max_attempts", 2):
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
