"""Adaptive P0-P3 workflow assembly with a legacy compatibility path.

The local policy node chooses the least expensive safe path before a model is
called. P1 bypasses model routing/planning/grading/verification; P2 buys one
plan; P3 retains bounded grading and verification. Old settings doubles that do
not expose ``adaptive_enabled`` follow the original graph for compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from linki.graph.nodes import (
    answer_risk_node,
    answer_risk_route,
    answer_node,
    chat_responder_node,
    clarify_node,
    direct_plan_node,
    evidence_pack_node,
    final_node,
    final_with_warning_node,
    local_responder_node,
    planner_node,
    policy_node,
    policy_route,
    retrieval_gate_node,
    retrieval_gate_route,
    router_node,
    router_route,
    rewrite_node,
    verifier_node,
    verifier_route,
)
from linki.graph.state import LinkiGraphState
from linki.graph.subgraph import retrieval_node


def dispatch_retrieval(state: LinkiGraphState) -> list[Send]:
    sub_queries = state.get("sub_queries") or [
        {
            "id": "q1",
            "query": state.get("rewritten_query") or state["question"],
            "target_kb": state.get("target_kb") or state["settings"].default_kb.tool_name,
            "reason": "fallback dispatch",
        }
    ]
    base = {
        "question": state["question"],
        "rewritten_query": state.get("rewritten_query") or state["question"],
        "model": state["model"],
        "judge": state.get("judge") or state["model"],
        "settings": state["settings"],
        "retrieve_fn": state["retrieve_fn"],
        "target_kb": state.get("target_kb"),
        "retrieval_keys": set(state.get("retrieval_keys") or set()),
        "policy": state.get("policy") or {},
        "policy_path": state.get("policy_path", "legacy"),
        "policy_escalated": bool(state.get("policy_escalated")),
        "policy_replanned": bool(state.get("policy_replanned")),
    }
    return [Send("retrieve", {**base, "sub_query": sq}) for sq in sub_queries]


def build_workflow():
    graph = StateGraph(LinkiGraphState)

    graph.add_node("policy", policy_node)
    graph.add_node("local_responder", local_responder_node)
    graph.add_node("router", router_node)
    graph.add_node("chat_responder", chat_responder_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("planner", planner_node)
    graph.add_node("direct_plan", direct_plan_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("retrieval_gate", retrieval_gate_node)
    graph.add_node("evidence_pack", evidence_pack_node)
    graph.add_node("answer", answer_node)
    graph.add_node("answer_risk", answer_risk_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)
    graph.add_node("final_with_warning", final_with_warning_node)

    graph.add_edge(START, "policy")
    graph.add_conditional_edges(
        "policy",
        policy_route,
        {
            "legacy": "router",
            "p0_local": "local_responder",
            "p0_model": "chat_responder",
            "p1": "direct_plan",
            "p2": "planner",
            "p3": "planner",
            "p3_rewrite": "rewrite",
        },
    )
    graph.add_edge("local_responder", END)
    graph.add_conditional_edges(
        "router",
        router_route,
        {"chat": "chat_responder", "clarify": "clarify", "retrieve": "rewrite"},
    )
    graph.add_edge("chat_responder", END)
    graph.add_edge("clarify", END)
    graph.add_edge("rewrite", "planner")
    graph.add_conditional_edges("planner", dispatch_retrieval, ["retrieve"])
    graph.add_conditional_edges("direct_plan", dispatch_retrieval, ["retrieve"])
    graph.add_edge("retrieve", "retrieval_gate")
    graph.add_conditional_edges(
        "retrieval_gate",
        retrieval_gate_route,
        {"answer": "evidence_pack", "planner": "planner"},
    )
    graph.add_edge("evidence_pack", "answer")
    graph.add_edge("answer", "answer_risk")
    graph.add_conditional_edges(
        "answer_risk",
        answer_risk_route,
        {"verifier": "verifier", "final": "final", "final_with_warning": "final_with_warning"},
    )
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {"final": "final", "planner": "planner", "final_with_warning": "final_with_warning"},
    )
    graph.add_edge("final", END)
    graph.add_edge("final_with_warning", END)

    return graph.compile()


_ANSWER_CACHE_FIELDS = (
    "route", "route_reason", "policy", "policy_path", "policy_escalated",
    "answer", "final_answer", "citations", "packed_evidence", "evidence_pack",
    "evidence_pack_id", "verified", "verification_error", "verify_issues",
    "retrieval_risk", "answer_risk", "gaps", "attempts",
)


def _run_id(value: str | None) -> str:
    import uuid

    return value or uuid.uuid4().hex[:12]


def _request_context(value):
    from linki.core.context import RequestContext

    return value or RequestContext()


def _snapshot_ids(settings: Any) -> dict[str, str]:
    from linki.knowledge.snapshots import SnapshotManifest

    path = getattr(settings, "snapshot_manifest_path", None)
    return SnapshotManifest(path).active_ids() if path is not None else {}


def _model_id(model: Any) -> str:
    for field in ("model_name", "model"):
        value = getattr(model, field, None)
        if value:
            return str(value)
    delegate = getattr(model, "delegate", None)
    return _model_id(delegate) if delegate is not None else model.__class__.__name__


def _predicted_policy(
    question: str,
    *,
    settings: Any,
    execution_mode: str,
    session_context: str,
    deadline_ms: int | None,
) -> dict[str, Any]:
    from linki.routing.policy import decide_policy, legacy_policy

    if not getattr(settings, "adaptive_enabled", False):
        return legacy_policy().as_dict()
    return decide_policy(
        question,
        mode=execution_mode,
        session_context=session_context,
        kb_count=len(getattr(settings, "knowledge_bases", []) or [None]),
        deadline_ms=deadline_ms if deadline_ms is not None else getattr(settings, "default_deadline_ms", None),
    ).as_dict()


def _answer_cache(
    question: str,
    *,
    settings: Any,
    model: Any,
    judge: Any,
    request_context: Any,
    snapshot_ids: dict[str, str],
    predicted_policy: dict[str, Any],
    execution_mode: str,
    deadline_ms: int | None,
    target_kb: str | None,
    session_context: str,
):
    if (
        not getattr(settings, "enable_persistent_cache", False)
        or not getattr(settings, "enable_answer_cache", False)
        or session_context.strip()
        or predicted_policy.get("path") not in {"p1", "p2"}
    ):
        return None, None
    kb_name = target_kb or settings.default_kb.name
    if kb_name.startswith("Retrieve_"):
        kb_name = kb_name[len("Retrieve_"):]
    # Never make an unversioned answer reachable: manual index mutations would
    # otherwise serve stale content indefinitely.
    if kb_name not in snapshot_ids:
        return None, None

    from linki.cache.base import make_cache_key, normalize_query
    from linki.cache.sqlite import get_sqlite_cache

    key = make_cache_key(
        "answer.v2",
        tenant_id=request_context.tenant_id,
        user_scope=request_context.user_scope,
        acl_hash=request_context.acl_hash,
        question=normalize_query(question),
        kb=kb_name,
        kb_snapshot_id=snapshot_ids[kb_name],
        memory_snapshot_id="none",
        policy_version="adaptive.v2",
        policy_path=predicted_policy.get("path"),
        execution_mode=execution_mode,
        prompt_versions="planner.v2|answer.v2|verifier.v2",
        model_id=_model_id(model),
        judge_model_id=_model_id(judge),
        evidence_pack_version="pack.v1",
        deadline_ms=deadline_ms,
    )
    return get_sqlite_cache(settings.cache_path), key


def _singleflight_key(
    question: str,
    *,
    model: Any,
    judge: Any,
    request_context: Any,
    snapshot_ids: dict[str, str],
    predicted_policy: dict[str, Any],
    execution_mode: str,
    deadline_ms: int | None,
    target_kb: str | None,
    session_context: str,
) -> str:
    from linki.cache.base import make_cache_key, normalize_query

    return make_cache_key(
        "answer-flight.v2",
        tenant_id=request_context.tenant_id,
        user_scope=request_context.user_scope,
        acl_hash=request_context.acl_hash,
        question=normalize_query(question),
        session_context=session_context,
        snapshots=snapshot_ids,
        policy_path=predicted_policy.get("path"),
        execution_mode=execution_mode,
        deadline_ms=deadline_ms,
        target_kb=target_kb,
        model_id=_model_id(model),
        judge_model_id=_model_id(judge),
    )


def _cache_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {field: result.get(field) for field in _ANSWER_CACHE_FIELDS if field in result}


def _cached_result(
    payload: dict[str, Any],
    *,
    run_id: str,
    request_context: Any,
    snapshot_ids: dict[str, str],
    source: str,
    wait_ms: float = 0.0,
) -> dict[str, Any]:
    result = dict(payload)
    policy = result.get("policy") or {"path": result.get("policy_path", "unknown")}
    result.update({
        "run_id": run_id,
        "policy": policy,
        "request_context": request_context.as_dict(),
        "snapshots": snapshot_ids,
        "cache": {"hit": source == "answer", "coalesced": source == "singleflight", "source": source},
        "cost": {
            "run_id": run_id,
            "policy_path": result.get("policy_path", policy.get("path", "unknown")),
            "question_type": str((policy.get("signals") or ["unknown"])[0]),
            "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "latency_ms": round(wait_ms, 3), "node_costs": [],
        },
    })
    return result


def _initial_state(
    question: str,
    *,
    model: Any,
    judge: Any,
    settings: Any,
    retrieve_fn: Any,
    session_context: str,
    execution_mode: str,
    deadline_ms: int | None,
    target_kb: str | None,
) -> LinkiGraphState:
    return {
        "question": question,
        "session_context": session_context,
        "execution_mode": execution_mode,
        "deadline_ms": deadline_ms,
        "target_kb": target_kb,
        "model": model,
        "judge": judge,
        "settings": settings,
        "retrieve_fn": retrieve_fn,
        "evidence": [],
        "retrieval_keys": set(),
        "gaps": [],
        "attempts": 0,
    }


def _run_resources(settings, run_id, request_context, snapshot_ids):
    from linki.core.telemetry import RunTelemetry
    from linki.core.trace import Tracer
    from linki.hooks.builtin import default_hooks

    telemetry = RunTelemetry(run_id)
    tracer = None
    data_dir = getattr(settings, "data_dir", None)
    if getattr(settings, "enable_trace", True) and data_dir is not None:
        tracer = Tracer(run_id, Path(data_dir) / "traces")
    hooks = default_hooks(
        settings, run_id=run_id, request_context=request_context, snapshot_ids=snapshot_ids,
    )
    return telemetry, tracer, hooks, data_dir


def _finalize_result(result, *, run_id, request_context, snapshot_ids, telemetry, data_dir):
    from linki.core.trace import emit_event

    result["run_id"] = run_id
    result["policy"] = result.get("policy") or {"path": result.get("policy_path", "legacy")}
    result["request_context"] = request_context.as_dict()
    result["snapshots"] = snapshot_ids
    result["cache"] = {"hit": False, "coalesced": False, "source": None}
    result["cost"] = telemetry.snapshot()
    emit_event({
        "node": "run", "type": "run_cost", "policy_path": result.get("policy_path"),
        "llm_calls": result["cost"]["llm_calls"],
        "input_tokens": result["cost"]["input_tokens"],
        "output_tokens": result["cost"]["output_tokens"],
        "total_ms": result["cost"]["latency_ms"],
    })
    if data_dir is not None:
        telemetry.persist(Path(data_dir) / "telemetry")
    return result


def _invoke_graph_sync(initial, *, app, settings, run_id, request_context, snapshot_ids):
    from linki.core.telemetry import _current_telemetry
    from linki.core.trace import _current_tracer
    from linki.hooks.base import _current_hooks

    telemetry, tracer, hooks, data_dir = _run_resources(
        settings, run_id, request_context, snapshot_ids,
    )
    t_tok = _current_tracer.set(tracer)
    h_tok = _current_hooks.set(hooks)
    m_tok = _current_telemetry.set(telemetry)
    try:
        return _finalize_result(
            app.invoke(initial, config={"max_concurrency": getattr(settings, "max_concurrency", 16)}),
            run_id=run_id, request_context=request_context,
            snapshot_ids=snapshot_ids, telemetry=telemetry, data_dir=data_dir,
        )
    finally:
        _current_telemetry.reset(m_tok)
        _current_hooks.reset(h_tok)
        if tracer is not None:
            tracer.finalize()
        _current_tracer.reset(t_tok)


async def _invoke_graph_async(initial, *, app, settings, run_id, request_context, snapshot_ids):
    from linki.core.telemetry import _current_telemetry
    from linki.core.trace import _current_tracer
    from linki.hooks.base import _current_hooks

    telemetry, tracer, hooks, data_dir = _run_resources(
        settings, run_id, request_context, snapshot_ids,
    )
    t_tok = _current_tracer.set(tracer)
    h_tok = _current_hooks.set(hooks)
    m_tok = _current_telemetry.set(telemetry)
    try:
        return _finalize_result(
            await app.ainvoke(
                initial, config={"max_concurrency": getattr(settings, "max_concurrency", 16)}
            ),
            run_id=run_id, request_context=request_context,
            snapshot_ids=snapshot_ids, telemetry=telemetry, data_dir=data_dir,
        )
    finally:
        _current_telemetry.reset(m_tok)
        _current_hooks.reset(h_tok)
        if tracer is not None:
            tracer.finalize()
        _current_tracer.reset(t_tok)


def answer_question(
    question: str,
    *,
    model: Any,
    settings: Any,
    retrieve_fn: Any,
    judge: Any = None,
    session_context: str = "",
    app=None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    deadline_ms: int | None = None,
    request_context: Any = None,
    target_kb: str | None = None,
) -> dict[str, Any]:
    """Synchronous compatibility entrypoint; async servers should use the async API."""
    run_id = _run_id(run_id)
    request_context = _request_context(request_context)
    judge = judge or model
    execution_mode = execution_mode or getattr(settings, "execution_mode", "auto")
    snapshots = _snapshot_ids(settings)
    predicted = _predicted_policy(
        question, settings=settings, execution_mode=execution_mode,
        session_context=session_context, deadline_ms=deadline_ms,
    )
    cache, cache_key = _answer_cache(
        question, settings=settings, model=model, judge=judge,
        request_context=request_context, snapshot_ids=snapshots,
        predicted_policy=predicted, execution_mode=execution_mode,
        deadline_ms=deadline_ms, target_kb=target_kb, session_context=session_context,
    )
    if cache is not None:
        cached = cache.get("answer.v2", cache_key)
        if cached is not None:
            return _cached_result(
                cached, run_id=run_id, request_context=request_context,
                snapshot_ids=snapshots, source="answer",
            )
    initial = _initial_state(
        question, model=model, judge=judge, settings=settings, retrieve_fn=retrieve_fn,
        session_context=session_context, execution_mode=execution_mode,
        deadline_ms=deadline_ms, target_kb=target_kb,
    )
    result = _invoke_graph_sync(
        initial, app=app or build_workflow(), settings=settings, run_id=run_id,
        request_context=request_context, snapshot_ids=snapshots,
    )
    if cache is not None and result.get("verified") and result.get("policy_path") in {"p1", "p2"}:
        cache.set(
            "answer.v2", cache_key, _cache_payload(result),
            ttl_seconds=getattr(settings, "answer_cache_ttl_seconds", 86_400),
        )
    return result


async def answer_question_async(
    question: str,
    *,
    model: Any,
    settings: Any,
    retrieve_fn: Any,
    judge: Any = None,
    session_context: str = "",
    app=None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    deadline_ms: int | None = None,
    request_context: Any = None,
    target_kb: str | None = None,
) -> dict[str, Any]:
    """Async primary entrypoint with exact-cache and identical-request single-flight."""
    import asyncio
    import time

    from linki.cache.singleflight import GLOBAL_SINGLE_FLIGHT

    run_id = _run_id(run_id)
    request_context = _request_context(request_context)
    judge = judge or model
    execution_mode = execution_mode or getattr(settings, "execution_mode", "auto")
    snapshots = _snapshot_ids(settings)
    predicted = _predicted_policy(
        question, settings=settings, execution_mode=execution_mode,
        session_context=session_context, deadline_ms=deadline_ms,
    )
    cache, cache_key = _answer_cache(
        question, settings=settings, model=model, judge=judge,
        request_context=request_context, snapshot_ids=snapshots,
        predicted_policy=predicted, execution_mode=execution_mode,
        deadline_ms=deadline_ms, target_kb=target_kb, session_context=session_context,
    )
    flight_key = _singleflight_key(
        question, model=model, judge=judge, request_context=request_context,
        snapshot_ids=snapshots, predicted_policy=predicted,
        execution_mode=execution_mode, deadline_ms=deadline_ms,
        target_kb=target_kb, session_context=session_context,
    )
    if cache is not None:
        cached = await asyncio.to_thread(cache.get, "answer.v2", cache_key)
        if cached is not None:
            return _cached_result(
                cached, run_id=run_id, request_context=request_context,
                snapshot_ids=snapshots, source="answer",
            )
    initial = _initial_state(
        question, model=model, judge=judge, settings=settings, retrieve_fn=retrieve_fn,
        session_context=session_context, execution_mode=execution_mode,
        deadline_ms=deadline_ms, target_kb=target_kb,
    )
    workflow = app or build_workflow()

    async def execute():
        return await _invoke_graph_async(
            initial, app=workflow, settings=settings, run_id=run_id,
            request_context=request_context, snapshot_ids=snapshots,
        )

    started = time.perf_counter()
    result, coalesced = await GLOBAL_SINGLE_FLIGHT.run(f"answer.v2:{flight_key}", execute)
    if coalesced:
        result = _cached_result(
            _cache_payload(result), run_id=run_id, request_context=request_context,
            snapshot_ids=snapshots, source="singleflight",
            wait_ms=(time.perf_counter() - started) * 1000.0,
        )
    elif cache is not None and result.get("verified") and result.get("policy_path") in {"p1", "p2"}:
        await asyncio.to_thread(
            cache.set, "answer.v2", cache_key, _cache_payload(result),
            ttl_seconds=getattr(settings, "answer_cache_ttl_seconds", 86_400),
        )
    return result
