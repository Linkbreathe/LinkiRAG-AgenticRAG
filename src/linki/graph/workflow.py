"""Phase 2 workflow assembly.

    router ─chat──▶ chat_responder ─▶ END
      │ ─clarify─▶ clarify ─▶ END
      └ ─retrieve▶ rewrite ─▶ planner ─Send×N▶ retrieve(loop) ─▶ answer ─▶ verifier
                                      ▲                                      ├ pass ─▶ final ─▶ END
                                      └──────── retry with issues ───────────┤
                                                                             └ giveup▶ final_with_warning ─▶ END
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from linki.graph.nodes import (
    answer_node,
    chat_responder_node,
    clarify_node,
    final_node,
    final_with_warning_node,
    planner_node,
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
    }
    return [Send("retrieve", {**base, "sub_query": sq}) for sq in sub_queries]


def build_workflow():
    graph = StateGraph(LinkiGraphState)

    graph.add_node("router", router_node)
    graph.add_node("chat_responder", chat_responder_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)
    graph.add_node("final_with_warning", final_with_warning_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        router_route,
        {"chat": "chat_responder", "clarify": "clarify", "retrieve": "rewrite"},
    )
    graph.add_edge("chat_responder", END)
    graph.add_edge("clarify", END)
    graph.add_edge("rewrite", "planner")
    graph.add_conditional_edges("planner", dispatch_retrieval, ["retrieve"])
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {"final": "final", "planner": "planner", "final_with_warning": "final_with_warning"},
    )
    graph.add_edge("final", END)
    graph.add_edge("final_with_warning", END)

    return graph.compile()


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
) -> dict[str, Any]:
    """Convenience one-shot: assemble initial state, invoke, return final state.

    Establishes the run-scoped observability context: a :class:`HookContext`
    (Cache/Dedup/TraceLog) and, when tracing is enabled and ``settings`` exposes
    a ``data_dir``, a :class:`Tracer` that persists a JSONL trace + timeline.md.
    Both live in contextvars for the duration of the invoke and are torn down in
    ``finally`` (the tracer's ``finalize`` renders the timeline).
    """
    import uuid

    from linki.core.trace import Tracer, _current_tracer
    from linki.hooks.base import _current_hooks
    from linki.hooks.builtin import default_hooks

    run_id = run_id or uuid.uuid4().hex[:12]
    hook_ctx = default_hooks(settings, run_id=run_id)

    tracer = None
    data_dir = getattr(settings, "data_dir", None)
    if getattr(settings, "enable_trace", True) and data_dir is not None:
        tracer = Tracer(run_id, Path(data_dir) / "traces")

    app = app or build_workflow()
    initial: LinkiGraphState = {
        "question": question,
        "session_context": session_context,
        "model": model,
        "judge": judge or model,
        "settings": settings,
        "retrieve_fn": retrieve_fn,
        "evidence": [],
        "retrieval_keys": set(),
        "gaps": [],
        "attempts": 0,
    }

    t_tok = _current_tracer.set(tracer)
    h_tok = _current_hooks.set(hook_ctx)
    try:
        return app.invoke(initial)
    finally:
        _current_hooks.reset(h_tok)
        if tracer is not None:
            tracer.finalize()
        _current_tracer.reset(t_tok)
