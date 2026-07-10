"""Retrieval with the grade -> refine loop (the most 'Agentic' step).

Phase 2 runs this as a single node containing an internal bounded loop:
``retrieve -> grade -> (insufficient) refine -> retrieve ...`` with global
de-duplication via ``retrieval_keys``. Phase 3 will refactor the loop body into a
compiled LangGraph subgraph fanned out with ``Send`` — the grading logic here
stays reusable either way.
"""

from __future__ import annotations

from typing import Any

from linki.core.jsonutil import extract_json
from linki.graph.prompts import GRADER_PROMPT
from linki.graph.state import Evidence, LinkiGraphState


def too_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Char-overlap guard against 'spinning in place' — a refined query that is
    nearly identical to the last one means the model has nothing new to try."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    sa, sb = set(a), set(b)
    overlap = len(sa & sb) / max(len(sa | sb), 1)
    return a == b or overlap >= threshold


def _render_hits(hits: list[Evidence]) -> str:
    return "\n\n".join(
        f"- id={h.get('chunk_id', '?')} (source={h.get('source', '?')})\n"
        f"  {h.get('text', '').strip()[:600]}"
        for h in hits
    )


def grade(judge: Any, query: str, hits: list[Evidence]) -> dict[str, Any]:
    """Structured retrieval-quality verdict for a sub-query against fresh hits."""
    if not hits:
        return {
            "sufficient": False,
            "relevant_chunk_ids": [],
            "missing": "no new evidence retrieved",
            "refined_query": query,
        }
    from langchain_core.messages import HumanMessage, SystemMessage

    raw = judge.invoke(
        [
            SystemMessage(content=GRADER_PROMPT),
            HumanMessage(content=f"Sub-query: {query}\n\nEvidence:\n{_render_hits(hits)}"),
        ]
    ).content
    # Fallback: don't stall the pipeline — accept everything if grading breaks.
    return extract_json(
        raw,
        fallback={
            "sufficient": True,
            "relevant_chunk_ids": [h.get("chunk_id") for h in hits],
            "missing": "",
            "refined_query": query,
        },
    )


def _stream_writer():
    """LangGraph custom-stream writer if we're inside a streaming run, else None.

    Lets the retrieval loop surface per-round events (each retrieve + grade) to a
    live UI without changing the node's return contract. Safely returns None when
    called via plain ``.invoke()`` or a direct unit-test call."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def retrieval_node(state: LinkiGraphState) -> dict[str, Any]:
    """Run the bounded retrieve->grade->refine loop for the current query.

    Reads ``retrieve_fn``/``judge``/``settings`` from state (injected at invoke
    time), so it is fully unit-testable with fakes. Emits ``retrieve_round`` /
    ``grade`` custom-stream events per round for live tracing.
    """
    settings = state["settings"]
    emit = _stream_writer()
    retrieve_fn = state["retrieve_fn"]
    judge = state.get("judge") or state["model"]
    max_rounds = getattr(settings, "max_rounds", 2)

    sub_queries = state.get("sub_queries") or []
    if sub_queries:
        base_query = sub_queries[0]["query"]
        target_kb = sub_queries[0].get("target_kb") or state.get("target_kb") or settings.default_kb.tool_name
    else:
        base_query = state.get("rewritten_query") or state["question"]
        target_kb = state.get("target_kb") or settings.default_kb.tool_name

    seen: set[str] = set(state.get("retrieval_keys") or set())
    collected: list[Evidence] = []
    gaps: list[str] = []
    current_query = base_query

    for round_no in range(1, max_rounds + 1):
        hits = retrieve_fn(current_query, target_kb)
        fresh = [h for h in hits if h.get("chunk_id") not in seen]
        seen |= {h.get("chunk_id") for h in fresh if h.get("chunk_id")}

        if emit:
            emit({
                "type": "retrieve_round", "round": round_no, "query": current_query,
                "kb": target_kb,
                "hits": [
                    {"chunk_id": h.get("chunk_id"), "source": h.get("source"),
                     "heading_path": h.get("heading_path"), "score": h.get("score")}
                    for h in fresh
                ],
            })

        verdict = grade(judge, base_query, fresh)
        relevant_ids = set(verdict.get("relevant_chunk_ids") or [])
        keep = [h for h in fresh if not relevant_ids or h.get("chunk_id") in relevant_ids]
        collected.extend(keep)

        if emit:
            emit({
                "type": "grade", "round": round_no,
                "sufficient": bool(verdict.get("sufficient")),
                "kept": len(keep), "missing": verdict.get("missing", ""),
                "refined_query": verdict.get("refined_query", ""),
            })

        if verdict.get("sufficient"):
            break
        if round_no >= max_rounds:
            gaps.append(verdict.get("missing") or "insufficient evidence")
            break
        refined = verdict.get("refined_query") or current_query
        if too_similar(refined, current_query):
            gaps.append(verdict.get("missing") or "insufficient evidence")
            break
        current_query = refined

    return {"evidence": collected, "retrieval_keys": seen, "gaps": gaps}
