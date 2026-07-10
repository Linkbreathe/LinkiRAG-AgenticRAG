"""The naive RAG baseline — the thing agentic Linki is measured against.

One straight line: retrieve top-k once → stuff the passages into a prompt →
generate. No router, no planner, no grade→refine loop, no citation discipline,
no honest-refusal check. It reuses the *same* retriever so the comparison
isolates the orchestration delta, not the index.
"""

from __future__ import annotations

from typing import Any

NAIVE_PROMPT = (
    "You are the fair single-shot RAG baseline. Answer using ONLY the numbered "
    "context passages. End every factual claim with the supporting marker [n]. "
    "If the passages do not contain enough information, explicitly say that the "
    "answer was not found in the knowledge base. Do not use outside knowledge.\n\n"
    "Numbered context passages:\n{context}"
)


def naive_answer(question: str, *, model: Any, settings: Any, retrieve_fn: Any,
                 kb: str | None = None) -> dict[str, Any]:
    """Single-shot retrieve-then-generate. Returns ``{answer, evidence}``."""
    from langchain_core.messages import HumanMessage, SystemMessage

    target_kb = kb or settings.default_kb.tool_name
    evidence = retrieve_fn(question, target_kb) or []
    context = "\n\n".join(
        f"[{i}] ({h.get('source', '?')}) {h.get('text', '').strip()}"
        for i, h in enumerate(evidence, start=1)
    ) or "(no passages retrieved)"

    answer = model.invoke([
        SystemMessage(content=NAIVE_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]).content
    return {"answer": answer, "evidence": evidence}
