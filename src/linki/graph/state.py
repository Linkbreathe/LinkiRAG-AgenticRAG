"""Graph state, evidence/citation shapes, and reducers.

Design notes:
- ``evidence`` and ``gaps`` use ``operator.add`` so parallel retrieval branches
  (Phase 3, LangGraph ``Send``) auto-merge; in Phase 2 there is a single branch.
- ``retrieval_keys`` is a set-union reducer for global de-duplication across
  retrieval rounds — a chunk seen once is never re-collected.
- Runtime dependencies (``model``, ``judge``, ``settings``, ``retrieve_fn``) are
  carried in state so nodes stay pure and are trivially unit-testable with fakes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict


class Evidence(TypedDict, total=False):
    chunk_id: str
    parent_id: str
    kb: str
    source: str
    heading_path: str
    text: str
    score: float


class Citation(TypedDict, total=False):
    index: int
    source: str
    heading_path: str
    parent_id: str
    chunk_id: str


class SubQuery(TypedDict, total=False):
    id: str
    query: str
    target_kb: str  # tool name, e.g. "Retrieve_default"
    reason: str


def _set_union(a: set[str], b: set[str]) -> set[str]:
    return (a or set()) | (b or set())


# ``retrieve_fn(query, kb_tool_name) -> list[Evidence]``
RetrieveFn = Callable[[str, str], list[Evidence]]


class LinkiGraphState(TypedDict, total=False):
    # —— input / session ——
    question: str
    session_context: str

    # —— runtime deps (injected at invoke time) ——
    model: Any
    judge: Any
    settings: Any
    retrieve_fn: RetrieveFn

    # —— router / rewrite ——
    route: str  # chat | retrieve | clarify
    route_reason: str
    clarify_question: str
    rewritten_query: str
    chat_response: str

    # —— planning (Phase 3) ——
    sub_queries: list[SubQuery]

    # —— retrieval results ——
    target_kb: str
    evidence: Annotated[list[Evidence], operator.add]
    retrieval_keys: Annotated[set[str], _set_union]
    gaps: Annotated[list[str], operator.add]

    # —— answer / verify ——
    answer: str
    citations: list[Citation]
    verified: bool
    verify_issues: list[dict]
    attempts: int
    final_answer: str
