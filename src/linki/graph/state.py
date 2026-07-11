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

from typing import Annotated, Any, Callable, TypedDict


class Evidence(TypedDict, total=False):
    evidence_id: str
    chunk_id: str
    parent_id: str
    kb: str
    source: str
    source_id: str
    source_version: str
    heading_path: str
    text: str
    quote: str
    char_start: int
    char_end: int
    score: float
    retrieval_score: float
    rerank_score: float | None
    rerank_backend: str
    supports: list[str]
    token_count: int
    content_hash: str
    offset_valid: bool


class Citation(TypedDict, total=False):
    index: int
    source: str
    heading_path: str
    parent_id: str
    chunk_id: str
    evidence_id: str
    quote: str
    char_start: int
    char_end: int


class SubQuery(TypedDict, total=False):
    id: str
    query: str
    target_kb: str  # tool name, e.g. "Retrieve_default"
    reason: str


def _set_union(a: set[str], b: set[str]) -> set[str]:
    return (a or set()) | (b or set())


def _evidence_union(a: list[Evidence], b: list[Evidence]) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[str] = set()
    for item in (a or []) + (b or []):
        key = item.get("chunk_id") or f"{item.get('source', '')}:{item.get('text', '')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _list_add(a: list[Any], b: list[Any]) -> list[Any]:
    return (a or []) + (b or [])


# ``retrieve_fn(query, kb_tool_name) -> list[Evidence]``
RetrieveFn = Callable[[str, str], list[Evidence]]


class LinkiGraphState(TypedDict, total=False):
    # —— input / session ——
    question: str
    session_context: str
    execution_mode: str
    deadline_ms: int | None
    memory_context: str
    memory_snapshot_id: str
    recalled_memories: list[dict[str, Any]]

    # —— runtime deps (injected at invoke time) ——
    model: Any
    judge: Any
    settings: Any
    retrieve_fn: RetrieveFn

    # —— local adaptive policy / budgets ——
    policy: dict[str, Any]
    policy_path: str  # legacy | p0 | p1 | p2 | p3
    policy_escalated: bool
    policy_replanned: bool
    retrieval_risk: dict[str, Any]
    answer_risk: dict[str, Any]
    verify_required: bool

    # —— router / rewrite ——
    route: str  # chat | retrieve | clarify
    route_reason: str
    clarify_question: str
    rewritten_query: str
    chat_response: str

    # —— planning (Phase 3) ——
    sub_queries: list[SubQuery]
    sub_query: SubQuery

    # —— retrieval results ——
    target_kb: str
    evidence: Annotated[list[Evidence], _evidence_union]
    packed_evidence: list[Evidence]
    evidence_pack: dict[str, Any]
    evidence_pack_id: str
    retrieval_keys: Annotated[set[str], _set_union]
    gaps: Annotated[list[str], _list_add]

    # —— answer / verify ——
    answer: str
    citations: list[Citation]
    verified: bool
    verification_error: bool
    verify_issues: list[dict]
    attempts: int
    final_answer: str
