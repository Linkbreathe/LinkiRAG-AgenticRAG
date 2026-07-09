"""Hybrid retrieval with parent expansion, returning structured ``Evidence``.

Retrieve child chunks (dense+sparse hybrid via Qdrant), then expand each hit to
its parent chunk for full context — de-duplicating so one parent is returned once
(highest-scoring child wins). Structured output (not concatenated text) lets the
grader/answer/verifier nodes process evidence programmatically.
"""

from __future__ import annotations

from typing import Any

from linki.config import Settings
from linki.graph.state import Evidence
from linki.ingestion.indexer import ParentStore, VectorStoreManager


class Retriever:
    def __init__(self, settings: Settings):
        self._s = settings
        self._vectors = VectorStoreManager(settings)
        self._parents = ParentStore(settings.parent_store_path)

    def retrieve(self, query: str, kb_tool_name: str | None = None) -> list[Evidence]:
        kb = self._s.kb(kb_tool_name or "") or self._s.default_kb
        vs = self._vectors.get_vectorstore(kb.collection)
        try:
            results = vs.similarity_search_with_score(query, k=self._s.retrieval_k)
        except Exception:
            # Collection may not exist yet / be empty.
            return []

        # Child hits -> best score per parent.
        best: dict[str, tuple[float, Any]] = {}
        for doc, score in results:
            if score < self._s.retrieval_score_threshold:
                continue
            pid = doc.metadata.get("parent_id", doc.metadata.get("chunk_id", ""))
            if pid not in best or score > best[pid][0]:
                best[pid] = (score, doc)

        evidence: list[Evidence] = []
        for pid, (score, doc) in sorted(best.items(), key=lambda kv: kv[1][0], reverse=True):
            parent = self._parents.load_content(pid)
            text = parent["content"] if parent else doc.page_content
            evidence.append(
                Evidence(
                    chunk_id=doc.metadata.get("chunk_id", pid),
                    parent_id=pid,
                    kb=kb.name,
                    source=doc.metadata.get("source", "?"),
                    heading_path=doc.metadata.get("heading_path", ""),
                    text=text,
                    score=float(score),
                )
            )
        return evidence


def make_retrieve_fn(settings: Settings):
    """Return a ``retrieve_fn(query, kb_tool_name) -> list[Evidence]`` bound to a
    single Retriever (embeddings load once)."""
    return Retriever(settings).retrieve
