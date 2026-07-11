"""High-recall hybrid candidates -> local rerank -> verbatim support spans."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from linki.config import Settings
from linki.core.kb_registry import KnowledgeBaseRegistry
from linki.graph.state import Evidence
from linki.ingestion.indexer import ParentStore, VectorStoreManager
from linki.retrieval.candidates import Candidate, deduplicate_parents, reciprocal_rank_fusion
from linki.retrieval.graph import GraphRetriever
from linki.retrieval.rerank import CrossEncoderReranker
from linki.retrieval.spans import select_supporting_span


class Retriever:
    def __init__(self, settings: Settings, *, graph_retriever: GraphRetriever | None = None):
        self._s = settings
        self._registry = KnowledgeBaseRegistry(settings)
        self._vectors = VectorStoreManager(settings)
        self._parents = ParentStore(settings.parent_store_path)
        self._reranker = CrossEncoderReranker(
            getattr(settings, "reranker_model", "Xenova/ms-marco-MiniLM-L-6-v2"),
            enabled=getattr(settings, "enable_local_reranker", True),
        )
        self._graph = graph_retriever

    def retrieve_candidates(
        self,
        query: str,
        kb_tool_name: str | None = None,
        *,
        top_n: int | None = None,
    ) -> list[Candidate]:
        """Return child candidates without loading parent bodies."""
        kb = self._registry.get(kb_tool_name) or self._s.default_kb
        vs = self._vectors.get_vectorstore(kb.collection)
        try:
            results = vs.similarity_search_with_score(
                query,
                k=top_n or getattr(self._s, "candidate_k", 30),
            )
        except Exception:
            return []
        candidates: list[Candidate] = []
        for doc, score in results:
            if score < self._s.retrieval_score_threshold:
                continue
            metadata = dict(doc.metadata or {})
            parent_id = metadata.get("parent_id", metadata.get("chunk_id", ""))
            candidates.append(Candidate(
                chunk_id=metadata.get("chunk_id", parent_id),
                parent_id=parent_id,
                kb=kb.name,
                source=metadata.get("source", "?"),
                heading_path=metadata.get("heading_path", ""),
                child_text=doc.page_content,
                retrieval_score=float(score),
                channel="hybrid",
                metadata=metadata,
            ))

        if self._graph is not None and getattr(self._s, "graph_retrieval", "none") == "ppr_pilot":
            seeds = _seed_entities(query)
            graph_hits = self._graph.retrieve(
                query, seeds, getattr(self._s, "knowledge_snapshot_id", "current"),
                getattr(self._s, "candidate_k", 30),
            )
            candidates = reciprocal_rank_fusion([candidates, graph_hits])
        return candidates

    def retrieve(self, query: str, kb_tool_name: str | None = None) -> list[Evidence]:
        candidates = self.retrieve_candidates(query, kb_tool_name)
        ranked, backend = self._reranker.rerank(query, candidates)
        ranked = deduplicate_parents(ranked)[:getattr(self._s, "rerank_k", 8)]
        evidence: list[Evidence] = []
        for candidate in ranked:
            parent = self._parents.load_content(candidate.parent_id)
            parent_text = parent["content"] if parent else candidate.child_text
            span = select_supporting_span(
                query,
                parent_text,
                anchor_text=candidate.child_text,
                max_chars=getattr(self._s, "supporting_span_chars", 1600),
            )
            source_version = str(
                candidate.metadata.get("source_version")
                or hashlib.sha256(parent_text.encode("utf-8")).hexdigest()[:16]
            )
            evidence.append(
                Evidence(
                    chunk_id=candidate.chunk_id,
                    parent_id=candidate.parent_id,
                    kb=candidate.kb,
                    source=candidate.source,
                    source_id=candidate.source,
                    source_version=source_version,
                    heading_path=candidate.heading_path,
                    text=span.quote,
                    quote=span.quote,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    score=float(candidate.score),
                    retrieval_score=float(candidate.retrieval_score),
                    rerank_score=candidate.rerank_score,
                    rerank_backend=backend,
                    content_hash=hashlib.sha256(span.quote.encode("utf-8")).hexdigest(),
                    offset_valid=span.validates(parent_text),
                )
            )
        return evidence

    def retrieve_legacy(self, query: str, kb_tool_name: str | None = None) -> list[Evidence]:
        """Frozen top-k/full-parent baseline for fair before/after evaluation."""
        kb = self._registry.get(kb_tool_name) or self._s.default_kb
        vs = self._vectors.get_vectorstore(kb.collection)
        try:
            results = vs.similarity_search_with_score(query, k=self._s.retrieval_k)
        except Exception:
            return []
        best: dict[str, tuple[float, Any]] = {}
        for doc, score in results:
            if score < self._s.retrieval_score_threshold:
                continue
            pid = doc.metadata.get("parent_id", doc.metadata.get("chunk_id", ""))
            if pid not in best or score > best[pid][0]:
                best[pid] = (score, doc)
        evidence: list[Evidence] = []
        for pid, (score, doc) in sorted(best.items(), key=lambda item: item[1][0], reverse=True):
            parent = self._parents.load_content(pid)
            text = parent["content"] if parent else doc.page_content
            evidence.append(Evidence(
                chunk_id=doc.metadata.get("chunk_id", pid), parent_id=pid, kb=kb.name,
                source=doc.metadata.get("source", "?"),
                heading_path=doc.metadata.get("heading_path", ""), text=text,
                score=float(score), retrieval_score=float(score),
            ))
        return evidence


def _seed_entities(query: str) -> list[str]:
    """Cheap pilot seed extraction; graph adapters may replace this upstream."""
    latin = re.findall(r"\b[A-Z][\w.-]{2,}\b", query or "")
    cjk = re.findall(r"[\u4e00-\u9fff]{2,8}", query or "")
    return list(dict.fromkeys(latin + cjk))[:8]


def make_retrieve_fn(settings: Settings, *, variant: str = "rerank_pack"):
    """Return a ``retrieve_fn(query, kb_tool_name) -> list[Evidence]`` bound to a
    single Retriever (embeddings load once)."""
    retriever = Retriever(settings)
    if variant == "legacy_k5":
        return retriever.retrieve_legacy
    if variant != "rerank_pack":
        raise ValueError("retrieval variant must be legacy_k5 or rerank_pack")
    return retriever.retrieve
