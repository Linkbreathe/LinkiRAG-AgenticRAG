"""Lazy local cross-encoder reranking with an explicit lexical fallback."""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from typing import Iterable

from linki.core.trace import emit_event
from linki.retrieval.candidates import Candidate

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\w+|[\u4e00-\u9fff]", (text or "").lower())


def lexical_score(query: str, document: str) -> float:
    """Deterministic fallback used only when the declared model cannot load."""
    q, d = Counter(_tokens(query)), Counter(_tokens(document))
    if not q or not d:
        return 0.0
    overlap = sum((q & d).values())
    return overlap / max(sum(q.values()), 1)


class CrossEncoderReranker:
    """Batch reranker. The ONNX model is loaded once per process, on demand."""

    _models: dict[str, object] = {}
    _model_errors: dict[str, str] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_RERANKER, *, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._scores: dict[tuple[str, str, str], float] = {}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _model(self):
        if not self.enabled:
            return None
        with self._lock:
            if self.model_name in self._models:
                return self._models[self.model_name]
            if self.model_name in self._model_errors:
                return None
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                model = TextCrossEncoder(model_name=self.model_name)
                self._models[self.model_name] = model
                return model
            except Exception as exc:  # model/cache/network availability is environmental
                self._model_errors[self.model_name] = f"{exc.__class__.__name__}: {exc}"
                return None

    def rerank(self, query: str, candidates: Iterable[Candidate]) -> tuple[list[Candidate], str]:
        candidates = list(candidates)
        if not candidates:
            return [], "empty"
        qh = self._hash(query)
        missing: list[Candidate] = []
        for candidate in candidates:
            key = (qh, self._hash(candidate.child_text), self.model_name)
            if key not in self._scores:
                missing.append(candidate)

        model = self._model()
        backend = self.model_name if model is not None else "lexical-fallback"
        if missing:
            if model is not None:
                try:
                    values = list(model.rerank(query, [item.child_text for item in missing]))
                    for candidate, value in zip(missing, values, strict=True):
                        key = (qh, self._hash(candidate.child_text), self.model_name)
                        self._scores[key] = float(value)
                except Exception as exc:
                    backend = "lexical-fallback"
                    self._model_errors[self.model_name] = f"{exc.__class__.__name__}: {exc}"
                    for candidate in missing:
                        key = (qh, self._hash(candidate.child_text), self.model_name)
                        self._scores[key] = lexical_score(query, candidate.child_text)
            else:
                for candidate in missing:
                    key = (qh, self._hash(candidate.child_text), self.model_name)
                    self._scores[key] = lexical_score(query, candidate.child_text)

        ranked = [
            candidate.with_rerank_score(
                self._scores[(qh, self._hash(candidate.child_text), self.model_name)]
            )
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: (item.score, item.retrieval_score), reverse=True)
        emit_event({
            "node": "rerank", "type": "rerank", "backend": backend,
            "model": self.model_name, "candidates": len(candidates),
            "error": self._model_errors.get(self.model_name),
        })
        return ranked, backend
