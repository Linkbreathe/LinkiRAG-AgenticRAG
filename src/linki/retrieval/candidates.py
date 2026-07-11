"""High-recall child candidates before parent expansion."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    parent_id: str
    kb: str
    source: str
    heading_path: str
    child_text: str
    retrieval_score: float
    rerank_score: float | None = None
    channel: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.retrieval_score

    def with_rerank_score(self, score: float) -> "Candidate":
        return replace(self, rerank_score=float(score))


def deduplicate_parents(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the best-ranked child for each parent."""
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        key = candidate.parent_id or candidate.chunk_id
        if key not in best or candidate.score > best[key].score:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def reciprocal_rank_fusion(rankings: list[list[Candidate]], *, rank_constant: int = 60) -> list[Candidate]:
    """Fuse lexical/dense/graph-style rankings without assuming score calibration."""
    scores: dict[str, float] = {}
    exemplars: dict[str, Candidate] = {}
    channels: dict[str, set[str]] = {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            key = candidate.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
            exemplars.setdefault(key, candidate)
            channels.setdefault(key, set()).add(candidate.channel)
    fused = [
        replace(
            exemplars[key],
            retrieval_score=score,
            channel="+".join(sorted(channels[key])),
        )
        for key, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.retrieval_score, reverse=True)
