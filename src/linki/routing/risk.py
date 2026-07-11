"""Deterministic retrieval and answer risk gates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_CITATION = re.compile(r"\[(\d+)\]")
_REFUSAL = re.compile(
    r"未找到|无法回答|信息不足|not found|cannot answer|insufficient|no relevant",
    re.I,
)


@dataclass(frozen=True)
class RiskDecision:
    risky: bool
    reasons: tuple[str, ...]
    action: str

    def as_dict(self) -> dict:
        return asdict(self)


def retrieval_risk(evidence: list[dict], gaps: list[str], settings: Any) -> RiskDecision:
    reasons: list[str] = []
    if not evidence:
        reasons.append("no-evidence")
    if gaps:
        reasons.append("coverage-gap")

    scores = sorted((float(e.get("score", 0.0)) for e in evidence), reverse=True)
    calibrated_floor = float(getattr(settings, "adaptive_low_score_threshold", 0.0) or 0.0)
    calibrated_margin = float(getattr(settings, "adaptive_min_score_margin", 0.0) or 0.0)
    if scores and calibrated_floor > 0 and scores[0] < calibrated_floor:
        reasons.append("low-rerank-score")
    if len(scores) > 1 and calibrated_margin > 0 and scores[0] - scores[1] < calibrated_margin:
        reasons.append("ambiguous-score-margin")
    return RiskDecision(bool(reasons), tuple(reasons), "upgrade" if reasons else "answer")


def answer_risk(answer: str, citations: list[dict], evidence: list[dict]) -> RiskDecision:
    reasons: list[str] = []
    markers = [int(n) for n in _CITATION.findall(answer or "")]
    if any(n < 1 or n > len(evidence) for n in markers):
        reasons.append("invalid-citation-index")
    if evidence and not markers and not _REFUSAL.search(answer or ""):
        reasons.append("factual-answer-without-citation")
    if markers and len(citations) != len(set(markers)):
        reasons.append("citation-mapping-incomplete")
    return RiskDecision(bool(reasons), tuple(reasons), "verify" if reasons else "accept")
