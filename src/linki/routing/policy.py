"""Local, explainable P0-P3 execution policy.

The policy deliberately runs before any model call.  Rules are conservative:
uncertainty upgrades a request, while an explicit user mode may only keep or
raise the amount of work.  This makes the decision cheap, inspectable and easy
to calibrate from benchmark traces.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

ExecutionMode = Literal["auto", "fast", "balanced", "deep"]
PolicyPath = Literal["p0", "p1", "p2", "p3", "legacy"]


@dataclass(frozen=True)
class PathBudget:
    max_model_calls: int
    max_input_tokens: int
    max_rounds: int
    evidence_tokens: int
    deadline_ms: int


PATH_BUDGETS: dict[str, PathBudget] = {
    "p0": PathBudget(1, 2_000, 0, 0, 5_000),
    "p1": PathBudget(1, 8_000, 1, 1_200, 12_000),
    "p2": PathBudget(3, 20_000, 1, 2_400, 25_000),
    "p3": PathBudget(7, 50_000, 2, 4_000, 60_000),
    # Compatibility path used by old callers and ablation tests.
    "legacy": PathBudget(100, 1_000_000, 4, 8_000, 120_000),
}


@dataclass(frozen=True)
class PolicyDecision:
    path: PolicyPath
    mode: ExecutionMode
    reason: str
    signals: tuple[str, ...]
    confidence: float
    needs_rewrite: bool
    local_response: str | None
    budget: PathBudget

    def as_dict(self) -> dict:
        return asdict(self)


_GREETINGS = {
    "hi": "Hi! Ask me a question about your knowledge base.",
    "hello": "Hello! Ask me a question about your knowledge base.",
    "hey": "Hi! Ask me a question about your knowledge base.",
    "你好": "你好！可以直接问我知识库中的问题。",
    "您好": "您好！可以直接问我知识库中的问题。",
    "嗨": "你好！可以直接问我知识库中的问题。",
    "谢谢": "不客气。",
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
}
_HIGH_RISK = re.compile(
    r"\b(medical|diagnos(?:e|is)|legal advice|lawsuit|investment advice|"
    r"prescription|dosage|strict(?:ly)? verify|double[- ]check)\b|"
    r"医疗|诊断|法律意见|诉讼|投资建议|处方|剂量|严格核验|仔细核验",
    re.I,
)
_COMPARE = re.compile(
    r"\b(compare|versus|vs\.?|difference between|both|respectively|relationship between)\b|"
    r"比较|对比|区别|分别|两者|关系",
    re.I,
)
_MULTIHOP = re.compile(
    r"\b(why.*then|how.*affect|based on.*what|before and after|timeline|across)\b|"
    r"为什么.*又|如何.*影响|基于.*什么|前后|时间线|跨库|多跳",
    re.I | re.S,
)
_UNRESOLVED = re.compile(
    r"^(it|that|this|they|he|she|those|these)\b|^(它|这|那|他们|她|他|上述|前者|后者)",
    re.I,
)


def _clean(question: str) -> str:
    return re.sub(r"[\s.!?。！？,，]+", " ", question or "").strip().lower()


def decide_policy(
    question: str,
    *,
    mode: ExecutionMode | str = "auto",
    session_context: str = "",
    kb_count: int = 1,
    deadline_ms: int | None = None,
) -> PolicyDecision:
    """Choose the cheapest safe path without invoking an LLM."""

    if mode not in {"auto", "fast", "balanced", "deep"}:
        raise ValueError("mode must be one of: auto, fast, balanced, deep")
    mode = mode  # narrow for type checkers
    normalized = _clean(question)
    signals: list[str] = []
    local_response = _GREETINGS.get(normalized)
    needs_rewrite = bool(session_context.strip() and _UNRESOLVED.search(normalized))

    if local_response is not None:
        path, reason, confidence = "p0", "exact local greeting/acknowledgement", 1.0
        signals.append("exact-local-intent")
    elif _HIGH_RISK.search(question or ""):
        path, reason, confidence = "p3", "high-risk or explicit verification request", 0.95
        signals.append("high-risk")
    elif needs_rewrite:
        path, reason, confidence = "p3", "unresolved reference requires history-aware reasoning", 0.85
        signals.append("unresolved-reference")
    elif mode == "deep":
        path, reason, confidence = "p3", "user forced deep mode", 1.0
        signals.append("forced:deep")
    elif _COMPARE.search(question or "") or _MULTIHOP.search(question or ""):
        path, reason, confidence = "p2", "comparison or multi-hop composition detected", 0.86
        signals.append("multi-facet")
    elif mode == "balanced":
        path, reason, confidence = "p2", "user requested at least balanced execution", 1.0
        signals.append("forced:balanced")
    elif (question or "").count("?") + (question or "").count("？") > 1:
        path, reason, confidence = "p2", "multiple explicit questions detected", 0.82
        signals.append("multiple-questions")
    elif kb_count > 1 and re.search(r"\b(across|all topics|multiple sources)\b|所有知识库|多个主题", question, re.I):
        path, reason, confidence = "p2", "cross-knowledge-base request detected", 0.88
        signals.append("cross-kb")
    else:
        path = "p1"
        if mode == "fast":
            reason, confidence = "user requested fast execution", 1.0
            signals.append("forced:fast")
        else:
            reason, confidence = "single-intent factual request", 0.78
            signals.append("single-intent")

    base = PATH_BUDGETS[path]
    budget = base
    if deadline_ms is not None:
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        budget = PathBudget(
            max_model_calls=base.max_model_calls,
            max_input_tokens=base.max_input_tokens,
            max_rounds=base.max_rounds,
            evidence_tokens=base.evidence_tokens,
            deadline_ms=min(base.deadline_ms, int(deadline_ms)),
        )
    return PolicyDecision(
        path=path,
        mode=mode,
        reason=reason,
        signals=tuple(signals),
        confidence=confidence,
        needs_rewrite=needs_rewrite,
        local_response=local_response,
        budget=budget,
    )


def legacy_policy() -> PolicyDecision:
    return PolicyDecision(
        path="legacy",
        mode="auto",
        reason="adaptive execution disabled for compatibility",
        signals=("legacy",),
        confidence=1.0,
        needs_rewrite=True,
        local_response=None,
        budget=PATH_BUDGETS["legacy"],
    )
