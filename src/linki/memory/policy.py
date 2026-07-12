"""Promotion policy: extraction confidence is never treated as truth."""

from __future__ import annotations

import re
from dataclasses import dataclass

from linki.memory.extractor import MemoryCandidate

_SENSITIVE = re.compile(
    r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b|\b(?:\+?\d[\d -]{8,}\d)\b|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"password|secret|api[_ -]?key|token|身份证|手机号|密码|密钥|病历|诊断",
    re.I,
)
_INJECTION = re.compile(
    r"ignore (?:all |the )?(?:previous|system)|override (?:system|policy)|"
    r"忽略(?:之前|系统)|覆盖(?:系统|策略)",
    re.I,
)


@dataclass(frozen=True)
class MemoryPolicyVerdict:
    status: str
    reason: str
    pii_class: str | None = None


def evaluate_candidate(candidate: MemoryCandidate, *, trusted_organization_source: bool = False) -> MemoryPolicyVerdict:
    rendered = str(candidate.content)
    if _INJECTION.search(rendered):
        return MemoryPolicyVerdict("REJECTED", "prompt/control instructions cannot become memory")
    if _SENSITIVE.search(rendered):
        return MemoryPolicyVerdict("REVIEW_REQUIRED", "potentially sensitive personal data", "sensitive")
    if candidate.type == "procedural":
        return MemoryPolicyVerdict("REVIEW_REQUIRED", "procedural changes require offline release review")
    if candidate.scope in {"team", "organization", "agent"} and not trusted_organization_source:
        return MemoryPolicyVerdict("REVIEW_REQUIRED", "non-user scope requires trusted evidence or approval")
    if candidate.type == "profile" and candidate.scope == "user":
        if candidate.explicit:
            return MemoryPolicyVerdict("ACTIVE", "explicit user-scoped preference")
        return MemoryPolicyVerdict("PROPOSED", "inferred preference awaits repetition or confirmation")
    if candidate.type == "episodic":
        return MemoryPolicyVerdict("ACTIVE", "bounded user episode with source and retention")
    return MemoryPolicyVerdict("PROPOSED", "candidate awaits confirmation")
