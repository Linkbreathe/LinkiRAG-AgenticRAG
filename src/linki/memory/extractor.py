"""Deterministic first-pass candidate extraction; assistant text is never fact."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from linki.memory.ledger import Episode

EXTRACTOR_VERSION = "rules.v1"


@dataclass(frozen=True)
class MemoryCandidate:
    type: str
    scope: str
    namespace: tuple[str, ...]
    content: dict[str, Any]
    source_spans: tuple[dict[str, Any], ...]
    confidence: float
    importance: float
    explicit: bool


_PREFERENCE_PATTERNS = (
    re.compile(r"(?:请)?记住[：:,， ]*(.+)", re.I),
    re.compile(r"(?:以后|今后)(.+?(?:优先|使用|采用|回答|示例).+)", re.I),
    re.compile(r"\bremember(?: that)?\s+(.+)", re.I),
    re.compile(r"\bi (?:prefer|always want|would like)\s+(.+)", re.I),
    re.compile(r"\bplease (?:always|prefer)\s+(.+)", re.I),
)
_FAILURE = re.compile(r"失败|报错|error|failed|exception", re.I)
_FIX = re.compile(r"修复|解决|fixed|resolved|workaround", re.I)
_INFERRED_PREFERENCE = re.compile(
    r"\b(?:use|provide|show)\s+(python|typescript|javascript|java)\s+(?:code|examples?)\b|"
    r"(?:用|使用)\s*(Python|TypeScript|JavaScript|Java)\s*(?:写|代码|示例)",
    re.I,
)


def preference_key(statement: str) -> str:
    low = statement.casefold()
    categories = {
        "code_language": ("python", "javascript", "typescript", "java", "代码", "示例"),
        "response_length": ("简洁", "详细", "short", "concise", "detailed", "verbose"),
        "language": ("中文", "英文", "chinese", "english"),
        "format": ("markdown", "表格", "列表", "table", "bullet"),
    }
    for key, markers in categories.items():
        if any(marker in low for marker in markers):
            return key
    return "preference:" + hashlib.sha256(low.encode("utf-8")).hexdigest()[:12]


def extract_candidates(episode: Episode) -> list[MemoryCandidate]:
    """Extract only from the user's original words; never from assistant output."""
    text = (episode.user_input or "").strip()
    if not text:
        return []
    candidates: list[MemoryCandidate] = []
    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        statement = match.group(1).strip(" 。.!！")
        if not statement:
            continue
        candidates.append(MemoryCandidate(
            type="profile", scope="user", namespace=(episode.tenant_id, episode.user_id, "profile"),
            content={"kind": "preference", "key": preference_key(statement), "value": statement,
                     "original": match.group(0)},
            source_spans=({"episode_id": episode.episode_id, "field": "user_input",
                           "char_start": match.start(), "char_end": match.end(),
                           "quote": match.group(0)},),
            confidence=0.98, importance=0.8, explicit=True,
        ))
        break

    if not candidates:
        inferred = _INFERRED_PREFERENCE.search(text)
        if inferred:
            statement = inferred.group(0)
            candidates.append(MemoryCandidate(
                type="profile", scope="user",
                namespace=(episode.tenant_id, episode.user_id, "profile"),
                content={"kind": "preference", "key": "code_language",
                         "value": statement, "original": statement},
                source_spans=({"episode_id": episode.episode_id, "field": "user_input",
                               "char_start": inferred.start(), "char_end": inferred.end(),
                               "quote": statement},),
                confidence=0.62, importance=0.5, explicit=False,
            ))

    # Episodic memories are narrow: a turn must contain both a failure signal
    # and an explicit resolution. They expire through the ledger retention rule.
    combined = f"{episode.user_input}\n{episode.assistant_output}"
    if _FAILURE.search(combined) and _FIX.search(combined):
        candidates.append(MemoryCandidate(
            type="episodic", scope="user", namespace=(episode.tenant_id, episode.user_id, "episodes"),
            content={"kind": "task_outcome", "key": f"episode:{episode.episode_id}",
                     "user_problem": episode.user_input[:800],
                     "outcome": episode.assistant_output[:1200]},
            source_spans=({"episode_id": episode.episode_id, "field": "turn", "char_start": 0,
                           "char_end": len(combined), "quote": combined[:2000]},),
            confidence=0.75, importance=0.55, explicit=False,
        ))
    return candidates
