"""Namespace-first memory recall with a hard injection token budget."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from linki.memory.ledger import MemoryItem, MemoryLedger
from linki.retrieval.evidence_pack import estimate_tokens


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+|[\u4e00-\u9fff]", (text or "").casefold()))


class MemoryRetriever:
    def __init__(self, ledger: MemoryLedger):
        self.ledger = ledger

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        point = (now or datetime.now(UTC)).astimezone(UTC)
        query_terms = _terms(query)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.ledger.list_current(
            tenant_id=tenant_id, user_id=user_id, statuses=("ACTIVE",),
        ):
            if item.valid_from and datetime.fromisoformat(item.valid_from) > point:
                continue
            if item.valid_to and datetime.fromisoformat(item.valid_to) <= point:
                continue
            rendered = " ".join(str(value) for value in item.content.values())
            memory_terms = _terms(rendered)
            relevance = len(query_terms & memory_terms) / max(len(query_terms), 1)
            if item.type == "profile":
                relevance = max(relevance, 0.12)
            if relevance <= 0:
                continue
            age_days = max(0.0, (point - datetime.fromisoformat(item.ingested_at)).total_seconds() / 86400)
            recency = 1.0 / (1.0 + age_days / 30.0)
            access_strength = math.log1p(item.access_count) / 10.0
            score = relevance + item.importance * 0.15 + recency * 0.05 + access_strength
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].ingested_at), reverse=True)
        selected = [
            {**item.as_dict(), "recall_score": round(score, 4)}
            for score, item in scored[:limit]
        ]
        self.ledger.mark_accessed([item["memory_id"] for item in selected])
        return selected


def render_memory_context(memories: list[dict[str, Any]], token_budget: int) -> str:
    if token_budget <= 0 or not memories:
        return ""
    lines: list[str] = []
    used = 0
    for item in memories:
        content = item.get("content") or {}
        if item.get("type") == "profile":
            text = f"[memory:{item['memory_id']}] User preference: {content.get('value', '')}"
        else:
            text = f"[memory:{item['memory_id']}] Prior task outcome: {content.get('outcome', '')}"
        tokens = estimate_tokens(text)
        if used + tokens > token_budget:
            remaining = token_budget - used
            if remaining <= 0:
                break
            text = text[:remaining * 4]
            tokens = estimate_tokens(text)
        lines.append(text)
        used += tokens
    return "\n".join(lines)
