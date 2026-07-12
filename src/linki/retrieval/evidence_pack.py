"""Source-diverse, token-budgeted immutable evidence packs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from linki.graph.state import Evidence


def estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4)


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidencePack:
    evidence_pack_id: str
    token_budget: int
    token_count: int
    units: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit(raw: Evidence, supports: list[str]) -> dict[str, Any]:
    quote = raw.get("quote") or raw.get("text") or ""
    content_hash = raw.get("content_hash") or _hash(quote)
    evidence_id = raw.get("evidence_id") or _hash(
        f"{raw.get('source', '')}:{raw.get('parent_id', '')}:{raw.get('char_start', 0)}:{content_hash}"
    )[:20]
    return {
        **raw,
        "evidence_id": evidence_id,
        "source_id": raw.get("source_id") or raw.get("source") or "?",
        "source_version": raw.get("source_version") or "unknown",
        "quote": quote,
        "text": quote,
        "char_start": int(raw.get("char_start") or 0),
        "char_end": int(raw.get("char_end") or len(quote)),
        "retrieval_score": float(raw.get("retrieval_score", raw.get("score", 0.0)) or 0.0),
        "rerank_score": (
            float(raw["rerank_score"]) if raw.get("rerank_score") is not None else None
        ),
        "supports": sorted(set(raw.get("supports") or supports)),
        "token_count": estimate_tokens(quote),
        "content_hash": content_hash,
    }


def _rank(unit: dict[str, Any]) -> tuple[float, float]:
    rerank = unit.get("rerank_score")
    return (float(rerank) if rerank is not None else float(unit.get("score", 0.0)), float(unit.get("retrieval_score", 0.0)))


def _truncate(unit: dict[str, Any], remaining_tokens: int) -> dict[str, Any] | None:
    if remaining_tokens <= 0:
        return None
    quote = unit["quote"]
    max_chars = remaining_tokens * 4
    if not quote:
        return None
    truncated = quote[:max_chars]
    out = dict(unit)
    out["quote"] = truncated
    out["text"] = truncated
    out["char_end"] = int(out.get("char_start", 0)) + len(truncated)
    out["token_count"] = estimate_tokens(truncated)
    out["content_hash"] = _hash(truncated)
    out["evidence_id"] = _hash(
        f"{out.get('source_id')}:{out.get('parent_id')}:{out.get('char_start')}:{out['content_hash']}"
    )[:20]
    return out


def build_evidence_pack(
    evidence: list[Evidence],
    *,
    token_budget: int,
    supports: list[str] | None = None,
) -> EvidencePack:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    units = [_unit(item, supports or []) for item in evidence]
    # Remove exact content duplicates before diversity selection.
    deduped: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for unit in sorted(units, key=_rank, reverse=True):
        if unit["content_hash"] in seen_hashes:
            continue
        seen_hashes.add(unit["content_hash"])
        deduped.append(unit)

    # First pass maximizes source and sub-query coverage, second pass fills by
    # marginal score. This ordering is deterministic for reproducible evals.
    coverage: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_supports: set[str] = set()
    for unit in deduped:
        source = str(unit.get("source_id"))
        unit_supports = set(unit.get("supports") or [])
        if source not in seen_sources or unit_supports - seen_supports:
            coverage.append(unit)
            seen_sources.add(source)
            seen_supports |= unit_supports
    remainder = [unit for unit in deduped if unit not in coverage]

    selected: list[dict[str, Any]] = []
    used = 0
    # Give every coverage unit a fair share before a long top-ranked span can
    # consume the whole pack. Unused shares naturally flow into the fill pass.
    for index, unit in enumerate(coverage):
        remaining = token_budget - used
        if remaining <= 0:
            break
        remaining_coverage = len(coverage) - index
        fair_share = max(1, remaining // remaining_coverage)
        allowance = min(remaining, max(fair_share, min(int(unit["token_count"]), fair_share)))
        candidate = unit if unit["token_count"] <= allowance else _truncate(unit, allowance)
        if candidate is None:
            continue
        selected.append(candidate)
        used += int(candidate["token_count"])

    for unit in remainder:
        remaining = token_budget - used
        if remaining <= 0:
            break
        candidate = unit if unit["token_count"] <= remaining else _truncate(unit, remaining)
        if candidate is None:
            continue
        selected.append(candidate)
        used += int(candidate["token_count"])

    identity = json.dumps(
        {"budget": token_budget, "units": [(u["evidence_id"], u["content_hash"]) for u in selected]},
        ensure_ascii=False,
        sort_keys=True,
    )
    pack_id = _hash(identity)[:24]
    return EvidencePack(pack_id, token_budget, used, tuple(selected))
