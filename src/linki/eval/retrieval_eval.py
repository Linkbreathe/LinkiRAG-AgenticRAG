"""Deterministic retrieval-only benchmark (no LLM or judge required)."""

from __future__ import annotations

import time
import math
from collections import defaultdict
from typing import Any

from linki.eval.run_eval import retrieval_metrics
from linki.retrieval.evidence_pack import estimate_tokens


def ranking_metrics(evidence: list[dict], gold: list[str] | None) -> dict[str, float | None]:
    if not gold:
        return {"mrr": None, "ndcg": None}
    gold_set = set(gold)
    ranked_sources: list[str] = []
    for item in evidence:
        source = item.get("source")
        if source and source not in ranked_sources:
            ranked_sources.append(source)
    relevant_ranks = [rank for rank, source in enumerate(ranked_sources, start=1) if source in gold_set]
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_n = min(len(gold_set), len(ranked_sources))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    return {"mrr": round(mrr, 4), "ndcg": round(dcg / idcg, 4) if idcg else 0.0}


def run_retrieval_eval(
    dataset: list[dict[str, Any]], retrieve_fn, kb: str, *, progress=None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(dataset, start=1):
        started = time.perf_counter()
        evidence = retrieve_fn(row["question"], kb) or []
        elapsed = time.perf_counter() - started
        metrics = retrieval_metrics(evidence, row.get("expect", {}).get("expect_sources"))
        metrics.update(ranking_metrics(evidence, row.get("expect", {}).get("expect_sources")))
        sources = [item.get("source") for item in evidence if item.get("source")]
        offset_flags = [item.get("offset_valid") for item in evidence if "offset_valid" in item]
        result = {
            "id": row.get("id"),
            "question_type": row.get("question_type", row.get("type", "unknown")),
            "latency_seconds": round(elapsed, 4),
            "n_retrieved": len(evidence),
            "retrieved_sources": [item.get("source") for item in evidence],
            "source_diversity": round(len(set(sources)) / len(sources), 4) if sources else 0.0,
            "evidence_tokens": sum(
                int(item.get("token_count") or estimate_tokens(item.get("text", "")))
                for item in evidence
            ),
            "offset_validity": (
                sum(bool(flag) for flag in offset_flags) / len(offset_flags) if offset_flags else None
            ),
            **metrics,
        }
        items.append(result)
        by_type[result["question_type"]].append(result)
        if progress and (index % 100 == 0 or index == len(dataset)):
            progress(f"retrieved {index}/{len(dataset)} questions")

    def aggregate(rows):
        answerable = [row for row in rows if row["retrieval_recall"] is not None]
        ranked = [row for row in rows if row["mrr"] is not None]
        offsets = [row["offset_validity"] for row in rows if row["offset_validity"] is not None]
        latencies = sorted(row["latency_seconds"] for row in rows)
        out = {
            "n": len(rows),
            "answerable_n": len(answerable),
            "retrieval_recall": round(sum(row["retrieval_recall"] for row in answerable) / len(answerable), 4) if answerable else None,
            "retrieval_precision": round(sum(row["retrieval_precision"] for row in answerable) / len(answerable), 4) if answerable else None,
            "all_support_recall": round(sum(bool(row["all_support_recall"]) for row in answerable) / len(answerable), 4) if answerable else None,
            "mrr": round(sum(row["mrr"] for row in ranked) / len(ranked), 4) if ranked else None,
            "ndcg": round(sum(row["ndcg"] for row in ranked) / len(ranked), 4) if ranked else None,
            "source_diversity": round(sum(row["source_diversity"] for row in rows) / len(rows), 4) if rows else 0.0,
            "mean_evidence_tokens": round(sum(row["evidence_tokens"] for row in rows) / len(rows), 2) if rows else 0.0,
            "offset_validity": round(sum(offsets) / len(offsets), 4) if offsets else None,
            "mean_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / len(rows), 4) if rows else 0.0,
            "p95_latency_seconds": latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else 0.0,
        }
        return out

    return {
        "aggregate": aggregate(items),
        "by_question_type": {name: aggregate(rows) for name, rows in sorted(by_type.items())},
        "items": items,
    }
