"""Repeated-run retrieval and answer-claim stability metrics."""

from __future__ import annotations

import re
import math
import statistics
import time
from itertools import combinations
from typing import Any


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def answer_claims(answer: str) -> set[str]:
    text = re.sub(r"\[\d+\]", "", answer or "").casefold()
    return {
        " ".join(sentence.split())
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if len(sentence.split()) >= 3
    }


def run_stability_eval(
    dataset: list[dict[str, Any]],
    retrieve_fn,
    kb: str,
    *,
    repeats: int = 3,
    answer_fn=None,
    progress=None,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("stability evaluation requires at least two repeats")
    items: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    for index, row in enumerate(dataset, start=1):
        rankings: list[list[str]] = []
        answers: list[str] = []
        latencies: list[float] = []
        errors = 0
        for _ in range(repeats):
            started = time.perf_counter()
            ranking: list[str] = []
            try:
                evidence = retrieve_fn(row["question"], kb) or []
                ranking = [
                    str(item.get("chunk_id") or item.get("source")) for item in evidence
                ]
                if answer_fn:
                    answers.append(str(answer_fn(row["question"], evidence)))
            except Exception:
                errors += 1
            rankings.append(ranking)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            all_latencies.append(elapsed)
        topk_scores = [
            jaccard(set(rankings[left]), set(rankings[right]))
            for left, right in combinations(range(repeats), 2)
        ]
        claim_scores = [
            jaccard(answer_claims(answers[left]), answer_claims(answers[right]))
            for left, right in combinations(range(len(answers)), 2)
        ] if answers else []
        items.append({
            "id": row.get("id"),
            "question_type": row.get("question_type", row.get("type", "unknown")),
            "topk_jaccard": statistics.fmean(topk_scores),
            "topk_exact_match": float(all(ranking == rankings[0] for ranking in rankings[1:])),
            "answer_claim_jaccard": statistics.fmean(claim_scores) if claim_scores else None,
            "latency_mean": statistics.fmean(latencies),
            "latency_variance": statistics.pvariance(latencies),
            "error_rate": errors / repeats,
        })
        if progress and (index % 25 == 0 or index == len(dataset)):
            progress(f"stability evaluated {index}/{len(dataset)} questions")
    return {
        "protocol": {
            "name": "linki-stability-v1", "repeats": repeats, "rows": len(items),
            "answer_claims_measured": answer_fn is not None,
        },
        "aggregate": {
            "topk_jaccard": statistics.fmean(item["topk_jaccard"] for item in items) if items else 1.0,
            "topk_exact_match_rate": statistics.fmean(item["topk_exact_match"] for item in items) if items else 1.0,
            "answer_claim_jaccard": statistics.fmean(
                item["answer_claim_jaccard"] for item in items if item["answer_claim_jaccard"] is not None
            ) if any(item["answer_claim_jaccard"] is not None for item in items) else None,
            "latency_variance": statistics.fmean(item["latency_variance"] for item in items) if items else 0.0,
            "p50_latency_seconds": sorted(all_latencies)[max(0, math.ceil(len(all_latencies) * 0.50) - 1)] if all_latencies else 0.0,
            "p95_latency_seconds": sorted(all_latencies)[max(0, math.ceil(len(all_latencies) * 0.95) - 1)] if all_latencies else 0.0,
            "error_rate": statistics.fmean(item["error_rate"] for item in items) if items else 0.0,
        },
        "items": items,
    }
