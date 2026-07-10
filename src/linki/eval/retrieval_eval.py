"""Deterministic retrieval-only benchmark (no LLM or judge required)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from linki.eval.run_eval import retrieval_metrics


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
        result = {
            "id": row.get("id"),
            "question_type": row.get("question_type", row.get("type", "unknown")),
            "latency_seconds": round(elapsed, 4),
            "n_retrieved": len(evidence),
            "retrieved_sources": [item.get("source") for item in evidence],
            **metrics,
        }
        items.append(result)
        by_type[result["question_type"]].append(result)
        if progress and (index % 100 == 0 or index == len(dataset)):
            progress(f"retrieved {index}/{len(dataset)} questions")

    def aggregate(rows):
        answerable = [row for row in rows if row["retrieval_recall"] is not None]
        return {
            "n": len(rows),
            "answerable_n": len(answerable),
            "retrieval_recall": round(sum(row["retrieval_recall"] for row in answerable) / len(answerable), 4) if answerable else None,
            "retrieval_precision": round(sum(row["retrieval_precision"] for row in answerable) / len(answerable), 4) if answerable else None,
            "all_support_recall": round(sum(bool(row["all_support_recall"]) for row in answerable) / len(answerable), 4) if answerable else None,
            "mean_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / len(rows), 4) if rows else 0.0,
        }

    return {
        "aggregate": aggregate(items),
        "by_question_type": {name: aggregate(rows) for name, rows in sorted(by_type.items())},
        "items": items,
    }
