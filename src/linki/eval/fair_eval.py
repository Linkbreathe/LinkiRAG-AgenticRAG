"""Checkpointed, counterbalanced adaptive-RAG end-to-end evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from linki.eval.run_eval import score_answer

SYSTEMS = ("fair_single_pass", "legacy_full", "adaptive_auto", "adaptive_deep")
QUALITY_METRICS = (
    "faithfulness", "quality", "has_citations", "citation_validity",
    "refusal_correct", "answer_contains_gold", "answer_token_f1",
    "retrieval_recall", "retrieval_precision", "all_support_recall",
)
TELEMETRY_METRICS = (
    "latency_seconds", "llm_calls", "input_tokens", "output_tokens", "total_tokens",
)


def _model_identifier(model: Any) -> str:
    for field in ("model_name", "model"):
        value = getattr(model, field, None)
        if value:
            return str(value)
    delegate = getattr(model, "delegate", None)
    return _model_identifier(delegate) if delegate is not None else model.__class__.__name__


def freeze_eval_settings(settings):
    """Disable stateful/runtime side effects that would make systems incomparable."""
    return replace(
        settings,
        enable_cache=False,
        enable_persistent_cache=False,
        enable_answer_cache=False,
        enable_semantic_cache=False,
        enable_memory=False,
        enable_feedback_ledger=False,
        enable_trace=False,
        enable_telemetry_persistence=False,
        enable_hook_trace=False,
    )


class UsageTracker:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.provider_usage_calls = 0

    def invoke(self, messages):
        self.calls += 1
        response = self.delegate.invoke(messages)
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        input_tokens = usage.get("input_tokens") or token_usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or token_usage.get("completion_tokens")
        if input_tokens is not None or output_tokens is not None:
            self.provider_usage_calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
        else:
            prompt = "\n".join(str(getattr(item, "content", item)) for item in messages)
            content = str(getattr(response, "content", response) or "")
            self.input_tokens += max(1, math.ceil(len(prompt) / 4))
            self.output_tokens += max(1, math.ceil(len(content) / 4))
        return response

    def telemetry(self, elapsed: float) -> dict[str, Any]:
        return {
            "latency_seconds": round(elapsed, 4),
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "provider_usage_fraction": round(self.provider_usage_calls / self.calls, 4) if self.calls else 1.0,
            "policy_path": "single_pass",
            "reflowed": False,
        }


def _single_pass(question, *, model, settings, retrieve_fn):
    from linki.eval.naive import naive_answer

    tracker = UsageTracker(model)
    started = time.perf_counter()
    result = naive_answer(
        question, model=tracker, settings=settings, retrieve_fn=retrieve_fn,
        token_budget=settings.evidence_budget_balanced,
    )
    return result["answer"], result["evidence"], tracker.telemetry(time.perf_counter() - started)


def _graph_run(
    question,
    *,
    model,
    judge,
    settings,
    retrieve_fn,
    mode,
    adaptive,
    run_id,
    app,
):
    from linki.graph.workflow import answer_question

    variant = replace(settings, adaptive_enabled=adaptive)
    started = time.perf_counter()
    state = answer_question(
        question, model=model, judge=judge, settings=variant,
        retrieve_fn=retrieve_fn, execution_mode=mode, run_id=run_id, app=app,
    )
    elapsed = time.perf_counter() - started
    cost = state.get("cost") or {}
    telemetry = {
        "latency_seconds": round(elapsed, 4),
        "llm_calls": int(cost.get("llm_calls", 0)),
        "input_tokens": int(cost.get("input_tokens", 0)),
        "output_tokens": int(cost.get("output_tokens", 0)),
        "total_tokens": int(cost.get("total_tokens", 0)),
        "provider_usage_fraction": (
            round(sum(row.get("usage_source") == "provider" for row in cost.get("node_costs", [])) / len(cost["node_costs"]), 4)
            if cost.get("node_costs") else 1.0
        ),
        "policy_path": state.get("policy_path", "legacy"),
        "reflowed": int(state.get("attempts", 0)) > 1,
        "verified": bool(state.get("verified")),
    }
    return (
        state.get("final_answer") or state.get("answer") or "",
        state.get("packed_evidence") or state.get("evidence") or [],
        telemetry,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def _bootstrap_ci(values: list[float], *, seed: int, samples: int = 1000) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return [round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4)]


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in SYSTEMS:
        rows = [item[system] for item in items if item.get(system, {}).get("status") == "ok"]
        aggregate_row: dict[str, Any] = {"n": len(rows)}
        confidence: dict[str, Any] = {}
        for metric in QUALITY_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                aggregate_row[metric] = round(statistics.fmean(values), 4)
                confidence[metric] = _bootstrap_ci(values, seed=20260711)
        for metric in TELEMETRY_METRICS:
            values = [float(row["telemetry"][metric]) for row in rows]
            if values:
                aggregate_row[metric] = round(statistics.fmean(values), 4)
        latencies = [float(row["telemetry"]["latency_seconds"]) for row in rows]
        token_values = [float(row["telemetry"]["total_tokens"]) for row in rows]
        total_calls = sum(int(row["telemetry"].get("llm_calls", 0)) for row in rows)
        provider_calls = sum(
            float(row["telemetry"].get("provider_usage_fraction", 0.0))
            * int(row["telemetry"].get("llm_calls", 0))
            for row in rows
        )
        provider_fraction = provider_calls / total_calls if total_calls else 1.0
        aggregate_row.update({
            "p50_latency_seconds": round(_percentile(latencies, 0.50), 4),
            "p95_latency_seconds": round(_percentile(latencies, 0.95), 4),
            "p50_total_tokens": round(_percentile(token_values, 0.50), 2),
            "p95_total_tokens": round(_percentile(token_values, 0.95), 2),
            "reflow_rate": round(statistics.fmean(float(row["telemetry"].get("reflowed", False)) for row in rows), 4) if rows else 0.0,
            "provider_usage_fraction": round(provider_fraction, 4),
            "token_usage_mode": (
                "provider" if provider_fraction == 1.0
                else "estimated" if provider_fraction == 0.0
                else "mixed"
            ),
            "path_distribution": {
                path: sum(row["telemetry"].get("policy_path") == path for row in rows)
                for path in sorted({str(row["telemetry"].get("policy_path")) for row in rows})
            },
            "confidence_95": confidence,
        })
        out[system] = aggregate_row
    return out


def paired_comparisons(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired candidate-minus-legacy deltas; negative is better for cost metrics."""
    comparisons: dict[str, Any] = {}
    metrics = QUALITY_METRICS + TELEMETRY_METRICS
    for system in ("fair_single_pass", "adaptive_auto", "adaptive_deep"):
        system_result: dict[str, Any] = {}
        pairs = [
            item for item in items
            if item.get(system, {}).get("status") == "ok"
            and item.get("legacy_full", {}).get("status") == "ok"
        ]
        for metric in metrics:
            deltas: list[float] = []
            for item in pairs:
                candidate = item[system]
                baseline = item["legacy_full"]
                if metric in TELEMETRY_METRICS:
                    candidate, baseline = candidate["telemetry"], baseline["telemetry"]
                if candidate.get(metric) is not None and baseline.get(metric) is not None:
                    deltas.append(float(candidate[metric]) - float(baseline[metric]))
            if deltas:
                system_result[metric] = {
                    "mean_delta": round(statistics.fmean(deltas), 4),
                    "confidence_95": _bootstrap_ci(deltas, seed=20260711),
                    "n": len(deltas),
                }
        comparisons[system] = system_result
    return comparisons


def _checkpoint(path: Path, protocol: dict[str, Any], items: list[dict[str, Any]], complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": protocol,
        "systems": list(SYSTEMS),
        "items": items,
        "aggregate": aggregate(items),
        "paired_vs_legacy_full": paired_comparisons(items),
        "complete": complete,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_fair_eval(
    dataset: list[dict[str, Any]],
    *,
    model: Any,
    judge: Any,
    settings: Any,
    retrieval: dict[str, Any],
    checkpoint_path: str | Path,
    corpus_snapshot_id: str,
    progress=None,
) -> dict[str, Any]:
    """Run four systems with per-question rotated order and durable resume."""
    from linki.graph.workflow import build_workflow

    settings = freeze_eval_settings(settings)
    path = Path(checkpoint_path)
    question_type_counts: dict[str, int] = {}
    for row in dataset:
        name = str(row.get("question_type", row.get("type", "unknown")))
        question_type_counts[name] = question_type_counts.get(name, 0) + 1
    protocol = {
        "name": "adaptive-rag-fair-v1",
        "corpus_snapshot_id": corpus_snapshot_id,
        "folds": sorted({(row.get("eval_fold"), row.get("eval_seed")) for row in dataset}),
        "rows": len(dataset),
        "question_type_counts": question_type_counts,
        "not_measured_by_multihop_rag": ["single", "multi_turn", "cross_kb"],
        "counterbalanced_order": True,
        "temperature": 0,
        "main_provider": getattr(settings, "provider", "unknown"),
        "main_model": _model_identifier(model),
        "judge_provider": getattr(settings, "judge_provider", "unknown"),
        "judge_model": _model_identifier(judge),
        "caches": "disabled_including_reranker_scores",
        "memory": "disabled",
        "feedback": "disabled",
        "judge_calls_in_system_telemetry": False,
        "single_pass_evidence_budget": settings.evidence_budget_balanced,
        "candidate_k": settings.candidate_k,
        "rerank_k": settings.rerank_k,
        "reranker_model": settings.reranker_model,
        "token_usage": "provider usage when returned; otherwise explicitly estimated",
    }
    experiment_material = {
        "protocol": protocol,
        "dataset": [
            {
                "id": row.get("id"), "question": row.get("question"),
                "expect": row.get("expect"), "fold": row.get("eval_fold"),
            }
            for row in dataset
        ],
        "systems": SYSTEMS,
    }
    protocol["experiment_id"] = hashlib.sha256(
        json.dumps(experiment_material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    existing_items: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_id = existing.get("protocol", {}).get("experiment_id")
            if existing_id != protocol["experiment_id"]:
                raise ValueError(
                    f"checkpoint protocol mismatch at {path}; archive or remove it before a new experiment"
                )
            existing_items = existing.get("items", [])
        except json.JSONDecodeError:
            existing_items = []
    completed = {
        item.get("id"): item
        for item in existing_items
        if all(item.get(system, {}).get("status") == "ok" for system in SYSTEMS)
    }
    items = [completed[row.get("id")] for row in dataset if row.get("id") in completed]
    app = build_workflow()
    for index, source_row in enumerate(dataset):
        item_id = source_row.get("id")
        if item_id in completed:
            continue
        question, expect = source_row["question"], source_row.get("expect", {})
        row: dict[str, Any] = {
            "id": item_id,
            "question_type": source_row.get("question_type", source_row.get("type", "unknown")),
            "eval_fold": source_row.get("eval_fold"),
            "eval_seed": source_row.get("eval_seed"),
        }
        order = list(SYSTEMS[index % len(SYSTEMS):] + SYSTEMS[:index % len(SYSTEMS)])
        outputs: dict[str, tuple[str, list[dict], dict[str, Any]]] = {}
        for system in order:
            try:
                if system == "fair_single_pass":
                    outputs[system] = _single_pass(
                        question, model=model, settings=settings,
                        retrieve_fn=retrieval["rerank_pack"],
                    )
                elif system == "legacy_full":
                    outputs[system] = _graph_run(
                        question, model=model, judge=judge, settings=settings,
                        retrieve_fn=retrieval["legacy_k5"], mode="auto", adaptive=False,
                        run_id=f"eval-{item_id}-legacy", app=app,
                    )
                elif system == "adaptive_auto":
                    outputs[system] = _graph_run(
                        question, model=model, judge=judge, settings=settings,
                        retrieve_fn=retrieval["rerank_pack"], mode="auto", adaptive=True,
                        run_id=f"eval-{item_id}-auto", app=app,
                    )
                else:
                    outputs[system] = _graph_run(
                        question, model=model, judge=judge, settings=settings,
                        retrieve_fn=retrieval["rerank_pack"], mode="deep", adaptive=True,
                        run_id=f"eval-{item_id}-deep", app=app,
                    )
            except Exception as exc:
                row[system] = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}

        for system, output in outputs.items():
            answer, evidence, telemetry = output
            try:
                scored = score_answer(judge, question, answer, evidence, expect)
                row[system] = {
                    "status": "ok",
                    **scored,
                    "telemetry": telemetry,
                    "answer": answer,
                    "retrieved_sources": [hit.get("source") for hit in evidence],
                }
            except Exception as exc:
                row[system] = {"status": "error", "error": f"judge:{exc.__class__.__name__}: {exc}"}
        items.append(row)
        _checkpoint(path, protocol, items, complete=False)
        if progress:
            done = sum(all(item.get(system, {}).get("status") == "ok" for system in SYSTEMS) for item in items)
            progress(f"fair-eval completed {done}/{len(dataset)} rows")

    complete = len(items) == len(dataset) and all(
        all(item.get(system, {}).get("status") == "ok" for system in SYSTEMS)
        for item in items
    )
    _checkpoint(path, protocol, items, complete=complete)
    return json.loads(path.read_text(encoding="utf-8"))


def format_fair_report(report: dict[str, Any]) -> str:
    lines = ["# Adaptive RAG fair evaluation", ""]
    for system in SYSTEMS:
        row = report.get("aggregate", {}).get(system, {})
        lines.extend([f"## {system}", f"- n: {row.get('n', 0)}"])
        for metric in QUALITY_METRICS + TELEMETRY_METRICS + (
            "p50_latency_seconds", "p95_latency_seconds", "p50_total_tokens",
            "p95_total_tokens", "reflow_rate",
        ):
            if metric in row:
                lines.append(f"- {metric}: {row[metric]}")
        lines.append(f"- path_distribution: {row.get('path_distribution', {})}")
        lines.append(f"- token_usage_mode: {row.get('token_usage_mode', 'unknown')}")
        lines.append("")
    return "\n".join(lines).rstrip()
