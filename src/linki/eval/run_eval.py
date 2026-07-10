"""Reproducible naive-vs-agentic evaluation with deterministic and judged metrics.

The baseline and Linki use the same retriever, model, grounding rule, refusal
policy, and citation format. The only intended difference is orchestration.
Strict mode also runs a no-reflow ablation (one retrieval round and one verifier
attempt) so gains can be attributed to Linki's correction loops.
"""

from __future__ import annotations

import copy
import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

from linki.core.jsonutil import extract_json

EVAL_JUDGE_PROMPT = (
    "You are an independent strict RAG evaluation judge. Given a question, a "
    "reference answer, a candidate answer, and the retrieved evidence, rate the "
    "candidate. Reply with ONLY JSON: {\"faithfulness\": <1-5, every claim is "
    "supported by evidence>, \"quality\": <1-5, correct and complete versus the "
    "reference answer>}. An honest refusal is correct only when the reference "
    "says the information is insufficient."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFUSAL_MARKERS = (
    "未找到", "没有找到", "无法找到", "知识库中没有", "知识库中未",
    "not found", "cannot find", "couldn't find", "don't have", "do not have",
    "no relevant", "not covered", "insufficient", "cannot answer",
    "do not contain", "does not contain", "未包含",
)


class _TrackedModel:
    """Small invoke proxy that records comparable call/token telemetry."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def invoke(self, messages):
        self.calls += 1
        response = self.delegate.invoke(messages)
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
        self.input_tokens += int(usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or token_usage.get("completion_tokens") or 0)
        return response

    def telemetry(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }


def _did_refuse(answer: str) -> bool:
    # Gap disclosure later in an otherwise useful answer is not a full refusal.
    # Classify from the first substantive sentence, where all supported Linki
    # refusal templates state that the answer is unavailable.
    text = re.sub(r"^[\s#>*_-]+", "", answer or "").strip()
    first_sentence = re.split(r"(?<=[.!?。！？])\s+|\n\s*\n", text, maxsplit=1)[0].lower()
    return any(marker.lower() in first_sentence for marker in _REFUSAL_MARKERS)


def _normalize(text: str) -> str:
    text = _CITATION_RE.sub(" ", (text or "").lower())
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return " ".join(text.translate(table).split())


def token_f1(answer: str, gold: str | None) -> float | None:
    if not gold:
        return None
    pred_tokens = _normalize(answer).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


def retrieval_metrics(evidence: list[dict], gold: list[str] | None) -> dict[str, float | bool | None]:
    if not gold:
        return {"retrieval_recall": None, "retrieval_precision": None, "all_support_recall": None}
    gold_set = set(gold)
    got = [e.get("source") for e in (evidence or []) if e.get("source")]
    got_set = set(got)
    hits = len(gold_set & got_set)
    return {
        "retrieval_recall": round(hits / len(gold_set), 3),
        "retrieval_precision": round(hits / len(got_set), 3) if got_set else 0.0,
        "all_support_recall": gold_set <= got_set,
    }


def score_answer(
    judge: Any,
    question: str,
    answer: str,
    evidence: list[dict],
    expect: dict[str, Any],
) -> dict[str, Any]:
    """Score one answer using gold-aware, structural, and retrieval metrics."""
    from langchain_core.messages import HumanMessage, SystemMessage

    rendered = "\n\n".join(
        f"[{i}] ({hit.get('source', '?')}) {hit.get('text', '').strip()}"
        for i, hit in enumerate(evidence or [], start=1)
    ) or "(none)"
    gold_answer = str(expect.get("gold_answer") or expect.get("gold") or "")
    raw = judge.invoke([
        SystemMessage(content=EVAL_JUDGE_PROMPT),
        HumanMessage(content=(
            f"Question: {question}\n\nReference answer: {gold_answer or '(not supplied)'}"
            f"\n\nCandidate answer: {answer}\n\nEvidence:\n{rendered}"
        )),
    ]).content
    verdict = extract_json(raw, fallback={"faithfulness": 0, "quality": 0})

    cited = [int(value) for value in _CITATION_RE.findall(answer or "")]
    valid = [index for index in cited if 1 <= index <= len(evidence or [])]
    did_refuse = _did_refuse(answer)
    should_refuse = bool(expect.get("should_refuse"))
    normalized_gold = _normalize(gold_answer)
    normalized_answer = _normalize(answer)

    result = {
        "faithfulness": float(verdict.get("faithfulness", 0) or 0),
        "quality": float(verdict.get("quality", 0) or 0),
        "has_citations": bool(cited),
        "citation_validity": round(len(valid) / len(cited), 3) if cited else (1.0 if did_refuse else 0.0),
        "did_refuse": did_refuse,
        "refusal_correct": did_refuse == should_refuse,
        "answer_contains_gold": bool(normalized_gold and normalized_gold in normalized_answer),
        "answer_token_f1": token_f1(answer, gold_answer),
    }
    result.update(retrieval_metrics(evidence, expect.get("expect_sources")))
    return result


def retrieval_recall(evidence: list[dict], gold: list[str] | None) -> float | None:
    """Backward-compatible helper retained for callers and unit tests."""
    return retrieval_metrics(evidence, gold)["retrieval_recall"]


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _variant_settings(settings: Any, *, no_reflow: bool) -> Any:
    out = copy.copy(settings)
    if no_reflow:
        object.__setattr__(out, "max_rounds", 1)
        object.__setattr__(out, "max_attempts", 1)
    return out


def _run_agentic(q, *, model, judge, settings, retrieve_fn, app=None):
    from linki.graph.workflow import answer_question

    main_tracker, judge_tracker = _TrackedModel(model), _TrackedModel(judge)
    started = time.perf_counter()
    state = answer_question(
        q, model=main_tracker, judge=judge_tracker, settings=settings,
        retrieve_fn=retrieve_fn, app=app,
    )
    elapsed = time.perf_counter() - started
    telemetry = main_tracker.telemetry()
    judge_usage = judge_tracker.telemetry()
    for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens"):
        telemetry[key] += judge_usage[key]
    telemetry["latency_seconds"] = round(elapsed, 3)
    return state.get("final_answer") or state.get("answer") or "", state.get("evidence") or [], telemetry


def _run_naive(q, *, model, settings, retrieve_fn):
    from linki.eval.naive import naive_answer

    tracker = _TrackedModel(model)
    started = time.perf_counter()
    result = naive_answer(q, model=tracker, settings=settings, retrieve_fn=retrieve_fn)
    telemetry = tracker.telemetry()
    telemetry["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result["answer"], result["evidence"], telemetry


def run_eval(
    dataset: list[dict], *, model: Any, judge: Any, settings: Any,
    retrieve_fn: Any, app=None, strict: bool = False, progress=None,
) -> dict[str, Any]:
    """Run a fair baseline, Linki, and optionally a no-reflow ablation."""
    systems = ["naive", "agentic_no_reflow", "agentic"] if strict else ["naive", "agentic"]
    items: list[dict[str, Any]] = []

    for item in dataset:
        q, expect = item["question"], item.get("expect", {})
        row: dict[str, Any] = {"id": item.get("id"), "type": item.get("type", "single"), "status": "ok"}
        outputs: dict[str, tuple[str, list[dict], dict[str, Any]]] = {}
        outputs["naive"] = _run_naive(q, model=model, settings=settings, retrieve_fn=retrieve_fn)
        if strict:
            outputs["agentic_no_reflow"] = _run_agentic(
                q, model=model, judge=judge,
                settings=_variant_settings(settings, no_reflow=True),
                retrieve_fn=retrieve_fn, app=None,
            )
        outputs["agentic"] = _run_agentic(
            q, model=model, judge=judge, settings=settings,
            retrieve_fn=retrieve_fn, app=app,
        )

        for name, (answer, evidence, telemetry) in outputs.items():
            row[name] = score_answer(judge, q, answer, evidence, expect)
            row[name]["telemetry"] = telemetry
            row[name]["answer"] = answer
            row[name]["retrieved_sources"] = [hit.get("source") for hit in evidence]
        items.append(row)
        if progress:
            progress(f"evaluated {len(items)}/{len(dataset)} questions")

    return {"systems": systems, "items": items, "aggregate": _aggregate(items, systems)}


_METRICS = (
    "faithfulness", "quality", "has_citations", "citation_validity",
    "refusal_correct", "answer_contains_gold", "answer_token_f1",
    "retrieval_recall", "retrieval_precision", "all_support_recall",
)
_TELEMETRY = ("latency_seconds", "llm_calls", "input_tokens", "output_tokens", "total_tokens")


def _aggregate(items: list[dict], systems: list[str]) -> dict[str, dict[str, float]]:
    scored = [item for item in items if item.get("status") == "ok"]
    out: dict[str, dict[str, float]] = {name: {} for name in systems}
    for name in systems:
        for metric in _METRICS:
            vals = [float(item[name][metric]) for item in scored if item[name].get(metric) is not None]
            if vals:
                out[name][metric] = round(sum(vals) / len(vals), 3)
        for metric in _TELEMETRY:
            vals = [float(item[name]["telemetry"][metric]) for item in scored]
            if vals:
                out[name][metric] = round(sum(vals) / len(vals), 3)
    return out


def format_report(report: dict[str, Any]) -> str:
    systems = report.get("systems") or ["naive", "agentic"]
    lines = ["# RAG benchmark", "", "Systems: " + " · ".join(systems), ""]
    for name in systems:
        agg = report.get("aggregate", {}).get(name, {})
        lines.append(f"## {name}")
        for metric in _METRICS + _TELEMETRY:
            if metric in agg:
                lines.append(f"- {metric}: {agg[metric]}")
        lines.append("")
    return "\n".join(lines).rstrip()
