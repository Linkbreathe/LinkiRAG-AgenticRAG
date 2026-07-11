"""Run-scoped model-call telemetry with honest provider/estimate attribution."""

from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ModelBudgetExceeded(RuntimeError):
    """Raised before a non-critical call would consume the answer reserve."""


@dataclass
class NodeCost:
    run_id: str
    node: str
    call_index: int
    model: str
    prompt_version: str
    policy_path: str
    input_tokens: int
    cached_input_tokens: int | None
    output_tokens: int
    queue_ms: float | None
    ttft_ms: float | None
    total_ms: float
    cost_usd: float | None
    status: str
    usage_source: str
    error: str | None = None


def _message_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for message in messages or []:
        parts.append(str(getattr(message, "content", message)))
    return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    # Explicitly labelled estimate.  Mixed CJK/Latin text is close enough for
    # enforcing a safety budget; benchmark reports keep usage_source visible.
    return max(1, (len(text or "") + 3) // 4)


def _usage(response: Any, input_text: str) -> tuple[int, int, int | None, str]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
    input_tokens = usage.get("input_tokens") or token_usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or token_usage.get("completion_tokens")
    details = usage.get("input_token_details", {}) if isinstance(usage, dict) else {}
    cached = (
        details.get("cache_read")
        or token_usage.get("prompt_cache_hit_tokens")
        or token_usage.get("cached_tokens")
    )
    if input_tokens is not None or output_tokens is not None:
        return int(input_tokens or 0), int(output_tokens or 0), int(cached) if cached is not None else None, "provider"
    content = str(getattr(response, "content", response) or "")
    return _estimate_tokens(input_text), _estimate_tokens(content), None, "estimated"


def _model_name(model: Any, response: Any = None) -> str:
    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        found = metadata.get("model_name") or metadata.get("model")
        if found:
            return str(found)
    for name in ("model_name", "model"):
        found = getattr(model, name, None)
        if found:
            return str(found)
    delegate = getattr(model, "delegate", None)
    if delegate is not None:
        return _model_name(delegate)
    return model.__class__.__name__


class RunTelemetry:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.started_at = time.time()
        self.started_perf = time.perf_counter()
        self.policy_path = "unclassified"
        self.question_type = "unknown"
        self.max_model_calls: int | None = None
        self.max_input_tokens: int | None = None
        self.deadline_ms: int | None = None
        self.records: list[NodeCost] = []
        self._started_calls = 0
        self._reserved_input_tokens = 0
        self._lock = threading.Lock()

    def configure(self, policy: dict[str, Any]) -> None:
        budget = policy.get("budget") or {}
        with self._lock:
            self.policy_path = str(policy.get("path") or "unclassified")
            self.question_type = str((policy.get("signals") or ["unknown"])[0])
            self.max_model_calls = int(budget["max_model_calls"]) if budget.get("max_model_calls") is not None else None
            self.max_input_tokens = int(budget["max_input_tokens"]) if budget.get("max_input_tokens") is not None else None
            self.deadline_ms = int(budget["deadline_ms"]) if budget.get("deadline_ms") is not None else None

    def begin_call(self, *, input_token_estimate: int = 0, reserve_after: int = 0) -> int:
        with self._lock:
            elapsed_ms = (time.perf_counter() - self.started_perf) * 1000.0
            if self.deadline_ms is not None and elapsed_ms >= self.deadline_ms:
                raise ModelBudgetExceeded(
                    f"request deadline exhausted before model call: elapsed={elapsed_ms:.1f}ms, "
                    f"deadline={self.deadline_ms}ms"
                )
            if self.max_model_calls is not None and self._started_calls + 1 + reserve_after > self.max_model_calls:
                raise ModelBudgetExceeded(
                    f"model-call budget exhausted for {self.policy_path}: "
                    f"started={self._started_calls}, max={self.max_model_calls}, reserve={reserve_after}"
                )
            if (
                self.max_input_tokens is not None
                and self._reserved_input_tokens + input_token_estimate > self.max_input_tokens
            ):
                raise ModelBudgetExceeded(
                    f"input-token budget exhausted for {self.policy_path}: "
                    f"estimated={self._reserved_input_tokens + input_token_estimate}, "
                    f"max={self.max_input_tokens}"
                )
            self._started_calls += 1
            self._reserved_input_tokens += input_token_estimate
            return self._started_calls

    def remaining_calls(self) -> int | None:
        with self._lock:
            if self.max_model_calls is None:
                return None
            return max(0, self.max_model_calls - self._started_calls)

    def add(self, record: NodeCost) -> None:
        with self._lock:
            self.records.append(record)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = [asdict(record) for record in sorted(self.records, key=lambda row: row.call_index)]
        input_tokens = sum(row["input_tokens"] for row in records)
        output_tokens = sum(row["output_tokens"] for row in records)
        return {
            "run_id": self.run_id,
            "policy_path": self.policy_path,
            "question_type": self.question_type,
            "llm_calls": len(records),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round((time.perf_counter() - self.started_perf) * 1000.0, 3),
            "node_costs": records,
        }

    def persist(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.cost.json"
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


_current_telemetry: ContextVar[RunTelemetry | None] = ContextVar("linki_telemetry", default=None)


def current_telemetry() -> RunTelemetry | None:
    return _current_telemetry.get()


def invoke_model(
    model: Any,
    messages: Any,
    *,
    node: str,
    prompt_version: str,
    policy_path: str = "unclassified",
    reserve_after: int = 0,
) -> Any:
    """Invoke a model and attribute latency/usage to one stable graph node."""

    from linki.core.trace import emit_event

    input_text = _message_text(messages)
    telemetry = _current_telemetry.get()
    call_index = telemetry.begin_call(
        input_token_estimate=_estimate_tokens(input_text),
        reserve_after=reserve_after,
    ) if telemetry else 0
    started = time.perf_counter()
    try:
        response = model.invoke(messages)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        if telemetry:
            telemetry.add(NodeCost(
                run_id=telemetry.run_id, node=node, call_index=call_index,
                model=_model_name(model), prompt_version=prompt_version,
                policy_path=policy_path, input_tokens=_estimate_tokens(input_text),
                cached_input_tokens=None, output_tokens=0, queue_ms=None,
                ttft_ms=None, total_ms=round(elapsed, 3), cost_usd=None,
                status="error", usage_source="estimated", error=exc.__class__.__name__,
            ))
        raise
    elapsed = (time.perf_counter() - started) * 1000.0
    input_tokens, output_tokens, cached, source = _usage(response, input_text)
    if telemetry:
        telemetry.add(NodeCost(
            run_id=telemetry.run_id, node=node, call_index=call_index,
            model=_model_name(model, response), prompt_version=prompt_version,
            policy_path=policy_path, input_tokens=input_tokens,
            cached_input_tokens=cached, output_tokens=output_tokens,
            queue_ms=None, ttft_ms=None, total_ms=round(elapsed, 3),
            cost_usd=None, status="ok", usage_source=source,
        ))
    emit_event({
        "node": node, "type": "model_call", "call_index": call_index,
        "prompt_version": prompt_version, "policy_path": policy_path,
        "model": _model_name(model, response), "input_tokens": input_tokens,
        "output_tokens": output_tokens, "usage_source": source,
        "total_ms": round(elapsed, 3),
    })
    return response
