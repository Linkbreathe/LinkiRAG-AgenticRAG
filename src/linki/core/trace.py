"""Persistent trace: turn the graph's ephemeral event stream into a durable
record.

Phase 6 formalises what was previously a UI-only custom stream. A ``Tracer``
appends one JSON object per event to ``<trace_dir>/<run_id>.jsonl`` and, on
``finalize()``, renders a human-readable ``<run_id>.timeline.md``.

The single ``emit()`` fans out to **two sinks**: the durable JSONL file *and*
the live LangGraph custom-stream writer (so the web UI keeps updating in real
time, unchanged). Nodes never touch a file — they call :func:`emit_event`,
which routes to the run's active tracer via a contextvar (or, absent a tracer,
straight to the stream writer). This is the seam that connects the Hook layer
to the Trace layer: one emit, both consumers.
"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# The active tracer for the current run. Set in the entrypoint (CLI/UI), read by
# nodes and hooks. contextvars propagate into LangGraph ``Send`` parallel
# branches (verified), so parallel retrieval shares one tracer.
_current_tracer: ContextVar["Tracer | None"] = ContextVar("linki_tracer", default=None)


def current_tracer() -> "Tracer | None":
    return _current_tracer.get()


def _stream_writer():
    """LangGraph custom-stream writer if inside a streaming run, else None."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def emit_event(event: dict[str, Any]) -> None:
    """Route one event to the active tracer (persist + live UI). Absent a
    tracer, fall back to the raw stream writer; absent that too, no-op. Safe to
    call from any node or hook without guarding."""
    tracer = _current_tracer.get()
    if tracer is not None:
        tracer.emit(event)
        return
    writer = _stream_writer()
    if writer:
        writer(event)


class Tracer:
    """Durable per-run event sink. One JSONL line per event + a timeline.md."""

    def __init__(self, run_id: str, trace_dir: str | Path):
        self.run_id = run_id
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self.timeline_path = self.dir / f"{run_id}.timeline.md"
        self.events: list[dict[str, Any]] = []
        # Start each run with a clean file so a re-used run_id doesn't append to
        # a stale trace.
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        enriched = {"ts": time.time(), "run_id": self.run_id, **event}
        self.events.append(enriched)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
        # Fan out to the live UI with the *original* event shape (UI schema
        # unchanged); only the persisted copy carries ts/run_id.
        writer = _stream_writer()
        if writer:
            writer(event)

    def finalize(self) -> Path:
        """Render the collected events into a readable markdown timeline."""
        lines = [f"# Trace timeline · `{self.run_id}`", ""]
        for ev in self.events:
            node = ev.get("node", "?")
            etype = ev.get("type", "event")
            bits = []
            for key in ("round", "query", "kb", "n_hits", "top_score", "n_citations",
                        "sufficient", "kept", "missing", "detail"):
                if ev.get(key) not in (None, "", []):
                    bits.append(f"{key}={ev[key]}")
            suffix = (" · " + ", ".join(bits)) if bits else ""
            lines.append(f"- **{node}** · `{etype}`{suffix}")
        lines.append("")
        self.timeline_path.write_text("\n".join(lines), encoding="utf-8")
        return self.timeline_path
