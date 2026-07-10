"""Tracer: persistent JSONL event stream + rendered timeline, plus the
``emit_event`` fan-out that routes through the active tracer (or degrades to a
no-op when none is set)."""

from __future__ import annotations

import json

from linki.core.trace import Tracer, current_tracer, emit_event


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_emit_persists_one_jsonl_line_per_event(tmp_path):
    tracer = Tracer("run1", tmp_path)
    tracer.emit({"node": "router", "type": "route", "detail": "retrieve"})
    tracer.emit({"node": "retrieve", "type": "hook_retrieve", "kb": "Retrieve_default", "n_hits": 3})

    rows = _read_lines(tracer.path)
    assert len(rows) == 2
    assert rows[0]["node"] == "router" and rows[0]["run_id"] == "run1" and "ts" in rows[0]
    assert rows[1]["n_hits"] == 3


def test_emit_event_routes_to_active_tracer(tmp_path):
    tracer = Tracer("run2", tmp_path)
    token = current_tracer  # sanity: symbol exists
    assert token is not None
    from linki.core.trace import _current_tracer

    reset = _current_tracer.set(tracer)
    try:
        assert current_tracer() is tracer
        emit_event({"node": "answer", "type": "answer", "n_citations": 2})
    finally:
        _current_tracer.reset(reset)

    rows = _read_lines(tracer.path)
    assert rows and rows[0]["type"] == "answer"


def test_emit_event_is_noop_without_active_tracer():
    # No tracer in context and not inside a streaming run -> must not raise.
    emit_event({"node": "router", "type": "route"})


def test_finalize_renders_timeline_markdown(tmp_path):
    tracer = Tracer("run3", tmp_path)
    tracer.emit({"node": "router", "type": "route", "detail": "retrieve"})
    tracer.emit({"node": "retrieve", "type": "hook_retrieve", "kb": "Retrieve_api", "n_hits": 2, "top_score": 0.91})
    tracer.emit({"node": "answer", "type": "answer", "n_citations": 1})

    path = tracer.finalize()
    text = path.read_text(encoding="utf-8")
    assert path.name == "run3.timeline.md"
    assert "run3" in text
    # each node appears in the rendered timeline
    for node in ("router", "retrieve", "answer"):
        assert node in text
