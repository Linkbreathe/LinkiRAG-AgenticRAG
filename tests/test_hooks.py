"""Retrieval Hook layer: the ``run_retrieval`` seam plus the Cache/Dedup/
TraceLog built-ins, all exercised with fakes (no LLM, no Qdrant)."""

from __future__ import annotations

from conftest import ev

from linki.hooks.base import HookContext, current_hooks, run_retrieval
from linki.hooks.builtin import CacheHook, DedupHook, TraceLogHook, default_hooks


def _with_ctx(ctx: HookContext):
    from linki.hooks.base import _current_hooks

    return _current_hooks.set(ctx)


def test_run_retrieval_without_context_calls_fn_directly():
    calls = []

    def retrieve_fn(q, kb):
        calls.append((q, kb))
        return [ev("a")]

    out = run_retrieval(retrieve_fn, "query", "Retrieve_default")
    assert [e["chunk_id"] for e in out] == ["a"]
    assert calls == [("query", "Retrieve_default")]


def test_cache_hook_short_circuits_second_identical_call():
    calls = []

    def retrieve_fn(q, kb):
        calls.append((q, kb))
        return [ev("a"), ev("b")]

    cache = CacheHook()
    ctx = HookContext(pre=[cache], post=[cache])
    tok = _with_ctx(ctx)
    try:
        first = run_retrieval(retrieve_fn, "q", "Retrieve_default")
        second = run_retrieval(retrieve_fn, "q", "Retrieve_default")
    finally:
        from linki.hooks.base import _current_hooks

        _current_hooks.reset(tok)

    assert len(calls) == 1  # second call served from cache
    assert {e["chunk_id"] for e in first} == {e["chunk_id"] for e in second} == {"a", "b"}


def test_dedup_hook_drops_parents_seen_earlier_in_run():
    ctx = HookContext(post=[DedupHook()])
    tok = _with_ctx(ctx)
    try:
        first = run_retrieval(lambda q, kb: [ev("a", parent_id="P1"), ev("b", parent_id="P2")], "q1", "kb")
        second = run_retrieval(lambda q, kb: [ev("c", parent_id="P2"), ev("d", parent_id="P3")], "q2", "kb")
    finally:
        from linki.hooks.base import _current_hooks

        _current_hooks.reset(tok)

    assert {e["parent_id"] for e in first} == {"P1", "P2"}
    # P2 already seen -> only the new parent P3 survives.
    assert {e["parent_id"] for e in second} == {"P3"}


def test_tracelog_hook_emits_a_retrieve_event(tmp_path):
    from linki.core.trace import Tracer, _current_tracer

    tracer = Tracer("hookrun", tmp_path)
    ctx = HookContext(post=[TraceLogHook()])
    t_tok = _current_tracer.set(tracer)
    h_tok = _with_ctx(ctx)
    try:
        run_retrieval(lambda q, kb: [ev("a", score=0.7), ev("b", score=0.4)], "the query", "Retrieve_api")
    finally:
        from linki.hooks.base import _current_hooks

        _current_hooks.reset(h_tok)
        _current_tracer.reset(t_tok)

    import json

    rows = [json.loads(l) for l in tracer.path.read_text().splitlines() if l.strip()]
    hook_events = [r for r in rows if r.get("type") == "hook_retrieve"]
    assert hook_events, "TraceLogHook should emit a hook_retrieve event"
    e = hook_events[0]
    assert e["kb"] == "Retrieve_api" and e["n_hits"] == 2 and e["query"] == "the query"
    assert e["top_score"] == 0.7


def test_default_hooks_toggle_off_via_settings():
    from conftest import FakeSettings

    s = FakeSettings()
    s.enable_cache = False
    s.enable_dedup = False
    s.enable_hook_trace = False
    ctx = default_hooks(s)
    assert ctx.pre == [] and ctx.post == []


def test_default_hooks_all_on_by_default():
    from conftest import FakeSettings

    ctx = default_hooks(FakeSettings())
    assert len(ctx.pre) == 1  # cache read
    # cache store + dedup + tracelog
    assert len(ctx.post) == 3
