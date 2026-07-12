"""Hook framework: the run-scoped context, the two hook protocols, and the
single ``run_retrieval`` seam that wraps every knowledge-base fetch.

Run-scoped state (cache, the cross-query ``seen_parents`` set) lives on a
:class:`HookContext` stashed in a contextvar. contextvars propagate into
LangGraph ``Send`` parallel branches (verified), so Cache and Dedup genuinely
share state across the parallel sub-query retrievals — which threading the
handles through graph state could not do, since ``Send`` copies the state map.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from linki.graph.state import Evidence


@runtime_checkable
class PreRetrieveHook(Protocol):
    """Runs before a fetch. Return a hit list to short-circuit the fetch
    (e.g. a cache hit), or ``None`` to let retrieval proceed."""

    def before(self, query: str, kb: str, ctx: "HookContext") -> list[Evidence] | None: ...


@runtime_checkable
class PostRetrieveHook(Protocol):
    """Runs after a fetch. Transforms (filters/annotates) the hit list."""

    def after(self, query: str, kb: str, hits: list[Evidence], ctx: "HookContext") -> list[Evidence]: ...


@dataclass
class HookContext:
    """Everything a run's hooks need to share. One per ``answer_question`` call."""

    run_id: str = "run"
    pre: list[PreRetrieveHook] = field(default_factory=list)
    post: list[PostRetrieveHook] = field(default_factory=list)
    # run-scoped mutable state
    cache: dict[tuple[str, str], list[Evidence]] = field(default_factory=dict)
    seen_parents: set[str] = field(default_factory=set)
    request_context: Any = None
    snapshot_ids: dict[str, str] = field(default_factory=dict)
    persistent_cache: Any = None
    cache_dimensions: dict[str, Any] = field(default_factory=dict)
    async_flights: Any = None
    # transient signals about the most recent fetch, for Post hooks (e.g. trace)
    last_latency_ms: float = 0.0
    last_from_cache: bool = False
    last_cache_level: str | None = None

    def run_pre(self, query: str, kb: str) -> list[Evidence] | None:
        for hook in self.pre:
            hits = hook.before(query, kb, self)
            if hits is not None:
                return hits
        return None

    def run_post(self, query: str, kb: str, hits: list[Evidence]) -> list[Evidence]:
        for hook in self.post:
            hits = hook.after(query, kb, hits, self)
        return hits


_current_hooks: ContextVar[HookContext | None] = ContextVar("linki_hooks", default=None)


def current_hooks() -> HookContext | None:
    return _current_hooks.get()


def run_retrieval(retrieve_fn, query: str, kb: str) -> list[Evidence]:
    """The single retrieval seam: ``pre → (fetch unless short-circuited) → post``.

    With no active :class:`HookContext` (e.g. a bare unit test), degrades to a
    plain ``retrieve_fn(query, kb)`` so retrieval stays trivially testable.
    """
    ctx = _current_hooks.get()
    if ctx is None:
        return retrieve_fn(query, kb)

    ctx.last_cache_level = None
    cached = ctx.run_pre(query, kb)
    ctx.last_from_cache = cached is not None
    start = time.perf_counter()
    hits = cached if cached is not None else retrieve_fn(query, kb)
    ctx.last_latency_ms = (time.perf_counter() - start) * 1000.0
    return ctx.run_post(query, kb, hits)
