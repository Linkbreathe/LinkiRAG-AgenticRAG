"""Built-in retrieval hooks: Cache (Pre+Post), Dedup (Post), TraceLog (Post).

Sensitive-word filtering and evidence compression are intentionally left as
documented extension points (no-op stubs) — no real pain point yet (YAGNI).
"""

from __future__ import annotations

from linki.graph.state import Evidence
from linki.hooks.base import HookContext


class CacheHook:
    """Memoize raw fetches by ``(query, kb)`` within a run.

    Same instance registers as both a Pre hook (read) and a Post hook (store).
    It must be **first** in the Post chain so it stores the *raw* hits before
    Dedup mutates the list — a cache hit then re-runs Dedup fresh against the
    current ``seen_parents``, which is what we want on refine/reflow retries.
    """

    def before(self, query: str, kb: str, ctx: HookContext) -> list[Evidence] | None:
        return ctx.cache.get((query, kb))

    def after(self, query: str, kb: str, hits: list[Evidence], ctx: HookContext) -> list[Evidence]:
        ctx.cache[(query, kb)] = list(hits)
        return hits


class DedupHook:
    """Drop hits whose parent chunk was already collected earlier in the run.

    Keys on ``parent_id`` (falling back to ``chunk_id``). Because the context is
    shared across parallel sub-query branches, this de-duplicates both across
    grade→refine rounds and across sub-queries."""

    def after(self, query: str, kb: str, hits: list[Evidence], ctx: HookContext) -> list[Evidence]:
        out: list[Evidence] = []
        for h in hits:
            key = h.get("parent_id") or h.get("chunk_id")
            if key and key in ctx.seen_parents:
                continue
            if key:
                ctx.seen_parents.add(key)
            out.append(h)
        return out


class TraceLogHook:
    """Emit one structured retrieval event per fetch — the Hook↔Trace closure."""

    def after(self, query: str, kb: str, hits: list[Evidence], ctx: HookContext) -> list[Evidence]:
        from linki.core.trace import emit_event

        emit_event({
            "node": "retrieve",
            "type": "hook_retrieve",
            "query": query,
            "kb": kb,
            "n_hits": len(hits),
            "top_score": max((h.get("score", 0.0) for h in hits), default=0.0),
            "latency_ms": round(ctx.last_latency_ms, 1),
            "from_cache": ctx.last_from_cache,
        })
        return hits


class SensitiveWordHook:
    """Extension point (Pre): block/redact queries containing sensitive terms.
    Not wired into the default chain — no real requirement yet."""

    def before(self, query: str, kb: str, ctx: HookContext) -> list[Evidence] | None:
        return None


class CompressHook:
    """Extension point (Post): compress over-long evidence before it reaches the
    answer node. Not wired into the default chain — knowledge-QA sessions are
    short enough that this isn't needed yet."""

    def after(self, query: str, kb: str, hits: list[Evidence], ctx: HookContext) -> list[Evidence]:
        return hits


def default_hooks(settings=None, run_id: str = "run") -> HookContext:
    """Assemble the default chain, honoring ``enable_cache`` / ``enable_dedup`` /
    ``enable_hook_trace`` on settings (all default True)."""
    enable_cache = getattr(settings, "enable_cache", True)
    enable_dedup = getattr(settings, "enable_dedup", True)
    enable_trace = getattr(settings, "enable_hook_trace", True)

    cache = CacheHook()
    pre = [cache] if enable_cache else []
    post: list = []
    if enable_cache:
        post.append(cache)  # store raw first
    if enable_dedup:
        post.append(DedupHook())
    if enable_trace:
        post.append(TraceLogHook())  # log final kept count last
    return HookContext(run_id=run_id, pre=pre, post=post)
