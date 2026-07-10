"""Retrieval Hook layer (Phase 6).

Retrieval is Linki's one high-frequency side effect, so it gets one Pre and one
Post hook point — the minimal dose of Claude Code's Pre/PostToolUse idea. Cross-
cutting logic (cache, dedup, trace) lives in hooks instead of rotting inside the
retrieval node.
"""

from linki.hooks.base import (
    HookContext,
    PostRetrieveHook,
    PreRetrieveHook,
    current_hooks,
    run_retrieval,
)
from linki.hooks.builtin import CacheHook, DedupHook, TraceLogHook, default_hooks

__all__ = [
    "HookContext",
    "PreRetrieveHook",
    "PostRetrieveHook",
    "current_hooks",
    "run_retrieval",
    "CacheHook",
    "DedupHook",
    "TraceLogHook",
    "default_hooks",
]
