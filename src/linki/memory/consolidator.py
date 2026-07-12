"""Exact-key consolidation, conflict detection and supersession."""

from __future__ import annotations

import threading

from linki.memory.ledger import MemoryItem, MemoryLedger
from linki.memory.policy import MemoryPolicyVerdict


class MemoryConsolidator:
    def __init__(self, ledger: MemoryLedger):
        self.ledger = ledger
        self._lock = threading.RLock()

    def consolidate(self, candidate: MemoryItem, verdict: MemoryPolicyVerdict) -> MemoryItem:
        with self._lock:
            if verdict.status != "ACTIVE":
                if verdict.status == "PROPOSED":
                    return candidate
                return self.ledger.transition(
                    candidate.memory_id, verdict.status,
                    reason=verdict.reason, actor="memory-policy",
                )

            key = candidate.content.get("key")
            active = self.ledger.list_current(
                tenant_id=candidate.tenant_id, user_id=candidate.user_id,
                statuses=("ACTIVE",),
            )
            matches = [
                item for item in active
                if item.type == candidate.type and item.scope == candidate.scope
                and item.content.get("key") == key
            ]
            if matches:
                current = matches[0]
                if current.content_hash == candidate.content_hash:
                    return self.ledger.transition(
                        candidate.memory_id, "MERGED",
                        reason=f"exact duplicate of {current.memory_id}", actor="consolidator",
                        supersedes=current.memory_id,
                    )
                self.ledger.transition(
                    current.memory_id, "SUPERSEDED",
                    reason=f"replaced by explicit newer memory {candidate.memory_id}",
                    actor="consolidator",
                )
                return self.ledger.transition(
                    candidate.memory_id, "ACTIVE", reason=verdict.reason,
                    actor="memory-policy", supersedes=current.memory_id,
                )
            return self.ledger.transition(
                candidate.memory_id, "ACTIVE", reason=verdict.reason, actor="memory-policy",
            )
