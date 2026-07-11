"""Memory formation service: episode -> extract -> policy -> consolidate."""

from __future__ import annotations

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from linki.core.context import RequestContext
from linki.memory.consolidator import MemoryConsolidator
from linki.memory.extractor import EXTRACTOR_VERSION, MemoryCandidate, extract_candidates, preference_key
from linki.memory.ledger import Episode, MemoryItem, MemoryLedger
from linki.memory.policy import evaluate_candidate


class MemoryService:
    def __init__(self, path: str | Path, *, retention_days: int = 90):
        self.ledger = MemoryLedger(path)
        self.retention_days = retention_days
        self.consolidator = MemoryConsolidator(self.ledger)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="linki-memory")
        self._pending: set[Future] = set()
        self._lock = threading.RLock()

    def _store_candidate(self, episode: Episode, candidate: MemoryCandidate) -> MemoryItem:
        verdict = evaluate_candidate(candidate)
        proposed = self.ledger.propose(
            tenant_id=episode.tenant_id, user_id=episode.user_id,
            type=candidate.type, scope=candidate.scope, namespace=candidate.namespace,
            content=candidate.content, source_episode_ids=[episode.episode_id],
            source_spans=list(candidate.source_spans), event_time=episode.event_time,
            valid_to=episode.expires_at if candidate.type == "episodic" else None,
            importance=candidate.importance, confidence=candidate.confidence,
            pii_class=verdict.pii_class,
        )
        return self.consolidator.consolidate(proposed, verdict)

    def process_episode(self, episode: Episode) -> list[MemoryItem]:
        if not self.ledger.claim_extraction(episode, EXTRACTOR_VERSION):
            return []
        return [self._store_candidate(episode, candidate) for candidate in extract_candidates(episode)]

    def record_turn(
        self,
        question: str,
        answer: str,
        *,
        context: RequestContext,
        thread_id: str = "default",
        run_id: str | None = None,
        background: bool = True,
    ) -> Episode:
        episode = self.ledger.append_episode(
            tenant_id=context.tenant_id, user_id=context.user_id,
            user_input=question, assistant_output=answer, thread_id=thread_id,
            retention_days=self.retention_days, idempotency_key=run_id,
        )
        if background:
            future = self._executor.submit(self.process_episode, episode)
            with self._lock:
                self._pending.add(future)
            future.add_done_callback(self._pending.discard)
        else:
            self.process_episode(episode)
        return episode

    def remember_explicit(self, statement: str, *, context: RequestContext) -> MemoryItem:
        episode = self.ledger.append_episode(
            tenant_id=context.tenant_id, user_id=context.user_id,
            user_input=f"Remember that {statement}", retention_days=self.retention_days,
        )
        candidate = MemoryCandidate(
            type="profile", scope="user",
            namespace=(context.tenant_id, context.user_id, "profile"),
            content={"kind": "preference", "key": preference_key(statement),
                     "value": statement, "original": statement},
            source_spans=({"episode_id": episode.episode_id, "field": "user_input",
                           "char_start": 0, "char_end": len(statement), "quote": statement},),
            confidence=1.0, importance=0.85, explicit=True,
        )
        return self._store_candidate(episode, candidate)

    def forget(self, identifier_or_query: str, *, context: RequestContext) -> list[MemoryItem]:
        target = identifier_or_query.strip().casefold()
        items = self.ledger.list_current(
            tenant_id=context.tenant_id, user_id=context.user_id,
            statuses=("ACTIVE", "PROPOSED", "REVIEW_REQUIRED", "CONFLICT"),
        )
        if target in {"all", "everything", "全部", "所有", "所有记忆"}:
            matches = items
        else:
            matches = [
                item for item in items
                if item.memory_id.casefold() == target
                or target in str(item.content).casefold()
            ]
        deleted: list[MemoryItem] = []
        for item in matches:
            deleted.append(self.ledger.delete(item.memory_id))
            for episode_id in item.source_episode_ids:
                self.ledger.redact_episode(episode_id)
        return deleted

    def edit(self, memory_id: str, statement: str, *, context: RequestContext) -> MemoryItem:
        current = self.ledger.current(memory_id)
        if current is None or current.tenant_id != context.tenant_id or current.user_id != context.user_id:
            raise KeyError(memory_id)
        if current.status != "ACTIVE":
            raise ValueError("only an ACTIVE memory can be edited")
        episode = self.ledger.append_episode(
            tenant_id=context.tenant_id, user_id=context.user_id,
            user_input=f"Update memory {memory_id}: {statement}",
            retention_days=self.retention_days,
        )
        candidate = MemoryCandidate(
            type=current.type, scope=current.scope, namespace=current.namespace,
            content={**current.content, "value": statement, "original": statement},
            source_spans=({"episode_id": episode.episode_id, "field": "user_input",
                           "char_start": 0, "char_end": len(statement), "quote": statement},),
            confidence=1.0, importance=current.importance, explicit=True,
        )
        return self._store_candidate(episode, candidate)

    def export(self, *, context: RequestContext, include_deleted: bool = False) -> list[dict[str, Any]]:
        return [
            item.as_dict() for item in self.ledger.list_current(
                tenant_id=context.tenant_id, user_id=context.user_id,
                include_deleted=include_deleted,
            )
        ]

    def drain(self) -> None:
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return
            for future in pending:
                future.result()


_SERVICES: dict[str, MemoryService] = {}
_SERVICES_LOCK = threading.Lock()


def get_memory_service(settings) -> MemoryService:
    path = str(Path(settings.memory_path).resolve())
    with _SERVICES_LOCK:
        service = _SERVICES.get(path)
        if service is None:
            service = MemoryService(
                path, retention_days=getattr(settings, "episodic_retention_days", 90),
            )
            _SERVICES[path] = service
        return service


def parse_memory_command(question: str) -> tuple[str, str] | None:
    text = (question or "").strip()
    patterns = (
        ("remember", re.compile(r"^(?:请)?(?:记住|remember(?: that)?)[：:,， ]*(.+)$", re.I)),
        ("forget", re.compile(r"^(?:请)?(?:忘记|删除记忆|forget)[：:,， ]*(.+)$", re.I)),
    )
    for action, pattern in patterns:
        match = pattern.match(text)
        if match and match.group(1).strip():
            return action, match.group(1).strip()
    return None
