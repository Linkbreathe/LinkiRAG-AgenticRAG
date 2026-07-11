"""SQLite-backed append/version ledger for episodes and governed memories."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MEMORY_STATUSES = {
    "OBSERVED", "PROPOSED", "MERGED", "REVIEW_REQUIRED", "CONFLICT",
    "REJECTED", "ACTIVE", "SUPERSEDED", "EXPIRED", "DELETED",
}
_TRANSITIONS = {
    "PROPOSED": {"MERGED", "REVIEW_REQUIRED", "CONFLICT", "REJECTED", "ACTIVE", "DELETED"},
    "REVIEW_REQUIRED": {"ACTIVE", "REJECTED", "DELETED"},
    "CONFLICT": {"ACTIVE", "REJECTED", "DELETED"},
    "ACTIVE": {"SUPERSEDED", "EXPIRED", "CONFLICT", "DELETED"},
    "MERGED": {"DELETED"},
    "REJECTED": {"DELETED"},
    "SUPERSEDED": {"DELETED"},
    "EXPIRED": {"DELETED"},
    "DELETED": set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Episode:
    episode_id: str
    tenant_id: str
    user_id: str
    thread_id: str
    user_input: str
    assistant_output: str
    source_hash: str
    event_time: str
    ingested_at: str
    expires_at: str
    redacted_at: str | None = None


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    tenant_id: str
    user_id: str
    type: str
    scope: str
    namespace: tuple[str, ...]
    content: dict[str, Any]
    content_hash: str
    status: str
    source_episode_ids: tuple[str, ...]
    source_spans: tuple[dict[str, Any], ...]
    event_time: str | None
    ingested_at: str
    valid_from: str | None
    valid_to: str | None
    importance: float
    confidence: float
    last_accessed_at: str | None
    access_count: int
    pii_class: str | None
    version: int
    supersedes: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                assistant_output TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                event_time TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                redacted_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episode_scope
                ON episodes(tenant_id, user_id, ingested_at);

            CREATE TABLE IF NOT EXISTS memory_versions (
                memory_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                scope TEXT NOT NULL,
                namespace_json TEXT NOT NULL,
                content_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                source_episode_ids_json TEXT NOT NULL,
                source_spans_json TEXT NOT NULL,
                event_time TEXT,
                ingested_at TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                pii_class TEXT,
                supersedes TEXT,
                PRIMARY KEY(memory_id, version)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_scope
                ON memory_versions(tenant_id, user_id, status, type);

            CREATE TABLE IF NOT EXISTS memory_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_access (
                access_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                accessed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_extractions (
                idempotency_key TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def append_episode(
        self,
        *,
        tenant_id: str,
        user_id: str,
        user_input: str,
        assistant_output: str = "",
        thread_id: str = "default",
        event_time: datetime | None = None,
        retention_days: int = 90,
        idempotency_key: str | None = None,
    ) -> Episode:
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        event = (event_time or datetime.now(UTC)).astimezone(UTC)
        source_hash = hashlib.sha256(
            _json({"tenant": tenant_id, "user": user_id, "thread": thread_id,
                   "input": user_input, "output": assistant_output,
                   "event": event.isoformat()}).encode("utf-8")
        ).hexdigest()
        episode_id = (
            hashlib.sha256(f"episode:{idempotency_key}".encode("utf-8")).hexdigest()[:24]
            if idempotency_key else uuid.uuid4().hex[:24]
        )
        episode = Episode(
            episode_id=episode_id, tenant_id=tenant_id, user_id=user_id,
            thread_id=thread_id, user_input=user_input,
            assistant_output=assistant_output, source_hash=source_hash,
            event_time=event.isoformat(), ingested_at=_now(),
            expires_at=(event + timedelta(days=retention_days)).isoformat(),
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(asdict(episode).values()),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return Episode(**dict(row))

    def get_episode(self, episode_id: str) -> Episode | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return Episode(**dict(row)) if row else None

    def claim_extraction(self, episode: Episode, extractor_version: str) -> bool:
        key = hashlib.sha256(
            f"{episode.source_hash}:{extractor_version}".encode("utf-8")
        ).hexdigest()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO processed_extractions VALUES (?, ?, ?, ?)",
                (key, episode.episode_id, extractor_version, _now()),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def propose(
        self,
        *,
        tenant_id: str,
        user_id: str,
        type: str,
        scope: str,
        namespace: tuple[str, ...],
        content: dict[str, Any],
        source_episode_ids: list[str],
        source_spans: list[dict[str, Any]] | None = None,
        event_time: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        pii_class: str | None = None,
    ) -> MemoryItem:
        if not source_episode_ids:
            raise ValueError("a memory proposal requires at least one source episode")
        with self._lock:
            for episode_id in source_episode_ids:
                episode = self.get_episode(episode_id)
                if episode is None:
                    raise ValueError(f"unknown source episode: {episode_id}")
                if episode.tenant_id != tenant_id or episode.user_id != user_id:
                    raise ValueError("memory source episode crosses tenant/user scope")
            memory_id = uuid.uuid4().hex[:24]
            now = _now()
            item = MemoryItem(
                memory_id=memory_id, tenant_id=tenant_id, user_id=user_id,
                type=type, scope=scope, namespace=tuple(namespace), content=dict(content),
                content_hash=hashlib.sha256(_json(content).encode("utf-8")).hexdigest(),
                status="PROPOSED", source_episode_ids=tuple(source_episode_ids),
                source_spans=tuple(source_spans or []), event_time=event_time,
                ingested_at=now, valid_from=valid_from or now, valid_to=valid_to,
                importance=max(0.0, min(1.0, float(importance))),
                confidence=max(0.0, min(1.0, float(confidence))),
                last_accessed_at=None, access_count=0, pii_class=pii_class,
                version=1, supersedes=None,
            )
            self._insert_version(item)
            self._event(item, "proposed", "candidate extracted from episode", "extractor")
            self._conn.commit()
            return item

    def _insert_version(self, item: MemoryItem) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_versions (
                memory_id, version, tenant_id, user_id, type, scope,
                namespace_json, content_json, content_hash, status,
                source_episode_ids_json, source_spans_json, event_time,
                ingested_at, valid_from, valid_to, importance, confidence,
                pii_class, supersedes
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                item.memory_id, item.version, item.tenant_id, item.user_id,
                item.type, item.scope, _json(item.namespace), _json(item.content),
                item.content_hash, item.status, _json(item.source_episode_ids),
                _json(item.source_spans), item.event_time, item.ingested_at,
                item.valid_from, item.valid_to, item.importance, item.confidence,
                item.pii_class, item.supersedes,
            ),
        )

    def _event(
        self, item: MemoryItem, action: str, reason: str, actor: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO memory_events(memory_id, version, action, reason, actor, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item.memory_id, item.version, action, reason, actor, _now(), _json(metadata or {})),
        )

    @staticmethod
    def _item(row: sqlite3.Row, *, access_count: int = 0, last_accessed_at: str | None = None) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"], version=int(row["version"]),
            tenant_id=row["tenant_id"], user_id=row["user_id"],
            type=row["type"], scope=row["scope"],
            namespace=tuple(json.loads(row["namespace_json"])),
            content=json.loads(row["content_json"]), content_hash=row["content_hash"],
            status=row["status"], source_episode_ids=tuple(json.loads(row["source_episode_ids_json"])),
            source_spans=tuple(json.loads(row["source_spans_json"])),
            event_time=row["event_time"], ingested_at=row["ingested_at"],
            valid_from=row["valid_from"], valid_to=row["valid_to"],
            importance=float(row["importance"]), confidence=float(row["confidence"]),
            last_accessed_at=last_accessed_at, access_count=access_count,
            pii_class=row["pii_class"], supersedes=row["supersedes"],
        )

    def current(self, memory_id: str) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? ORDER BY version DESC LIMIT 1",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            access = self._conn.execute(
                "SELECT COUNT(*) AS n, MAX(accessed_at) AS last FROM memory_access WHERE memory_id=?",
                (memory_id,),
            ).fetchone()
        return self._item(row, access_count=int(access["n"]), last_accessed_at=access["last"])

    def transition(
        self,
        memory_id: str,
        status: str,
        *,
        reason: str,
        actor: str,
        supersedes: str | None = None,
    ) -> MemoryItem:
        if status not in MEMORY_STATUSES:
            raise ValueError(f"unknown memory status: {status}")
        with self._lock:
            current = self.current(memory_id)
            if current is None:
                raise KeyError(memory_id)
            if status not in _TRANSITIONS.get(current.status, set()):
                raise ValueError(f"invalid memory transition: {current.status} -> {status}")
            if status == "ACTIVE" and not current.source_episode_ids:
                raise ValueError("memory without a source episode cannot become ACTIVE")
            updated = replace(
                current,
                status=status,
                version=current.version + 1,
                ingested_at=_now(),
                supersedes=supersedes if supersedes is not None else current.supersedes,
            )
            self._insert_version(updated)
            self._event(updated, status.lower(), reason, actor)
            self._conn.commit()
            return updated

    def list_current(
        self,
        *,
        tenant_id: str,
        user_id: str,
        statuses: tuple[str, ...] | None = None,
        include_deleted: bool = False,
    ) -> list[MemoryItem]:
        params: list[Any] = [tenant_id, user_id]
        status_clause = ""
        if statuses:
            status_clause = f" AND m.status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        elif not include_deleted:
            status_clause = " AND m.status != 'DELETED'"
        query = f"""
            SELECT m.*,
                   (SELECT COUNT(*) FROM memory_access a WHERE a.memory_id=m.memory_id) AS derived_access_count,
                   (SELECT MAX(accessed_at) FROM memory_access a WHERE a.memory_id=m.memory_id) AS derived_last_access
            FROM memory_versions m
            JOIN (SELECT memory_id, MAX(version) AS version FROM memory_versions GROUP BY memory_id) latest
              ON latest.memory_id=m.memory_id AND latest.version=m.version
            WHERE m.tenant_id=? AND m.user_id=? {status_clause}
            ORDER BY m.ingested_at DESC
        """
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            self._item(
                row,
                access_count=int(row["derived_access_count"] or 0),
                last_accessed_at=row["derived_last_access"],
            )
            for row in rows
        ]

    def events(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_events WHERE memory_id=? ORDER BY event_id", (memory_id,)
            ).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in rows]

    def mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO memory_access(memory_id, accessed_at) VALUES (?, ?)",
                [(memory_id, _now()) for memory_id in memory_ids],
            )
            self._conn.commit()

    def delete(self, memory_id: str, *, actor: str = "user", purge_payload: bool = True) -> MemoryItem:
        deleted = self.transition(memory_id, "DELETED", reason="user requested deletion", actor=actor)
        if purge_payload:
            # Legal deletion is the one explicit exception to immutable payloads:
            # retain hashes/events as non-sensitive proof, remove recoverable text.
            with self._lock:
                tombstone_hash = hashlib.sha256(f"deleted:{memory_id}".encode("utf-8")).hexdigest()
                self._conn.execute(
                    "UPDATE memory_versions SET content_json='{}', source_spans_json='[]', content_hash=? WHERE memory_id=?",
                    (tombstone_hash, memory_id),
                )
                self._event(deleted, "payload_purged", "recoverable memory payload removed", actor)
                self._conn.commit()
            deleted = replace(deleted, content={}, source_spans=())
        return deleted

    def redact_episode(self, episode_id: str, *, actor: str = "user") -> None:
        with self._lock:
            tombstone_hash = hashlib.sha256(f"redacted:{episode_id}".encode("utf-8")).hexdigest()
            self._conn.execute(
                "UPDATE episodes SET user_input='', assistant_output='', source_hash=?, redacted_at=? WHERE episode_id=?",
                (tombstone_hash, _now(), episode_id),
            )
            self._conn.commit()

    def erase_user(self, tenant_id: str, user_id: str) -> dict[str, int]:
        items = self.list_current(
            tenant_id=tenant_id, user_id=user_id, include_deleted=False,
        )
        for item in items:
            self.delete(item.memory_id, actor="legal-delete", purge_payload=True)
        with self._lock:
            episode_rows = self._conn.execute(
                "SELECT episode_id FROM episodes WHERE tenant_id=? AND user_id=? AND redacted_at IS NULL",
                (tenant_id, user_id),
            ).fetchall()
        for row in episode_rows:
            self.redact_episode(row["episode_id"], actor="legal-delete")
        return {"memories_tombstoned": len(items), "episodes_redacted": len(episode_rows)}

    def expire_due(self, now: datetime | None = None) -> int:
        point = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.memory_id FROM memory_versions m
                JOIN (SELECT memory_id, MAX(version) AS version FROM memory_versions GROUP BY memory_id) latest
                  ON latest.memory_id=m.memory_id AND latest.version=m.version
                WHERE m.status='ACTIVE' AND m.valid_to IS NOT NULL AND m.valid_to <= ?
                """,
                (point,),
            ).fetchall()
        for row in rows:
            self.transition(row["memory_id"], "EXPIRED", reason="retention window elapsed", actor="retention-worker")
        return len(rows)

    def snapshot_id(self, tenant_id: str, user_id: str) -> str:
        items = self.list_current(
            tenant_id=tenant_id, user_id=user_id, statuses=("ACTIVE",),
        )
        payload = [(item.memory_id, item.version, item.content_hash) for item in sorted(items, key=lambda x: x.memory_id)]
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:24]
