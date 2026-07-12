"""Immutable SourceArtifact and bitemporal knowledge Episode repositories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from linki.knowledge.store import KnowledgeDatabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SourceArtifact:
    source_id: str
    tenant_id: str
    source_key: str
    version: int
    uri: str
    content_hash: str
    content_text: str
    scope: str
    acl: tuple[str, ...]
    parser_version: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeEpisode:
    episode_id: str
    tenant_id: str
    source_id: str
    kind: str
    event_time: str
    ingested_at: str
    source_span: dict[str, Any]
    idempotency_key: str


class SourceRepository:
    def __init__(self, database: KnowledgeDatabase):
        self.db = database

    @staticmethod
    def _source(row) -> SourceArtifact:
        return SourceArtifact(
            source_id=row["source_id"], tenant_id=row["tenant_id"],
            source_key=row["source_key"], version=int(row["version"]), uri=row["uri"],
            content_hash=row["content_hash"], content_text=row["content_text"],
            scope=row["scope"], acl=tuple(json.loads(row["acl_json"])),
            parser_version=row["parser_version"], created_at=row["created_at"],
        )

    def ingest(
        self,
        *,
        tenant_id: str,
        source_key: str,
        uri: str,
        content: str,
        scope: str = "organization",
        acl: tuple[str, ...] = ("public",),
        parser_version: str = "text.v1",
    ) -> SourceArtifact:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self.db.lock:
            existing = self.db.conn.execute(
                "SELECT * FROM source_artifacts WHERE tenant_id=? AND source_key=? AND content_hash=?",
                (tenant_id, source_key, content_hash),
            ).fetchone()
            if existing:
                return self._source(existing)
            row = self.db.conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM source_artifacts WHERE tenant_id=? AND source_key=?",
                (tenant_id, source_key),
            ).fetchone()
            version = int(row["version"]) + 1
            source_id = hashlib.sha256(
                f"{tenant_id}:{source_key}:{content_hash}".encode("utf-8")
            ).hexdigest()[:24]
            artifact = SourceArtifact(
                source_id=source_id, tenant_id=tenant_id, source_key=source_key,
                version=version, uri=uri, content_hash=content_hash,
                content_text=content, scope=scope,
                acl=tuple(sorted(set(acl))), parser_version=parser_version,
                created_at=_now(),
            )
            self.db.conn.execute(
                "INSERT INTO source_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact.source_id, artifact.tenant_id, artifact.source_key,
                    artifact.version, artifact.uri, artifact.content_hash,
                    artifact.content_text, artifact.scope, json.dumps(artifact.acl),
                    artifact.parser_version, artifact.created_at,
                ),
            )
            self.db.conn.commit()
            return artifact

    def get(self, source_id: str) -> SourceArtifact | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM source_artifacts WHERE source_id=?", (source_id,)
            ).fetchone()
        return self._source(row) if row else None

    def observe(
        self,
        source_id: str,
        *,
        tenant_id: str,
        event_time: datetime | None = None,
        source_span: dict[str, Any] | None = None,
        kind: str = "source_ingest",
        idempotency_key: str | None = None,
    ) -> KnowledgeEpisode:
        source = self.get(source_id)
        if source is None or source.tenant_id != tenant_id:
            raise ValueError("source does not exist in this tenant")
        span = source_span or {
            "source_id": source_id, "char_start": 0,
            "char_end": len(source.content_text), "quote": source.content_text,
        }
        if not self.validate_span(span, tenant_id=tenant_id):
            raise ValueError("episode source span does not match the immutable source")
        event = (event_time or datetime.now(UTC)).astimezone(UTC).isoformat()
        key = idempotency_key or hashlib.sha256(
            json.dumps({"source": source_id, "kind": kind, "event": event, "span": span}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        episode_id = hashlib.sha256(f"knowledge-episode:{key}".encode("utf-8")).hexdigest()[:24]
        with self.db.lock:
            existing = self.db.conn.execute(
                "SELECT * FROM knowledge_episodes WHERE idempotency_key=?", (key,)
            ).fetchone()
            if existing:
                return self._episode(existing)
            episode = KnowledgeEpisode(
                episode_id=episode_id, tenant_id=tenant_id, source_id=source_id,
                kind=kind, event_time=event, ingested_at=_now(),
                source_span=span, idempotency_key=key,
            )
            self.db.conn.execute(
                "INSERT INTO knowledge_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.episode_id, episode.tenant_id, episode.source_id,
                    episode.kind, episode.event_time, episode.ingested_at,
                    json.dumps(span, ensure_ascii=False), episode.idempotency_key,
                ),
            )
            self.db.conn.commit()
            return episode

    @staticmethod
    def _episode(row) -> KnowledgeEpisode:
        return KnowledgeEpisode(
            episode_id=row["episode_id"], tenant_id=row["tenant_id"],
            source_id=row["source_id"], kind=row["kind"],
            event_time=row["event_time"], ingested_at=row["ingested_at"],
            source_span=json.loads(row["source_span_json"]),
            idempotency_key=row["idempotency_key"],
        )

    def get_episode(self, episode_id: str) -> KnowledgeEpisode | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM knowledge_episodes WHERE episode_id=?", (episode_id,)
            ).fetchone()
        return self._episode(row) if row else None

    def validate_span(self, span: dict[str, Any], *, tenant_id: str) -> bool:
        source = self.get(str(span.get("source_id") or ""))
        if source is None or source.tenant_id != tenant_id:
            return False
        try:
            start, end = int(span["char_start"]), int(span["char_end"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            0 <= start <= end <= len(source.content_text)
            and source.content_text[start:end] == span.get("quote")
        )
