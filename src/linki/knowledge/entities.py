"""Conservative exact entity normalization and alias resolution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from linki.knowledge.store import KnowledgeDatabase


def normalize_entity(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", " ", name.casefold()).strip()


@dataclass(frozen=True)
class Entity:
    entity_id: str
    tenant_id: str
    canonical_name: str
    normalized_name: str
    entity_type: str
    aliases: tuple[str, ...]
    created_at: str


class EntityResolver:
    def __init__(self, database: KnowledgeDatabase):
        self.db = database

    @staticmethod
    def _entity(row) -> Entity:
        return Entity(
            entity_id=row["entity_id"], tenant_id=row["tenant_id"],
            canonical_name=row["canonical_name"], normalized_name=row["normalized_name"],
            entity_type=row["entity_type"], aliases=tuple(json.loads(row["aliases_json"])),
            created_at=row["created_at"],
        )

    def resolve(self, name: str, *, tenant_id: str, entity_type: str = "concept") -> Entity:
        normalized = normalize_entity(name)
        if not normalized:
            raise ValueError("entity name must not be empty")
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM entities WHERE tenant_id=? AND entity_type=?",
                (tenant_id, entity_type),
            ).fetchall()
            for row in rows:
                entity = self._entity(row)
                aliases = {normalize_entity(alias) for alias in entity.aliases}
                if normalized == entity.normalized_name or normalized in aliases:
                    return entity
            entity_id = hashlib.sha256(
                f"{tenant_id}:{entity_type}:{normalized}".encode("utf-8")
            ).hexdigest()[:24]
            entity = Entity(
                entity_id=entity_id, tenant_id=tenant_id, canonical_name=name.strip(),
                normalized_name=normalized, entity_type=entity_type,
                aliases=(name.strip(),), created_at=datetime.now(UTC).isoformat(),
            )
            self.db.conn.execute(
                "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.entity_id, entity.tenant_id, entity.canonical_name,
                    entity.normalized_name, entity.entity_type,
                    json.dumps(entity.aliases, ensure_ascii=False), entity.created_at,
                ),
            )
            self.db.conn.commit()
            return entity

    def get(self, entity_id: str) -> Entity | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM entities WHERE entity_id=?", (entity_id,)
            ).fetchone()
        return self._entity(row) if row else None

    def add_alias(self, entity_id: str, alias: str) -> Entity:
        with self.db.lock:
            entity = self.get(entity_id)
            if entity is None:
                raise KeyError(entity_id)
            aliases = tuple(sorted(set(entity.aliases) | {alias.strip()}))
            self.db.conn.execute(
                "UPDATE entities SET aliases_json=? WHERE entity_id=?",
                (json.dumps(aliases, ensure_ascii=False), entity_id),
            )
            self.db.conn.commit()
        return Entity(**{**entity.__dict__, "aliases": aliases})
