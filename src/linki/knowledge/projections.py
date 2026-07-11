"""Staging, promotion and rollback gate for rebuildable projections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from linki.knowledge.store import KnowledgeDatabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ProjectionSnapshot:
    snapshot_id: str
    tenant_id: str
    status: str
    parent_snapshot_id: str | None
    claim_snapshot_id: str
    created_at: str
    promoted_at: str | None
    validation: dict[str, Any]


class ProjectionRegistry:
    def __init__(self, database: KnowledgeDatabase):
        self.db = database

    @staticmethod
    def _snapshot(row) -> ProjectionSnapshot:
        return ProjectionSnapshot(
            snapshot_id=row["snapshot_id"], tenant_id=row["tenant_id"],
            status=row["status"], parent_snapshot_id=row["parent_snapshot_id"],
            claim_snapshot_id=row["claim_snapshot_id"], created_at=row["created_at"],
            promoted_at=row["promoted_at"], validation=json.loads(row["validation_json"]),
        )

    def active(self, tenant_id: str) -> ProjectionSnapshot | None:
        with self.db.lock:
            row = self.db.conn.execute(
                """SELECT s.* FROM projection_active a
                   JOIN projection_snapshots s ON s.snapshot_id=a.snapshot_id
                   WHERE a.tenant_id=?""",
                (tenant_id,),
            ).fetchone()
        return self._snapshot(row) if row else None

    def stage(self, tenant_id: str, claim_snapshot_id: str) -> ProjectionSnapshot:
        parent = self.active(tenant_id)
        snapshot_id = hashlib.sha256(
            f"{tenant_id}:{claim_snapshot_id}:projection.v1".encode("utf-8")
        ).hexdigest()[:24]
        with self.db.lock:
            existing = self.db.conn.execute(
                "SELECT * FROM projection_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if existing:
                return self._snapshot(existing)
            snapshot = ProjectionSnapshot(
                snapshot_id=snapshot_id, tenant_id=tenant_id, status="staging",
                parent_snapshot_id=parent.snapshot_id if parent else None,
                claim_snapshot_id=claim_snapshot_id, created_at=_now(),
                promoted_at=None, validation={},
            )
            self.db.conn.execute(
                "INSERT INTO projection_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id, snapshot.tenant_id, snapshot.status,
                    snapshot.parent_snapshot_id, snapshot.claim_snapshot_id,
                    snapshot.created_at, snapshot.promoted_at, json.dumps({}),
                ),
            )
            self.db.conn.commit()
            return snapshot

    @staticmethod
    def _passes(validation: dict[str, Any]) -> bool:
        required = (
            ("claims", "provenance_coverage"),
            ("wiki", "claim_coverage"),
            ("wiki", "source_span_validity"),
            ("graph", "provenance_validity"),
        )
        return all(float(validation.get(group, {}).get(metric, 0.0)) == 1.0 for group, metric in required)

    def promote(self, snapshot_id: str, validation: dict[str, Any]) -> ProjectionSnapshot:
        if not self._passes(validation):
            with self.db.lock:
                self.db.conn.execute(
                    "UPDATE projection_snapshots SET status='rejected', validation_json=? WHERE snapshot_id=?",
                    (json.dumps(validation), snapshot_id),
                )
                self.db.conn.commit()
            raise ValueError("projection snapshot failed provenance/rebuild gates")
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM projection_snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise KeyError(snapshot_id)
            snapshot = self._snapshot(row)
            if snapshot.status not in {"staging", "active"}:
                raise ValueError(f"cannot promote snapshot in status {snapshot.status}")
            old = self.active(snapshot.tenant_id)
            if old and old.snapshot_id != snapshot_id:
                self.db.conn.execute(
                    "UPDATE projection_snapshots SET status='superseded' WHERE snapshot_id=?",
                    (old.snapshot_id,),
                )
            promoted_at = _now()
            self.db.conn.execute(
                "UPDATE projection_snapshots SET status='active', promoted_at=?, validation_json=? WHERE snapshot_id=?",
                (promoted_at, json.dumps(validation), snapshot_id),
            )
            self.db.conn.execute(
                "INSERT INTO projection_active VALUES (?, ?) ON CONFLICT(tenant_id) DO UPDATE SET snapshot_id=excluded.snapshot_id",
                (snapshot.tenant_id, snapshot_id),
            )
            self.db.conn.commit()
            return ProjectionSnapshot(
                **{**snapshot.__dict__, "status": "active", "promoted_at": promoted_at, "validation": validation}
            )

    def rollback(self, tenant_id: str, snapshot_id: str) -> ProjectionSnapshot:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM projection_snapshots WHERE snapshot_id=? AND tenant_id=?",
                (snapshot_id, tenant_id),
            ).fetchone()
            if row is None:
                raise KeyError(snapshot_id)
            target = self._snapshot(row)
            if target.status not in {"active", "superseded"}:
                raise ValueError("rollback target was never an active validated snapshot")
            current = self.active(tenant_id)
            if current and current.snapshot_id != snapshot_id:
                self.db.conn.execute(
                    "UPDATE projection_snapshots SET status='superseded' WHERE snapshot_id=?",
                    (current.snapshot_id,),
                )
            self.db.conn.execute(
                "UPDATE projection_snapshots SET status='active', promoted_at=? WHERE snapshot_id=?",
                (_now(), snapshot_id),
            )
            self.db.conn.execute(
                "INSERT INTO projection_active VALUES (?, ?) ON CONFLICT(tenant_id) DO UPDATE SET snapshot_id=excluded.snapshot_id",
                (tenant_id, snapshot_id),
            )
            self.db.conn.commit()
            return ProjectionSnapshot(**{**target.__dict__, "status": "active"})
