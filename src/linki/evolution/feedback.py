"""Versioned, tenant-scoped feedback ledger. Feedback is a signal, not truth."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from linki.evolution.store import EvolutionDatabase

FEEDBACK_KINDS = {
    "correction", "not_found", "verifier_issue", "citation_open",
    "thumbs_up", "thumbs_down", "cache", "trace_example",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Feedback:
    feedback_id: str
    version: int
    tenant_id: str
    user_id: str
    run_id: str | None
    kind: str
    payload: dict[str, Any]
    source: str
    acl: tuple[str, ...]
    status: str
    created_at: str
    idempotency_key: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackLedger:
    def __init__(self, database: EvolutionDatabase):
        self.db = database

    @staticmethod
    def _feedback(row) -> Feedback:
        return Feedback(
            feedback_id=row["feedback_id"], version=int(row["version"]),
            tenant_id=row["tenant_id"], user_id=row["user_id"], run_id=row["run_id"],
            kind=row["kind"], payload=json.loads(row["payload_json"]),
            source=row["source"], acl=tuple(json.loads(row["acl_json"])),
            status=row["status"], created_at=row["created_at"],
            idempotency_key=row["idempotency_key"],
        )

    def _insert(self, item: Feedback) -> None:
        self.db.conn.execute(
            "INSERT INTO feedback_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.feedback_id, item.version, item.tenant_id, item.user_id,
                item.run_id, item.kind, json.dumps(item.payload, ensure_ascii=False),
                item.source, json.dumps(item.acl), item.status, item.created_at,
                item.idempotency_key,
            ),
        )

    def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kind: str,
        payload: dict[str, Any],
        run_id: str | None = None,
        source: str = "system",
        acl: tuple[str, ...] = ("public",),
        idempotency_key: str | None = None,
    ) -> Feedback:
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"unsupported feedback kind: {kind}")
        canonical = json.dumps(
            {"tenant": tenant_id, "user": user_id, "kind": kind, "payload": payload,
             "run": run_id, "source": source},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        key = idempotency_key or hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM feedback_versions WHERE idempotency_key=? ORDER BY version DESC LIMIT 1",
                (key,),
            ).fetchone()
            if row:
                return self._feedback(row)
            status = "observed"
            if kind == "correction" and not (
                payload.get("source_id") and payload.get("quote") and payload.get("correction")
            ):
                status = "needs_source"
            item = Feedback(
                feedback_id=uuid.uuid4().hex[:24], version=1,
                tenant_id=tenant_id, user_id=user_id, run_id=run_id,
                kind=kind, payload=dict(payload), source=source,
                acl=tuple(sorted(set(acl))), status=status,
                created_at=_now(), idempotency_key=key,
            )
            self._insert(item)
            self.db.conn.commit()
            return item

    def current(self, feedback_id: str) -> Feedback | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM feedback_versions WHERE feedback_id=? ORDER BY version DESC LIMIT 1",
                (feedback_id,),
            ).fetchone()
        return self._feedback(row) if row else None

    def transition(self, feedback_id: str, status: str) -> Feedback:
        if status not in {"observed", "needs_source", "diagnosed", "processed", "dismissed"}:
            raise ValueError(status)
        with self.db.lock:
            current = self.current(feedback_id)
            if current is None:
                raise KeyError(feedback_id)
            updated = replace(current, version=current.version + 1, status=status, created_at=_now())
            self._insert(updated)
            self.db.conn.commit()
            return updated

    def list_current(
        self,
        *,
        tenant_id: str,
        kinds: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] = ("observed", "diagnosed"),
        acl: tuple[str, ...] = ("public",),
    ) -> list[Feedback]:
        with self.db.lock:
            rows = self.db.conn.execute(
                """
                SELECT f.* FROM feedback_versions f
                JOIN (SELECT feedback_id, MAX(version) AS version FROM feedback_versions GROUP BY feedback_id) latest
                  ON latest.feedback_id=f.feedback_id AND latest.version=f.version
                WHERE f.tenant_id=? ORDER BY f.created_at
                """,
                (tenant_id,),
            ).fetchall()
        allowed = set(acl) | {"public"}
        out = [self._feedback(row) for row in rows]
        return [
            item for item in out
            if item.status in statuses
            and (not kinds or item.kind in kinds)
            and bool(set(item.acl) & allowed)
        ]
