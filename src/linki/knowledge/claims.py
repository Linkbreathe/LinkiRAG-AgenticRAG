"""Immutable bitemporal ClaimVersion ledger with provenance-gated activation."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from linki.knowledge.sources import SourceRepository
from linki.knowledge.store import KnowledgeDatabase

CLAIM_STATUSES = {"candidate", "active", "superseded", "rejected"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ClaimVersion:
    claim_id: str
    version: int
    tenant_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str | None
    literal_value: Any | None
    qualifiers: dict[str, Any]
    status: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str
    source_spans: tuple[dict[str, Any], ...]
    confidence: float
    supersedes: str | None
    schema_version: str
    episode_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimLedger:
    def __init__(self, database: KnowledgeDatabase, sources: SourceRepository):
        self.db = database
        self.sources = sources

    @staticmethod
    def _claim(row) -> ClaimVersion:
        return ClaimVersion(
            claim_id=row["claim_id"], version=int(row["version"]),
            tenant_id=row["tenant_id"], subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"], object_entity_id=row["object_entity_id"],
            literal_value=json.loads(row["literal_json"]) if row["literal_json"] is not None else None,
            qualifiers=json.loads(row["qualifiers_json"]), status=row["status"],
            valid_from=row["valid_from"], valid_to=row["valid_to"],
            recorded_at=row["recorded_at"], source_spans=tuple(json.loads(row["source_spans_json"])),
            confidence=float(row["confidence"]), supersedes=row["supersedes"],
            schema_version=row["schema_version"], episode_id=row["episode_id"],
        )

    def _insert(self, claim: ClaimVersion) -> None:
        self.db.conn.execute(
            """
            INSERT INTO claim_versions (
                claim_id, version, tenant_id, subject_entity_id, predicate,
                object_entity_id, literal_json, qualifiers_json, status,
                valid_from, valid_to, recorded_at, source_spans_json,
                confidence, supersedes, schema_version, episode_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id, claim.version, claim.tenant_id,
                claim.subject_entity_id, claim.predicate, claim.object_entity_id,
                _json(claim.literal_value) if claim.literal_value is not None else None,
                _json(claim.qualifiers), claim.status, claim.valid_from, claim.valid_to,
                claim.recorded_at, _json(claim.source_spans), claim.confidence,
                claim.supersedes, claim.schema_version, claim.episode_id,
            ),
        )

    def _event(self, claim: ClaimVersion, action: str, reason: str, actor: str) -> None:
        self.db.conn.execute(
            "INSERT INTO claim_events(claim_id, version, action, reason, actor, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (claim.claim_id, claim.version, action, reason, actor, _now()),
        )

    def propose(
        self,
        *,
        tenant_id: str,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str | None = None,
        literal_value: Any | None = None,
        qualifiers: dict[str, Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_spans: list[dict[str, Any]] | None = None,
        confidence: float = 0.5,
        episode_id: str | None = None,
        schema_version: str = "claim.v1",
    ) -> ClaimVersion:
        if bool(object_entity_id) == (literal_value is not None):
            raise ValueError("claim requires exactly one object_entity_id or literal_value")
        if not predicate.strip():
            raise ValueError("predicate must not be empty")
        episode = self.sources.get_episode(episode_id) if episode_id else None
        if episode_id and (episode is None or episode.tenant_id != tenant_id):
            raise ValueError("claim episode does not exist in this tenant")
        fingerprint_payload = {
            "tenant": tenant_id, "subject": subject_entity_id,
            "predicate": predicate.strip().casefold(), "object": object_entity_id,
            "literal": literal_value, "qualifiers": qualifiers or {},
            "valid_from": valid_from, "valid_to": valid_to,
            "spans": source_spans or [], "episode": episode_id,
        }
        fingerprint = hashlib.sha256(_json(fingerprint_payload).encode("utf-8")).hexdigest()
        with self.db.lock:
            existing = self.db.conn.execute(
                "SELECT claim_id FROM claim_fingerprints WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            if existing:
                current = self.current(existing["claim_id"])
                assert current is not None
                return current
            claim = ClaimVersion(
                claim_id=uuid.uuid4().hex[:24], version=1, tenant_id=tenant_id,
                subject_entity_id=subject_entity_id,
                predicate=predicate.strip().casefold(), object_entity_id=object_entity_id,
                literal_value=literal_value, qualifiers=qualifiers or {}, status="candidate",
                valid_from=valid_from, valid_to=valid_to, recorded_at=_now(),
                source_spans=tuple(source_spans or []),
                confidence=max(0.0, min(1.0, float(confidence))),
                supersedes=None, schema_version=schema_version, episode_id=episode_id,
            )
            self._insert(claim)
            self.db.conn.execute(
                "INSERT INTO claim_fingerprints VALUES (?, ?)", (fingerprint, claim.claim_id)
            )
            self._event(claim, "proposed", "structured candidate", "extractor")
            self.db.conn.commit()
            return claim

    def current(self, claim_id: str) -> ClaimVersion | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM claim_versions WHERE claim_id=? ORDER BY version DESC LIMIT 1",
                (claim_id,),
            ).fetchone()
        return self._claim(row) if row else None

    def _transition(
        self,
        claim: ClaimVersion,
        status: str,
        *,
        reason: str,
        actor: str,
        valid_to: str | None = None,
        supersedes: str | None = None,
    ) -> ClaimVersion:
        if status not in CLAIM_STATUSES:
            raise ValueError(status)
        updated = replace(
            claim, version=claim.version + 1, status=status,
            valid_to=valid_to if valid_to is not None else claim.valid_to,
            recorded_at=_now(),
            supersedes=supersedes if supersedes is not None else claim.supersedes,
        )
        self._insert(updated)
        self._event(updated, status, reason, actor)
        return updated

    @staticmethod
    def _same_value(left: ClaimVersion, right: ClaimVersion) -> bool:
        return (
            left.object_entity_id == right.object_entity_id
            and left.literal_value == right.literal_value
            and left.qualifiers == right.qualifiers
        )

    def _provenance_valid(self, claim: ClaimVersion) -> bool:
        if not claim.source_spans or not claim.episode_id:
            return False
        episode = self.sources.get_episode(claim.episode_id)
        if episode is None or episode.tenant_id != claim.tenant_id:
            return False
        return all(
            str(span.get("source_id")) == episode.source_id
            and self.sources.validate_span(span, tenant_id=claim.tenant_id)
            for span in claim.source_spans
        )

    def activate(self, claim_id: str, *, actor: str = "reviewer") -> ClaimVersion:
        with self.db.lock:
            candidate = self.current(claim_id)
            if candidate is None:
                raise KeyError(claim_id)
            if candidate.status != "candidate":
                raise ValueError("only a candidate claim can be activated")
            if not self._provenance_valid(candidate):
                raise ValueError("active claim requires an episode and exact valid source spans")
            active = self.active_claims(
                tenant_id=candidate.tenant_id,
                subject_entity_id=candidate.subject_entity_id,
                predicate=candidate.predicate,
            )
            same_qualifiers = [item for item in active if item.qualifiers == candidate.qualifiers]
            for old in same_qualifiers:
                if self._same_value(old, candidate):
                    self._transition(
                        candidate, "rejected", reason=f"duplicate of active claim {old.claim_id}",
                        actor="claim-validator",
                    )
                    self.db.conn.commit()
                    return old
            superseded = same_qualifiers[0] if same_qualifiers else None
            if superseded:
                boundary = candidate.valid_from or candidate.recorded_at
                self._transition(
                    superseded, "superseded",
                    reason=f"superseded by {candidate.claim_id}", actor=actor,
                    valid_to=boundary,
                )
            activated = self._transition(
                candidate, "active", reason="provenance and schema validated", actor=actor,
                supersedes=superseded.claim_id if superseded else None,
            )
            self.db.conn.commit()
            return activated

    def reject(self, claim_id: str, *, reason: str, actor: str = "reviewer") -> ClaimVersion:
        with self.db.lock:
            claim = self.current(claim_id)
            if claim is None:
                raise KeyError(claim_id)
            if claim.status != "candidate":
                raise ValueError("only a candidate may be rejected")
            rejected = self._transition(claim, "rejected", reason=reason, actor=actor)
            self.db.conn.commit()
            return rejected

    def _latest_rows(self, tenant_id: str) -> list[ClaimVersion]:
        with self.db.lock:
            rows = self.db.conn.execute(
                """
                SELECT c.* FROM claim_versions c
                JOIN (SELECT claim_id, MAX(version) AS version FROM claim_versions GROUP BY claim_id) latest
                  ON latest.claim_id=c.claim_id AND latest.version=c.version
                WHERE c.tenant_id=?
                """,
                (tenant_id,),
            ).fetchall()
        return [self._claim(row) for row in rows]

    def active_claims(
        self,
        *,
        tenant_id: str,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
    ) -> list[ClaimVersion]:
        claims = [claim for claim in self._latest_rows(tenant_id) if claim.status == "active"]
        if subject_entity_id:
            claims = [claim for claim in claims if claim.subject_entity_id == subject_entity_id]
        if predicate:
            claims = [claim for claim in claims if claim.predicate == predicate.casefold()]
        return claims

    def query_at(
        self,
        *,
        tenant_id: str,
        valid_at: str,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        recorded_at: str | None = None,
        acl: tuple[str, ...] = ("public",),
    ) -> list[ClaimVersion]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM claim_versions WHERE tenant_id=? ORDER BY claim_id, version",
                (tenant_id,),
            ).fetchall()
        grouped: dict[str, list[ClaimVersion]] = {}
        for row in rows:
            claim = self._claim(row)
            if recorded_at is None or claim.recorded_at <= recorded_at:
                grouped.setdefault(claim.claim_id, []).append(claim)
        visible: list[ClaimVersion] = []
        allowed = set(acl) | {"public"}
        for versions in grouped.values():
            claim = versions[-1]
            if claim.status not in {"active", "superseded"}:
                continue
            if claim.valid_from and claim.valid_from > valid_at:
                continue
            if claim.valid_to and claim.valid_to <= valid_at:
                continue
            if subject_entity_id and claim.subject_entity_id != subject_entity_id:
                continue
            if predicate and claim.predicate != predicate.casefold():
                continue
            sources = [self.sources.get(str(span.get("source_id"))) for span in claim.source_spans]
            if not sources or any(source is None or not (set(source.acl) & allowed) for source in sources):
                continue
            visible.append(claim)
        return sorted(visible, key=lambda item: (item.subject_entity_id, item.predicate))

    def provenance_coverage(self, tenant_id: str) -> float:
        active = self.active_claims(tenant_id=tenant_id)
        return (
            sum(self._provenance_valid(claim) for claim in active) / len(active)
            if active else 1.0
        )

    def snapshot_id(self, tenant_id: str) -> str:
        active = self.active_claims(tenant_id=tenant_id)
        payload = [
            (claim.claim_id, claim.version, claim.valid_from, claim.valid_to)
            for claim in sorted(active, key=lambda item: item.claim_id)
        ]
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:24]

    def events(self, claim_id: str) -> list[dict[str, Any]]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM claim_events WHERE claim_id=? ORDER BY event_id", (claim_id,)
            ).fetchall()
        return [dict(row) for row in rows]
