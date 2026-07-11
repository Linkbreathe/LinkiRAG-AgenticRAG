"""Constraint gates, stable canaries, promotion and automatic rollback."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from linki.evolution.store import EvolutionDatabase

ARTIFACT_TYPES = {
    "router", "retrieval", "prompt", "memory_rule", "wiki_summary",
    "claim_extractor", "index",
}
HUMAN_APPROVAL_REQUIRED = {"prompt", "memory_rule", "claim_extractor"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _metric(metrics: dict[str, Any], group: str, name: str) -> float | None:
    value = metrics.get(group, {}).get(name)
    return float(value) if value is not None else None


@dataclass(frozen=True)
class ReleaseCriteria:
    min_all_support_recall: float
    min_faithfulness: float
    max_p95_latency_ms: float
    max_error_rate: float
    max_memory_conflict_rate: float = 0.0
    min_topk_jaccard: float = 0.8
    max_error_variance: float = 0.02

    @classmethod
    def from_baseline(
        cls, baseline: dict[str, Any], *, quality_tolerance: float = 0.02,
        latency_ratio: float = 1.10,
    ) -> "ReleaseCriteria":
        return cls(
            min_all_support_recall=max(0.0, _metric(baseline, "quality", "all_support_recall") - quality_tolerance),
            min_faithfulness=max(0.0, _metric(baseline, "quality", "faithfulness") - quality_tolerance),
            max_p95_latency_ms=_metric(baseline, "efficiency", "p95_latency_ms") * latency_ratio,
            max_error_rate=_metric(baseline, "stability", "error_rate") + 0.005,
        )


@dataclass(frozen=True)
class GateReport:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(metrics: dict[str, Any], criteria: ReleaseCriteria) -> GateReport:
    reasons: list[str] = []

    def minimum(group: str, name: str, floor: float):
        value = _metric(metrics, group, name)
        if value is None or value < floor:
            reasons.append(f"{group}.{name}={value} below floor {floor}")

    def maximum(group: str, name: str, ceiling: float):
        value = _metric(metrics, group, name)
        if value is None or value > ceiling:
            reasons.append(f"{group}.{name}={value} above ceiling {ceiling}")

    minimum("quality", "all_support_recall", criteria.min_all_support_recall)
    minimum("quality", "faithfulness", criteria.min_faithfulness)
    minimum("quality", "citation_integrity", 1.0)
    maximum("quality", "privacy_leakage", 0.0)
    maximum("efficiency", "p95_latency_ms", criteria.max_p95_latency_ms)
    maximum("stability", "error_rate", criteria.max_error_rate)
    minimum("stability", "topk_jaccard", criteria.min_topk_jaccard)
    maximum("stability", "error_variance", criteria.max_error_variance)
    maximum("memory", "conflict_rate", criteria.max_memory_conflict_rate)
    minimum("memory", "delete_completeness", 1.0)
    minimum("knowledge", "provenance_coverage", 1.0)
    minimum("knowledge", "rollback_test", 1.0)
    return GateReport(not reasons, tuple(reasons), metrics)


@dataclass(frozen=True)
class ReleaseVersion:
    release_id: str
    version: int
    tenant_id: str
    artifact_type: str
    artifact_version: str
    artifact: dict[str, Any]
    parent_release_id: str | None
    status: str
    train_hash: str
    dev_hash: str
    test_hash: str
    evaluations: dict[str, Any]
    change_card: dict[str, Any]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReleaseManager:
    def __init__(self, database: EvolutionDatabase):
        self.db = database

    @staticmethod
    def _release(row) -> ReleaseVersion:
        return ReleaseVersion(
            release_id=row["release_id"], version=int(row["version"]),
            tenant_id=row["tenant_id"], artifact_type=row["artifact_type"],
            artifact_version=row["artifact_version"], artifact=json.loads(row["artifact_json"]),
            parent_release_id=row["parent_release_id"], status=row["status"],
            train_hash=row["train_hash"], dev_hash=row["dev_hash"], test_hash=row["test_hash"],
            evaluations=json.loads(row["evaluations_json"]),
            change_card=json.loads(row["change_card_json"]), created_at=row["created_at"],
        )

    def _insert(self, item: ReleaseVersion) -> None:
        self.db.conn.execute(
            "INSERT INTO registry_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.release_id, item.version, item.tenant_id, item.artifact_type,
                item.artifact_version, json.dumps(item.artifact, ensure_ascii=False),
                item.parent_release_id, item.status, item.train_hash, item.dev_hash,
                item.test_hash, json.dumps(item.evaluations, ensure_ascii=False),
                json.dumps(item.change_card, ensure_ascii=False), item.created_at,
            ),
        )

    def current(self, release_id: str) -> ReleaseVersion | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM registry_versions WHERE release_id=? ORDER BY version DESC LIMIT 1",
                (release_id,),
            ).fetchone()
        return self._release(row) if row else None

    def active(self, tenant_id: str, artifact_type: str) -> ReleaseVersion | None:
        with self.db.lock:
            row = self.db.conn.execute(
                """SELECT r.* FROM registry_active a
                   JOIN registry_versions r ON r.release_id=a.release_id
                   WHERE a.tenant_id=? AND a.artifact_type=?
                   ORDER BY r.version DESC LIMIT 1""",
                (tenant_id, artifact_type),
            ).fetchone()
        return self._release(row) if row else None

    def register_candidate(
        self,
        *,
        tenant_id: str,
        artifact_type: str,
        artifact_version: str,
        artifact: dict[str, Any],
        train_hash: str,
        dev_hash: str,
        test_hash: str,
        change_card: dict[str, Any],
    ) -> ReleaseVersion:
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unsupported artifact type: {artifact_type}")
        if not all((train_hash, dev_hash, test_hash)) or len({train_hash, dev_hash, test_hash}) != 3:
            raise ValueError("train/dev/test hashes must be non-empty and mutually distinct")
        required_card = {"what_changed", "why", "data_scope", "known_limitations"}
        if not required_card <= change_card.keys():
            raise ValueError(f"change card missing: {sorted(required_card - change_card.keys())}")
        with self.db.lock:
            existing = self.db.conn.execute(
                "SELECT * FROM registry_versions WHERE tenant_id=? AND artifact_type=? AND artifact_version=?",
                (tenant_id, artifact_type, artifact_version),
            ).fetchone()
            if existing:
                return self.current(existing["release_id"])
            parent = self.active(tenant_id, artifact_type)
            item = ReleaseVersion(
                release_id=uuid.uuid4().hex[:24], version=1,
                tenant_id=tenant_id, artifact_type=artifact_type,
                artifact_version=artifact_version, artifact=dict(artifact),
                parent_release_id=parent.release_id if parent else None,
                status="proposal", train_hash=train_hash, dev_hash=dev_hash,
                test_hash=test_hash, evaluations={}, change_card=dict(change_card),
                created_at=_now(),
            )
            self._insert(item)
            self.db.conn.commit()
            return item

    def _transition(
        self, item: ReleaseVersion, status: str,
        *,
        stage: str | None = None,
        report: GateReport | None = None,
    ) -> ReleaseVersion:
        evaluations = dict(item.evaluations)
        if stage and report:
            evaluations[stage] = report.as_dict()
        updated = replace(
            item, version=item.version + 1, status=status,
            evaluations=evaluations, created_at=_now(),
        )
        self._insert(updated)
        return updated

    def evaluate(
        self,
        release_id: str,
        *,
        stage: str,
        dataset_hash: str,
        metrics: dict[str, Any],
        criteria: ReleaseCriteria,
    ) -> ReleaseVersion:
        expected = {
            "offline_dev": ("proposal", "dev_hash", "offline_passed"),
            "test": ("offline_passed", "test_hash", "test_passed"),
            "shadow": ("test_passed", None, "shadow_passed"),
            "canary": ("canary", None, "canary_passed"),
        }
        if stage not in expected:
            raise ValueError("stage must be offline_dev, test, shadow, or canary")
        with self.db.lock:
            item = self.current(release_id)
            if item is None:
                raise KeyError(release_id)
            required_status, hash_field, passed_status = expected[stage]
            if item.status != required_status:
                raise ValueError(f"{stage} requires status {required_status}, got {item.status}")
            if hash_field and dataset_hash != getattr(item, hash_field):
                raise ValueError(f"{stage} dataset hash does not match locked {hash_field}")
            report = evaluate_gate(metrics, criteria)
            status = passed_status if report.passed else "rejected"
            updated = self._transition(item, status, stage=stage, report=report)
            self.db.conn.commit()
            return updated

    def start_canary(self, release_id: str) -> ReleaseVersion:
        with self.db.lock:
            item = self.current(release_id)
            if item is None:
                raise KeyError(release_id)
            if item.status != "shadow_passed":
                raise ValueError("canary requires a passed test and shadow replay")
            updated = self._transition(item, "canary")
            self.db.conn.commit()
            return updated

    @staticmethod
    def in_canary(
        release_id: str, tenant_id: str, user_id: str,
        *, percentage: float,
    ) -> bool:
        if not 0 <= percentage <= 100:
            raise ValueError("percentage must be in [0, 100]")
        bucket = int(hashlib.sha256(
            f"{release_id}:{tenant_id}:{user_id}".encode("utf-8")
        ).hexdigest()[:8], 16) / 0xFFFFFFFF * 100
        return bucket < percentage

    def promote(self, release_id: str, *, human_approved: bool = False) -> ReleaseVersion:
        with self.db.lock:
            item = self.current(release_id)
            if item is None:
                raise KeyError(release_id)
            if item.status != "canary_passed":
                raise ValueError("promotion requires offline test, shadow, and canary gates")
            if item.artifact_type in HUMAN_APPROVAL_REQUIRED and not human_approved:
                raise ValueError(f"{item.artifact_type} promotion requires human approval")
            old = self.active(item.tenant_id, item.artifact_type)
            if old and old.release_id != item.release_id:
                self._transition(old, "superseded")
            card = {
                **item.change_card,
                "rollback_release_id": old.release_id if old else None,
                "evaluations": item.evaluations,
            }
            promoted = replace(item, change_card=card)
            promoted = self._transition(promoted, "active")
            self.db.conn.execute(
                "INSERT INTO registry_active VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id, artifact_type) DO UPDATE SET release_id=excluded.release_id",
                (item.tenant_id, item.artifact_type, item.release_id),
            )
            self.db.conn.commit()
            return promoted

    def rollback(self, release_id: str, *, reason: str) -> ReleaseVersion:
        with self.db.lock:
            item = self.current(release_id)
            if item is None:
                raise KeyError(release_id)
            if item.status != "active" or not item.parent_release_id:
                raise ValueError("only an active release with a prior active version can roll back")
            rolled = replace(
                self._transition(item, "rolled_back"),
                change_card={**item.change_card, "rollback_reason": reason},
            )
            # Persist the card-bearing rollback version instead of mutating the
            # row that was just appended.
            rolled = replace(rolled, version=rolled.version + 1, created_at=_now())
            self._insert(rolled)
            parent = self.current(item.parent_release_id)
            if parent is None:
                raise RuntimeError("rollback parent is missing")
            restored = self._transition(parent, "active")
            self.db.conn.execute(
                "UPDATE registry_active SET release_id=? WHERE tenant_id=? AND artifact_type=?",
                (restored.release_id, item.tenant_id, item.artifact_type),
            )
            self.db.conn.commit()
            return rolled

    def monitor_active(
        self,
        release_id: str,
        *,
        metrics: dict[str, Any],
        criteria: ReleaseCriteria,
    ) -> tuple[GateReport, ReleaseVersion | None]:
        report = evaluate_gate(metrics, criteria)
        if report.passed:
            return report, None
        return report, self.rollback(
            release_id, reason="; ".join(report.reasons),
        )
