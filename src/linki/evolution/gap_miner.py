"""Cluster failure observations into a backlog; never synthesize missing facts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from linki.evolution.feedback import Feedback, FeedbackLedger
from linki.evolution.store import EvolutionDatabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _terms(text: str) -> set[str]:
    stop = {"what", "which", "where", "when", "how", "the", "a", "an", "is", "are", "of", "to", "什么", "如何", "哪里"}
    return {
        token for token in re.findall(r"\w+|[\u4e00-\u9fff]", (text or "").casefold())
        if token not in stop and len(token) > 1
    }


@dataclass(frozen=True)
class KnowledgeGap:
    gap_id: str
    version: int
    tenant_id: str
    cluster_label: str
    example_questions: tuple[str, ...]
    frequency: int
    affected_users: int
    business_impact: str
    missing_entities_or_claims: tuple[str, ...]
    nearest_sources: tuple[str, ...]
    suggested_source_to_add: str
    status: str
    feedback_ids: tuple[str, ...]
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GapMiner:
    def __init__(self, database: EvolutionDatabase, feedback: FeedbackLedger):
        self.db, self.feedback = database, feedback

    @staticmethod
    def _gap(row) -> KnowledgeGap:
        return KnowledgeGap(
            gap_id=row["gap_id"], version=int(row["version"]), tenant_id=row["tenant_id"],
            cluster_label=row["cluster_label"],
            example_questions=tuple(json.loads(row["example_questions_json"])),
            frequency=int(row["frequency"]), affected_users=int(row["affected_users"]),
            business_impact=row["business_impact"],
            missing_entities_or_claims=tuple(json.loads(row["missing_entities_json"])),
            nearest_sources=tuple(json.loads(row["nearest_sources_json"])),
            suggested_source_to_add=row["suggested_source_to_add"], status=row["status"],
            feedback_ids=tuple(json.loads(row["feedback_ids_json"])), updated_at=row["updated_at"],
        )

    def _insert(self, gap: KnowledgeGap) -> None:
        self.db.conn.execute(
            "INSERT INTO gap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                gap.gap_id, gap.version, gap.tenant_id, gap.cluster_label,
                json.dumps(gap.example_questions, ensure_ascii=False), gap.frequency,
                gap.affected_users, gap.business_impact,
                json.dumps(gap.missing_entities_or_claims, ensure_ascii=False),
                json.dumps(gap.nearest_sources, ensure_ascii=False),
                gap.suggested_source_to_add, gap.status,
                json.dumps(gap.feedback_ids), gap.updated_at,
            ),
        )

    @staticmethod
    def _similar(left: set[str], right: set[str]) -> bool:
        return bool(left and right) and len(left & right) / len(left | right) >= 0.55

    def mine(self, tenant_id: str, *, min_frequency: int = 2) -> list[KnowledgeGap]:
        observations = self.feedback.list_current(
            tenant_id=tenant_id, kinds=("not_found", "verifier_issue"),
            statuses=("observed", "diagnosed"), acl=("public",),
        )
        clusters: list[list[Feedback]] = []
        signatures: list[set[str]] = []
        for item in observations:
            payload = item.payload
            signature = _terms(
                " ".join(str(payload.get(key, "")) for key in ("question", "normalized_query", "missing_support", "entities"))
            )
            placed = False
            for index, existing in enumerate(signatures):
                same_kb = clusters[index][0].payload.get("target_kb") == payload.get("target_kb")
                if same_kb and self._similar(signature, existing):
                    clusters[index].append(item)
                    signatures[index] |= signature
                    placed = True
                    break
            if not placed:
                clusters.append([item])
                signatures.append(set(signature))

        gaps: list[KnowledgeGap] = []
        with self.db.lock:
            for cluster, signature in zip(clusters, signatures, strict=True):
                if len(cluster) < min_frequency:
                    continue
                label = " ".join(sorted(signature)[:8]) or "unclassified knowledge gap"
                kb = str(cluster[0].payload.get("target_kb") or "default")
                gap_id = hashlib.sha256(f"{tenant_id}:{kb}:{label}".encode("utf-8")).hexdigest()[:24]
                row = self.db.conn.execute(
                    "SELECT * FROM gap_versions WHERE gap_id=? ORDER BY version DESC LIMIT 1", (gap_id,)
                ).fetchone()
                previous = self._gap(row) if row else None
                questions = tuple(dict.fromkeys(str(item.payload.get("question") or "") for item in cluster if item.payload.get("question")))[:5]
                nearest = tuple(sorted({str(source) for item in cluster for source in item.payload.get("nearest_sources", [])}))
                impact = "high" if len(cluster) >= 10 else ("medium" if len(cluster) >= 4 else "low")
                gap = KnowledgeGap(
                    gap_id=gap_id, version=(previous.version + 1 if previous else 1),
                    tenant_id=tenant_id, cluster_label=label,
                    example_questions=questions, frequency=len(cluster),
                    affected_users=len({item.user_id for item in cluster}), business_impact=impact,
                    missing_entities_or_claims=tuple(sorted(signature)), nearest_sources=nearest,
                    suggested_source_to_add=f"Add a reviewed source covering: {label}",
                    status=previous.status if previous else "open",
                    feedback_ids=tuple(item.feedback_id for item in cluster), updated_at=_now(),
                )
                if previous is None or (
                    previous.frequency, previous.feedback_ids, previous.status
                ) != (gap.frequency, gap.feedback_ids, gap.status):
                    self._insert(gap)
                else:
                    gap = previous
                gaps.append(gap)
            self.db.conn.commit()
        return gaps

    def list_current(self, tenant_id: str, status: str | None = None) -> list[KnowledgeGap]:
        with self.db.lock:
            rows = self.db.conn.execute(
                """
                SELECT g.* FROM gap_versions g
                JOIN (SELECT gap_id, MAX(version) AS version FROM gap_versions GROUP BY gap_id) latest
                  ON latest.gap_id=g.gap_id AND latest.version=g.version
                WHERE g.tenant_id=? ORDER BY g.frequency DESC, g.updated_at DESC
                """,
                (tenant_id,),
            ).fetchall()
        gaps = [self._gap(row) for row in rows]
        return [gap for gap in gaps if status is None or gap.status == status]

    def set_status(self, gap_id: str, status: str) -> KnowledgeGap:
        if status not in {"open", "documented", "wont_fix"}:
            raise ValueError(status)
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM gap_versions WHERE gap_id=? ORDER BY version DESC LIMIT 1", (gap_id,)
            ).fetchone()
            if row is None:
                raise KeyError(gap_id)
            current = self._gap(row)
            updated = replace(current, version=current.version + 1, status=status, updated_at=_now())
            self._insert(updated)
            self.db.conn.commit()
            return updated
