"""Rebuildable temporal graph projection and retrieval adapter."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from linki.knowledge.claims import ClaimLedger, ClaimVersion
from linki.knowledge.entities import EntityResolver, normalize_entity
from linki.knowledge.sources import SourceRepository
from linki.knowledge.store import KnowledgeDatabase
from linki.knowledge.wiki import WikiPage
from linki.retrieval.candidates import Candidate


class GraphStore(Protocol):
    def neighbors(
        self, node_id: str, *, tenant_id: str, snapshot_id: str,
        acl: tuple[str, ...] = ("public",), limit: int = 20,
    ) -> list[dict[str, Any]]: ...


class TemporalGraphProjection:
    def __init__(
        self,
        database: KnowledgeDatabase,
        claims: ClaimLedger,
        entities: EntityResolver,
        sources: SourceRepository,
    ):
        self.db = database
        self.claims = claims
        self.entities = entities
        self.sources = sources

    def _node(self, snapshot: str, tenant: str, node_id: str, node_type: str, label: str, properties=None):
        self.db.conn.execute(
            "INSERT OR REPLACE INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot, node_id, tenant, node_type, label, json.dumps(properties or {}, ensure_ascii=False)),
        )

    def _edge(
        self,
        snapshot: str,
        tenant: str,
        from_id: str,
        relation: str,
        to_id: str,
        claim: ClaimVersion,
    ):
        identity = f"{snapshot}:{from_id}:{relation}:{to_id}:{claim.claim_id}"
        edge_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        self.db.conn.execute(
            "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot, edge_id, tenant, from_id, relation, to_id,
                claim.valid_from, claim.valid_to, claim.recorded_at,
                json.dumps(claim.source_spans, ensure_ascii=False),
            ),
        )

    def build(self, tenant_id: str, snapshot_id: str, pages: list[WikiPage]) -> dict[str, int]:
        active = self.claims.active_claims(tenant_id=tenant_id)
        page_by_claim = {
            claim_id: page for page in pages for claim_id in page.claim_ids
        }
        with self.db.lock:
            self.db.conn.execute("DELETE FROM graph_edges WHERE snapshot_id=?", (snapshot_id,))
            self.db.conn.execute("DELETE FROM graph_nodes WHERE snapshot_id=?", (snapshot_id,))
            for claim in active:
                subject = self.entities.get(claim.subject_entity_id)
                if subject is None:
                    raise ValueError(f"missing subject entity {claim.subject_entity_id}")
                claim_node = f"claim:{claim.claim_id}"
                subject_node = f"entity:{subject.entity_id}"
                self._node(snapshot_id, tenant_id, subject_node, "Entity", subject.canonical_name)
                self._node(snapshot_id, tenant_id, claim_node, "Claim", claim.predicate, claim.as_dict())
                self._edge(snapshot_id, tenant_id, claim_node, "SUBJECT", subject_node, claim)
                self._edge(snapshot_id, tenant_id, subject_node, "HAS_CLAIM", claim_node, claim)
                if claim.object_entity_id:
                    obj = self.entities.get(claim.object_entity_id)
                    if obj is None:
                        raise ValueError(f"missing object entity {claim.object_entity_id}")
                    object_node = f"entity:{obj.entity_id}"
                    self._node(snapshot_id, tenant_id, object_node, "Entity", obj.canonical_name)
                    self._edge(snapshot_id, tenant_id, claim_node, "OBJECT", object_node, claim)
                    self._edge(snapshot_id, tenant_id, subject_node, claim.predicate.upper(), object_node, claim)
                for span in claim.source_spans:
                    source = self.sources.get(str(span["source_id"]))
                    source_node = f"source:{span['source_id']}"
                    self._node(
                        snapshot_id, tenant_id, source_node, "Source",
                        source.uri if source else str(span["source_id"]),
                    )
                    self._edge(snapshot_id, tenant_id, claim_node, "SUPPORTED_BY", source_node, claim)
                if claim.episode_id:
                    episode_node = f"episode:{claim.episode_id}"
                    self._node(snapshot_id, tenant_id, episode_node, "Episode", claim.episode_id)
                    self._edge(snapshot_id, tenant_id, claim_node, "OBSERVED_IN", episode_node, claim)
                if claim.supersedes:
                    old_node = f"claim:{claim.supersedes}"
                    self._node(snapshot_id, tenant_id, old_node, "Claim", "superseded claim")
                    self._edge(snapshot_id, tenant_id, claim_node, "SUPERSEDES", old_node, claim)
                page = page_by_claim.get(claim.claim_id)
                if page:
                    page_node = f"wiki:{page.page_id}"
                    self._node(snapshot_id, tenant_id, page_node, "WikiPage", page.title)
                    self._edge(snapshot_id, tenant_id, page_node, "SUMMARIZES", claim_node, claim)
            self.db.conn.commit()
            node_count = self.db.conn.execute(
                "SELECT COUNT(*) AS n FROM graph_nodes WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()["n"]
            edge_count = self.db.conn.execute(
                "SELECT COUNT(*) AS n FROM graph_edges WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()["n"]
        return {"nodes": int(node_count), "edges": int(edge_count)}

    def neighbors(
        self, node_id: str, *, tenant_id: str, snapshot_id: str,
        acl: tuple[str, ...] = ("public",), limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self.db.lock:
            rows = self.db.conn.execute(
                """
                SELECT e.*, n.node_type AS to_type, n.label AS to_label, n.properties_json
                FROM graph_edges e
                JOIN graph_nodes n ON n.snapshot_id=e.snapshot_id AND n.node_id=e.to_id
                WHERE e.snapshot_id=? AND e.tenant_id=? AND e.from_id=?
                ORDER BY e.relation, e.to_id LIMIT ?
                """,
                (snapshot_id, tenant_id, node_id, limit),
            ).fetchall()
        allowed = set(acl) | {"public"}
        out: list[dict[str, Any]] = []
        for row in rows:
            spans = json.loads(row["source_spans_json"])
            sources = [self.sources.get(str(span.get("source_id"))) for span in spans]
            if not sources or any(source is None or not (set(source.acl) & allowed) for source in sources):
                continue
            out.append({
                **dict(row),
                "source_spans": spans,
                "properties": json.loads(row["properties_json"]),
            })
        return out

    def validate(self, tenant_id: str, snapshot_id: str) -> dict[str, Any]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM graph_edges WHERE snapshot_id=? AND tenant_id=?",
                (snapshot_id, tenant_id),
            ).fetchall()
        valid = 0
        for row in rows:
            spans = json.loads(row["source_spans_json"])
            if spans and all(self.sources.validate_span(span, tenant_id=tenant_id) for span in spans):
                valid += 1
        return {
            "edges": len(rows),
            "provenance_validity": valid / len(rows) if rows else 1.0,
        }


class LedgerGraphRetriever:
    """Adapter from the temporal graph to the common Candidate protocol."""

    def __init__(
        self,
        graph: TemporalGraphProjection,
        database: KnowledgeDatabase,
        claims: ClaimLedger,
        sources: SourceRepository,
        tenant_id: str,
        acl: tuple[str, ...] = ("public",),
    ):
        self.graph, self.db, self.claims, self.sources = graph, database, claims, sources
        self.tenant_id = tenant_id
        self.acl = set(acl) | {"public"}

    def _seed_ids(self, seeds: list[str]) -> list[str]:
        normalized = {normalize_entity(seed) for seed in seeds}
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM entities WHERE tenant_id=?", (self.tenant_id,)
            ).fetchall()
        return [f"entity:{row['entity_id']}" for row in rows if row["normalized_name"] in normalized]

    def retrieve(self, query: str, seed_entities: list[str], snapshot_id: str, limit: int) -> list[Candidate]:
        seeds = self._seed_ids(seed_entities)
        queue = deque((seed, 0) for seed in seeds)
        visited = set(seeds)
        claims_with_hop: dict[str, int] = {}
        while queue:
            node, hop = queue.popleft()
            if hop >= 2:
                continue
            for edge in self.graph.neighbors(
                node, tenant_id=self.tenant_id, snapshot_id=snapshot_id,
                acl=tuple(self.acl), limit=50,
            ):
                target = edge["to_id"]
                if target.startswith("claim:"):
                    claims_with_hop[target.split(":", 1)[1]] = hop + 1
                if target not in visited:
                    visited.add(target)
                    queue.append((target, hop + 1))
        candidates: list[Candidate] = []
        for claim_id, hop in claims_with_hop.items():
            claim = self.claims.current(claim_id)
            if claim is None or claim.status != "active" or not claim.source_spans:
                continue
            span = claim.source_spans[0]
            source = self.sources.get(str(span["source_id"]))
            if source is None or not (set(source.acl) & self.acl):
                continue
            candidates.append(Candidate(
                chunk_id=f"claim:{claim_id}", parent_id=str(span["source_id"]),
                kb="knowledge", source=source.uri if source else str(span["source_id"]),
                heading_path=claim.predicate, child_text=str(span["quote"]),
                retrieval_score=1.0 / hop, channel="temporal-graph",
                metadata={"claim_id": claim_id, "source_id": span["source_id"]},
            ))
        return sorted(candidates, key=lambda item: item.retrieval_score, reverse=True)[:limit]


class ContextualLedgerGraphRetriever:
    """Resolve tenant/ACL from the active request instead of binding them globally."""

    def __init__(self, service):
        self.service = service

    def retrieve(self, query: str, seed_entities: list[str], snapshot_id: str, limit: int) -> list[Candidate]:
        from linki.core.context import RequestContext
        from linki.hooks.base import current_hooks
        from linki.retrieval.graph import PersonalizedPageRankRetriever

        hooks = current_hooks()
        context = getattr(hooks, "request_context", None) or RequestContext()
        active = self.service.projections.active(context.tenant_id)
        if active is None or active.snapshot_id != snapshot_id:
            return []
        base = LedgerGraphRetriever(
            self.service.graph, self.service.db, self.service.claims,
            self.service.sources, context.tenant_id, context.acl,
        )
        seed_ids = base._seed_ids(seed_entities)
        if not seed_ids:
            return []
        with self.service.db.lock:
            rows = self.service.db.conn.execute(
                "SELECT from_id, to_id FROM graph_edges WHERE snapshot_id=? AND tenant_id=?",
                (snapshot_id, context.tenant_id),
            ).fetchall()
        adjacency: dict[str, set[str]] = {}
        for row in rows:
            adjacency.setdefault(row["from_id"], set()).add(row["to_id"])
            adjacency.setdefault(row["to_id"], set()).add(row["from_id"])
        candidates: dict[str, Candidate] = {}
        for claim in self.service.claims.active_claims(tenant_id=context.tenant_id):
            if not claim.source_spans:
                continue
            span = claim.source_spans[0]
            source = self.service.sources.get(str(span["source_id"]))
            if source is None or not (set(source.acl) & (set(context.acl) | {"public"})):
                continue
            node_id = f"claim:{claim.claim_id}"
            candidates[node_id] = Candidate(
                chunk_id=node_id, parent_id=str(span["source_id"]), kb="knowledge",
                source=source.uri, heading_path=claim.predicate,
                child_text=str(span["quote"]), retrieval_score=0.0,
                channel="temporal-graph",
                metadata={"claim_id": claim.claim_id, "source_id": span["source_id"]},
            )
        return PersonalizedPageRankRetriever(
            adjacency, candidates, max_hops=2,
        ).retrieve(query, seed_ids, snapshot_id, limit)
