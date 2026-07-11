"""Facade for source/claim truth and gated Wiki/Graph publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linki.knowledge.claims import ClaimLedger
from linki.knowledge.entities import EntityResolver
from linki.knowledge.graph import TemporalGraphProjection
from linki.knowledge.projections import ProjectionRegistry
from linki.knowledge.sources import SourceRepository
from linki.knowledge.store import get_knowledge_database
from linki.knowledge.wiki import WikiProjection


class KnowledgeService:
    def __init__(self, path: str | Path):
        self.db = get_knowledge_database(path)
        self.sources = SourceRepository(self.db)
        self.entities = EntityResolver(self.db)
        self.claims = ClaimLedger(self.db, self.sources)
        self.wiki = WikiProjection(self.db, self.claims, self.entities, self.sources)
        self.graph = TemporalGraphProjection(self.db, self.claims, self.entities, self.sources)
        self.projections = ProjectionRegistry(self.db)

    def propose_claim(
        self,
        *,
        tenant_id: str,
        subject: str,
        predicate: str,
        source_id: str,
        quote: str,
        literal_value: Any | None = None,
        object_name: str | None = None,
        qualifiers: dict[str, Any] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 0.5,
        event_time=None,
        idempotency_key: str | None = None,
    ):
        source = self.sources.get(source_id)
        if source is None or source.tenant_id != tenant_id:
            raise ValueError("source not found in tenant")
        start = source.content_text.find(quote)
        if start < 0:
            raise ValueError("quote is not a verbatim source span")
        span = {
            "source_id": source_id, "char_start": start,
            "char_end": start + len(quote), "quote": quote,
        }
        episode = self.sources.observe(
            source_id, tenant_id=tenant_id, event_time=event_time,
            source_span=span, kind="claim_observation", idempotency_key=idempotency_key,
        )
        subject_entity = self.entities.resolve(subject, tenant_id=tenant_id)
        object_entity = self.entities.resolve(object_name, tenant_id=tenant_id) if object_name else None
        return self.claims.propose(
            tenant_id=tenant_id, subject_entity_id=subject_entity.entity_id,
            predicate=predicate, object_entity_id=object_entity.entity_id if object_entity else None,
            literal_value=literal_value, qualifiers=qualifiers,
            valid_from=valid_from or episode.event_time, valid_to=valid_to,
            source_spans=[span], confidence=confidence, episode_id=episode.episode_id,
        )

    def publish(self, tenant_id: str):
        claim_snapshot = self.claims.snapshot_id(tenant_id)
        staged = self.projections.stage(tenant_id, claim_snapshot)
        pages = self.wiki.build(tenant_id, staged.snapshot_id)
        graph_stats = self.graph.build(tenant_id, staged.snapshot_id, pages)
        validation = {
            "claims": {"provenance_coverage": self.claims.provenance_coverage(tenant_id)},
            "wiki": self.wiki.validate(tenant_id, staged.snapshot_id),
            "graph": self.graph.validate(tenant_id, staged.snapshot_id),
            "graph_stats": graph_stats,
        }
        return self.projections.promote(staged.snapshot_id, validation)


_SERVICES: dict[str, KnowledgeService] = {}


def get_knowledge_service(settings) -> KnowledgeService:
    path = str(Path(settings.knowledge_path).resolve())
    service = _SERVICES.get(path)
    if service is None:
        service = KnowledgeService(path)
        _SERVICES[path] = service
    return service
