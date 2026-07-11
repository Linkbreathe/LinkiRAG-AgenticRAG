import pytest

from linki.knowledge.graph import LedgerGraphRetriever
from linki.knowledge.service import KnowledgeService


def setup_claim(service, *, value="PostgreSQL 15", quote=None, valid_from="2026-01-01T00:00:00+00:00", acl=("public",)):
    quote = quote or f"Production uses {value}."
    source = service.sources.ingest(
        tenant_id="tenant-a", source_key=f"decision-{value}", uri=f"meeting://{value}",
        content=quote, acl=acl,
    )
    claim = service.propose_claim(
        tenant_id="tenant-a", subject="Production database", predicate="uses",
        literal_value=value, source_id=source.source_id, quote=quote,
        qualifiers={"domain": "Deployment", "topic": "Database"},
        valid_from=valid_from, confidence=0.99,
        idempotency_key=f"episode-{value}",
    )
    return source, claim


def test_source_and_claim_ingestion_are_idempotent(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    source, claim = setup_claim(service)
    same_source = service.sources.ingest(
        tenant_id="tenant-a", source_key="decision-PostgreSQL 15",
        uri="meeting://PostgreSQL 15", content="Production uses PostgreSQL 15.",
    )
    _, same_claim = setup_claim(service)
    assert source.source_id == same_source.source_id
    assert same_source.version == 1
    assert claim.claim_id == same_claim.claim_id


def test_active_claim_requires_exact_source_provenance(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    subject = service.entities.resolve("Service", tenant_id="tenant-a")
    candidate = service.claims.propose(
        tenant_id="tenant-a", subject_entity_id=subject.entity_id,
        predicate="uses", literal_value="Redis", source_spans=[],
    )
    with pytest.raises(ValueError, match="source spans"):
        service.claims.activate(candidate.claim_id)
    assert service.claims.current(candidate.claim_id).status == "candidate"


def test_supersession_preserves_current_historical_and_recorded_time_views(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    _, old_candidate = setup_claim(service)
    old = service.claims.activate(old_candidate.claim_id)
    _, new_candidate = setup_claim(
        service, value="PostgreSQL 16", valid_from="2026-06-01T00:00:00+00:00",
    )
    new = service.claims.activate(new_candidate.claim_id)

    old_latest = service.claims.current(old.claim_id)
    assert old_latest.status == "superseded"
    assert old_latest.valid_to == "2026-06-01T00:00:00+00:00"
    assert new.status == "active"
    assert new.supersedes == old.claim_id
    subject = new.subject_entity_id

    historical = service.claims.query_at(
        tenant_id="tenant-a", subject_entity_id=subject, predicate="uses",
        valid_at="2026-03-01T00:00:00+00:00",
    )
    current = service.claims.query_at(
        tenant_id="tenant-a", subject_entity_id=subject, predicate="uses",
        valid_at="2026-07-01T00:00:00+00:00",
    )
    known_before_update = service.claims.query_at(
        tenant_id="tenant-a", subject_entity_id=subject, predicate="uses",
        valid_at="2026-07-01T00:00:00+00:00", recorded_at=old.recorded_at,
    )
    assert [claim.literal_value for claim in historical] == ["PostgreSQL 15"]
    assert [claim.literal_value for claim in current] == ["PostgreSQL 16"]
    assert [claim.literal_value for claim in known_before_update] == ["PostgreSQL 15"]
    assert service.claims.provenance_coverage("tenant-a") == 1.0


def test_claim_query_enforces_source_acl_before_returning_fact(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    _, candidate = setup_claim(service, acl=("confidential",))
    active = service.claims.activate(candidate.claim_id)
    query = dict(
        tenant_id="tenant-a", subject_entity_id=active.subject_entity_id,
        predicate="uses", valid_at="2026-03-01T00:00:00+00:00",
    )
    assert service.claims.query_at(**query, acl=("public",)) == []
    assert service.claims.query_at(**query, acl=("confidential",)) == [active]
    snapshot = service.publish("tenant-a")
    assert service.wiki.list("tenant-a", snapshot.snapshot_id, ("public",)) == []
    assert service.wiki.list("tenant-a", snapshot.snapshot_id, ("confidential",))
    assert service.graph.neighbors(
        f"entity:{active.subject_entity_id}", tenant_id="tenant-a",
        snapshot_id=snapshot.snapshot_id, acl=("public",),
    ) == []
    assert service.graph.neighbors(
        f"entity:{active.subject_entity_id}", tenant_id="tenant-a",
        snapshot_id=snapshot.snapshot_id, acl=("confidential",),
    )


def test_publish_builds_cited_wiki_and_provenance_graph_then_can_rollback(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    _, candidate = setup_claim(service)
    active = service.claims.activate(candidate.claim_id)
    first = service.publish("tenant-a")
    assert first.status == "active"
    assert first.validation["claims"]["provenance_coverage"] == 1.0
    assert first.validation["wiki"]["claim_coverage"] == 1.0
    assert first.validation["graph"]["provenance_validity"] == 1.0

    pages = service.wiki.list("tenant-a", first.snapshot_id)
    assert len(pages) == 1
    assert f"claim:{active.claim_id}" in pages[0].markdown
    assert "source:meeting://" in pages[0].markdown
    neighbors = service.graph.neighbors(
        f"entity:{active.subject_entity_id}", tenant_id="tenant-a",
        snapshot_id=first.snapshot_id,
    )
    assert any(edge["relation"] == "HAS_CLAIM" for edge in neighbors)
    graph_retriever = LedgerGraphRetriever(
        service.graph, service.db, service.claims, service.sources, "tenant-a",
    )
    hits = graph_retriever.retrieve(
        "database", ["Production database"], first.snapshot_id, 5,
    )
    assert hits and hits[0].metadata["claim_id"] == active.claim_id

    _, replacement = setup_claim(
        service, value="PostgreSQL 16", valid_from="2026-06-01T00:00:00+00:00",
    )
    service.claims.activate(replacement.claim_id)
    second = service.publish("tenant-a")
    assert second.snapshot_id != first.snapshot_id
    service.projections.rollback("tenant-a", first.snapshot_id)
    assert service.projections.active("tenant-a").snapshot_id == first.snapshot_id


def test_failed_projection_gate_never_switches_active_alias(tmp_path):
    service = KnowledgeService(tmp_path / "knowledge.sqlite3")
    _, candidate = setup_claim(service)
    service.claims.activate(candidate.claim_id)
    active = service.publish("tenant-a")
    staged = service.projections.stage("tenant-a", "deliberately-bad-claim-snapshot")
    with pytest.raises(ValueError, match="failed"):
        service.projections.promote(staged.snapshot_id, {
            "claims": {"provenance_coverage": 1.0},
            "wiki": {"claim_coverage": 0.9, "source_span_validity": 1.0},
            "graph": {"provenance_validity": 1.0},
        })
    assert service.projections.active("tenant-a").snapshot_id == active.snapshot_id
