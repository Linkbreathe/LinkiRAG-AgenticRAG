from fastapi.testclient import TestClient

from conftest import FakeLLM, ev

from linki.config import Settings
from linki.knowledge.service import get_knowledge_service
from linki.ui.app import build_app


def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        qdrant_path=tmp_path / "qdrant",
        parent_store_path=tmp_path / "parents",
        markdown_dir=tmp_path / "markdown",
        cache_path=tmp_path / "cache.sqlite3",
        snapshot_manifest_path=tmp_path / "snapshots.json",
        memory_path=tmp_path / "memory.sqlite3",
        knowledge_path=tmp_path / "knowledge.sqlite3",
        evolution_path=tmp_path / "evolution.sqlite3",
        enable_trace=False,
        enable_persistent_cache=False,
        enable_answer_cache=False,
        memory_background_formation=False,
        gap_min_frequency=2,
    )


def test_chat_api_exposes_policy_cost_snapshots_pack_and_versions(tmp_path):
    config = settings(tmp_path)
    llm = FakeLLM(answer=lambda s, h: "The answer is grounded [1].")
    client = TestClient(build_app(config, llm, llm, lambda q, kb: [ev("doc")]))
    response = client.post("/api/chat", json={
        "message": "What is the answer?", "topic": "default", "mode": "auto",
        "tenant": "tenant-a", "user": "user-a", "acl": ["public"],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["policy_path"] == "p1"
    assert data["cost"]["llm_calls"] == 1
    assert data["run_id"]
    assert data["evidence_pack_id"]
    assert data["evidence_pack"]["token_count"] <= 1200
    assert data["versions"]["policy"] == "adaptive.v2"
    assert data["citations"][0]["label"] == "doc.pdf · Section"
    assert any(step["kind"] == "cost" for step in data["trace"])
    assert client.post("/api/chat", json={
        "message": "q", "topic": "default", "mode": "unsafe",
    }).status_code == 400


def test_memory_feedback_gap_and_wiki_apis_are_scoped_and_auditable(tmp_path):
    config = settings(tmp_path)
    llm = FakeLLM()
    client = TestClient(build_app(config, llm, llm, lambda q, kb: []))

    remembered = client.post("/api/memory", json={
        "statement": "code examples should use Python",
        "tenant": "tenant-a", "user": "user-a",
    })
    assert remembered.status_code == 200
    memory_id = remembered.json()["item"]["memory_id"]
    listed = client.get("/api/memory", params={"tenant": "tenant-a", "user": "user-a"}).json()
    assert [item["memory_id"] for item in listed["items"]] == [memory_id]
    assert client.get("/api/memory", params={"tenant": "tenant-b", "user": "user-a"}).json()["items"] == []
    provenance = client.get(
        f"/api/memory/{memory_id}/provenance",
        params={"tenant": "tenant-a", "user": "user-a"},
    ).json()
    assert provenance["episodes"] and provenance["events"]

    for run_id, question in (("r1", "What is project X retention policy?"), ("r2", "Where is project X retention policy documented?")):
        response = client.post("/api/feedback", json={
            "kind": "not_found", "run_id": run_id,
            "tenant": "tenant-a", "user": f"user-{run_id}",
            "payload": {"question": question, "target_kb": "Retrieve_policy", "missing_support": "retention policy"},
        })
        assert response.status_code == 200
    gaps = client.get("/api/gaps", params={"tenant": "tenant-a"}).json()["gaps"]
    assert len(gaps) == 1 and gaps[0]["frequency"] == 2

    knowledge = get_knowledge_service(config)
    source = knowledge.sources.ingest(
        tenant_id="tenant-a", source_key="decision", uri="meeting://decision",
        content="Production uses PostgreSQL 16.",
    )
    claim = knowledge.propose_claim(
        tenant_id="tenant-a", subject="Production database", predicate="uses",
        literal_value="PostgreSQL 16", source_id=source.source_id,
        quote="Production uses PostgreSQL 16.",
    )
    knowledge.claims.activate(claim.claim_id)
    knowledge.publish("tenant-a")
    pages = client.get("/api/wiki", params={"tenant": "tenant-a"}).json()["pages"]
    assert len(pages) == 1
    page = client.get(
        f"/api/wiki/{pages[0]['slug']}", params={"tenant": "tenant-a"},
    ).json()["page"]
    assert f"claim:{claim.claim_id}" in page["markdown"]

    deleted = client.delete(
        f"/api/memory/{memory_id}", params={"tenant": "tenant-a", "user": "user-a"},
    )
    assert deleted.status_code == 200
