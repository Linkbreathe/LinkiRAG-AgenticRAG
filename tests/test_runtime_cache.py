import asyncio
import time

from conftest import FakeLLM, FakeSettings, ev

from linki.cache.base import make_cache_key
from linki.cache.sqlite import SQLiteCache
from linki.core.context import RequestContext
from linki.graph.workflow import answer_question_async, build_workflow
from linki.knowledge.snapshots import SnapshotManifest


def test_request_context_and_cache_key_isolate_tenant_acl_and_snapshot():
    context = RequestContext("tenant-a", "user-a", ("read:z", "read:a", "read:a"))
    assert context.acl == ("read:a", "read:z")
    base = dict(query="q", tenant=context.tenant_id, acl=context.acl_hash)
    key = make_cache_key("answer", **base, snapshot="s1")
    assert key != make_cache_key("answer", **{**base, "tenant": "tenant-b"}, snapshot="s1")
    assert key != make_cache_key("answer", **base, snapshot="s2")


def test_sqlite_cache_expires_without_deserializing_pickle(tmp_path, monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr("linki.cache.sqlite.time.time", lambda: now["value"])
    cache = SQLiteCache(tmp_path / "cache.sqlite3")
    cache.set("test", "key", {"safe": [1, 2]}, ttl_seconds=10)
    assert cache.get("test", "key") == {"safe": [1, 2]}
    now["value"] = 111.0
    assert cache.get("test", "key") is None


def test_snapshot_manifest_is_immutable_and_rollback_moves_only_pointer(tmp_path):
    manifest = SnapshotManifest(tmp_path / "manifest.json")
    first = manifest.promote(
        "default", artifact_digest="a", index_version="v1", stats={"children": 1}
    )
    second = manifest.promote(
        "default", artifact_digest="b", index_version="v1", stats={"children": 2}
    )
    assert first.snapshot_id != second.snapshot_id
    assert second.parent_snapshot_id == first.snapshot_id
    assert manifest.active_id("default") == second.snapshot_id
    manifest.rollback("default", first.snapshot_id)
    assert manifest.active_id("default") == first.snapshot_id
    assert manifest.get(second.snapshot_id) == second


def _cached_settings(tmp_path):
    settings = FakeSettings()
    settings.adaptive_enabled = True
    settings.execution_mode = "auto"
    settings.default_deadline_ms = None
    settings.adaptive_low_score_threshold = 0.0
    settings.adaptive_min_score_margin = 0.0
    settings.data_dir = tmp_path
    settings.enable_trace = False
    settings.enable_cache = True
    settings.enable_dedup = True
    settings.enable_hook_trace = False
    settings.enable_persistent_cache = True
    settings.enable_answer_cache = True
    settings.answer_cache_ttl_seconds = 3600
    settings.retrieval_cache_ttl_seconds = 3600
    settings.cache_path = tmp_path / "cache.sqlite3"
    settings.snapshot_manifest_path = tmp_path / "manifest.json"
    SnapshotManifest(settings.snapshot_manifest_path).promote(
        "default", artifact_digest="initial", index_version="v1", stats={"children": 1}
    )
    return settings


def test_async_singleflight_exact_cache_tenant_and_snapshot_correctness(tmp_path):
    settings = _cached_settings(tmp_path)
    answer_calls = {"n": 0}
    retrieval_calls = {"n": 0}

    def answer(system, human):
        answer_calls["n"] += 1
        time.sleep(0.05)
        return "The release is June [1]."

    llm = FakeLLM(answer=answer)
    app = build_workflow()

    def retrieve(query, kb):
        retrieval_calls["n"] += 1
        return [ev("release")]

    async def ask(context):
        return await answer_question_async(
            "What is the release date?", model=llm, judge=llm, settings=settings,
            retrieve_fn=retrieve, app=app,
            request_context=context,
        )

    tenant_a = RequestContext("tenant-a", "user-a", ("public",))
    async def concurrent_requests():
        return await asyncio.gather(*(ask(tenant_a) for _ in range(16)))

    concurrent = asyncio.run(concurrent_requests())
    assert answer_calls["n"] == 1
    assert retrieval_calls["n"] == 1
    assert sum(result["cache"]["coalesced"] for result in concurrent) == 15
    assert sorted(result["cost"]["llm_calls"] for result in concurrent) == [0] * 15 + [1]

    hot = asyncio.run(ask(tenant_a))
    assert hot["cache"] == {"hit": True, "coalesced": False, "source": "answer"}
    assert hot["cost"]["llm_calls"] == 0
    assert answer_calls["n"] == 1

    tenant_b = RequestContext("tenant-b", "user-a", ("public",))
    isolated = asyncio.run(ask(tenant_b))
    assert isolated["cache"]["hit"] is False
    assert answer_calls["n"] == 2

    SnapshotManifest(settings.snapshot_manifest_path).promote(
        "default", artifact_digest="changed", index_version="v1", stats={"children": 2}
    )
    invalidated = asyncio.run(ask(tenant_a))
    assert invalidated["cache"]["hit"] is False
    assert answer_calls["n"] == 3
    assert retrieval_calls["n"] == 3
