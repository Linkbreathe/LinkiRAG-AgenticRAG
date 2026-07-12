import pytest

from conftest import FakeLLM, FakeSettings

from linki.evolution.feedback import FeedbackLedger
from linki.evolution.gap_miner import GapMiner
from linki.evolution.optimizer import shadow_replay
from linki.evolution.release import ReleaseCriteria, ReleaseManager, evaluate_gate
from linki.evolution.store import EvolutionDatabase
from linki.evolution.service import get_evolution_service
from linki.graph.workflow import answer_question, build_workflow


def database(tmp_path):
    return EvolutionDatabase(tmp_path / "evolution.sqlite3")


def good_metrics():
    return {
        "quality": {
            "all_support_recall": 0.82, "faithfulness": 0.91,
            "citation_integrity": 1.0, "privacy_leakage": 0.0,
        },
        "efficiency": {"p95_latency_ms": 900.0},
        "memory": {"conflict_rate": 0.0, "delete_completeness": 1.0},
        "knowledge": {"provenance_coverage": 1.0, "rollback_test": 1.0},
        "stability": {"error_rate": 0.01, "topk_jaccard": 0.9, "error_variance": 0.005},
    }


def criteria():
    return ReleaseCriteria(
        min_all_support_recall=0.8, min_faithfulness=0.9,
        max_p95_latency_ms=1000, max_error_rate=0.02,
    )


def test_feedback_is_idempotent_and_correction_without_source_stays_signal(tmp_path):
    ledger = FeedbackLedger(database(tmp_path))
    item = ledger.record(
        tenant_id="tenant-a", user_id="user-a", kind="correction",
        payload={"correction": "Use PostgreSQL 16"}, idempotency_key="same",
    )
    duplicate = ledger.record(
        tenant_id="tenant-a", user_id="user-a", kind="correction",
        payload={"correction": "Use PostgreSQL 16"}, idempotency_key="same",
    )
    assert item.feedback_id == duplicate.feedback_id
    assert item.status == "needs_source"
    assert item.payload["correction"] == "Use PostgreSQL 16"


def test_gap_miner_only_creates_backlog_and_counts_distinct_users(tmp_path):
    db = database(tmp_path)
    feedback = FeedbackLedger(db)
    for index, (user, question) in enumerate((
        ("u1", "What is project X retention policy?"),
        ("u2", "Where is project X retention policy documented?"),
    )):
        feedback.record(
            tenant_id="tenant-a", user_id=user, kind="not_found",
            payload={
                "question": question, "target_kb": "Retrieve_policy",
                "missing_support": "retention policy", "nearest_sources": ["handbook.md"],
            },
            idempotency_key=f"gap-{index}",
        )
    miner = GapMiner(db, feedback)
    gaps = miner.mine("tenant-a", min_frequency=2)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.frequency == 2
    assert gap.affected_users == 2
    assert gap.status == "open"
    assert "Add a reviewed source" in gap.suggested_source_to_add
    assert "answer" not in gap.as_dict()
    documented = miner.set_status(gap.gap_id, "documented")
    assert documented.version == 2
    assert documented.status == "documented"


def _register(manager, version, artifact_type="retrieval"):
    return manager.register_candidate(
        tenant_id="tenant-a", artifact_type=artifact_type,
        artifact_version=version, artifact={"candidate_k": 30},
        train_hash=f"train-{version}", dev_hash=f"dev-{version}", test_hash=f"test-{version}",
        change_card={
            "what_changed": "candidate count", "why": "recall gap",
            "data_scope": "locked benchmark folds", "known_limitations": "English corpus",
        },
    )


def _pass_to_canary(manager, release, *, human=False):
    metrics, gate = good_metrics(), criteria()
    release = manager.evaluate(
        release.release_id, stage="offline_dev", dataset_hash=release.dev_hash,
        metrics=metrics, criteria=gate,
    )
    release = manager.evaluate(
        release.release_id, stage="test", dataset_hash=release.test_hash,
        metrics=metrics, criteria=gate,
    )
    release = manager.evaluate(
        release.release_id, stage="shadow", dataset_hash="trace-snapshot",
        metrics=metrics, criteria=gate,
    )
    release = manager.start_canary(release.release_id)
    release = manager.evaluate(
        release.release_id, stage="canary", dataset_hash="canary-window",
        metrics=metrics, criteria=gate,
    )
    return manager.promote(release.release_id, human_approved=human)


def test_release_cannot_skip_locked_test_shadow_canary_or_human_prompt_review(tmp_path):
    manager = ReleaseManager(database(tmp_path))
    release = _register(manager, "prompt-v2", artifact_type="prompt")
    with pytest.raises(ValueError, match="promotion requires"):
        manager.promote(release.release_id, human_approved=True)
    with pytest.raises(ValueError, match="dataset hash"):
        manager.evaluate(
            release.release_id, stage="offline_dev", dataset_hash="wrong",
            metrics=good_metrics(), criteria=criteria(),
        )
    metrics, gate = good_metrics(), criteria()
    release = manager.evaluate(
        release.release_id, stage="offline_dev", dataset_hash=release.dev_hash,
        metrics=metrics, criteria=gate,
    )
    release = manager.evaluate(
        release.release_id, stage="test", dataset_hash=release.test_hash,
        metrics=metrics, criteria=gate,
    )
    release = manager.evaluate(
        release.release_id, stage="shadow", dataset_hash="trace",
        metrics=metrics, criteria=gate,
    )
    release = manager.start_canary(release.release_id)
    release = manager.evaluate(
        release.release_id, stage="canary", dataset_hash="window",
        metrics=metrics, criteria=gate,
    )
    with pytest.raises(ValueError, match="human approval"):
        manager.promote(release.release_id)
    promoted = manager.promote(release.release_id, human_approved=True)
    assert promoted.status == "active"


def test_gate_failure_rejects_candidate_and_active_regression_rolls_back(tmp_path):
    manager = ReleaseManager(database(tmp_path))
    first = _pass_to_canary(manager, _register(manager, "retrieval-v1"))
    second = _pass_to_canary(manager, _register(manager, "retrieval-v2"))
    assert manager.active("tenant-a", "retrieval").release_id == second.release_id

    broken = good_metrics()
    broken["quality"] = {**broken["quality"], "citation_integrity": 0.99}
    report, rollback = manager.monitor_active(
        second.release_id, metrics=broken, criteria=criteria(),
    )
    assert report.passed is False
    assert rollback.status == "rolled_back"
    assert manager.active("tenant-a", "retrieval").release_id == first.release_id


def test_canary_assignment_is_stable_and_shadow_never_returns_candidate_to_user(tmp_path):
    release_id = "release-1"
    decisions = {
        ReleaseManager.in_canary(release_id, "tenant-a", "user-a", percentage=10)
        for _ in range(20)
    }
    assert len(decisions) == 1
    replay = shadow_replay(
        [{"q": "x"}],
        active=lambda row: "active answer",
        candidate=lambda row: "candidate answer",
        score=lambda active, candidate, cases: {"compared": len(cases)},
    )
    assert replay.active_outputs == ("active answer",)
    assert replay.candidate_outputs == ("candidate answer",)
    assert replay.metrics == {"compared": 1}


def test_gate_requires_all_five_metric_categories_without_scalar_score():
    incomplete = good_metrics()
    del incomplete["memory"]
    report = evaluate_gate(incomplete, criteria())
    assert report.passed is False
    assert any("memory." in reason for reason in report.reasons)


def test_runtime_not_found_becomes_gap_observation_not_generated_knowledge(tmp_path):
    settings = FakeSettings()
    settings.adaptive_enabled = True
    settings.execution_mode = "auto"
    settings.default_deadline_ms = None
    settings.adaptive_low_score_threshold = 0.0
    settings.adaptive_min_score_margin = 0.0
    settings.data_dir = tmp_path
    settings.enable_trace = False
    settings.enable_memory = False
    settings.enable_persistent_cache = False
    settings.enable_answer_cache = False
    settings.enable_feedback_ledger = True
    settings.evolution_path = tmp_path / "evolution.sqlite3"
    settings.snapshot_manifest_path = tmp_path / "snapshots.json"
    settings.gap_min_frequency = 2
    llm = FakeLLM()
    for run_id in ("missing-1", "missing-2"):
        result = answer_question(
            "What is the undocumented retention policy?", model=llm, judge=llm,
            settings=settings, retrieve_fn=lambda q, kb: [], app=build_workflow(),
            run_id=run_id,
        )
        assert result["packed_evidence"] == []
        assert result["versions"]["policy"] == "adaptive.v2"
        assert result["versions"]["evidence_pack"] == result["evidence_pack_id"]
    service = get_evolution_service(settings)
    observations = service.feedback.list_current(
        tenant_id="default", kinds=("not_found",),
    )
    assert len(observations) == 2
    gaps = service.gaps.mine("default", min_frequency=2)
    assert len(gaps) == 1
    assert "answer" not in gaps[0].as_dict()
