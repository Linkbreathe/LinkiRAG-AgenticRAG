from conftest import FakeLLM, ev

from linki.config import Settings
from linki.eval.fair_eval import SYSTEMS, run_fair_eval


def test_fair_eval_runs_four_systems_and_resumes_without_new_calls(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        qdrant_path=tmp_path / "qdrant",
        parent_store_path=tmp_path / "parents",
        markdown_dir=tmp_path / "markdown",
        cache_path=tmp_path / "cache.sqlite3",
        snapshot_manifest_path=tmp_path / "snapshots.json",
        memory_path=tmp_path / "memory.sqlite3",
        knowledge_path=tmp_path / "knowledge.sqlite3",
        evolution_path=tmp_path / "evolution.sqlite3",
    )
    llm = FakeLLM()
    rows = [
        {
            "id": "row-1", "question": "What is one?", "question_type": "inference_query",
            "eval_fold": 1, "eval_seed": 42,
            "expect": {"gold_answer": "Answer", "expect_sources": ["gold.pdf"], "should_refuse": False},
        },
        {
            "id": "row-2", "question": "Compare one versus two", "question_type": "comparison_query",
            "eval_fold": 2, "eval_seed": 123,
            "expect": {"gold_answer": "Answer", "expect_sources": ["gold.pdf"], "should_refuse": False},
        },
    ]
    retrieval = {
        "legacy_k5": lambda q, kb: [ev("legacy", source="gold.pdf")],
        "rerank_pack": lambda q, kb: [ev("rerank", source="gold.pdf")],
    }
    path = tmp_path / "fair.json"
    first = run_fair_eval(
        rows, model=llm, judge=llm, settings=settings, retrieval=retrieval,
        checkpoint_path=path, corpus_snapshot_id="snapshot-1",
    )
    calls = len(llm.calls)
    second = run_fair_eval(
        rows, model=llm, judge=llm, settings=settings, retrieval=retrieval,
        checkpoint_path=path, corpus_snapshot_id="snapshot-1",
    )
    assert first["complete"] is True and second["complete"] is True
    assert first["systems"] == list(SYSTEMS)
    assert all(first["aggregate"][system]["n"] == 2 for system in SYSTEMS)
    assert all("p95_latency_seconds" in first["aggregate"][system] for system in SYSTEMS)
    assert all("token_usage_mode" in first["aggregate"][system] for system in SYSTEMS)
    assert first["protocol"]["experiment_id"]
    assert first["paired_vs_legacy_full"]["adaptive_auto"]["total_tokens"]["n"] == 2
    assert len(llm.calls) == calls
