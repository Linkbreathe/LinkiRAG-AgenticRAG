from linki.eval.memory_eval import run_memory_eval
from linki.eval.stability_eval import answer_claims, jaccard, run_stability_eval


def test_memory_governance_suite_reports_separate_quality_dimensions(tmp_path):
    report = run_memory_eval(path=tmp_path / "memory.sqlite3")
    metrics = report["metrics"]
    assert metrics["status_accuracy"] == 1.0
    assert metrics["write_precision"] == 1.0
    assert metrics["update_correctness"] == 1.0
    assert metrics["stale_memory_rate"] == 0.0
    assert metrics["deletion_completeness"] == 1.0
    assert report["protocol"]["external_benchmark_claim"] is None


def test_stability_eval_repeats_and_reports_jaccard_variance_and_errors():
    rows = [{"id": "q1", "question": "q"}]
    report = run_stability_eval(
        rows,
        lambda q, kb: [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "Retrieve_default",
        repeats=3,
        answer_fn=lambda q, evidence: "Stable factual claim [1].",
    )
    assert report["aggregate"]["topk_jaccard"] == 1.0
    assert report["aggregate"]["topk_exact_match_rate"] == 1.0
    assert report["aggregate"]["answer_claim_jaccard"] == 1.0
    assert report["aggregate"]["error_rate"] == 0.0
    assert report["items"][0]["question_type"] == "unknown"
    assert jaccard(set(), set()) == 1.0
    assert answer_claims("One useful claim [1].") == {"one useful claim ."}
