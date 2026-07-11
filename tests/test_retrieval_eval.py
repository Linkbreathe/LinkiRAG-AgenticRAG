from linki.eval.retrieval_eval import run_retrieval_eval


def test_retrieval_eval_reports_full_chain_recall_by_type():
    dataset = [{
        "id": "a", "question": "q", "question_type": "inference_query",
        "expect": {"expect_sources": ["one", "two"]},
    }]
    report = run_retrieval_eval(
        dataset,
        lambda q, kb: [{"source": "one"}, {"source": "noise"}],
        "Retrieve_multihop",
    )
    assert report["aggregate"]["retrieval_recall"] == 0.5
    assert report["aggregate"]["all_support_recall"] == 0.0
    assert report["aggregate"]["mrr"] == 1.0
    assert report["aggregate"]["ndcg"] > 0
    assert report["aggregate"]["source_diversity"] == 1.0
    assert report["by_question_type"]["inference_query"]["n"] == 1


def test_retrieval_eval_checkpoint_resumes_completed_ids(tmp_path):
    calls = {"n": 0}

    def retrieve(question, kb):
        calls["n"] += 1
        return [{"source": question, "text": question}]

    rows = [
        {"id": "a", "question": "one", "expect": {"expect_sources": ["one"]}},
        {"id": "b", "question": "two", "expect": {"expect_sources": ["two"]}},
    ]
    path = tmp_path / "checkpoint.json"
    first = run_retrieval_eval(rows, retrieve, "Retrieve_default", checkpoint_path=path, checkpoint_every=1)
    second = run_retrieval_eval(rows, retrieve, "Retrieve_default", checkpoint_path=path, checkpoint_every=1)
    assert first["complete"] is True and second["complete"] is True
    assert first["protocol"]["dataset_fingerprint"]
    assert calls["n"] == 2
