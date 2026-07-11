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
