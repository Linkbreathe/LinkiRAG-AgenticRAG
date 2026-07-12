import json

from linki.eval.benchmarks.multihop_rag import (
    CorpusPPRGraphRetriever,
    build_dataset,
    select_disjoint_stratified_folds,
    select_stratified,
)


def test_build_dataset_maps_gold_urls_and_answers():
    rows = [{
        "query": "Who?", "answer": "Ada", "question_type": "inference_query",
        "evidence_list": [{"url": "https://a"}, {"url": "https://b"}],
    }]
    item = build_dataset(rows)[0]
    assert item["type"] == "multihop"
    assert item["expect"]["gold_answer"] == "Ada"
    assert item["expect"]["expect_sources"] == ["https://a", "https://b"]


def test_build_dataset_maps_null_queries_to_refusal():
    item = build_dataset([{
        "query": "Unknown?", "answer": "Insufficient information",
        "question_type": "null_query", "evidence_list": [],
    }])[0]
    assert item["type"] == "out_of_kb"
    assert item["expect"]["should_refuse"] is True
    assert item["expect"]["expect_sources"] is None


def test_stratified_sample_is_deterministic_and_covers_types():
    rows = [
        {"id": f"{kind}-{i}", "question_type": kind}
        for kind in ("comparison_query", "inference_query", "temporal_query", "null_query")
        for i in range(10)
    ]
    one = select_stratified(rows, 8, seed=7)
    two = select_stratified(rows, 8, seed=7)
    assert [row["id"] for row in one] == [row["id"] for row in two]
    assert {row["question_type"] for row in one} == {row["question_type"] for row in rows}


def test_disjoint_folds_have_fixed_seeds_and_no_repeated_ids():
    rows = [
        {"id": f"{kind}-{i}", "question_type": kind}
        for kind in ("comparison_query", "inference_query", "temporal_query", "null_query")
        for i in range(40)
    ]
    selected = select_disjoint_stratified_folds(rows, fold_size=10, seeds=(42, 123, 2026))
    assert len(selected) == 30
    assert len({row["id"] for row in selected}) == 30
    assert {row["eval_seed"] for row in selected} == {42, 123, 2026}


def test_corpus_ppr_uses_only_corpus_entities_and_returns_document_candidates(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([
        {"url": "https://one", "title": "Alpha Project", "body": "Alpha Project uses PostgreSQL."},
        {"url": "https://two", "title": "Other News", "body": "Unrelated report."},
    ]), encoding="utf-8")
    retriever = CorpusPPRGraphRetriever(path)
    hits = retriever.retrieve("What does Alpha use?", ["Alpha"], "snapshot", 5)
    assert hits
    assert hits[0].source == "https://one"
    assert hits[0].channel == "graph-ppr"
