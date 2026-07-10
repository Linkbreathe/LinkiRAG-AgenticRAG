"""MIRACL loader — pure parsing + Linki-schema dataset construction (no network)."""

from __future__ import annotations

import json

from linki.eval.benchmarks.miracl import (
    build_dataset,
    collect_needed_docids,
    parse_corpus_line,
    parse_qrels,
    parse_topics,
    select_subset,
)

TOPICS = "0\tWhat is the capital of France?\n1\t中国的首都是哪里？\n"
# qid, Q0, docid, relevance  (tab-separated)
QRELS = (
    "0\tQ0\tdoc-a\t1\n"
    "0\tQ0\tdoc-b\t0\n"
    "0\tQ0\tdoc-c\t1\n"
    "1\tQ0\tdoc-x\t1\n"
)


def test_parse_topics_maps_qid_to_query():
    topics = parse_topics(TOPICS)
    assert topics == {"0": "What is the capital of France?", "1": "中国的首都是哪里？"}


def test_parse_qrels_groups_docid_relevance_per_query():
    qrels = parse_qrels(QRELS)
    assert qrels["0"] == {"doc-a": 1, "doc-b": 0, "doc-c": 1}
    assert qrels["1"] == {"doc-x": 1}


def test_build_dataset_puts_positive_docids_in_expect_sources():
    rows = build_dataset(parse_topics(TOPICS), parse_qrels(QRELS), lang="en")
    by_id = {r["id"]: r for r in rows}

    r0 = by_id["miracl-en-0"]
    assert r0["question"] == "What is the capital of France?"
    assert r0["type"] == "single"
    assert r0["expect"]["should_refuse"] is False
    assert r0["expect"]["lang"] == "en"
    # only relevance>0 docids are gold sources; doc-b (rel 0) excluded
    assert sorted(r0["expect"]["expect_sources"]) == ["doc-a", "doc-c"]


def test_build_dataset_skips_queries_with_no_positive():
    topics = {"9": "orphan query with only a judged negative"}
    qrels = {"9": {"doc-neg": 0}}
    assert build_dataset(topics, qrels, lang="zh") == []


def test_parse_corpus_line_reads_docid_title_text():
    line = json.dumps({"docid": "doc-a", "title": "Paris", "text": "Paris is the capital of France."})
    p = parse_corpus_line(line)
    assert p == {"docid": "doc-a", "title": "Paris", "text": "Paris is the capital of France."}


def test_parse_corpus_line_tolerates_blank():
    assert parse_corpus_line("   ") is None


def _rows(n):
    return [
        {"id": f"miracl-zh-{i}", "question": f"q{i}", "type": "single",
         "expect": {"expect_sources": [f"doc-{i}a", f"doc-{i}b"]}}
        for i in range(n)
    ]


def test_select_subset_is_deterministic_and_sized():
    rows = _rows(50)
    a = select_subset(rows, 10, seed=7)
    b = select_subset(rows, 10, seed=7)
    assert len(a) == 10
    assert [r["id"] for r in a] == [r["id"] for r in b]  # same seed -> same pick


def test_select_subset_caps_at_available():
    rows = _rows(5)
    assert len(select_subset(rows, 20, seed=1)) == 5


def test_collect_needed_docids_unions_all_gold():
    rows = _rows(3)
    got = collect_needed_docids(rows)
    assert got == {"doc-0a", "doc-0b", "doc-1a", "doc-1b", "doc-2a", "doc-2b"}
