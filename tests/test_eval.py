"""Eval harness: the naive baseline path and the naive-vs-agentic comparison
with LLM-judged + structural metrics, all driven by fakes."""

from __future__ import annotations

import json

from conftest import FakeLLM, FakeSettings, ev

from linki.eval.naive import naive_answer
from linki.eval.run_eval import format_report, load_dataset, run_eval, score_answer
from linki.eval.run_eval import _did_refuse


def _agentic_llm():
    return FakeLLM(
        router=lambda s, h: json.dumps({"route": "retrieve", "reason": "kb"}),
        grader=lambda s, h: json.dumps({"sufficient": True, "relevant_chunk_ids": []}),
        answer=lambda s, h: "Grounded answer [1].",
        verifier=lambda s, h: json.dumps({"passed": True, "issues": []}),
        naive=lambda s, h: "Naive says X.",
        evalscore=lambda s, h: json.dumps({"faithfulness": 5, "quality": 4}),
    )


def test_naive_answer_single_shot_no_router_or_planner():
    calls = []

    def retrieve_fn(q, kb):
        calls.append((q, kb))
        return [ev("a", text="ctx")]

    llm = FakeLLM(naive=lambda s, h: "Naive answer.")
    out = naive_answer("what is X?", model=llm, settings=FakeSettings(), retrieve_fn=retrieve_fn)

    assert out["answer"] == "Naive answer."
    assert out["evidence"] and out["evidence"][0]["chunk_id"] == "a"
    # naive retrieves exactly once (no grade->refine loop, no planning)
    assert len(calls) == 1


def test_score_answer_marks_refusal_correctness():
    llm = FakeLLM(evalscore=lambda s, h: json.dumps({"faithfulness": 1, "quality": 3}))
    # An out-of-kb item that SHOULD be refused, and an answer that refuses.
    scored = score_answer(
        llm, "obscure q", "知识库中未找到相关内容。", evidence=[],
        expect={"should_refuse": True},
    )
    assert scored["did_refuse"] is True
    assert scored["refusal_correct"] is True

    # A hallucinated answer to the same item is refusal-incorrect.
    scored2 = score_answer(
        llm, "obscure q", "The answer is definitely 42.", evidence=[],
        expect={"should_refuse": True},
    )
    assert scored2["did_refuse"] is False
    assert scored2["refusal_correct"] is False


def test_partial_gap_after_answer_is_not_a_full_refusal():
    answer = "The company is Google [1]. A secondary detail was not found in the knowledge base."
    assert _did_refuse(answer) is False
    assert _did_refuse("The answer was not found in the knowledge base.") is True
    assert _did_refuse("The provided passages do not contain the requested information.") is True


def test_score_answer_computes_retrieval_recall_vs_gold():
    llm = FakeLLM(evalscore=lambda s, h: json.dumps({"faithfulness": 5, "quality": 5}))
    # gold = {g1, g2}; evidence covers g1 (source) but not g2 -> recall 0.5
    evidence = [ev("c1", source="g1"), ev("c2", source="noise")]
    scored = score_answer(llm, "q", "Answer [1].", evidence=evidence,
                          expect={"expect_sources": ["g1", "g2"]})
    assert scored["retrieval_recall"] == 0.5


def test_score_answer_recall_none_when_no_gold():
    llm = FakeLLM(evalscore=lambda s, h: json.dumps({"faithfulness": 5, "quality": 5}))
    scored = score_answer(llm, "q", "Answer.", evidence=[ev("a")], expect={})
    assert scored["retrieval_recall"] is None


def test_score_answer_citation_coverage_structural():
    llm = FakeLLM(evalscore=lambda s, h: json.dumps({"faithfulness": 5, "quality": 5}))
    with_cite = score_answer(llm, "q", "Answer [1].", evidence=[ev("a")], expect={})
    without_cite = score_answer(llm, "q", "Answer with no marker.", evidence=[ev("a")], expect={})
    assert with_cite["has_citations"] is True
    assert without_cite["has_citations"] is False


def test_run_eval_compares_both_paths_and_aggregates():
    dataset = [
        {"id": "s1", "question": "single fact?", "type": "single", "expect": {"expect_keywords": ["X"]}},
        {"id": "o1", "question": "out of kb?", "type": "out_of_kb", "expect": {"should_refuse": True}},
    ]
    llm = _agentic_llm()
    report = run_eval(dataset, model=llm, judge=llm, settings=FakeSettings(),
                      retrieve_fn=lambda q, kb: [ev("a")])

    assert len(report["items"]) == 2
    row = report["items"][0]
    assert "agentic" in row and "naive" in row
    assert "agentic" in report["aggregate"] and "naive" in report["aggregate"]
    # format_report returns a printable string mentioning both paths
    text = format_report(report)
    assert "agentic" in text.lower() and "naive" in text.lower()


def test_run_eval_counts_empty_retrieval_as_a_real_failure():
    dataset = [{"id": "s1", "question": "needs a KB", "type": "single", "expect": {"expect_keywords": ["X"]}}]
    llm = _agentic_llm()
    # empty retriever == KB not ingested
    report = run_eval(dataset, model=llm, judge=llm, settings=FakeSettings(),
                      retrieve_fn=lambda q, kb: [])
    row = report["items"][0]
    assert row["status"] == "ok"
    assert row["agentic"]["retrieved_sources"] == []
    assert row["naive"]["retrieved_sources"] == []


def test_run_eval_strict_includes_no_reflow_ablation_and_telemetry():
    dataset = [{
        "id": "m1", "question": "multi?", "type": "multihop",
        "expect": {"gold_answer": "X", "expect_sources": ["doc.pdf"]},
    }]
    llm = _agentic_llm()
    report = run_eval(
        dataset, model=llm, judge=llm, settings=FakeSettings(),
        retrieve_fn=lambda q, kb: [ev("a")], strict=True,
    )
    assert report["systems"] == ["naive", "agentic_no_reflow", "agentic"]
    assert report["items"][0]["agentic"]["telemetry"]["llm_calls"] >= 1


def test_load_dataset_reads_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"id": "a", "question": "q", "type": "single", "expect": {}}) + "\n"
        + "\n"  # blank line tolerated
        + json.dumps({"id": "b", "question": "q2", "type": "chat", "expect": {}}) + "\n",
        encoding="utf-8",
    )
    rows = load_dataset(p)
    assert [r["id"] for r in rows] == ["a", "b"]
