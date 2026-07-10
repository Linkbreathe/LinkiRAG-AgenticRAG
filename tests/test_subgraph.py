import json

from conftest import FakeLLM, FakeSettings, ev

from linki.graph.subgraph import retrieval_node, too_similar


def _state(retrieve_fn, judge, **kw):
    return {
        "question": "q",
        "rewritten_query": "q",
        "settings": FakeSettings(**kw),
        "retrieve_fn": retrieve_fn,
        "judge": judge,
        "retrieval_keys": set(),
    }


def test_too_similar_guard():
    assert too_similar("same query", "same query")
    assert not too_similar("apples", "oranges bananas grapes")


def test_dedup_across_rounds_and_refine():
    rounds = [[ev("a"), ev("b")], [ev("b"), ev("c")]]  # b repeats -> deduped in round 2

    def retrieve_fn(query, kb):
        return rounds.pop(0) if rounds else []

    calls = {"n": 0}

    def grader(system, human):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"sufficient": False, "relevant_chunk_ids": [], "missing": "need more", "refined_query": "different query entirely"})
        return json.dumps({"sufficient": True, "relevant_chunk_ids": []})

    out = retrieval_node(_state(retrieve_fn, FakeLLM(grader=grader)))
    assert out["retrieval_keys"] == {"a", "b", "c"}
    assert out["gaps"] == []
    collected_ids = {e["chunk_id"] for e in out["evidence"]}
    assert "c" in collected_ids  # round 2 fresh chunk collected


def test_giveup_records_gaps():
    counter = {"n": 0}

    def retrieve_fn(query, kb):
        counter["n"] += 1
        return [ev(f"x{counter['n']}")]  # distinct fresh chunk each round

    def grader(system, human):
        return json.dumps({"sufficient": False, "relevant_chunk_ids": [], "missing": "still missing", "refined_query": f"brand new distinct query {counter['n']}"})

    out = retrieval_node(_state(retrieve_fn, FakeLLM(grader=grader)))
    # After max_rounds with evidence still judged insufficient, the grader's
    # 'missing' is recorded as a gap for honest downstream refusal.
    assert out["gaps"] and "still missing" in out["gaps"][0]


def test_relevant_ids_filter_noise():
    def retrieve_fn(query, kb):
        return [ev("keep"), ev("noise")]

    def grader(system, human):
        return json.dumps({"sufficient": True, "relevant_chunk_ids": ["keep"]})

    out = retrieval_node(_state(retrieve_fn, FakeLLM(grader=grader)))
    assert {e["chunk_id"] for e in out["evidence"]} == {"keep"}


def test_grader_fallback_on_bad_json_does_not_stall():
    def retrieve_fn(query, kb):
        return [ev("a")]

    out = retrieval_node(_state(retrieve_fn, FakeLLM(grader=lambda s, h: "not json at all")))
    # fallback treats as sufficient, keeps evidence, no gaps
    assert out["evidence"] and out["gaps"] == []


def test_target_kb_is_passed_to_retriever():
    seen = {}

    def retrieve_fn(query, kb):
        seen["kb"] = kb
        return [ev("a")]

    state = _state(retrieve_fn, FakeLLM(grader=lambda s, h: json.dumps({"sufficient": True})))
    state["target_kb"] = "Retrieve_sales"
    retrieval_node(state)

    assert seen["kb"] == "Retrieve_sales"


def test_sub_query_target_kb_is_passed_to_retriever():
    seen = {}

    def retrieve_fn(query, kb):
        seen["query"] = query
        seen["kb"] = kb
        return [ev("a")]

    state = _state(retrieve_fn, FakeLLM(grader=lambda s, h: json.dumps({"sufficient": True})))
    state["sub_query"] = {"id": "q1", "query": "api workers", "target_kb": "Retrieve_api"}
    retrieval_node(state)

    assert seen == {"query": "api workers", "kb": "Retrieve_api"}
