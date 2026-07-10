import json

from conftest import FakeLLM, FakeSettings, ev

from linki.graph.nodes import (
    answer_node,
    planner_node,
    router_node,
    verifier_node,
    verifier_route,
)


def test_router_falls_back_to_retrieve_on_bad_json():
    out = router_node({"question": "hi", "model": FakeLLM(router=lambda s, h: "garbage")})
    assert out["route"] == "retrieve"


def test_router_respects_valid_chat():
    llm = FakeLLM(router=lambda s, h: json.dumps({"route": "chat", "reason": "greeting"}))
    out = router_node({"question": "hello", "model": llm})
    assert out["route"] == "chat"


def test_answer_node_builds_citations():
    llm = FakeLLM(answer=lambda s, h: "Point one [1]. Point two [2].")
    out = answer_node({"question": "q", "model": llm, "evidence": [ev("a"), ev("b")], "gaps": []})
    assert [c["index"] for c in out["citations"]] == [1, 2]


def test_planner_node_builds_validated_query_plan():
    llm = FakeLLM(planner=lambda s, h: json.dumps({
        "sub_queries": [
            {"id": "api", "query": "FastAPI deployment", "target_kb": "Retrieve_api", "reason": "api docs"},
            {"id": "bad", "query": "Meeting decision", "target_kb": "Retrieve_missing", "reason": "bad tool"},
        ]
    }))
    out = planner_node({
        "question": "Compare deployment and meeting decision",
        "rewritten_query": "Compare deployment and meeting decision",
        "model": llm,
        "settings": FakeSettings(kbs=["default", "api"]),
    })

    assert [q["query"] for q in out["sub_queries"]] == ["FastAPI deployment", "Meeting decision"]
    assert out["sub_queries"][0]["target_kb"] == "Retrieve_api"
    assert out["sub_queries"][1]["target_kb"] == "Retrieve_default"


def test_planner_uses_verifier_issues_for_fallback_query():
    out = planner_node({
        "question": "q",
        "rewritten_query": "q",
        "model": FakeLLM(planner=lambda s, h: "{}"),
        "settings": FakeSettings(),
        "verify_issues": [{"claim": "x", "problem": "unsupported", "fix_instruction": "retrieve X mechanism"}],
    })

    assert out["sub_queries"][0]["query"] == "retrieve X mechanism"


def test_planner_respects_explicit_target_kb_constraint():
    llm = FakeLLM(planner=lambda s, h: json.dumps({
        "sub_queries": [
            {"id": "q1", "query": "anything", "target_kb": "Retrieve_api"}
        ]
    }))
    out = planner_node({
        "question": "q",
        "rewritten_query": "q",
        "model": llm,
        "settings": FakeSettings(kbs=["default", "api"]),
        "target_kb": "Retrieve_runtime_topic",
    })

    assert out["sub_queries"][0]["target_kb"] == "Retrieve_runtime_topic"


def test_verifier_node_marks_failed_and_increments_attempts():
    llm = FakeLLM(verifier=lambda s, h: json.dumps(
        {"passed": False, "issues": [{"claim": "x", "problem": "unsupported"}]}))
    out = verifier_node({"question": "q", "model": llm, "judge": llm, "evidence": [ev("a")], "attempts": 0})
    assert out["verified"] is False
    assert out["attempts"] == 1
    assert out["verify_issues"]


def test_verifier_route_logic():
    s = FakeSettings(max_attempts=2)
    assert verifier_route({"verified": True, "settings": s}) == "final"
    assert verifier_route({"verified": False, "attempts": 2, "settings": s}) == "final_with_warning"
    assert verifier_route({"verified": False, "attempts": 1, "settings": s}) == "planner"
