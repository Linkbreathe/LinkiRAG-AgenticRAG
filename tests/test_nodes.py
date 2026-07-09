import json

from conftest import FakeLLM, FakeSettings, ev

from linki.graph.nodes import (
    answer_node,
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
    assert verifier_route({"verified": False, "attempts": 1, "settings": s}) == "retrieve"
