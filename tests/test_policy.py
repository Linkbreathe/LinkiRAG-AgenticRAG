import json

from conftest import FakeLLM, FakeSettings, ev

from linki.graph.nodes import policy_node
from linki.graph.workflow import answer_question, build_workflow
from linki.routing.policy import decide_policy


def _adaptive_settings(tmp_path):
    settings = FakeSettings()
    settings.adaptive_enabled = True
    settings.execution_mode = "auto"
    settings.default_deadline_ms = None
    settings.adaptive_low_score_threshold = 0.0
    settings.adaptive_min_score_margin = 0.0
    settings.data_dir = tmp_path
    settings.enable_trace = True
    return settings


def test_policy_is_local_and_conservatively_classifies_paths():
    assert decide_policy("hello").path == "p0"
    assert decide_policy("What is the release date?").path == "p1"
    assert decide_policy("Compare alpha versus beta").path == "p2"
    assert decide_policy("Strictly verify this medical diagnosis").path == "p3"
    assert decide_policy("Strictly verify this medical diagnosis", mode="fast").path == "p3"
    assert decide_policy("simple", mode="deep").path == "p3"


def test_failed_release_gate_keeps_auto_in_shadow_but_explicit_mode_opts_in():
    settings = FakeSettings()
    settings.adaptive_enabled = False
    settings.execution_mode = "auto"
    settings.default_deadline_ms = None
    shadowed = policy_node({"question": "What is the release date?", "settings": settings})
    assert shadowed["policy_path"] == "legacy"
    assert shadowed["shadow_policy"]["path"] == "p1"

    opted_in = policy_node({
        "question": "What is the release date?", "settings": settings,
        "execution_mode": "fast",
    })
    assert opted_in["policy_path"] == "p1"
    assert "shadow_policy" not in opted_in


def test_p0_exact_greeting_uses_no_model_call(tmp_path):
    settings = _adaptive_settings(tmp_path)
    llm = FakeLLM()
    state = answer_question(
        "hello", model=llm, judge=llm, settings=settings,
        retrieve_fn=lambda q, kb: (_ for _ in ()).throw(AssertionError("no retrieval")),
        app=build_workflow(),
    )
    assert state["policy_path"] == "p0"
    assert state["cost"]["llm_calls"] == 0
    assert "Hello" in state["final_answer"]


def test_p1_single_fact_is_one_model_call(tmp_path):
    settings = _adaptive_settings(tmp_path)
    llm = FakeLLM(answer=lambda s, h: "The release date is June [1].")
    state = answer_question(
        "What is the release date?", model=llm, judge=llm, settings=settings,
        retrieve_fn=lambda q, kb: [ev("release")], app=build_workflow(),
    )
    assert state["policy_path"] == "p1"
    assert state["verified"] is True
    assert state["cost"]["llm_calls"] == 1
    assert state["evidence_pack_id"]
    assert state["evidence_pack"]["token_count"] <= 1200
    assert state["citations"][0]["evidence_id"]
    assert state["cost"]["node_costs"][0]["node"] == "answer"
    assert state["cost"]["node_costs"][0]["usage_source"] == "estimated"
    assert list((tmp_path / "telemetry").glob("*.cost.json"))


def test_p2_comparison_pays_for_plan_and_answer_only_when_low_risk(tmp_path):
    settings = _adaptive_settings(tmp_path)
    llm = FakeLLM(
        planner=lambda s, h: json.dumps({"sub_queries": [
            {"id": "a", "query": "alpha", "target_kb": "Retrieve_default"},
            {"id": "b", "query": "beta", "target_kb": "Retrieve_default"},
        ]}),
        answer=lambda s, h: "Alpha and beta differ in scope [1] [2].",
    )
    state = answer_question(
        "Compare alpha versus beta", model=llm, judge=llm, settings=settings,
        retrieve_fn=lambda q, kb: [ev(q, text=q)], app=build_workflow(),
    )
    assert state["policy_path"] == "p2"
    assert state["verified"] is True
    assert [row["node"] for row in state["cost"]["node_costs"]] == ["planner", "answer"]


def test_p3_keeps_grader_and_verifier_inside_seven_call_budget(tmp_path):
    settings = _adaptive_settings(tmp_path)
    llm = FakeLLM(
        planner=lambda s, h: json.dumps({"sub_queries": [
            {"id": "q1", "query": "dosage evidence", "target_kb": "Retrieve_default"}
        ]}),
        grader=lambda s, h: json.dumps({"sufficient": True, "relevant_chunk_ids": []}),
        answer=lambda s, h: "The source reports the dosage [1].",
        verifier=lambda s, h: json.dumps({"passed": True, "issues": []}),
    )
    state = answer_question(
        "Strictly verify the medical dosage", model=llm, judge=llm, settings=settings,
        retrieve_fn=lambda q, kb: [ev("dose")], app=build_workflow(),
    )
    assert state["policy_path"] == "p3"
    assert state["verified"] is True
    assert state["cost"]["llm_calls"] <= 7
    assert {row["node"] for row in state["cost"]["node_costs"]} == {
        "planner", "grader", "answer", "verifier"
    }
