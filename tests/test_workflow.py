import json

from conftest import FakeLLM, FakeSettings, ev

from linki.graph.workflow import answer_question, build_workflow


def _run(llm, retrieve_fn, **kw):
    return answer_question(
        "What is X?",
        model=llm,
        judge=llm,
        settings=FakeSettings(**kw),
        retrieve_fn=retrieve_fn,
        app=build_workflow(),
    )


def test_planner_fans_out_sub_queries_to_target_kbs():
    llm = FakeLLM(
        router=lambda s, h: json.dumps({"route": "retrieve", "reason": "needs kb"}),
        planner=lambda s, h: json.dumps({
            "sub_queries": [
                {"id": "q1", "query": "FastAPI deployment", "target_kb": "Retrieve_api", "reason": "api"},
                {"id": "q2", "query": "meeting limit decision", "target_kb": "Retrieve_meetings", "reason": "meeting"},
            ]
        }),
        grader=lambda s, h: json.dumps({"sufficient": True, "relevant_chunk_ids": []}),
        answer=lambda s, h: "The answer combines both sources [1] [2].",
        verifier=lambda s, h: json.dumps({"passed": True, "issues": []}),
    )
    calls = []

    def retrieve_fn(query, kb):
        calls.append((query, kb))
        return [ev(f"{kb}-{len(calls)}", kb=kb, text=query)]

    state = _run(llm, retrieve_fn, kbs=["default", "api", "meetings"])

    assert ("FastAPI deployment", "Retrieve_api") in calls
    assert ("meeting limit decision", "Retrieve_meetings") in calls
    assert len(state["evidence"]) == 2
    assert state["verified"] is True


def test_retrieve_path_end_to_end():
    llm = FakeLLM(
        router=lambda s, h: json.dumps({"route": "retrieve", "reason": "needs kb"}),
        grader=lambda s, h: json.dumps({"sufficient": True, "relevant_chunk_ids": []}),
        answer=lambda s, h: "X is a thing [1].",
        verifier=lambda s, h: json.dumps({"passed": True, "issues": []}),
    )
    state = _run(llm, lambda q, kb: [ev("a")])
    assert state["evidence"]
    assert state["verified"] is True
    assert "X is a thing" in state["final_answer"]
    assert state["citations"][0]["index"] == 1


def test_chat_path_skips_retrieval():
    llm = FakeLLM(
        router=lambda s, h: json.dumps({"route": "chat", "reason": "greeting"}),
        chat=lambda s, h: "Hello, I'm Linki!",
    )
    called = {"n": 0}

    def retrieve_fn(q, kb):
        called["n"] += 1
        return [ev("a")]

    state = _run(llm, retrieve_fn)
    assert called["n"] == 0
    assert state["final_answer"] == "Hello, I'm Linki!"


def test_no_evidence_still_completes():
    llm = FakeLLM(
        router=lambda s, h: json.dumps({"route": "retrieve", "reason": "kb"}),
        grader=lambda s, h: json.dumps({"sufficient": False, "relevant_chunk_ids": [], "missing": "nothing found", "refined_query": "totally different phrasing here"}),
        answer=lambda s, h: "知识库中未找到 X 的相关内容。",
        verifier=lambda s, h: json.dumps({"passed": True, "issues": []}),
    )
    state = _run(llm, lambda q, kb: [])
    assert state["evidence"] == []
    assert state["gaps"]
    assert "未找到" in state["final_answer"]


def test_verifier_reflow_to_warning_when_attempts_exhausted():
    planner_calls = {"n": 0}

    def planner(system, human):
        planner_calls["n"] += 1
        query = "initial query" if planner_calls["n"] == 1 else "supplemental query"
        return json.dumps({
            "sub_queries": [
                {"id": f"q{planner_calls['n']}", "query": query, "target_kb": "Retrieve_default"}
            ]
        })

    llm = FakeLLM(
        router=lambda s, h: json.dumps({"route": "retrieve", "reason": "kb"}),
        planner=planner,
        grader=lambda s, h: json.dumps({"sufficient": True, "relevant_chunk_ids": []}),
        answer=lambda s, h: "Dubious claim [1].",
        verifier=lambda s, h: json.dumps({"passed": False, "issues": [{"claim": "Dubious claim", "problem": "unsupported"}]}),
    )
    rounds = [[ev("a")], [ev("b")], [ev("c")]]

    def retrieve_fn(q, kb):
        return rounds.pop(0) if rounds else []

    state = _run(llm, retrieve_fn, max_attempts=2)
    assert state["attempts"] == 2
    assert planner_calls["n"] == 2
    assert "未完全通过验证" in state["final_answer"] or "did not fully pass" in state["final_answer"]
