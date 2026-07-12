from conftest import ev

from linki.retrieval.candidates import Candidate, reciprocal_rank_fusion
from linki.retrieval.evidence_pack import build_evidence_pack
from linki.retrieval.graph import PersonalizedPageRankRetriever
from linki.retrieval.rerank import CrossEncoderReranker
from linki.retrieval.spans import select_supporting_span
from linki.tools.retrieve import fuse_candidate_channels


def candidate(name: str, score: float = 1.0, *, channel: str = "hybrid") -> Candidate:
    return Candidate(
        chunk_id=name, parent_id=f"{name}-p", kb="default", source=f"{name}.md",
        heading_path="", child_text=f"content about {name}",
        retrieval_score=score, channel=channel,
    )


def test_supporting_span_is_verbatim_and_offsets_validate():
    parent = "Intro sentence.\n" + ("background " * 100) + "The release date is June 7.\nTail."
    anchor = "The release date is June 7."
    span = select_supporting_span("release date", parent, anchor_text=anchor, max_chars=160)
    assert anchor in span.quote
    assert span.validates(parent)
    assert span.char_end - span.char_start <= 280  # natural-boundary allowance


def test_evidence_pack_is_deterministic_diverse_and_budgeted():
    rows = [
        ev("a", text="alpha " * 80, source="one.md", score=0.9, supports=["q1"]),
        ev("b", text="beta " * 80, source="two.md", score=0.8, supports=["q2"]),
        ev("c", text="gamma " * 80, source="one.md", score=0.7, supports=["q1"]),
    ]
    one = build_evidence_pack(rows, token_budget=100)
    two = build_evidence_pack(rows, token_budget=100)
    assert one.evidence_pack_id == two.evidence_pack_id
    assert one.token_count <= 100
    assert one.units
    assert all(unit["content_hash"] and unit["evidence_id"] for unit in one.units)
    assert {unit["source_id"] for unit in one.units[:2]} == {"one.md", "two.md"}


def test_disabled_cross_encoder_uses_labelled_lexical_fallback():
    reranker = CrossEncoderReranker(enabled=False)
    ranked, backend = reranker.rerank(
        "release date", [candidate("noise"), candidate("release-date")]
    )
    assert backend == "lexical-fallback"
    assert ranked[0].chunk_id == "release-date"
    assert ranked[0].rerank_score is not None


def test_reranker_score_cache_can_be_disabled_for_fair_latency_measurement(monkeypatch):
    class Backend:
        calls = 0

        def rerank(self, query, documents):
            self.calls += 1
            return [0.5 for _ in documents]

    backend = Backend()
    reranker = CrossEncoderReranker(cache_scores=False)
    monkeypatch.setattr(reranker, "_model", lambda: backend)
    reranker.rerank("same query", [candidate("same")])
    reranker.rerank("same query", [candidate("same")])
    assert backend.calls == 2


def test_rrf_fuses_channels_without_treating_raw_scores_as_calibrated():
    hybrid = [candidate("a", 99), candidate("b", 1)]
    graph = [candidate("b", 0.01, channel="graph"), candidate("c", 0.009, channel="graph")]
    fused = reciprocal_rank_fusion([hybrid, graph])
    assert fused[0].chunk_id == "b"
    assert fused[0].channel == "graph+hybrid"


def test_channel_fusion_honors_the_total_candidate_budget():
    hybrid = [candidate(f"hybrid-{index}") for index in range(30)]
    graph = [candidate(f"graph-{index}", channel="graph") for index in range(30)]
    fused = fuse_candidate_channels([hybrid, graph], limit=30)
    assert len(fused) == 30
    assert {item.channel for item in fused} == {"hybrid", "graph"}


def test_ppr_pilot_is_bounded_to_two_hops():
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}
    candidates = {name: candidate(name) for name in graph}
    retriever = PersonalizedPageRankRetriever(graph, candidates, max_hops=2)
    hits = retriever.retrieve("q", ["a"], "snapshot", 10)
    assert {hit.chunk_id for hit in hits} == {"a", "b", "c"}
    assert all(hit.channel == "graph-ppr" for hit in hits)
