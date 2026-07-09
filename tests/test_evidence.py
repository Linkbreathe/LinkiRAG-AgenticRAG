from conftest import ev

from linki.graph.evidence import build_citations, number_evidence, render_evidence


def test_number_evidence_assigns_1_based_indices():
    numbered, mapping = number_evidence([ev("a"), ev("b")])
    assert [n["index"] for n in numbered] == [1, 2]
    assert mapping[1]["chunk_id"] == "a"


def test_build_citations_only_used_markers():
    _, mapping = number_evidence([ev("a", source="a.pdf"), ev("b", source="b.pdf"), ev("c")])
    cites = build_citations("Claim one [2] and claim two [1]. Repeat [1].", mapping)
    assert [c["index"] for c in cites] == [1, 2]  # sorted, deduped, only used
    assert cites[0]["source"] == "a.pdf"


def test_render_evidence_includes_markers_and_source():
    numbered, _ = number_evidence([ev("a", text="hello", source="doc.pdf")])
    out = render_evidence(numbered)
    assert "[1]" in out and "doc.pdf" in out and "hello" in out
