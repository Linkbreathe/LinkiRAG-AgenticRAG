"""Evidence numbering, rendering, and citation back-mapping.

The answer node assigns each evidence item a stable ``[n]`` marker; after
generation we scan the answer for the markers the model actually used and map
them back to their source, giving the CLI a verifiable "Sources" panel.
"""

from __future__ import annotations

import re
from typing import Any

from linki.graph.state import Citation, Evidence

_MARKER = re.compile(r"\[(\d+)\]")


def number_evidence(evidence: list[Evidence]) -> tuple[list[dict[str, Any]], dict[int, Evidence]]:
    """Assign 1-based indices to evidence. Returns the numbered list (for
    rendering) and an index->Evidence mapping (for citation back-mapping)."""
    numbered: list[dict[str, Any]] = []
    mapping: dict[int, Evidence] = {}
    for i, ev in enumerate(evidence, start=1):
        numbered.append({"index": i, **ev})
        mapping[i] = ev
    return numbered, mapping


def render_evidence(numbered: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in numbered:
        src = item.get("source", "?")
        heading = item.get("heading_path", "")
        loc = f"{src} · {heading}" if heading else src
        lines.append(f"[{item['index']}] ({loc})\n{item.get('text', '').strip()}")
    return "\n\n".join(lines) if lines else "(no evidence)"


def build_citations(answer_text: str, mapping: dict[int, Evidence]) -> list[Citation]:
    """Extract the [n] markers the answer used and map them back to sources."""
    used = sorted({int(m) for m in _MARKER.findall(answer_text or "")})
    citations: list[Citation] = []
    for n in used:
        ev = mapping.get(n)
        if ev is None:
            continue
        citations.append(
            Citation(
                index=n,
                source=ev.get("source", "?"),
                heading_path=ev.get("heading_path", ""),
                parent_id=ev.get("parent_id", ""),
                chunk_id=ev.get("chunk_id", ""),
            )
        )
    return citations
