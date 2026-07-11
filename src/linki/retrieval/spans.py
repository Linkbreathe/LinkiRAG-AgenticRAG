"""Extract verbatim supporting spans with offsets into the parent text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SupportingSpan:
    quote: str
    char_start: int
    char_end: int

    def validates(self, parent_text: str) -> bool:
        return parent_text[self.char_start:self.char_end] == self.quote


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+|[\u4e00-\u9fff]", (text or "").lower()))


def _bounded_window(text: str, center_start: int, center_end: int, max_chars: int) -> tuple[int, int]:
    if len(text) <= max_chars:
        return 0, len(text)
    center = (center_start + center_end) // 2
    start = max(0, center - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    # Prefer natural boundaries while never changing the source text.
    left = max(text.rfind("\n", 0, start + 120), text.rfind(". ", 0, start + 120))
    if left >= max(0, start - 120):
        start = left + 1
    right_candidates = [pos for pos in (text.find("\n", end - 120), text.find(". ", end - 120)) if pos >= 0]
    if right_candidates:
        right = min(right_candidates)
        if right <= min(len(text), end + 120):
            end = right + 1
    return start, end


def select_supporting_span(
    query: str,
    parent_text: str,
    *,
    anchor_text: str = "",
    max_chars: int = 1_600,
) -> SupportingSpan:
    """Select a verbatim parent slice, preferring the retrieved child anchor."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = parent_text or anchor_text or ""
    if not text:
        return SupportingSpan("", 0, 0)

    anchor_start = text.find(anchor_text) if anchor_text else -1
    if anchor_start >= 0:
        start, end = _bounded_window(text, anchor_start, anchor_start + len(anchor_text), max_chars)
        return SupportingSpan(text[start:end], start, end)

    terms = _terms(query)
    best = (0.0, 0, min(len(text), max_chars))
    # Sentence-level lexical selection is a cheap fallback when the child text
    # is not a literal substring (for example after loader normalization).
    for match in re.finditer(r"[^\n.!?。！？]+(?:[.!?。！？]+|$)", text):
        sentence_terms = _terms(match.group(0))
        score = len(terms & sentence_terms) / max(len(terms), 1)
        if score > best[0]:
            best = (score, match.start(), match.end())
    start, end = _bounded_window(text, best[1], best[2], max_chars)
    return SupportingSpan(text[start:end], start, end)
