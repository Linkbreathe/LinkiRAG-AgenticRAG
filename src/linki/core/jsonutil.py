"""Tolerant JSON extraction from LLM output.

Models wrap JSON in prose or ```json fences. ``extract_json`` digs out the first
balanced object and falls back to a caller-supplied default when parsing fails —
so a flaky structured response degrades gracefully instead of crashing a node.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if text is None:
        return dict(fallback or {})
    candidates: list[str] = []
    for m in _FENCE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    # Also try the substring from the first '{' to the last '}'.
    lo, hi = text.find("{"), text.rfind("}")
    if lo != -1 and hi != -1 and hi > lo:
        candidates.append(text[lo : hi + 1])

    for cand in candidates:
        try:
            value = json.loads(cand)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            continue
    return dict(fallback or {})
