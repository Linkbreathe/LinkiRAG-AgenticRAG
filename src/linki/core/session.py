"""Minimal multi-turn session: append Q/A pairs to a JSON file and render the
recent history as context for the router/rewrite nodes."""

from __future__ import annotations

import json
from pathlib import Path


class Session:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._turns: list[dict] = []
        if self._path.exists():
            try:
                self._turns = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._turns = []

    def add(self, question: str, answer: str) -> None:
        self._turns.append({"q": question, "a": answer})
        self._path.write_text(json.dumps(self._turns, ensure_ascii=False, indent=2), encoding="utf-8")

    def context(self, n: int = 5) -> str:
        recent = self._turns[-n:]
        if not recent:
            return ""
        blocks = []
        for t in recent:
            answer = t["a"]
            if len(answer) > 300:
                answer = answer[:300] + "…"
            blocks.append(f"Q: {t['q']}\nA: {answer}")
        return "\n\n".join(blocks)
