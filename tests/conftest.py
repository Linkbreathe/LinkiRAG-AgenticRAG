"""Test doubles: a scriptable fake chat model and evidence/retriever helpers.

The fake inspects the SystemMessage to decide which node is calling it, so one
object can stand in for the whole graph. Per-node behaviour is overridable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FakeResponse:
    content: str


def _system_text(messages) -> str:
    for m in messages:
        if m.__class__.__name__ == "SystemMessage":
            return m.content
    return ""


def _human_text(messages) -> str:
    for m in reversed(messages):
        if m.__class__.__name__ == "HumanMessage":
            return m.content
    return ""


@dataclass
class FakeLLM:
    """Routes ``.invoke(messages)`` to a handler chosen by the system prompt.

    Override any of the handlers with a callable ``(system, human) -> str``.
    """

    router: Callable[[str, str], str] | None = None
    rewrite: Callable[[str, str], str] | None = None
    grader: Callable[[str, str], str] | None = None
    answer: Callable[[str, str], str] | None = None
    verifier: Callable[[str, str], str] | None = None
    chat: Callable[[str, str], str] | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def invoke(self, messages) -> FakeResponse:
        system, human = _system_text(messages), _human_text(messages)
        self.calls.append((system, human))
        if "router" in system:
            return FakeResponse(self._call(self.router, system, human,
                                            json.dumps({"route": "retrieve", "reason": "default"})))
        if "Rewrite the user" in system:
            return FakeResponse(self._call(self.rewrite, system, human, "rewritten query"))
        if "retrieval-quality grader" in system:
            return FakeResponse(self._call(self.grader, system, human,
                                            json.dumps({"sufficient": True, "relevant_chunk_ids": []})))
        if "numbered <evidence>" in system:
            return FakeResponse(self._call(self.answer, system, human, "Answer [1]."))
        if "answer auditor" in system:
            return FakeResponse(self._call(self.verifier, system, human,
                                            json.dumps({"passed": True, "issues": []})))
        if "friendly knowledge assistant" in system:
            return FakeResponse(self._call(self.chat, system, human, "Hi there!"))
        return FakeResponse("{}")

    @staticmethod
    def _call(handler, system, human, default):
        return handler(system, human) if handler else default


def ev(chunk_id: str, text: str = "some text", source: str = "doc.pdf", **kw) -> dict:
    base = {
        "chunk_id": chunk_id,
        "parent_id": kw.get("parent_id", chunk_id + "_p"),
        "kb": kw.get("kb", "default"),
        "source": source,
        "heading_path": kw.get("heading_path", "Section"),
        "text": text,
        "score": kw.get("score", 0.9),
    }
    return base


class FakeSettings:
    """Minimal settings stand-in for graph nodes."""

    def __init__(self, max_rounds: int = 2, max_attempts: int = 2):
        self.max_rounds = max_rounds
        self.max_attempts = max_attempts

        class _KB:
            tool_name = "Retrieve_default"
            name = "default"

        self._kb = _KB()

    @property
    def default_kb(self):
        return self._kb
