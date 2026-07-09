"""Tool registry: per-KB retrieval tools + description rendering.

The planner (Phase 3) selects a target KB purely from these descriptions, so the
rendered text is the single routing signal — mirror ``usage_hint`` verbatim.
Phase 2's graph calls the retriever directly; these StructuredTools exist for the
planner and for parity with linki-agent-i's registry.
"""

from __future__ import annotations

import json
from typing import Any

from linki.config import Settings


def render_tool_descriptions(settings: Settings) -> str:
    lines = []
    for kb in settings.knowledge_bases:
        lines.append(f"- {kb.tool_name}: 检索「{kb.title}」。{kb.usage_hint}")
    return "\n".join(lines)


def build_retrieval_tools(settings: Settings, retriever: Any) -> list:
    from langchain_core.tools import StructuredTool

    tools = []
    for kb in settings.knowledge_bases:
        def _fn(query: str, _kb=kb) -> str:
            return json.dumps(retriever.retrieve(query, _kb.tool_name), ensure_ascii=False)

        tools.append(
            StructuredTool.from_function(
                func=_fn,
                name=kb.tool_name,
                description=f"检索「{kb.title}」。{kb.usage_hint}",
            )
        )
    return tools
