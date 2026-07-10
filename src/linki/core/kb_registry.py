"""Persistent runtime knowledge-base/topic registry.

Static knowledge bases still come from ``linki.yaml``. Topics created in the web
UI are stored under ``.linki/topics.json`` and map directly to Qdrant
collections via ``KnowledgeBase.collection``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from linki.config import KnowledgeBase, Settings


def topic_slug(title: str) -> str:
    base = re.sub(r"[^0-9a-zA-Z]+", "_", (title or "").strip().lower()).strip("_")
    if base:
        return base[:48]
    digest = hashlib.sha1((title or "topic").encode("utf-8")).hexdigest()[:8]
    return f"topic_{digest}"


class KnowledgeBaseRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._path = Path(settings.data_dir) / "topics.json"

    def list(self) -> list[KnowledgeBase]:
        merged: dict[str, KnowledgeBase] = {kb.name: kb for kb in self._settings.knowledge_bases}
        for kb in self._load_runtime():
            merged.setdefault(kb.name, kb)
        return list(merged.values())

    def get(self, name_or_tool: str | None) -> KnowledgeBase | None:
        if not name_or_tool:
            return None
        want = name_or_tool[len("Retrieve_"):] if name_or_tool.startswith("Retrieve_") else name_or_tool
        for kb in self.list():
            if kb.name == want or kb.collection == want or kb.tool_name == name_or_tool:
                return kb
        return None

    def create(self, title: str, usage_hint: str = "") -> KnowledgeBase:
        title = (title or "").strip()
        if not title:
            raise ValueError("Topic name is required.")

        existing = self.get(title)
        if existing:
            return existing

        names = {kb.name for kb in self.list()}
        base = topic_slug(title)
        name = base
        suffix = 2
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1

        kb = KnowledgeBase(
            name=name,
            title=title,
            usage_hint=usage_hint.strip() or f"Documents about {title}.",
        )
        runtime = self._load_runtime()
        runtime.append(kb)
        self._save_runtime(runtime)
        return kb

    def _load_runtime(self) -> list[KnowledgeBase]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = raw.get("knowledge_bases", raw if isinstance(raw, list) else [])
        out: list[KnowledgeBase] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            out.append(
                KnowledgeBase(
                    name=str(item["name"]),
                    title=str(item.get("title") or item["name"]),
                    usage_hint=str(item.get("usage_hint") or ""),
                )
            )
        return out

    def _save_runtime(self, kbs: list[KnowledgeBase]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "knowledge_bases": [
                {"name": kb.name, "title": kb.title, "usage_hint": kb.usage_hint}
                for kb in kbs
            ]
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
