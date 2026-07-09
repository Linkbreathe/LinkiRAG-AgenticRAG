"""Hierarchical parent/child chunking.

Ported from the bundled reference implementation (``project/document_chunker.py``)
and decoupled from its global ``config`` module: sizes/headers now come from a
``Settings`` object. Children are what get embedded ("被搜到"); parents are what the
model reads ("给模型看"). Each parent gets a stable ``parent_id`` and a
``heading_path`` derived from the Markdown header trail.

``langchain_text_splitters`` is a light dependency (no torch), so this module and
its tests import cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linki.config import Settings


def _heading_path(metadata: dict[str, Any]) -> str:
    parts = [str(metadata[k]).strip() for k in ("h1", "h2", "h3", "h4") if metadata.get(k)]
    # Header values themselves can contain " -> " chains from merges; flatten.
    flat: list[str] = []
    for part in parts:
        flat.extend(seg.strip() for seg in part.split(" -> ") if seg.strip())
    return " > ".join(dict.fromkeys(flat))


class DocumentChunker:
    def __init__(self, settings: Settings):
        s = settings
        if s.min_parent_size <= 0 or s.max_parent_size < s.min_parent_size:
            raise ValueError("Parent sizes must be positive and min <= max.")
        if not 0 <= s.child_chunk_overlap < s.child_chunk_size:
            raise ValueError("child_chunk_overlap must be < child_chunk_size.")
        self._s = s

        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=s.headers_to_split_on, strip_headers=False
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.child_chunk_size, chunk_overlap=s.child_chunk_overlap
        )

    # —— public API ——

    def chunk_text(self, md_text: str, source_name: str) -> tuple[list, list]:
        """Return ``(parent_pairs, child_docs)`` for one document's Markdown.

        ``parent_pairs`` is a list of ``(parent_id, Document)``; ``child_docs`` is a
        list of child ``Document`` s carrying ``parent_id``/``source``/``heading_path``.
        """
        parents = self._parent_splitter.split_text(md_text)
        parents = self._merge_small_parents(parents)
        parents = self._split_large_parents(parents)
        parents = self._clean_small_chunks(parents)
        if any(len(c.page_content) > self._s.max_parent_size for c in parents):
            raise ValueError("Parent chunking produced a chunk larger than max_parent_size.")

        stem = Path(source_name).stem
        parent_pairs: list = []
        child_docs: list = []
        for i, p in enumerate(parents):
            parent_id = f"{stem}_p{i}"
            heading = _heading_path(p.metadata)
            p.metadata.update({"source": source_name, "parent_id": parent_id, "heading_path": heading})
            parent_pairs.append((parent_id, p))
            for child in self._child_splitter.split_documents([p]):
                child.metadata.setdefault("heading_path", heading)
                child_docs.append(child)
        return parent_pairs, child_docs

    # —— private helpers (ported from reference) ——

    @staticmethod
    def _merge_metadata(target, source, prepend=False):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                first, second = (value, target[key]) if prepend else (target[key], value)
                values = [
                    item.strip()
                    for raw in (first, second)
                    for item in str(raw).split(" -> ")
                    if item.strip()
                ]
                target[key] = " -> ".join(dict.fromkeys(values))

    def _merge_small_parents(self, chunks):
        if not chunks:
            return []
        merged, current = [], None
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self._merge_metadata(current.metadata, chunk.metadata)
            if len(current.page_content) >= self._s.min_parent_size:
                merged.append(current)
                current = None
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self._merge_metadata(merged[-1].metadata, current.metadata)
            else:
                merged.append(current)
        return merged

    def _split_large_parents(self, chunks):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        out = []
        for chunk in chunks:
            if len(chunk.page_content) <= self._s.max_parent_size:
                out.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._s.max_parent_size, chunk_overlap=self._s.child_chunk_overlap
                )
                out.extend(splitter.split_documents([chunk]))
        return out

    def _rebalance_pair(self, first, second):
        combined = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()
        lower = max(1, len(combined) - self._s.max_parent_size)
        upper = min(self._s.max_parent_size, len(combined) - 1)
        if len(combined) >= 2 * self._s.min_parent_size:
            lower = max(lower, self._s.min_parent_size)
            upper = min(upper, len(combined) - self._s.min_parent_size)
        preferred = min(max(len(combined) // 2, lower), upper)
        split_at = preferred
        for separator in ("\n\n", "\n", " "):
            before = combined.rfind(separator, lower, preferred + 1)
            after = combined.find(separator, preferred, upper + 1)
            if before >= lower:
                split_at = before
                break
            if after != -1:
                split_at = after
                break
        left_text = combined[:split_at].rstrip()
        right_text = combined[split_at:].lstrip()
        if len(combined) >= 2 * self._s.min_parent_size and (
            len(left_text) < self._s.min_parent_size or len(right_text) < self._s.min_parent_size
        ):
            split_at = preferred
            left_text, right_text = combined[:split_at], combined[split_at:]
        if not left_text or not right_text:
            return first, second
        metadata = dict(first.metadata)
        self._merge_metadata(metadata, second.metadata)
        first.page_content, first.metadata = left_text, dict(metadata)
        second.page_content, second.metadata = right_text, dict(metadata)
        return first, second

    def _clean_small_chunks(self, chunks):
        cleaned = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self._s.min_parent_size:
                if cleaned and len(cleaned[-1].page_content) + 2 + len(chunk.page_content) <= self._s.max_parent_size:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self._merge_metadata(cleaned[-1].metadata, chunk.metadata)
                elif (
                    i < len(chunks) - 1
                    and len(chunk.page_content) + 2 + len(chunks[i + 1].page_content) <= self._s.max_parent_size
                ):
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    self._merge_metadata(chunks[i + 1].metadata, chunk.metadata, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        for i, chunk in enumerate(cleaned):
            if len(chunk.page_content) >= self._s.min_parent_size or len(cleaned) == 1:
                continue
            if i < len(cleaned) - 1:
                cleaned[i], cleaned[i + 1] = self._rebalance_pair(chunk, cleaned[i + 1])
            else:
                cleaned[i - 1], cleaned[i] = self._rebalance_pair(cleaned[i - 1], chunk)
        return cleaned
