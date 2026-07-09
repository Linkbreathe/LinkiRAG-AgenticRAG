"""Indexing: local on-disk Qdrant (dense + sparse hybrid) + a JSON parent store.

Ported/condensed from ``project/db/*``. **No Docker**: Qdrant runs in embedded
on-disk mode via ``QdrantClient(path=...)``. All heavy deps (qdrant_client,
embeddings) are imported lazily inside methods.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from linki.config import KnowledgeBase, Settings


def _dense_embeddings(model_name: str):
    """Local dense embeddings via ``fastembed`` (ONNX, no torch), returned as a
    proper ``langchain_core.embeddings.Embeddings`` subclass (QdrantVectorStore
    type-checks the class, so duck typing is not enough)."""
    from langchain_core.embeddings import Embeddings

    class _FastEmbedDense(Embeddings):
        def __init__(self, name: str):
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [v.tolist() for v in self._model.embed(list(texts))]

        def embed_query(self, text: str) -> list[float]:
            return next(iter(self._model.embed([text]))).tolist()

    return _FastEmbedDense(model_name)


class ParentStore:
    """One JSON file per parent chunk; loaded to expand child hits into full
    context at answer time."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def save_many(self, parent_pairs: list[tuple[str, Any]]) -> None:
        for parent_id, doc in parent_pairs:
            (self._path / f"{parent_id}.json").write_text(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def load_content(self, parent_id: str) -> dict[str, Any] | None:
        fp = self._path / (parent_id if parent_id.endswith(".json") else f"{parent_id}.json")
        if not fp.exists():
            return None
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {"content": data["page_content"], "parent_id": parent_id, "metadata": data["metadata"]}

    def list_sources(self) -> list[str]:
        sources: set[str] = set()
        for fp in self._path.glob("*.json"):
            try:
                src = json.loads(fp.read_text(encoding="utf-8")).get("metadata", {}).get("source")
            except (OSError, json.JSONDecodeError):
                continue
            if src:
                sources.add(src)
        return sorted(sources)

    def clear(self) -> None:
        for fp in self._path.glob("*.json"):
            fp.unlink()


class VectorStoreManager:
    """Wraps the embedded Qdrant client + hybrid vector store. Embeddings load
    once on first use."""

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = None
        self._dense = None
        self._sparse = None

    def _ensure(self):
        if self._client is not None:
            return
        from langchain_qdrant import FastEmbedSparse
        from qdrant_client import QdrantClient

        Path(self._s.qdrant_path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self._s.qdrant_path))
        self._dense = _dense_embeddings(self._s.dense_model)
        self._sparse = FastEmbedSparse(model_name=self._s.sparse_model)

    def _dense_size(self) -> int:
        self._ensure()
        return len(self._dense.embed_query("test"))

    def create_collection(self, collection: str) -> None:
        self._ensure()
        from qdrant_client.http import models as qm

        if self._client.collection_exists(collection):
            return
        self._client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=self._dense_size(), distance=qm.Distance.COSINE),
            sparse_vectors_config={self._s.sparse_vector_name: qm.SparseVectorParams()},
        )

    def delete_collection(self, collection: str) -> None:
        self._ensure()
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)

    def get_vectorstore(self, collection: str):
        self._ensure()
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        return QdrantVectorStore(
            client=self._client,
            collection_name=collection,
            embedding=self._dense,
            sparse_embedding=self._sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=self._s.sparse_vector_name,
        )


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")


class Indexer:
    def __init__(self, settings: Settings):
        self._s = settings
        self.vectors = VectorStoreManager(settings)
        self.parents = ParentStore(settings.parent_store_path)

    def ingest_document(self, path: str | Path, kb: KnowledgeBase) -> dict[str, int]:
        from linki.ingestion.chunker import DocumentChunker
        from linki.ingestion.loader import load_document

        md_text = load_document(path)
        source_name = Path(path).name
        parent_pairs, child_docs = DocumentChunker(self._s).chunk_text(md_text, source_name)

        stem = _slug(Path(path).stem)
        for idx, child in enumerate(child_docs):
            child.metadata["chunk_id"] = f"{stem}_c{idx}"
            child.metadata["kb"] = kb.name

        self.vectors.create_collection(kb.collection)
        vs = self.vectors.get_vectorstore(kb.collection)
        # Qdrant point ids must be UUID/int; derive a stable uuid5 from chunk_id so
        # re-ingesting a document overwrites rather than duplicates. The human
        # chunk_id stays in the payload metadata (used for dedup and citations).
        import uuid

        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, c.metadata["chunk_id"])) for c in child_docs]
        vs.add_documents(child_docs, ids=point_ids)
        self.parents.save_many(parent_pairs)

        return {"parents": len(parent_pairs), "children": len(child_docs)}

    def clear(self, kb: KnowledgeBase) -> None:
        """Drop a knowledge base's vector collection and parent store."""
        self.vectors.delete_collection(kb.collection)
        self.parents.clear()
