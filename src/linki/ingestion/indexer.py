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
from linki.core.kb_registry import topic_slug


def _dense_embeddings(model_name: str, query_prefix: str = "", passage_prefix: str = ""):
    """Local dense embeddings via ``fastembed`` (ONNX, no torch), returned as a
    proper ``langchain_core.embeddings.Embeddings`` subclass (QdrantVectorStore
    type-checks the class, so duck typing is not enough).

    ``query_prefix``/``passage_prefix`` support e5-family models that require
    ``"query: "`` / ``"passage: "`` markers; empty for bge and friends."""
    from langchain_core.embeddings import Embeddings

    class _FastEmbedDense(Embeddings):
        def __init__(self, name: str):
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=name)
            self._qp = query_prefix
            self._pp = passage_prefix

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            prefixed = [self._pp + t for t in texts]
            return [v.tolist() for v in self._model.embed(prefixed)]

        def embed_query(self, text: str) -> list[float]:
            return next(iter(self._model.embed([self._qp + text]))).tolist()

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

    def list_sources(self, kb_name: str | None = None) -> list[str]:
        sources: set[str] = set()
        for fp in self._path.glob("*.json"):
            try:
                metadata = json.loads(fp.read_text(encoding="utf-8")).get("metadata", {})
            except (OSError, json.JSONDecodeError):
                continue
            if kb_name:
                item_kb = metadata.get("kb")
                if item_kb != kb_name and not (kb_name == "default" and item_kb is None):
                    continue
            src = metadata.get("source")
            if src:
                sources.add(src)
        return sorted(sources)

    def clear(self, kb_name: str | None = None) -> None:
        for fp in self._path.glob("*.json"):
            if kb_name is None:
                fp.unlink()
                continue
            try:
                metadata = json.loads(fp.read_text(encoding="utf-8")).get("metadata", {})
            except (OSError, json.JSONDecodeError):
                continue
            item_kb = metadata.get("kb")
            if item_kb == kb_name or (kb_name == "default" and item_kb is None):
                fp.unlink()


class VectorStoreManager:
    """Wraps the embedded Qdrant client + hybrid vector store. Embeddings load
    once on first use."""

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = None
        self._dense = None
        self._sparse = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import os

        from qdrant_client import QdrantClient

        if getattr(self._s, "qdrant_url", None):
            key = os.getenv(getattr(self._s, "qdrant_api_key_env", "QDRANT_API_KEY"))
            self._client = QdrantClient(
                url=self._s.qdrant_url,
                api_key=key,
                prefer_grpc=getattr(self._s, "qdrant_prefer_grpc", False),
            )
            return
        Path(self._s.qdrant_path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self._s.qdrant_path))

    def _ensure(self):
        self._ensure_client()
        if self._dense is not None and self._sparse is not None:
            return
        from langchain_qdrant import FastEmbedSparse

        self._dense = _dense_embeddings(
            self._s.dense_model,
            getattr(self._s, "dense_query_prefix", ""),
            getattr(self._s, "dense_passage_prefix", ""),
        )
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
        self._ensure_client()
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)

    def collection_points(self, collection: str) -> int | None:
        self._ensure_client()
        if not self._client.collection_exists(collection):
            return None
        return self._client.get_collection(collection).points_count

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

    def ingest_document(self, path: str | Path, kb: KnowledgeBase) -> dict[str, Any]:
        from linki.ingestion.chunker import DocumentChunker
        from linki.ingestion.loader import load_document

        md_text = load_document(path)
        source_name = Path(path).name
        parent_pairs, child_docs = DocumentChunker(self._s).chunk_text(md_text, source_name)

        stem = _slug(Path(path).stem)
        kb_prefix = topic_slug(kb.name)
        parent_id_map: dict[str, str] = {}
        for parent_id, parent in parent_pairs:
            new_parent_id = f"{kb_prefix}_{parent_id}"
            parent_id_map[parent_id] = new_parent_id
            parent.metadata["parent_id"] = new_parent_id
            parent.metadata["kb"] = kb.name
        parent_pairs = [(parent_id_map[parent_id], parent) for parent_id, parent in parent_pairs]

        for idx, child in enumerate(child_docs):
            old_parent_id = child.metadata.get("parent_id", "")
            child.metadata["parent_id"] = parent_id_map.get(old_parent_id, old_parent_id)
            child.metadata["chunk_id"] = f"{kb_prefix}_{stem}_c{idx}"
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

        stats = {"parents": len(parent_pairs), "children": len(child_docs)}
        from linki.knowledge.snapshots import SnapshotManifest, file_digest, index_version

        snapshot = SnapshotManifest(self._s.snapshot_manifest_path).promote(
            kb.name,
            artifact_digest=file_digest(path),
            index_version=index_version(self._s),
            stats=stats,
            metadata={"source": source_name, "collection": kb.collection},
        )
        return {**stats, "snapshot_id": snapshot.snapshot_id}

    def clear(self, kb: KnowledgeBase) -> None:
        """Drop a knowledge base's vector collection and parent store."""
        self.vectors.delete_collection(kb.collection)
        self.parents.clear(kb.name)
