"""Official MultiHop-RAG adapter for Linki.

The benchmark contains 2,556 questions over 609 news articles. Answerable
questions require evidence from 2-4 documents; ``null_query`` items have no
supporting evidence and are the abstention set. URLs are stable document IDs.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from linki.config import KnowledgeBase

QUESTIONS_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json"
CORPUS_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/corpus.json"


def build_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question_type = row.get("question_type", "multihop")
        sources = list(dict.fromkeys(
            evidence.get("url", "") for evidence in row.get("evidence_list", [])
            if evidence.get("url")
        ))
        should_refuse = question_type == "null_query" or not sources
        dataset.append({
            "id": f"multihop-rag-{index:04d}",
            "question": row["query"],
            "type": "out_of_kb" if should_refuse else "multihop",
            "question_type": question_type,
            "expect": {
                "should_refuse": should_refuse,
                "expect_sources": sources or None,
                "gold_answer": row.get("answer", ""),
            },
        })
    return dataset


def select_stratified(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic round-robin sample across all official question types."""
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type", row.get("type", "unknown")), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    while len(selected) < n:
        progressed = False
        for name in names:
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)


def prepare(dest: str | Path, *, progress=print) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw_questions, corpus_path = dest / "MultiHopRAG.json", dest / "corpus.json"
    for url, path in ((QUESTIONS_URL, raw_questions), (CORPUS_URL, corpus_path)):
        if not path.exists():
            progress(f"downloading {path.name} …")
            _download(url, path)
    questions = json.loads(raw_questions.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    dataset = build_dataset(questions)
    dataset_path = dest / "dataset.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset),
        encoding="utf-8",
    )
    manifest = {
        "benchmark": "MultiHop-RAG",
        "questions": len(dataset),
        "corpus_documents": len(corpus),
        "answerable": sum(not row["expect"]["should_refuse"] for row in dataset),
        "unanswerable": sum(row["expect"]["should_refuse"] for row in dataset),
        "dataset": str(dataset_path),
        "corpus": str(corpus_path),
        "source": "yixuantt/MultiHop-RAG",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def benchmark_settings(settings: Any, dest: str | Path):
    dest = Path(dest)
    kb = KnowledgeBase(
        name="multihop",
        title="MultiHop-RAG news corpus",
        usage_hint="English news articles; use for every MultiHop-RAG benchmark question.",
    )
    return replace(
        settings,
        data_dir=dest / "runtime",
        qdrant_path=dest / "qdrant",
        parent_store_path=dest / "parent_store",
        markdown_dir=dest / "markdown",
        cache_path=dest / "runtime" / "cache" / "linki.sqlite3",
        snapshot_manifest_path=dest / "runtime" / "snapshots" / "manifest.json",
        knowledge_bases=[kb],
    )


def ingest_corpus(settings: Any, corpus_path: str | Path, *, batch_size: int = 256, progress=print) -> dict[str, Any]:
    """Chunk and index all 609 documents through Linki's production chunker."""
    from linki.ingestion.chunker import DocumentChunker
    from linki.ingestion.indexer import ParentStore, VectorStoreManager

    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    kb = settings.default_kb
    chunker = DocumentChunker(settings)
    parent_pairs, child_docs = [], []
    for document in corpus:
        url = document["url"]
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        text = f"# {document.get('title', '')}\n\n{document.get('body', '')}"
        parents, children = chunker.chunk_text(text, url)
        parent_map: dict[str, str] = {}
        for local_id, parent in parents:
            parent_id = f"multihop_{digest}_{local_id.rsplit('_p', 1)[-1]}"
            parent_map[local_id] = parent_id
            parent.metadata.update({"parent_id": parent_id, "source": url, "kb": kb.name})
            parent_pairs.append((parent_id, parent))
        for index, child in enumerate(children):
            child.metadata.update({
                "parent_id": parent_map[child.metadata["parent_id"]],
                "chunk_id": f"multihop_{digest}_c{index}",
                "source": url,
                "kb": kb.name,
            })
            child_docs.append(child)

    vectors = VectorStoreManager(settings)
    parents = ParentStore(settings.parent_store_path)
    vectors.delete_collection(kb.collection)
    parents.clear(kb.name)
    vectors.create_collection(kb.collection)
    store = vectors.get_vectorstore(kb.collection)
    for start in range(0, len(child_docs), batch_size):
        batch = child_docs[start:start + batch_size]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata["chunk_id"])) for doc in batch]
        store.add_documents(batch, ids=ids)
        if progress:
            progress(f"indexed {min(start + batch_size, len(child_docs))}/{len(child_docs)} child chunks")
    parents.save_many(parent_pairs)
    stats = {"documents": len(corpus), "parents": len(parent_pairs), "children": len(child_docs)}
    from linki.knowledge.snapshots import SnapshotManifest, file_digest, index_version

    snapshot = SnapshotManifest(settings.snapshot_manifest_path).promote(
        kb.name,
        artifact_digest=file_digest(corpus_path),
        index_version=index_version(settings),
        stats=stats,
        metadata={"source": str(corpus_path), "collection": kb.collection, "benchmark": "multihop-rag"},
    )
    return {**stats, "snapshot_id": snapshot.snapshot_id}
