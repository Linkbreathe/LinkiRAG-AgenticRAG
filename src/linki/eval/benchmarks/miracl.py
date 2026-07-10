"""MIRACL adapter — download materials and map them into Linki's eval schema.

MIRACL (Making a MIRACL, TACL 2023) is an 18-language ad-hoc retrieval benchmark
with native-speaker relevance judgments and a public leaderboard. Layout on the
Hugging Face Hub (both are plain-file datasets, no loading script needed):

- ``miracl/miracl``        — ``topics/*.tsv`` (qid ⇥ query) and ``qrels/*.tsv``
  (qid ⇥ Q0 ⇥ docid ⇥ relevance), per language / split.
- ``miracl/miracl-corpus`` — ``miracl-corpus-v1.0-<lang>/docs-*.jsonl.gz``, each
  line ``{"docid", "title", "text"}``.

We keep the corpus shards as-downloaded (they *are* the material) and generate a
``dataset-<lang>.jsonl`` in Linki's schema. Gold ``docid``s (relevance > 0) become
``expect_sources`` — so when the corpus is later ingested with each passage's
``source`` set to its MIRACL ``docid``, retrieved evidence lines up with the gold
labels and citation/recall metrics are computed against the real qrels.

The pure parsing/building functions take strings (unit-tested, no network); the
``download_*`` / ``prepare`` helpers touch the Hub.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

MIRACL_REPO = "miracl/miracl"
CORPUS_REPO = "miracl/miracl-corpus"


# ————————————————————————— pure parsing / building —————————————————————————

def parse_topics(text: str) -> dict[str, str]:
    """``qid ⇥ query`` lines -> ``{qid: query}``."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0].strip()] = parts[1].strip()
    return out


def parse_qrels(text: str) -> dict[str, dict[str, int]]:
    """``qid ⇥ Q0 ⇥ docid ⇥ relevance`` lines -> ``{qid: {docid: relevance}}``."""
    out: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            parts = line.split()  # tolerate space-separated qrels
        if len(parts) < 4:
            continue
        qid, _, docid, rel = parts[0].strip(), parts[1], parts[2].strip(), parts[3].strip()
        try:
            out.setdefault(qid, {})[docid] = int(rel)
        except ValueError:
            continue
    return out


def build_dataset(topics: dict[str, str], qrels: dict[str, dict[str, int]],
                  lang: str) -> list[dict[str, Any]]:
    """Build Linki-schema rows. Gold (relevance > 0) docids -> ``expect_sources``;
    queries with no positive judgment are skipped."""
    rows: list[dict[str, Any]] = []
    for qid, query in topics.items():
        gold = [docid for docid, rel in qrels.get(qid, {}).items() if rel > 0]
        if not gold:
            continue
        rows.append({
            "id": f"miracl-{lang}-{qid}",
            "question": query,
            "type": "single",
            "expect": {
                "should_refuse": False,
                "expect_sources": gold,
                "lang": lang,
            },
        })
    return rows


def parse_corpus_line(line: str) -> dict[str, str] | None:
    """One corpus JSONL line -> ``{docid, title, text}`` (None for blanks)."""
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    return {
        "docid": obj.get("docid", ""),
        "title": obj.get("title", ""),
        "text": obj.get("text", ""),
    }


def select_subset(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministically sample ``n`` dataset rows (capped at availability)."""
    import random

    if n >= len(rows):
        return list(rows)
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def collect_needed_docids(rows: Iterable[dict[str, Any]]) -> set[str]:
    """Union of every row's gold ``expect_sources`` — the passages a subset must
    contain to keep those queries answerable."""
    needed: set[str] = set()
    for r in rows:
        needed.update(r.get("expect", {}).get("expect_sources", []))
    return needed


def build_corpus_subset(shard_paths: Iterable[str | Path], needed_docids: set[str],
                        n_distractors: int, seed: int = 0) -> list[dict[str, str]]:
    """Stream the full corpus once; keep all ``needed_docids`` (gold) passages and
    reservoir-sample ``n_distractors`` others, so retrieval isn't trivial."""
    import random

    rng = random.Random(seed)
    gold: list[dict[str, str]] = []
    seen_gold: set[str] = set()
    distractors: list[dict[str, str]] = []
    seen_count = 0

    for p in iter_corpus_passages(shard_paths):
        docid = p["docid"]
        if docid in needed_docids:
            if docid not in seen_gold:
                seen_gold.add(docid)
                gold.append(p)
            continue
        # reservoir sampling for distractors
        seen_count += 1
        if len(distractors) < n_distractors:
            distractors.append(p)
        else:
            j = rng.randint(0, seen_count - 1)
            if j < n_distractors:
                distractors[j] = p
    return gold + distractors


def ingest_passages(settings: Any, passages: list[dict[str, str]], kb: Any,
                    batch_size: int = 256, progress=print) -> int:
    """Index MIRACL passages into ``kb``'s collection, one retrievable unit per
    passage with ``source = docid`` — so retrieved evidence's ``source`` matches
    the gold ``expect_sources`` and recall is measured against the real qrels."""
    import uuid

    from langchain_core.documents import Document

    from linki.ingestion.indexer import ParentStore, VectorStoreManager

    vectors = VectorStoreManager(settings)
    parents = ParentStore(settings.parent_store_path)
    vectors.create_collection(kb.collection)
    vs = vectors.get_vectorstore(kb.collection)

    total = 0
    for start in range(0, len(passages), batch_size):
        chunk = passages[start:start + batch_size]
        docs, pairs, ids = [], [], []
        for p in chunk:
            docid = p["docid"]
            cid = f"{kb.name}::{docid}"
            content = f"{p['title']}\n\n{p['text']}" if p.get("title") else p["text"]
            meta = {"chunk_id": cid, "parent_id": cid, "kb": kb.name,
                    "source": docid, "heading_path": p.get("title", "")}
            doc = Document(page_content=content, metadata=meta)
            docs.append(doc)
            pairs.append((cid, doc))
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, cid)))
        vs.add_documents(docs, ids=ids)
        parents.save_many(pairs)
        total += len(docs)
        if progress:
            progress(f"  ingested {total}/{len(passages)} passages")
    return total


def iter_corpus_passages(shard_paths: Iterable[str | Path]) -> Iterator[dict[str, str]]:
    """Stream passages from ``.jsonl.gz`` corpus shards (for later ingestion)."""
    for shard in shard_paths:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                passage = parse_corpus_line(line)
                if passage:
                    yield passage


# ————————————————————————————— download (Hub) —————————————————————————————

def _topics_path(lang: str, split: str) -> str:
    return f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv"


def _qrels_path(lang: str, split: str) -> str:
    return f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv"


def download_topics_qrels(lang: str, dest: Path, split: str = "dev") -> tuple[Path, Path]:
    """Fetch the (small) topics + qrels TSVs for a language into ``dest``."""
    from huggingface_hub import hf_hub_download

    topics = hf_hub_download(MIRACL_REPO, _topics_path(lang, split),
                             repo_type="dataset", local_dir=dest)
    qrels = hf_hub_download(MIRACL_REPO, _qrels_path(lang, split),
                            repo_type="dataset", local_dir=dest)
    return Path(topics), Path(qrels)


def list_corpus_shards(lang: str) -> list[str]:
    from huggingface_hub import HfApi

    prefix = f"miracl-corpus-v1.0-{lang}/"
    files = HfApi().list_repo_files(CORPUS_REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".jsonl.gz"))


def download_corpus(lang: str, dest: Path, progress=print) -> list[Path]:
    """Download every corpus shard for ``lang`` into ``dest`` (full corpus)."""
    from huggingface_hub import hf_hub_download

    shards = list_corpus_shards(lang)
    paths: list[Path] = []
    for i, rel in enumerate(shards, 1):
        if progress:
            progress(f"  corpus[{lang}] shard {i}/{len(shards)}: {rel}")
        local = hf_hub_download(CORPUS_REPO, rel, repo_type="dataset", local_dir=dest)
        paths.append(Path(local))
    return paths


def prepare(dest: Path, *, dataset_langs: list[str], corpus_langs: list[str],
            split: str = "dev", progress=print) -> dict[str, Any]:
    """Download materials and write ``dataset-<lang>.jsonl`` + a manifest.

    ``dataset_langs`` get topics+qrels+dataset files; ``corpus_langs`` also get
    the full corpus shards downloaded. Does not ingest or evaluate anything.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"benchmark": "miracl", "split": split, "languages": {}}

    for lang in dataset_langs:
        progress(f"[miracl] {lang}: topics + qrels …")
        topics_p, qrels_p = download_topics_qrels(lang, raw, split)
        topics = parse_topics(topics_p.read_text(encoding="utf-8"))
        qrels = parse_qrels(qrels_p.read_text(encoding="utf-8"))
        rows = build_dataset(topics, qrels, lang)

        ds_path = dest / f"dataset-{lang}.jsonl"
        with ds_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        lang_manifest: dict[str, Any] = {
            "topics": str(topics_p), "qrels": str(qrels_p),
            "dataset": str(ds_path), "n_queries": len(topics),
            "n_dataset_rows": len(rows), "corpus": None,
        }

        if lang in corpus_langs:
            progress(f"[miracl] {lang}: full corpus …")
            shard_paths = download_corpus(lang, raw, progress)
            lang_manifest["corpus"] = {
                "n_shards": len(shard_paths),
                "shards": [str(p) for p in shard_paths],
                "bytes": sum(p.stat().st_size for p in shard_paths),
            }

        manifest["languages"][lang] = lang_manifest

    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
