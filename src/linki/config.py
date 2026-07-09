"""Linki configuration.

A single ``Settings`` object carries every knob the pipeline needs: paths,
chunking sizes, model choices, retrieval and loop limits, and the knowledge-base
registry. Defaults are sensible for a first run; ``linki.yaml`` (if present in the
working directory or pointed at by ``LINKI_CONFIG``) overrides any field, and a
few env vars override the model choices so keys/models stay out of source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# MarkdownHeaderTextSplitter split points — kept as (marker, metadata-key) pairs.
DEFAULT_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass(frozen=True)
class KnowledgeBase:
    """One knowledge source. ``usage_hint`` is the *sole* routing signal the
    planner sees — write it as "what this KB holds / which questions belong
    here"."""

    name: str
    title: str
    usage_hint: str

    @property
    def collection(self) -> str:
        return f"kb_{self.name}" if not self.name.startswith("kb_") else self.name

    @property
    def tool_name(self) -> str:
        return f"Retrieve_{self.name}"


@dataclass(frozen=True)
class Settings:
    # —— paths (all under data_dir unless absolute) ——
    data_dir: Path = Path(".linki")
    qdrant_path: Path = Path(".linki/qdrant")
    parent_store_path: Path = Path(".linki/parent_store")
    markdown_dir: Path = Path(".linki/markdown")

    # —— chunking ——
    child_chunk_size: int = 500
    child_chunk_overlap: int = 100
    min_parent_size: int = 2000
    max_parent_size: int = 4000
    headers_to_split_on: list[tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_HEADERS_TO_SPLIT_ON)
    )
    sparse_vector_name: str = "sparse"

    # —— models ——
    dense_model: str = "BAAI/bge-small-en-v1.5"  # ~130MB via sentence-transformers
    sparse_model: str = "Qdrant/bm25"
    provider: str = "openai"  # reuses linki provider factory (openai|deepseek)
    llm_model: str | None = None  # None -> provider default
    judge_provider: str = "openai"
    judge_model: str | None = None  # verifier/grader-as-judge; keep != llm_model

    # —— retrieval & loops ——
    retrieval_k: int = 5
    retrieval_score_threshold: float = 0.0
    max_rounds: int = 2  # grader -> refine loop cap (per sub-query)
    max_attempts: int = 2  # verifier -> reflow cap (whole answer)

    # —— knowledge bases ——
    knowledge_bases: list[KnowledgeBase] = field(
        default_factory=lambda: [
            KnowledgeBase(
                name="default",
                title="Default knowledge base",
                usage_hint="General ingested documents. Query here for anything not covered by a more specific base.",
            )
        ]
    )

    def resolved(self, root: Path) -> "Settings":
        """Return a copy with relative paths anchored under ``root``."""

        def anchor(p: Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (root / p)

        return replace(
            self,
            data_dir=anchor(self.data_dir),
            qdrant_path=anchor(self.qdrant_path),
            parent_store_path=anchor(self.parent_store_path),
            markdown_dir=anchor(self.markdown_dir),
        )

    def kb(self, name: str) -> KnowledgeBase | None:
        want = name[len("Retrieve_"):] if name.startswith("Retrieve_") else name
        for k in self.knowledge_bases:
            if k.name == want or k.collection == want or k.tool_name == name:
                return k
        return None

    @property
    def default_kb(self) -> KnowledgeBase:
        return self.knowledge_bases[0]


def _coerce_kbs(raw: Any) -> list[KnowledgeBase]:
    kbs: list[KnowledgeBase] = []
    for item in raw or []:
        kbs.append(
            KnowledgeBase(
                name=item["name"],
                title=item.get("title", item["name"]),
                usage_hint=item.get("usage_hint", ""),
            )
        )
    return kbs


def load_settings(path: str | Path | None = None, *, root: str | Path | None = None) -> Settings:
    """Load settings from defaults, overlay ``linki.yaml``, then env overrides.

    ``root`` (default: cwd) anchors relative paths. Missing yaml is fine — the
    defaults stand up a working single-KB setup.
    """

    # Load .env first so LINKI_PROVIDER/model env overrides are visible here (not
    # just later when the provider factory runs) — otherwise provider selection
    # silently falls back to the default.
    from dotenv import load_dotenv

    load_dotenv()

    root_path = Path(root or Path.cwd())
    cfg_path = Path(path or os.getenv("LINKI_CONFIG", "linki.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = root_path / cfg_path

    data: dict[str, Any] = {}
    if cfg_path.exists():
        import yaml  # light dep; import lazily so importing config never fails

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    kwargs: dict[str, Any] = {}
    for key in (
        "child_chunk_size", "child_chunk_overlap", "min_parent_size",
        "max_parent_size", "sparse_vector_name", "dense_model", "sparse_model",
        "provider", "llm_model", "judge_provider", "judge_model",
        "retrieval_k", "retrieval_score_threshold", "max_rounds", "max_attempts",
    ):
        if key in data:
            kwargs[key] = data[key]
    for key in ("data_dir", "qdrant_path", "parent_store_path", "markdown_dir"):
        if key in data:
            kwargs[key] = Path(data[key])
    if "knowledge_bases" in data:
        kwargs["knowledge_bases"] = _coerce_kbs(data["knowledge_bases"])

    # env overrides for model/provider (keys live in .env, never in yaml)
    if os.getenv("LINKI_PROVIDER"):
        kwargs["provider"] = os.environ["LINKI_PROVIDER"]
    if os.getenv("LINKI_MODEL"):
        kwargs["llm_model"] = os.environ["LINKI_MODEL"]
    if os.getenv("LINKI_JUDGE_PROVIDER"):
        kwargs["judge_provider"] = os.environ["LINKI_JUDGE_PROVIDER"]
    if os.getenv("LINKI_JUDGE_MODEL"):
        kwargs["judge_model"] = os.environ["LINKI_JUDGE_MODEL"]

    # The judge defaults to the same provider as the main model unless the config
    # or LINKI_JUDGE_PROVIDER set it explicitly — so a single-provider setup works.
    if "judge_provider" not in kwargs:
        kwargs["judge_provider"] = kwargs.get("provider", Settings().provider)

    return Settings(**kwargs).resolved(root_path)
