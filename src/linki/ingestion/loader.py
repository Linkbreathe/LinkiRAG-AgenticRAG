"""Document loading: PDF/Markdown -> Markdown text (heading hierarchy preserved).

``pymupdf4llm`` is imported lazily so the package (and unit tests) import without
the heavy PDF stack installed. ``.md``/``.markdown`` files pass through verbatim.
"""

from __future__ import annotations

from pathlib import Path


def load_document(path: str | Path) -> str:
    """Return Markdown text for a PDF or Markdown source."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return p.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import pymupdf4llm  # lazy heavy dep

        return pymupdf4llm.to_markdown(str(p))
    raise ValueError(f"Unsupported document type: {suffix} ({p.name}). Use .pdf or .md.")
