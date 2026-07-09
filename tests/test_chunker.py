from linki.config import Settings
from linki.ingestion.chunker import DocumentChunker

MD = """# FastAPI

Intro paragraph about FastAPI and how it works in production settings.

## Deployment

### Workers

Use the --workers flag to run multiple uvicorn worker processes. Each worker is a
separate process and they do not share memory, so use external storage for shared
state across workers in production deployments.

## Middleware

### TrustedHostMiddleware

Configure TrustedHostMiddleware by passing allowed_hosts to guard against host
header attacks in your application configuration.
"""


def _settings() -> Settings:
    return Settings(child_chunk_size=120, child_chunk_overlap=20, min_parent_size=80, max_parent_size=400)


def test_chunk_text_produces_parents_and_children():
    parents, children = DocumentChunker(_settings()).chunk_text(MD, "fastapi.pdf")
    assert parents and children
    assert all(len(p.page_content) <= 400 for _, p in parents)


def test_children_carry_parent_metadata():
    _, children = DocumentChunker(_settings()).chunk_text(MD, "fastapi.pdf")
    for c in children:
        assert c.metadata.get("parent_id")
        assert c.metadata.get("source") == "fastapi.pdf"
        assert "heading_path" in c.metadata


def test_heading_path_reflects_header_trail():
    parents, _ = DocumentChunker(_settings()).chunk_text(MD, "fastapi.pdf")
    paths = [p.metadata.get("heading_path", "") for _, p in parents]
    assert any("FastAPI" in p for p in paths)
