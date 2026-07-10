"""FastAPI backend for the TypeScript Linki web console."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from linki.config import Settings

_HERE = Path(__file__).resolve().parent
_DIST = _HERE / "frontend" / "dist"


def _model_names(settings: Settings) -> tuple[str, str]:
    main = settings.llm_model or ("gpt-4o-mini" if settings.provider == "openai" else "deepseek-chat")
    return main, settings.judge_model or main


def _is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return "knowledge base" in low and "not found" in low or "知识库中未找到" in (text or "")


def _source_label(citation: dict[str, Any]) -> str:
    source = citation.get("source") or "?"
    heading = citation.get("heading_path") or ""
    return f"{source} · {heading}" if heading else source


def _history_context(history: list[dict[str, str]], n: int = 5) -> str:
    turns: list[str] = []
    pending_user: str | None = None
    for msg in history or []:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append(f"Q: {pending_user}\nA: {content[:300]}")
            pending_user = None
    return "\n\n".join(turns[-n:])


def build_app(settings: Settings, model: Any, judge: Any, retrieve_fn: Any):
    from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from linki.core.kb_registry import KnowledgeBaseRegistry
    from linki.graph.workflow import build_workflow
    from linki.ingestion.indexer import Indexer

    workflow = build_workflow()
    indexer = Indexer(settings)
    registry = KnowledgeBaseRegistry(settings)
    main_model, judge_model = _model_names(settings)

    app = FastAPI(title="Linki Web API")

    class TopicCreate(BaseModel):
        title: str
        usage_hint: str = ""

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        message: str
        topic: str = "default"
        history: list[ChatMessage] = []

    def selected_kb(topic: str | None):
        return registry.get(topic) or settings.default_kb

    def topic_payload(kb) -> dict[str, Any]:
        try:
            points = indexer.vectors.collection_points(kb.collection)
        except Exception:
            points = None
        docs = indexer.parents.list_sources(kb.name)
        return {
            "name": kb.name,
            "title": kb.title,
            "usage_hint": kb.usage_hint,
            "tool_name": kb.tool_name,
            "collection": kb.collection,
            "documents": docs,
            "document_count": len(docs),
            "vector_count": points,
        }

    def trace_step(kind: str, title: str, status: str, detail: str = "", **extra) -> dict[str, Any]:
        return {"kind": kind, "title": title, "status": status, "detail": detail, **extra}

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "provider": settings.provider,
            "model": main_model,
            "judge": judge_model,
            "default_topic": settings.default_kb.name,
        }

    @app.get("/api/topics")
    def list_topics():
        return {"topics": [topic_payload(kb) for kb in registry.list()]}

    @app.post("/api/topics")
    def create_topic(payload: TopicCreate):
        try:
            kb = registry.create(payload.title, payload.usage_hint)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"topic": topic_payload(kb), "topics": [topic_payload(item) for item in registry.list()]}

    @app.get("/api/topics/{topic}/documents")
    def list_documents(topic: str):
        kb = selected_kb(topic)
        return {"topic": topic_payload(kb), "documents": indexer.parents.list_sources(kb.name)}

    @app.delete("/api/topics/{topic}/documents")
    def clear_documents(topic: str):
        kb = selected_kb(topic)
        indexer.clear(kb)
        return {"topic": topic_payload(kb), "topics": [topic_payload(item) for item in registry.list()]}

    @app.post("/api/documents")
    async def upload_documents(topic: str = Form("default"), files: list[UploadFile] = File(...)):
        kb = selected_kb(topic)
        added = 0
        failures: list[dict[str, str]] = []
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in {".pdf", ".md", ".markdown"}:
                failures.append({"name": upload.filename or "file", "error": "Unsupported file type"})
                continue

            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await upload.read())
                    tmp_path = tmp.name
                indexer.ingest_document(tmp_path, kb)
                added += 1
            except Exception as exc:
                failures.append({"name": upload.filename or "file", "error": str(exc)})
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                await upload.close()

        return {
            "topic": topic_payload(kb),
            "added": added,
            "failed": failures,
            "topics": [topic_payload(item) for item in registry.list()],
        }

    @app.post("/api/chat")
    def chat(payload: ChatRequest):
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")

        kb = selected_kb(payload.topic)
        init = {
            "question": message,
            "session_context": _history_context([m.model_dump() for m in payload.history]),
            "model": model,
            "judge": judge,
            "settings": settings,
            "retrieve_fn": retrieve_fn,
            "target_kb": kb.tool_name,
            "evidence": [],
            "retrieval_keys": set(),
            "gaps": [],
            "attempts": 0,
        }

        trace: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        final = ""

        # Run-scoped observability: hooks (Cache/Dedup/TraceLog) + persistent
        # trace. The tracer fans events out to the live custom stream too, so the
        # UI trace panel below keeps working unchanged.
        import uuid as _uuid

        from linki.core.trace import Tracer, _current_tracer
        from linki.hooks.base import _current_hooks
        from linki.hooks.builtin import default_hooks

        run_id = _uuid.uuid4().hex[:12]
        hook_ctx = default_hooks(settings, run_id=run_id)
        tracer = None
        data_dir = getattr(settings, "data_dir", None)
        if getattr(settings, "enable_trace", True) and data_dir is not None:
            tracer = Tracer(run_id, Path(data_dir) / "traces")
        t_tok = _current_tracer.set(tracer)
        h_tok = _current_hooks.set(hook_ctx)

        try:
            for mode, chunk in workflow.stream(init, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    event_type = chunk.get("type")
                    if event_type == "retrieve_round":
                        hits = chunk.get("hits") or []
                        trace.append(
                            trace_step(
                                "retrieve",
                                f"Retrieve round {chunk.get('round')}",
                                f"{len(hits)} hits",
                                f"topic: {chunk.get('kb')}\nquery: {chunk.get('query')}",
                                round=chunk.get("round"),
                                query=chunk.get("query"),
                                kb=chunk.get("kb"),
                                hits=hits,
                            )
                        )
                    elif event_type == "grade":
                        ok = bool(chunk.get("sufficient"))
                        trace.append(
                            trace_step(
                                "grade",
                                "Grade",
                                "sufficient" if ok else "insufficient",
                                (
                                    f"kept: {chunk.get('kept')}\n"
                                    f"missing: {chunk.get('missing') or ''}\n"
                                    f"refined_query: {chunk.get('refined_query') or ''}"
                                ),
                                sufficient=ok,
                                kept=chunk.get("kept"),
                                missing=chunk.get("missing") or "",
                                refined_query=chunk.get("refined_query") or "",
                            )
                        )
                    continue

                for node, node_payload in chunk.items():
                    node_payload = node_payload or {}
                    if node == "router":
                        trace.append(
                            trace_step(
                                "route",
                                "Route",
                                node_payload.get("route", ""),
                                node_payload.get("route_reason", ""),
                            )
                        )
                    elif node == "rewrite" and node_payload.get("rewritten_query"):
                        trace.append(
                            trace_step("rewrite", "Rewrite", "ready", node_payload["rewritten_query"])
                        )
                    elif node == "planner":
                        sub_queries = node_payload.get("sub_queries") or []
                        detail = "\n".join(
                            (
                                f"{sq.get('id', '?')}: {sq.get('query', '')}\n"
                                f"  target: {sq.get('target_kb', '')}\n"
                                f"  reason: {sq.get('reason', '')}"
                            )
                            for sq in sub_queries
                        )
                        trace.append(
                            trace_step(
                                "plan",
                                "Plan",
                                f"{len(sub_queries)} sub-queries",
                                detail or "No retrieval plan returned.",
                                sub_queries=sub_queries,
                            )
                        )
                    elif node == "retrieve":
                        evidence = node_payload.get("evidence") or evidence
                        gaps = node_payload.get("gaps") or []
                        if gaps:
                            trace.append(trace_step("retrieve", "Retrieve", "gaps", "\n".join(gaps)))
                    elif node == "answer":
                        citations = node_payload.get("citations") or []
                        trace.append(
                            trace_step("answer", "Answer", f"{len(citations)} citations", "Generated from evidence.")
                        )
                    elif node == "verifier":
                        ok = bool(node_payload.get("verified"))
                        trace.append(
                            trace_step(
                                "verify",
                                "Verify",
                                "passed" if ok else "failed",
                                str(node_payload.get("verify_issues") or "all claims backed by evidence"),
                                verified=ok,
                                issues=node_payload.get("verify_issues") or [],
                            )
                        )
                    elif node in {"final", "final_with_warning", "chat_responder"}:
                        final = node_payload.get("final_answer") or final
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _current_hooks.reset(h_tok)
            if tracer is not None:
                tracer.finalize()
            _current_tracer.reset(t_tok)

        used_sources = [] if _is_refusal(final) else [
            {**citation, "label": _source_label(citation)} for citation in citations
        ]
        return {
            "answer": final or "(no answer)",
            "trace": trace,
            "evidence": evidence,
            "citations": used_sources,
            "topic": topic_payload(kb),
        }

    if (_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        if (_DIST / "index.html").exists():
            return FileResponse(_DIST / "index.html")
        return HTMLResponse(
            "<h1>Linki frontend is not built</h1>"
            "<p>Run <code>npm install && npm run build</code> in "
            "<code>src/linki/ui/frontend</code>.</p>",
            status_code=503,
        )

    @app.head("/")
    def index_head():
        return Response(status_code=200 if (_DIST / "index.html").exists() else 503)

    @app.get("/{path:path}")
    def spa_fallback(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return index()

    return app


def launch(*, server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False, **kw) -> None:
    if share:
        raise ValueError("The TypeScript web UI does not support Gradio share links.")

    import uvicorn

    from linki.cli.app import _build_runtime  # lazy: avoids import cycle

    settings, model, judge, retrieve_fn = _build_runtime()
    app = build_app(settings, model, judge, retrieve_fn)
    uvicorn.run(app, host=server_name, port=server_port, **kw)
