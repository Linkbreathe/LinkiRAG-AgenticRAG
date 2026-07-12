"""FastAPI backend for the TypeScript Linki web console."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from linki.config import Settings

_HERE = Path(__file__).resolve().parent
_DIST = _HERE / "frontend" / "dist"


class TopicCreate(BaseModel):
    title: str
    usage_hint: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    topic: str = "default"
    history: list[ChatMessage] = Field(default_factory=list)
    mode: str = "auto"
    tenant: str = "default"
    user: str = "anonymous"
    acl: list[str] = Field(default_factory=lambda: ["public"])
    deadline_ms: int | None = Field(default=None, ge=1)


class MemoryCreate(BaseModel):
    statement: str
    tenant: str = "default"
    user: str = "anonymous"


class MemoryUpdate(BaseModel):
    statement: str
    tenant: str = "default"
    user: str = "anonymous"


class FeedbackCreate(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    tenant: str = "default"
    user: str = "anonymous"
    acl: list[str] = Field(default_factory=lambda: ["public"])


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
    from linki.core.context import RequestContext
    from linki.core.kb_registry import KnowledgeBaseRegistry
    from linki.evolution.service import get_evolution_service
    from linki.graph.workflow import answer_question_async, build_workflow
    from linki.ingestion.indexer import Indexer
    from linki.knowledge.service import get_knowledge_service
    from linki.memory.service import get_memory_service

    workflow = build_workflow()
    indexer = Indexer(settings)
    registry = KnowledgeBaseRegistry(settings)
    memory_service = get_memory_service(settings)
    knowledge_service = get_knowledge_service(settings)
    evolution_service = get_evolution_service(settings)
    main_model, judge_model = _model_names(settings)

    app = FastAPI(title="Linki Web API")

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
            "execution_modes": ["auto", "fast", "balanced", "deep"],
            "features": {
                "adaptive": settings.adaptive_enabled,
                "memory": settings.enable_memory,
                "answer_cache": settings.enable_answer_cache,
                "semantic_cache": settings.enable_semantic_cache,
            },
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
        result = indexer.clear(kb)
        return {
            "topic": topic_payload(kb), "topics": [topic_payload(item) for item in registry.list()],
            "snapshot_id": result["snapshot_id"],
        }

    @app.post("/api/documents")
    async def upload_documents(
        topic: str = Form("default"),
        tenant: str = Form("default"),
        acl: str = Form("public"),
        files: list[UploadFile] = File(...),
    ):
        import asyncio

        kb = selected_kb(topic)
        added = 0
        failures: list[dict[str, str]] = []
        snapshots: list[str] = []
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
                stats = await asyncio.to_thread(
                    indexer.ingest_document,
                    tmp_path,
                    kb,
                    tenant_id=tenant,
                    acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
                    source_name=upload.filename or Path(tmp_path).name,
                    source_uri=f"upload://{tenant}/{upload.filename or Path(tmp_path).name}",
                )
                snapshots.append(stats["snapshot_id"])
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
            "snapshot_ids": snapshots,
            "topics": [topic_payload(item) for item in registry.list()],
        }

    @app.post("/api/chat")
    async def chat(payload: ChatRequest):
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message is required.")
        if payload.mode not in {"auto", "fast", "balanced", "deep"}:
            raise HTTPException(status_code=400, detail="mode must be auto, fast, balanced, or deep")
        kb = registry.get(payload.topic)
        if kb is None:
            raise HTTPException(status_code=404, detail=f"Unknown topic: {payload.topic}")
        try:
            context = RequestContext(
                tenant_id=payload.tenant,
                user_id=payload.user,
                acl=tuple(payload.acl),
            )
            state = await answer_question_async(
                message,
                model=model,
                judge=judge,
                settings=settings,
                retrieve_fn=retrieve_fn,
                session_context=_history_context([item.model_dump() for item in payload.history]),
                app=workflow,
                execution_mode=payload.mode,
                deadline_ms=payload.deadline_ms,
                request_context=context,
                target_kb=kb.tool_name,
                thread_id=f"web:{payload.tenant}:{payload.user}",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            from linki.core.telemetry import ModelBudgetExceeded

            status = 408 if isinstance(exc, ModelBudgetExceeded) else 500
            raise HTTPException(status_code=status, detail=str(exc)) from exc

        policy = state.get("policy") or {}
        evidence = state.get("packed_evidence") or state.get("evidence") or []
        citations = state.get("citations") or []
        final = state.get("final_answer") or state.get("answer") or "(no answer)"
        trace: list[dict[str, Any]] = [
            trace_step(
                "policy", "Policy", state.get("policy_path", "unknown"),
                policy.get("reason", ""),
            )
        ]
        shadow_policy = state.get("shadow_policy") or {}
        if shadow_policy:
            trace.append(trace_step(
                "shadow", "Shadow policy", shadow_policy.get("path", "unknown"),
                shadow_policy.get("reason", ""), policy=shadow_policy,
            ))
        sub_queries = state.get("sub_queries") or []
        if sub_queries:
            trace.append(trace_step(
                "plan", "Plan", f"{len(sub_queries)} sub-queries",
                "\n".join(f"{item.get('id')}: {item.get('query')}" for item in sub_queries),
                sub_queries=sub_queries,
            ))
        retrieval_risk = state.get("retrieval_risk") or {}
        trace.append(trace_step(
            "retrieve", "Retrieve", f"{len(evidence)} packed units",
            ", ".join(retrieval_risk.get("reasons") or []) or "Coverage gate passed.",
            hits=[
                {
                    "chunk_id": item.get("chunk_id"), "source": item.get("source"),
                    "heading_path": item.get("heading_path"), "score": item.get("score"),
                }
                for item in evidence
            ],
        ))
        pack = state.get("evidence_pack") or {}
        if pack:
            trace.append(trace_step(
                "evidence", "Evidence Pack", state.get("evidence_pack_id", ""),
                f"{pack.get('token_count', 0)} / {pack.get('token_budget', 0)} tokens",
            ))
        if state.get("recalled_memories"):
            trace.append(trace_step(
                "memory", "Memory", f"{len(state['recalled_memories'])} recalled",
                "User-scoped context; never used as factual citation evidence.",
            ))
        trace.append(trace_step(
            "verify", "Validation", "passed" if state.get("verified") else "needs review",
            str(state.get("verify_issues") or "Deterministic/LLM gate passed."),
            verified=bool(state.get("verified")), issues=state.get("verify_issues") or [],
        ))
        cost = state.get("cost") or {}
        trace.append(trace_step(
            "cost", "Cost", f"{cost.get('llm_calls', 0)} model calls",
            f"tokens: {cost.get('total_tokens', 0)} · latency: {cost.get('latency_ms', 0):.1f} ms",
        ))

        used_sources = [] if _is_refusal(final) else [
            {**citation, "label": _source_label(citation)} for citation in citations
        ]
        return {
            "answer": final,
            "trace": trace,
            "evidence": evidence,
            "citations": used_sources,
            "topic": topic_payload(kb),
            "run_id": state.get("run_id"),
            "policy": policy,
            "policy_path": state.get("policy_path"),
            "shadow_policy": shadow_policy or None,
            "cost": cost,
            "cache": state.get("cache") or {},
            "snapshots": state.get("snapshots") or {},
            "versions": state.get("versions") or {},
            "evidence_pack_id": state.get("evidence_pack_id"),
            "evidence_pack": pack,
            "memory": {
                "snapshot_id": state.get("memory_snapshot_id"),
                "recalled": state.get("recalled_memories") or [],
                "changes": state.get("memory_changes") or [],
            },
        }

    @app.get("/api/memory")
    def list_memory(tenant: str = "default", user: str = "anonymous", include_deleted: bool = False):
        context = RequestContext(tenant, user, ("public",))
        return {
            "items": memory_service.export(context=context, include_deleted=include_deleted),
            "snapshot_id": memory_service.ledger.snapshot_id(tenant, user),
        }

    @app.post("/api/memory")
    def remember(payload: MemoryCreate):
        context = RequestContext(payload.tenant, payload.user, ("public",))
        item = memory_service.remember_explicit(payload.statement, context=context)
        return {"item": item.as_dict(), "snapshot_id": memory_service.ledger.snapshot_id(payload.tenant, payload.user)}

    @app.patch("/api/memory/{memory_id}")
    def edit_memory(memory_id: str, payload: MemoryUpdate):
        context = RequestContext(payload.tenant, payload.user, ("public",))
        try:
            item = memory_service.edit(memory_id, payload.statement, context=context)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Memory not found in this scope") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"item": item.as_dict(), "snapshot_id": memory_service.ledger.snapshot_id(payload.tenant, payload.user)}

    @app.delete("/api/memory/{memory_id}")
    def forget_memory(memory_id: str, tenant: str = "default", user: str = "anonymous"):
        context = RequestContext(tenant, user, ("public",))
        deleted = memory_service.forget(memory_id, context=context)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found in this scope")
        return {"deleted": [item.memory_id for item in deleted], "snapshot_id": memory_service.ledger.snapshot_id(tenant, user)}

    @app.get("/api/memory/{memory_id}/provenance")
    def memory_provenance(memory_id: str, tenant: str = "default", user: str = "anonymous"):
        item = memory_service.ledger.current(memory_id)
        if item is None or item.tenant_id != tenant or item.user_id != user:
            raise HTTPException(status_code=404, detail="Memory not found in this scope")
        return {
            "item": item.as_dict(),
            "episodes": [
                episode.__dict__ for episode in (
                    memory_service.ledger.get_episode(source) for source in item.source_episode_ids
                ) if episode is not None
            ],
            "events": memory_service.ledger.events(memory_id),
        }

    @app.get("/api/wiki")
    def list_wiki(tenant: str = "default", acl: str = "public"):
        snapshot = knowledge_service.projections.active(tenant)
        scopes = tuple(item.strip() for item in acl.split(",") if item.strip())
        pages = knowledge_service.wiki.list(tenant, snapshot.snapshot_id, scopes) if snapshot else []
        return {"snapshot_id": snapshot.snapshot_id if snapshot else None, "pages": [page.as_dict() for page in pages]}

    @app.get("/api/wiki/{slug}")
    def get_wiki(slug: str, tenant: str = "default", acl: str = "public"):
        snapshot = knowledge_service.projections.active(tenant)
        scopes = tuple(item.strip() for item in acl.split(",") if item.strip())
        page = knowledge_service.wiki.get(tenant, snapshot.snapshot_id, slug, scopes) if snapshot else None
        if page is None:
            raise HTTPException(status_code=404, detail="Wiki page not found in the active snapshot")
        return {"snapshot_id": snapshot.snapshot_id, "page": page.as_dict()}

    @app.post("/api/feedback")
    def feedback(payload: FeedbackCreate):
        try:
            item = evolution_service.feedback.record(
                tenant_id=payload.tenant, user_id=payload.user,
                kind=payload.kind, payload=payload.payload,
                run_id=payload.run_id, source="web",
                acl=tuple(payload.acl),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"feedback": item.as_dict()}

    @app.get("/api/gaps")
    def gaps(tenant: str = "default", mine: bool = True):
        if mine:
            evolution_service.gaps.mine(tenant, min_frequency=settings.gap_min_frequency)
        return {"gaps": [gap.as_dict() for gap in evolution_service.gaps.list_current(tenant)]}

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
