"""Gradio web UI for Linki.

Layout and CSS are reused from the bundled reference implementation
(``project/ui``); the chat handler is rewritten to drive **our** agentic graph
(``linki.graph.workflow``) via ``stream_mode="updates"``, surfacing each node's
decision (router / rewrite / retrieve / verifier) as a collapsible step and then
the evidence-cited answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linki.config import Settings
from linki.ui.css import custom_css

_AVATAR = Path(__file__).resolve().parent / "assets" / "chatbot_avatar.png"


def _step(title: str, content: str) -> dict:
    """A collapsible assistant message (Gradio renders metadata.title collapsed)."""
    return {"role": "assistant", "content": content, "metadata": {"title": title}}


def _history_to_context(history: list[dict], n: int = 5) -> str:
    """Turn Gradio 'messages' history into the Q/A context our rewrite node reads.
    Skips collapsible step messages (those carry a metadata.title)."""
    turns: list[str] = []
    pending_user: str | None = None
    for m in history:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and "metadata" not in m and pending_user is not None:
            ans = content if len(content) <= 300 else content[:300] + "…"
            turns.append(f"Q: {pending_user}\nA: {ans}")
            pending_user = None
    return "\n\n".join(turns[-n:])


def _sources_block(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = [
        f"[{c['index']}] {c.get('source', '?')}"
        + (f" · {c['heading_path']}" if c.get("heading_path") else "")
        for c in citations
    ]
    return "\n\n**来源 / Sources**\n" + "\n".join(lines)


def build_demo(settings: Settings, model: Any, judge: Any, retrieve_fn: Any):
    import gradio as gr

    from linki.graph.workflow import build_workflow
    from linki.ingestion.indexer import Indexer

    workflow = build_workflow()
    indexer = Indexer(settings)
    default_kb = settings.default_kb

    def list_sources() -> str:
        srcs = indexer.parents.list_sources()
        return "\n".join(f"• {s}" for s in srcs) if srcs else "📭 知识库为空 / No documents yet."

    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, list_sources()
        added = 0
        for i, fp in enumerate(files):
            progress((i + 1) / len(files), desc=f"Ingesting {Path(fp).name}")
            try:
                indexer.ingest_document(fp, default_kb)
                added += 1
            except Exception as exc:  # surface, don't crash the UI
                gr.Warning(f"Failed {Path(fp).name}: {exc}")
        gr.Info(f"✅ Added {added} / {len(files)}")
        return None, list_sources()

    def clear_handler():
        try:
            indexer.clear(default_kb)
            gr.Info("🗑️ Knowledge base cleared")
        except Exception as exc:
            gr.Error(f"Unable to clear: {exc}")
        return list_sources()

    def chat_handler(message, history):
        init = {
            "question": message.strip(),
            "session_context": _history_to_context(history),
            "model": model,
            "judge": judge,
            "settings": settings,
            "retrieve_fn": retrieve_fn,
            "evidence": [],
            "retrieval_keys": set(),
            "gaps": [],
            "attempts": 0,
        }
        steps: list[dict] = []
        final_answer, citations, route = "", [], ""

        def render() -> list[dict]:
            msgs = list(steps)
            if final_answer:
                msgs.append({"role": "assistant", "content": final_answer + _sources_block(citations)})
            return msgs

        try:
            for update in workflow.stream(init, stream_mode="updates"):
                for node, payload in update.items():
                    payload = payload or {}
                    if node == "router":
                        route = payload.get("route", "")
                        steps.append(_step("🧭 Router", f"route = **{route}** — {payload.get('route_reason', '')}"))
                    elif node == "rewrite" and payload.get("rewritten_query"):
                        steps.append(_step("✏️ Rewrite", payload["rewritten_query"]))
                    elif node == "retrieve":
                        ev = payload.get("evidence") or []
                        gaps = payload.get("gaps") or []
                        detail = f"{len(ev)} evidence chunk(s)" + (f"; gaps: {gaps}" if gaps else "")
                        steps.append(_step("🔎 Retrieve", detail))
                    elif node == "answer":
                        citations = payload.get("citations") or citations
                    elif node == "verifier":
                        steps.append(_step("🔬 Verifier", f"verified = **{payload.get('verified')}**"))
                    elif node in ("final", "final_with_warning", "chat_responder"):
                        final_answer = payload.get("final_answer") or final_answer
                    yield render()
            if not final_answer:
                final_answer = "(no answer produced)"
            yield render()
        except Exception as exc:  # keep the UI responsive on failure
            yield steps + [{"role": "assistant", "content": f"❌ Error: {exc}"}]

    with gr.Blocks(title="Linki · Agentic RAG") as demo:  # css passed at launch() in Gradio 6
        with gr.Tab("Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add documents")
            gr.Markdown("Upload PDF or Markdown files into the knowledge base.")
            files_input = gr.File(
                label="Drop PDF or Markdown files here",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False,
            )
            add_btn = gr.Button("Add Documents", variant="primary")
            gr.Markdown("## Current documents")
            file_list = gr.Textbox(
                value=list_sources(), interactive=False, lines=7, max_lines=10,
                elem_id="file-list-box", show_label=False,
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear All", variant="stop")
            add_btn.click(upload_handler, [files_input], [files_input, file_list], show_progress="corner")
            refresh_btn.click(list_sources, None, file_list)
            clear_btn.click(clear_handler, None, file_list)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=680, show_label=False,
                avatar_images=(None, str(_AVATAR) if _AVATAR.exists() else None),
                placeholder="<strong>Ask me anything!</strong><br><em>I'll route, retrieve, grade, cite — or honestly say when the knowledge base has no answer.</em>",
            )
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot)

    return demo


def launch(*, server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False, **kw) -> None:
    """Build runtime (settings/model/judge/retriever) and launch the web UI."""
    from linki.cli.app import _build_runtime  # lazy: avoids import cycle

    settings, model, judge, retrieve_fn = _build_runtime()
    demo = build_demo(settings, model, judge, retrieve_fn)
    demo.launch(server_name=server_name, server_port=server_port, share=share, css=custom_css, **kw)
