"""Gradio web UI for Linki — a cockpit that makes the agentic reasoning visible.

Layout: a Knowledge Base bar (upload/manage) above a two-pane cockpit — the
conversation on the left, a live **Agent Trace** on the right whose steps appear
as the graph executes (``stream_mode=["updates","custom"]``) and expand to reveal
exactly what each step did: the router's decision, the retrieved chunks + scores,
the grader's verdict, the verifier's checks. Evidence cards sit below the trace.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from linki.config import Settings

_AVATAR = Path(__file__).resolve().parent / "assets" / "chatbot_avatar.png"


# ————————————————————————————————— theme / css —————————————————————————————————

def _theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "JetBrains Mono", "Menlo", "monospace"],
    )


CSS = """
:root {
  --page:#f5f7fb;
  --panel:#ffffff;
  --panel-subtle:#f8fafc;
  --panel-strong:#eef4ff;
  --line:#d9e0ec;
  --line-soft:#e8edf5;
  --text:#182033;
  --muted:#657186;
  --faint:#8b96a8;
  --blue:#2764d8;
  --blue-soft:#eaf1ff;
  --teal:#0f766e;
  --teal-soft:#e6f5f2;
  --amber:#b7791f;
  --amber-soft:#fff4df;
  --red:#c33a2b;
  --red-soft:#ffebe8;
  --violet:#6757d6;
  --violet-soft:#f0edff;
  --shadow:0 12px 30px rgba(31,42,68,.08);
  --mono:ui-monospace,"SFMono-Regular","JetBrains Mono","Menlo",monospace;
}

html, body, .gradio-container {
  background:var(--page) !important;
  color:var(--text) !important;
}
.gradio-container {
  max-width:1360px !important;
  min-height:100vh !important;
  padding:18px 20px 24px !important;
}
footer, .progress-text { display:none !important; }
.gradio-container button {
  border-radius:6px !important;
  box-shadow:none !important;
  font-weight:650 !important;
}
.gradio-container button.primary {
  background:var(--blue) !important;
  border-color:var(--blue) !important;
}
.gradio-container button.stop {
  background:#ffffff !important;
  color:var(--red) !important;
  border:1px solid #f0b7ae !important;
}
.gradio-container button:not(.primary):not(.stop) {
  background:#ffffff !important;
  border:1px solid var(--line) !important;
  color:var(--text) !important;
}
.gradio-container button:hover { filter:brightness(.98); }
input, textarea {
  border-radius:8px !important;
  border:1px solid var(--line) !important;
  background:#ffffff !important;
  color:var(--text) !important;
}
input:focus, textarea:focus {
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px rgba(39,100,216,.12) !important;
}

/* app frame */
.linki-head {
  display:flex;
  align-items:center;
  gap:14px;
  flex-wrap:wrap;
  padding:12px 14px;
  margin-bottom:12px;
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:8px;
  box-shadow:var(--shadow);
}
.linki-mark {
  width:34px;
  height:34px;
  border-radius:8px;
  display:grid;
  place-items:center;
  background:var(--blue);
  color:#ffffff;
  font-family:var(--mono);
  font-weight:800;
  font-size:12px;
}
.linki-title {
  display:flex;
  flex-direction:column;
  gap:2px;
  min-width:150px;
}
.linki-title .product {
  font-weight:800;
  font-size:17px;
  line-height:1;
}
.linki-title .sub {
  color:var(--muted);
  font-size:12px;
}
.linki-head .chips {
  margin-left:auto;
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  justify-content:flex-end;
}
.linki-head .chip {
  font-family:var(--mono);
  font-size:11px;
  color:var(--muted);
  background:var(--panel-subtle);
  border:1px solid var(--line-soft);
  padding:5px 8px;
  border-radius:6px;
  white-space:nowrap;
}
.linki-head .chip b { color:var(--text); font-weight:750; }

#kb-panel, #chat-panel, #trace-panel {
  background:var(--panel) !important;
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  box-shadow:var(--shadow);
}
#kb-panel {
  margin-bottom:12px !important;
}
#chat-panel {
  padding:14px !important;
}
#kb-panel .label-wrap, #kb-panel label {
  color:var(--muted) !important;
}
#linki-chatbot {
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%) !important;
}
#linki-chatbot .message {
  border-radius:8px !important;
  box-shadow:none !important;
}
#linki-chatbot .message.user {
  background:var(--blue) !important;
  color:#ffffff !important;
}
#linki-chatbot .message.bot {
  background:#ffffff !important;
  border:1px solid var(--line-soft) !important;
  color:var(--text) !important;
}
#chat-composer {
  background:var(--panel) !important;
  border:1px solid var(--line) !important;
  border-radius:8px !important;
  padding:8px !important;
}
#chat-composer textarea {
  min-height:44px !important;
  border:none !important;
  box-shadow:none !important;
  background:transparent !important;
}
#chat-composer textarea:focus {
  box-shadow:none !important;
}
#trace-panel {
  padding:14px !important;
}

/* trace */
.trace-wrap { font-family:var(--mono); }
.trace-title {
  font-family:var(--mono);
  font-size:11px;
  letter-spacing:1.1px;
  text-transform:uppercase;
  color:var(--muted);
  margin:0 0 12px;
  display:flex;
  align-items:center;
  gap:8px;
}
.trace-title .r {
  margin-left:auto;
  color:var(--faint);
  font-size:11px;
  text-transform:none;
  letter-spacing:0;
}
.trace { position:relative; }
.stage {
  position:relative;
  padding-left:30px;
  margin-bottom:2px;
}
.stage::before {
  content:"";
  position:absolute;
  left:9px;
  top:24px;
  bottom:-2px;
  width:2px;
  background:var(--line);
}
.stage:last-child::before { display:none; }
.stage.on::before { background:#9eb6ed; }
.stage.loop::before { background:#e4c17a; }
.stage .node {
  position:absolute;
  left:1px;
  top:5px;
  width:18px;
  height:18px;
  border-radius:50%;
  display:grid;
  place-items:center;
  background:#ffffff;
  border:2px solid var(--line);
  color:var(--muted);
  z-index:1;
}
.stage .node svg {
  width:10px;
  height:10px;
}
.stage.on .node {
  border-color:var(--blue);
  background:var(--blue);
  color:#ffffff;
}
.stage.warn .node {
  border-color:var(--amber);
  background:var(--amber-soft);
  color:var(--amber);
}
.stage.bad .node {
  border-color:var(--red);
  background:var(--red-soft);
  color:var(--red);
}
details.stage > summary {
  list-style:none;
  cursor:pointer;
  display:flex;
  align-items:center;
  gap:8px;
  min-height:30px;
  padding:5px 8px 5px 3px;
  border-radius:6px;
}
details.stage > summary::-webkit-details-marker { display:none; }
details.stage > summary:hover { background:var(--panel-subtle); }
.stage .nm {
  font-size:12px;
  color:var(--text);
}
.stage .nm .sub { color:var(--faint); }
.stage .vd {
  font-size:10.5px;
  padding:1px 7px;
  border-radius:999px;
  white-space:nowrap;
}
.vd.good { color:var(--teal); background:var(--teal-soft); }
.vd.warn { color:var(--amber); background:var(--amber-soft); }
.vd.info { color:var(--blue); background:var(--blue-soft); }
.vd.bad { color:var(--red); background:var(--red-soft); }
.stage .ms {
  margin-left:auto;
  font-size:10.5px;
  color:var(--faint);
}
.stage .chev {
  width:12px;
  height:12px;
  color:var(--faint);
  transition:transform .16s ease;
  flex:0 0 auto;
}
details.stage[open] > summary .chev { transform:rotate(90deg); }
.payload {
  margin:0 0 10px 3px;
  padding:9px 10px;
  background:var(--panel-subtle);
  border:1px solid var(--line-soft);
  border-radius:8px;
  font-size:11.5px;
  line-height:1.6;
  color:var(--muted);
  white-space:pre-wrap;
  word-break:break-word;
}
.payload .k { color:var(--blue); }
.payload .s { color:var(--teal); }
.payload .w { color:var(--amber); }
.payload .p { color:var(--violet); }
.hit {
  display:flex;
  gap:8px;
  align-items:flex-start;
  padding:4px 0;
  border-bottom:1px dashed var(--line);
}
.hit:last-child { border:none; }
.hit span:nth-child(2) {
  min-width:0;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.hit .id { color:var(--violet); flex:0 0 auto; }
.hit .sc {
  margin-left:auto;
  color:var(--teal);
  flex:0 0 auto;
}
.hit .sc.low { color:var(--faint); }
.trace-empty {
  font-family:var(--mono);
  font-size:12px;
  color:var(--faint);
  padding:18px;
  text-align:center;
  border:1px dashed var(--line);
  border-radius:8px;
  background:var(--panel-subtle);
}
@keyframes fadein { from{opacity:0; transform:translateY(5px);} to{opacity:1; transform:none;} }
.stage:last-child { animation:fadein .22s ease; }
@media (prefers-reduced-motion:reduce){ .stage:last-child{ animation:none; } }

/* evidence */
.ev-h {
  font-family:var(--mono);
  font-size:11px;
  letter-spacing:1.1px;
  text-transform:uppercase;
  color:var(--muted);
  margin:16px 0 6px;
}
.ev {
  background:#ffffff;
  border:1px solid var(--line-soft);
  border-radius:8px;
  padding:10px;
  margin-top:8px;
}
.ev .top {
  display:flex;
  align-items:center;
  gap:8px;
  margin-bottom:6px;
  font-family:var(--mono);
  min-width:0;
}
.ev .n {
  font-size:11px;
  font-weight:800;
  color:var(--violet);
  background:var(--violet-soft);
  padding:1px 6px;
  border-radius:5px;
  flex:0 0 auto;
}
.ev .src {
  font-size:11px;
  color:var(--muted);
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  min-width:0;
}
.ev .score {
  margin-left:auto;
  font-size:10.5px;
  color:var(--teal);
  flex:0 0 auto;
}
.ev .bar {
  height:4px;
  border-radius:3px;
  background:var(--line-soft);
  overflow:hidden;
  margin-bottom:7px;
}
.ev .bar i {
  display:block;
  height:100%;
  background:linear-gradient(90deg,var(--violet),var(--teal));
}
.ev .snip {
  font-size:12px;
  color:var(--muted);
  line-height:1.5;
}
.ev-empty {
  font-family:var(--mono);
  font-size:12px;
  color:var(--faint);
  text-align:center;
  padding:14px;
  border:1px dashed var(--line);
  border-radius:8px;
  background:var(--panel-subtle);
}
.kbdoc {
  display:flex;
  align-items:center;
  gap:8px;
  font-family:var(--mono);
  font-size:12px;
  color:var(--text);
  padding:8px 10px;
  background:var(--panel-subtle);
  border:1px solid var(--line-soft);
  border-radius:8px;
  margin-bottom:6px;
}
.kbdoc::before {
  content:"";
  width:8px;
  height:8px;
  border-radius:50%;
  background:var(--teal);
  flex:0 0 auto;
}
.kbdoc .doc-name {
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.kb-empty {
  font-family:var(--mono);
  font-size:12px;
  color:var(--faint);
  padding:12px;
  border:1px dashed var(--line);
  border-radius:8px;
  background:var(--panel-subtle);
}

@media (max-width:900px) {
  .gradio-container { padding:12px !important; }
  .linki-head .chips {
    width:100%;
    margin-left:0;
    justify-content:flex-start;
  }
  #trace-panel { margin-top:10px !important; }
  .hit {
    display:grid;
    grid-template-columns:1fr auto;
  }
  .hit .id { grid-column:1 / -1; }
  .trace-title { align-items:flex-start; }
  .trace-title .r { display:none; }
}
"""

_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6 9 17l-5-5"/></svg>'
_PLUS = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M12 5v14M5 12h14"/></svg>'
_CHEV = '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="m9 18 6-6-6-6"/></svg>'


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


# ————————————————————————————————— renderers —————————————————————————————————

def _stage(cls: str, icon: str, name: str, vd: str, payload: str, *, open_: bool = False) -> str:
    op = " open" if open_ else ""
    return (
        f'<details class="stage {cls}"{op}><summary>'
        f'<span class="node">{icon}</span><span class="nm">{name}</span>{vd}'
        f'<span class="ms"></span>{_CHEV}</summary>'
        f'<div class="payload">{payload}</div></details>'
    )


def _render_trace(stages: list[str]) -> str:
    if not stages:
        body = '<div class="trace-empty">Ask a question. The agent steps will appear here.</div>'
    else:
        opened = stages[:-1] + [stages[-1].replace('<details class="stage', '<details open class="stage', 1)]
        body = '<div class="trace">' + "".join(opened) + "</div>"
    return f'<div class="trace-wrap"><div class="trace-title">Agent Trace<span class="r">click a step to inspect</span></div>{body}</div>'


def _render_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return '<div class="ev-h">Evidence</div><div class="ev-empty">No evidence yet</div>'
    scores = [float(e.get("score") or 0) for e in evidence] or [1.0]
    top = max(scores) or 1.0
    cards = []
    for i, e in enumerate(evidence, 1):
        sc = float(e.get("score") or 0)
        w = max(6, min(100, round(sc / top * 100)))
        loc = _esc(e.get("source", "?"))
        if e.get("heading_path"):
            loc += " · " + _esc(e["heading_path"])
        snip = _esc((e.get("text") or "").strip()[:150])
        suffix = "..." if snip else ""
        cards.append(
            f'<div class="ev"><div class="top"><span class="n">[{i}]</span>'
            f'<span class="src">{loc}</span><span class="score">{sc:.3f}</span></div>'
            f'<div class="bar"><i style="width:{w}%"></i></div>'
            f'<div class="snip">{snip}{suffix}</div></div>'
        )
    return '<div class="ev-h">Evidence</div>' + "".join(cards)


# ————————————————————————————————— app —————————————————————————————————

def build_demo(settings: Settings, model: Any, judge: Any, retrieve_fn: Any):
    import gradio as gr

    from linki.graph.workflow import build_workflow
    from linki.ingestion.indexer import Indexer

    workflow = build_workflow()
    indexer = Indexer(settings)
    default_kb = settings.default_kb
    main = settings.llm_model or ("gpt-4o-mini" if settings.provider == "openai" else "deepseek-chat")
    judge_name = settings.judge_model or main

    header_html = (
        '<div class="linki-head"><div class="linki-mark">LI</div>'
        '<div class="linki-title"><div class="product">Linki</div><div class="sub">Agentic RAG console</div></div>'
        f'<div class="chips"><span class="chip">provider <b>{_esc(settings.provider)}</b></span>'
        f'<span class="chip">model <b>{_esc(main)}</b></span>'
        f'<span class="chip">judge <b>{_esc(judge_name)}</b></span>'
        f'<span class="chip">kb <b>{_esc(default_kb.name)}</b></span></div></div>'
    )

    def kb_docs_html() -> str:
        srcs = indexer.parents.list_sources()
        if not srcs:
            return '<div class="kb-empty">No documents indexed.</div>'
        return "".join(f'<div class="kbdoc"><span class="doc-name">{_esc(s)}</span></div>' for s in srcs)

    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, kb_docs_html()
        added = 0
        for i, fp in enumerate(files):
            progress((i + 1) / len(files), desc=f"Ingesting {Path(fp).name}")
            try:
                indexer.ingest_document(fp, default_kb)
                added += 1
            except Exception as exc:
                gr.Warning(f"Failed {Path(fp).name}: {exc}")
        gr.Info(f"Indexed {added} / {len(files)}")
        return None, kb_docs_html()

    def clear_handler():
        try:
            indexer.clear(default_kb)
            gr.Info("Knowledge base cleared")
        except Exception as exc:
            gr.Error(f"Unable to clear: {exc}")
        return kb_docs_html()

    def clear_chat_handler():
        return [], _render_trace([]), _render_evidence([]), ""

    def respond(message, chat):
        message = (message or "").strip()
        if not message:
            yield chat, _render_trace([]), _render_evidence([]), ""
            return
        chat = list(chat or [])
        chat.append({"role": "user", "content": message})
        # history context = previous Q/A pairs
        ctx_pairs, pend = [], None
        for m in chat[:-1]:
            if m["role"] == "user":
                pend = m["content"]
            elif m["role"] == "assistant" and pend is not None:
                a = m["content"][:300]
                ctx_pairs.append(f"Q: {pend}\nA: {a}")
                pend = None
        init = {
            "question": message, "session_context": "\n\n".join(ctx_pairs[-5:]),
            "model": model, "judge": judge, "settings": settings, "retrieve_fn": retrieve_fn,
            "evidence": [], "retrieval_keys": set(), "gaps": [], "attempts": 0,
        }

        stages: list[str] = []
        evidence: list[dict] = []
        citations: list[dict] = []
        final = ""
        yield chat, _render_trace(stages), _render_evidence(evidence), ""

        try:
            for mode, chunk in workflow.stream(init, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    t = chunk.get("type")
                    if t == "retrieve_round":
                        hits = chunk.get("hits") or []
                        rows = "".join(
                            f'<div class="hit"><span class="id">{_esc(h.get("chunk_id"))}</span>'
                            f'<span>{_esc(h.get("heading_path") or h.get("source"))}</span>'
                            f'<span class="sc{" low" if (h.get("score") or 0) < 0.5 else ""}">'
                            f'{float(h.get("score") or 0):.3f}</span></div>'
                            for h in hits
                        ) or '<span class="w">no fresh chunks</span>'
                        stages.append(_stage(
                            "on", _CHECK, f'Retrieve <span class="sub">round {chunk.get("round")}</span>',
                            f'<span class="vd info">{len(hits)} hits</span>',
                            f'<div class="p" style="margin-bottom:6px">query: {_esc(chunk.get("query"))}</div>{rows}',
                        ))
                    elif t == "grade":
                        ok = chunk.get("sufficient")
                        if ok:
                            vd = '<span class="vd good">sufficient</span>'
                            pl = f'<span class="k">"sufficient"</span>: <span class="s">true</span>, kept {chunk.get("kept")} chunk(s)'
                            cls = "on"
                        else:
                            vd = '<span class="vd warn">insufficient</span><span class="vd warn">↻ refine</span>'
                            pl = (f'<span class="k">"sufficient"</span>: <span class="w">false</span>\n'
                                  f'<span class="k">"missing"</span>: <span class="w">{_esc(chunk.get("missing"))}</span>\n'
                                  f'<span class="k">"refined_query"</span>: <span class="p">{_esc(chunk.get("refined_query"))}</span>')
                            cls = "loop warn"
                        stages.append(_stage(cls, _CHECK if ok else _PLUS, "Grade", vd, pl))
                    yield chat, _render_trace(stages), _render_evidence(evidence), ""
                    continue

                # mode == "updates": {node: payload}
                for node, payload in chunk.items():
                    payload = payload or {}
                    if node == "router":
                        r = payload.get("route", "")
                        stages.append(_stage("on", _CHECK, "Route", f'<span class="vd info">{_esc(r)}</span>',
                            f'<span class="k">"route"</span>: <span class="p">{_esc(r)}</span>\n<span class="k">"reason"</span>: {_esc(payload.get("route_reason"))}'))
                    elif node == "rewrite" and payload.get("rewritten_query"):
                        stages.append(_stage("on", _CHECK, "Rewrite", "",
                            f'<span class="p">{_esc(payload["rewritten_query"])}</span>'))
                    elif node == "retrieve":
                        evidence = payload.get("evidence") or evidence
                        gaps = payload.get("gaps") or []
                        if gaps:
                            stages.append(_stage("warn", _PLUS, "Retrieve", '<span class="vd warn">gaps</span>',
                                f'<span class="w">{_esc("; ".join(gaps))}</span>'))
                    elif node == "answer":
                        citations = payload.get("citations") or []
                        stages.append(_stage("on", _CHECK, "Answer",
                            f'<span class="vd info">{len(citations)} citation(s)</span>',
                            f'Generated from evidence only · {len(citations)} claim(s) cited [n].'))
                    elif node == "verifier":
                        ok = payload.get("verified")
                        stages.append(_stage("on" if ok else "bad", _CHECK, "Verify",
                            f'<span class="vd {"good" if ok else "bad"}">{"passed" if ok else "failed"}</span>',
                            _esc(payload.get("verify_issues") or "all claims backed by evidence")))
                    elif node in ("final", "final_with_warning", "chat_responder"):
                        final = payload.get("final_answer") or final
                    yield chat, _render_trace(stages), _render_evidence(evidence), ""

            # append the assistant answer
            refusal = "知识库中未找到" in final or "not found in the knowledge base" in final.lower()
            if citations and not refusal:
                src = "\n\n---\n**Sources**\n" + "\n".join(
                    f"`[{c['index']}]` {c.get('source','?')}" + (f" · {c['heading_path']}" if c.get("heading_path") else "")
                    for c in citations)
                final = final + src
            chat.append({"role": "assistant", "content": final or "_(no answer)_"})
            yield chat, _render_trace(stages), _render_evidence(evidence), ""
        except Exception as exc:
            chat.append({"role": "assistant", "content": f"Error: {exc}"})
            yield chat, _render_trace(stages), _render_evidence(evidence), ""

    with gr.Blocks(title="Linki · Agentic RAG", fill_height=True) as demo:
        gr.HTML(header_html)

        with gr.Accordion("Knowledge Base", open=not bool(indexer.parents.list_sources()), elem_id="kb-panel"):
            with gr.Row():
                with gr.Column(scale=3):
                    files_in = gr.File(label="Drop PDF or Markdown files", file_count="multiple",
                                       type="filepath", height=150)
                    add_btn = gr.Button("Add documents", variant="primary")
                with gr.Column(scale=2):
                    docs_html = gr.HTML(kb_docs_html())
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh", size="sm")
                        clear_btn = gr.Button("Clear all", size="sm", variant="stop")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6, elem_id="chat-panel"):
                chatbot = gr.Chatbot(height=520, show_label=False,
                                     avatar_images=(None, str(_AVATAR) if _AVATAR.exists() else None),
                                     placeholder="<strong>Ask about your documents.</strong>",
                                     elem_id="linki-chatbot")
                with gr.Row(elem_id="chat-composer"):
                    msg = gr.Textbox(placeholder="Ask about your documents…", show_label=False, scale=8, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90)
                    clear_chat = gr.Button("Clear", scale=1, min_width=88)
                gr.Examples(
                    examples=["How do I configure TrustedHostMiddleware?",
                              "Do uvicorn workers share memory, and how do I share state?",
                              "What is our project's canary release process?"],
                    inputs=msg, label="Try",
                )
            with gr.Column(scale=5, elem_id="trace-panel"):
                trace_html = gr.HTML(_render_trace([]))
                evidence_html = gr.HTML(_render_evidence([]))

        outs = [chatbot, trace_html, evidence_html, msg]
        send.click(respond, [msg, chatbot], outs)
        msg.submit(respond, [msg, chatbot], outs)
        clear_chat.click(clear_chat_handler, None, outs)
        add_btn.click(upload_handler, [files_in], [files_in, docs_html])
        refresh_btn.click(kb_docs_html, None, docs_html)
        clear_btn.click(clear_handler, None, docs_html)

    return demo.queue()


def launch(*, server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False, **kw) -> None:
    from linki.cli.app import _build_runtime  # lazy: avoids import cycle

    settings, model, judge, retrieve_fn = _build_runtime()
    demo = build_demo(settings, model, judge, retrieve_fn)
    demo.launch(server_name=server_name, server_port=server_port, share=share,
                theme=_theme(), css=CSS, **kw)
