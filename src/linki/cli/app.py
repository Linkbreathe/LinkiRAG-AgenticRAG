"""Typer CLI: ``linki ingest`` / ``linki ask`` / ``linki`` (multi-turn REPL)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from linki.config import load_settings

app = typer.Typer(add_completion=False, help="Linki — Agentic RAG knowledge assistant.")
console = Console()


def _build_runtime(debug: bool = False):
    """Assemble (settings, model, judge, retrieve_fn), failing loudly with a
    helpful message when provider keys or heavy deps are missing."""
    from linki.core.providers import create_judge_model, create_main_model
    from linki.tools.retrieve import make_retrieve_fn

    settings = load_settings()
    model = create_main_model(settings)
    judge = create_judge_model(settings)
    retrieve_fn = make_retrieve_fn(settings)
    return settings, model, judge, retrieve_fn


def _render_result(state: dict, debug: bool) -> None:
    if debug:
        console.print(f"[dim][router][/dim] {state.get('route')} — {state.get('route_reason', '')}")
        if state.get("rewritten_query"):
            console.print(f"[dim][rewrite][/dim] {state['rewritten_query']}")
        console.print(f"[dim][evidence][/dim] {len(state.get('evidence') or [])} chunk(s); "
                      f"verified={state.get('verified')}")
        for issue in state.get("verify_issues") or []:
            console.print(f"[yellow][verify-issue][/yellow] {issue}")

    console.print(Panel(state.get("final_answer") or state.get("answer") or "(no answer)", title="Linki"))

    citations = state.get("citations") or []
    if citations:
        lines = [
            f"[{c['index']}] {c.get('source', '?')}"
            + (f" · {c['heading_path']}" if c.get("heading_path") else "")
            for c in citations
        ]
        console.print(Panel("\n".join(lines), title="Sources", border_style="dim"))


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a .pdf or .md document."),
    kb: str = typer.Option("default", "--kb", help="Target knowledge base name."),
) -> None:
    """Ingest a document into a knowledge base (chunk -> local Qdrant hybrid index)."""
    from linki.ingestion.indexer import Indexer

    settings = load_settings()
    kb_obj = settings.kb(kb)
    if kb_obj is None:
        console.print(f"[red]Unknown knowledge base '{kb}'. Configure it in linki.yaml.[/red]")
        raise typer.Exit(1)
    if not Path(path).exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"📄 Ingesting {path} → {kb_obj.collection} …")
    stats = Indexer(settings).ingest_document(path, kb_obj)
    console.print(f"🧩 parents={stats['parents']} / children={stats['children']}")
    console.print(f"📦 indexed into local Qdrant at {settings.qdrant_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    debug: bool = typer.Option(False, "--debug", help="Print routing/retrieval decisions."),
) -> None:
    """Answer a single question through the full Agentic RAG graph."""
    from linki.graph.workflow import answer_question

    try:
        settings, model, judge, retrieve_fn = _build_runtime(debug)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    state = answer_question(
        question, model=model, judge=judge, settings=settings, retrieve_fn=retrieve_fn
    )
    _render_result(state, debug)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(7860, "--port", help="Port."),
    share: bool = typer.Option(False, "--share", help="Unsupported; kept for CLI compatibility."),
) -> None:
    """Launch the TypeScript web UI (Topics + Documents + Chat)."""
    try:
        from linki.ui.app import launch
    except ImportError:
        console.print("[red]Web UI dependencies are missing. Run: uv pip install -e '.[ui]'[/red]")
        raise typer.Exit(1)
    try:
        launch(server_name=host, server_port=port, share=share)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, debug: bool = typer.Option(False, "--debug")) -> None:
    """With no subcommand, start a multi-turn REPL."""
    if ctx.invoked_subcommand is not None:
        return

    from linki.core.session import Session
    from linki.graph.workflow import answer_question, build_workflow

    try:
        settings, model, judge, retrieve_fn = _build_runtime(debug)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    session = Session(settings.data_dir / "session.json")
    workflow = build_workflow()
    console.print("[bold]🔗 Linki[/bold] · knowledge assistant (multi-turn). Ctrl-C to exit.")
    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye 👋")
            break
        if not question:
            continue
        state = answer_question(
            question,
            model=model,
            judge=judge,
            settings=settings,
            retrieve_fn=retrieve_fn,
            session_context=session.context(),
            app=workflow,
        )
        _render_result(state, debug)
        session.add(question, state.get("final_answer") or state.get("answer") or "")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
