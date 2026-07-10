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


def _build_runtime(debug: bool = False, settings=None):
    """Assemble (settings, model, judge, retrieve_fn), failing loudly with a
    helpful message when provider keys or heavy deps are missing."""
    from linki.core.providers import create_judge_model, create_main_model
    from linki.tools.retrieve import make_retrieve_fn

    settings = settings or load_settings()
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


@app.command("bench-prep")
def bench_prep(
    benchmark: str = typer.Option("multihop-rag", "--benchmark", help="Benchmark to prepare (multihop-rag|miracl)."),
    langs: str = typer.Option("zh,en", "--langs", help="Languages to fetch topics+qrels+dataset for."),
    corpus_langs: str = typer.Option("zh", "--corpus-langs", help="Languages to also download the full corpus for (heavy)."),
    split: str = typer.Option("dev", "--split", help="MIRACL split."),
) -> None:
    """Download a benchmark's materials and map them into Linki's eval schema.

    Downloads only — it does not ingest or evaluate anything.
    """
    if benchmark not in {"miracl", "multihop-rag"}:
        console.print(f"[red]Unknown benchmark '{benchmark}'. Supported: multihop-rag, miracl.[/red]")
        raise typer.Exit(1)

    settings = load_settings()
    dest = settings.data_dir / "benchmarks" / benchmark
    if benchmark == "multihop-rag":
        from linki.eval.benchmarks import multihop_rag

        manifest = multihop_rag.prepare(dest, progress=console.print)
        console.print(
            f"✅ MultiHop-RAG: {manifest['questions']} questions "
            f"({manifest['answerable']} answerable / {manifest['unanswerable']} null), "
            f"{manifest['corpus_documents']} documents → {dest}"
        )
        return

    from linki.eval.benchmarks import miracl

    ds_langs = [x.strip() for x in langs.split(",") if x.strip()]
    cp_langs = [x.strip() for x in corpus_langs.split(",") if x.strip()]

    console.print(f"⬇️  Preparing {benchmark}: dataset langs={ds_langs}, corpus langs={cp_langs} → {dest}")
    manifest = miracl.prepare(dest, dataset_langs=ds_langs, corpus_langs=cp_langs,
                              split=split, progress=console.print)
    for lang, m in manifest["languages"].items():
        corpus = m.get("corpus")
        corpus_note = f", corpus {corpus['n_shards']} shards ({corpus['bytes'] / 1e9:.2f} GB)" if corpus else ""
        console.print(f"  {lang}: {m['n_dataset_rows']} eval rows{corpus_note}")
    console.print(f"✅ materials under {dest} (run tests later with: linki eval --dataset {dest}/dataset-zh.jsonl)")


@app.command("bench-ingest")
def bench_ingest(
    benchmark: str = typer.Option("multihop-rag", "--benchmark"),
) -> None:
    """Build an isolated benchmark index using Linki's production chunker/retriever."""
    if benchmark != "multihop-rag":
        console.print("[red]bench-ingest currently supports multihop-rag.[/red]")
        raise typer.Exit(1)
    from linki.eval.benchmarks import multihop_rag

    base = load_settings()
    dest = base.data_dir / "benchmarks" / benchmark
    corpus = dest / "corpus.json"
    if not corpus.exists():
        console.print("[red]Benchmark is not prepared. Run: linki bench-prep[/red]")
        raise typer.Exit(1)
    settings = multihop_rag.benchmark_settings(base, dest)
    console.print(f"📚 Indexing official MultiHop-RAG corpus → {settings.qdrant_path}")
    stats = multihop_rag.ingest_corpus(settings, corpus, progress=console.print)
    console.print(f"✅ indexed documents={stats['documents']} parents={stats['parents']} children={stats['children']}")


@app.command("eval")
def eval_cmd(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Path to a .jsonl eval set (defaults to the packaged sample)."),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", help="Use an isolated prepared benchmark (multihop-rag)."),
    strict: bool = typer.Option(False, "--strict", help="Also run the no-reflow ablation and detailed metrics."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Deterministic stratified sample size."),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
) -> None:
    """Run the naive-vs-agentic evaluation and print a comparison table."""
    import json
    from datetime import datetime

    from linki.eval.run_eval import format_report, load_dataset, run_eval

    settings = load_settings()
    if benchmark:
        if benchmark != "multihop-rag":
            console.print("[red]Supported benchmark: multihop-rag.[/red]")
            raise typer.Exit(1)
        from linki.eval.benchmarks import multihop_rag

        dest = settings.data_dir / "benchmarks" / benchmark
        settings = multihop_rag.benchmark_settings(settings, dest)
        dataset = dataset or str(dest / "dataset.jsonl")
    try:
        settings, model, judge, retrieve_fn = _build_runtime(settings=settings)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    ds_path = Path(dataset) if dataset else Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"
    if not ds_path.exists():
        console.print(f"[red]Dataset not found: {ds_path}[/red]")
        raise typer.Exit(1)

    rows = load_dataset(ds_path)
    if limit is not None:
        from linki.eval.benchmarks.multihop_rag import select_stratified

        rows = select_stratified(rows, limit, seed=seed)
    console.print(f"🧪 Evaluating {len(rows)} rows from {ds_path} (fair baseline vs agentic) …")
    report = run_eval(rows, model=model, judge=judge, settings=settings,
                      retrieve_fn=retrieve_fn, strict=strict)
    console.print(format_report(report))

    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"💾 saved report → {out_path}")


@app.command("bench-retrieval")
def bench_retrieval(
    benchmark: str = typer.Option("multihop-rag", "--benchmark"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run deterministic retrieval metrics over a prepared benchmark without LLM calls."""
    import json
    from datetime import datetime

    if benchmark != "multihop-rag":
        console.print("[red]Supported benchmark: multihop-rag.[/red]")
        raise typer.Exit(1)
    from linki.eval.benchmarks import multihop_rag
    from linki.eval.retrieval_eval import run_retrieval_eval
    from linki.eval.run_eval import load_dataset
    from linki.tools.retrieve import make_retrieve_fn

    base = load_settings()
    dest = base.data_dir / "benchmarks" / benchmark
    settings = multihop_rag.benchmark_settings(base, dest)
    rows = load_dataset(dest / "dataset.jsonl")
    if limit is not None:
        rows = multihop_rag.select_stratified(rows, limit, seed=seed)
    report = run_retrieval_eval(
        rows, make_retrieve_fn(settings), settings.default_kb.tool_name,
        progress=console.print,
    )
    console.print(json.dumps({
        "aggregate": report["aggregate"],
        "by_question_type": report["by_question_type"],
    }, indent=2))
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"💾 saved report → {out_path}")


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
