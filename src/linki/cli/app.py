"""Typer CLI: ``linki ingest`` / ``linki ask`` / ``linki`` (multi-turn REPL)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from linki.config import load_settings

app = typer.Typer(add_completion=False, help="Linki — Agentic RAG knowledge assistant.")
memory_app = typer.Typer(help="Inspect and govern long-term user memory.")
knowledge_app = typer.Typer(help="Manage sourced claims and projection releases.")
wiki_app = typer.Typer(help="Browse and review Wiki projections.")
evolve_app = typer.Typer(help="Inspect gaps and run gated candidate releases.")
app.add_typer(memory_app, name="memory")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(wiki_app, name="wiki")
app.add_typer(evolve_app, name="evolve")
console = Console()


def _memory_runtime(tenant: str, user: str):
    from linki.core.context import RequestContext
    from linki.memory.service import get_memory_service

    settings = load_settings()
    return get_memory_service(settings), RequestContext(tenant, user, ("public",))


def _knowledge_runtime():
    from linki.knowledge.service import get_knowledge_service

    return get_knowledge_service(load_settings())


def _evolution_runtime():
    from linki.evolution.service import get_evolution_service

    return get_evolution_service(load_settings())


@evolve_app.command("feedback")
def evolve_feedback(
    kind: str = typer.Argument(...),
    payload: str = typer.Option(..., "--payload", help="JSON signal payload."),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """Append a feedback observation; it never mutates active knowledge."""
    import json

    try:
        data = json.loads(payload)
        item = _evolution_runtime().feedback.record(
            tenant_id=tenant, user_id=user, kind=kind, payload=data,
            run_id=run_id, source="cli",
            acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@evolve_app.command("gaps")
def evolve_gaps(
    tenant: str = typer.Option("default", "--tenant"),
    mine: bool = typer.Option(True, "--mine/--no-mine"),
    min_frequency: Optional[int] = typer.Option(None, "--min-frequency", min=1),
    status: Optional[str] = typer.Option(None, "--status"),
) -> None:
    """Mine or list knowledge-gap backlog items; no answer text is generated."""
    settings = load_settings()
    service = _evolution_runtime()
    if mine:
        service.gaps.mine(
            tenant, min_frequency=min_frequency or settings.gap_min_frequency,
        )
    console.print_json(data=[gap.as_dict() for gap in service.gaps.list_current(tenant, status)])


@evolve_app.command("gap-status")
def evolve_gap_status(
    gap_id: str = typer.Argument(...),
    status: str = typer.Argument(..., help="open|documented|wont_fix"),
) -> None:
    """Version a backlog status change."""
    try:
        gap = _evolution_runtime().gaps.set_status(gap_id, status)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=gap.as_dict())


@evolve_app.command("release-register")
def evolve_release_register(
    artifact_type: str = typer.Argument(...),
    artifact_version: str = typer.Argument(...),
    artifact: str = typer.Option(..., "--artifact", help="Candidate artifact JSON."),
    train_hash: str = typer.Option(..., "--train-hash"),
    dev_hash: str = typer.Option(..., "--dev-hash"),
    test_hash: str = typer.Option(..., "--test-hash"),
    change_card: str = typer.Option(..., "--change-card", help="Model-card-style JSON."),
    tenant: str = typer.Option("default", "--tenant"),
) -> None:
    """Register an offline candidate; registration cannot change production."""
    import json

    try:
        item = _evolution_runtime().releases.register_candidate(
            tenant_id=tenant, artifact_type=artifact_type,
            artifact_version=artifact_version, artifact=json.loads(artifact),
            train_hash=train_hash, dev_hash=dev_hash, test_hash=test_hash,
            change_card=json.loads(change_card),
        )
    except (ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@evolve_app.command("release-evaluate")
def evolve_release_evaluate(
    release_id: str = typer.Argument(...),
    stage: str = typer.Argument(..., help="offline_dev|test|shadow|canary"),
    dataset_hash: str = typer.Option(..., "--dataset-hash"),
    metrics: str = typer.Option(..., "--metrics", help="Five-category metrics JSON."),
    criteria: str = typer.Option(..., "--criteria", help="ReleaseCriteria JSON."),
) -> None:
    """Apply constraint gates to one immutable evaluation result."""
    import json

    from linki.evolution.release import ReleaseCriteria

    try:
        item = _evolution_runtime().releases.evaluate(
            release_id, stage=stage, dataset_hash=dataset_hash,
            metrics=json.loads(metrics), criteria=ReleaseCriteria(**json.loads(criteria)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@evolve_app.command("canary-start")
def evolve_canary_start(release_id: str = typer.Argument(...)) -> None:
    """Move a shadow-passed release into stable-bucket canary state."""
    try:
        item = _evolution_runtime().releases.start_canary(release_id)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@evolve_app.command("canary-check")
def evolve_canary_check(
    release_id: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
    percentage: float = typer.Option(5.0, "--percentage", min=0, max=100),
) -> None:
    """Show the deterministic assignment for one tenant/user."""
    from linki.evolution.release import ReleaseManager

    console.print_json(data={
        "release_id": release_id,
        "candidate": ReleaseManager.in_canary(
            release_id, tenant, user, percentage=percentage,
        ),
    })


@evolve_app.command("promote")
def evolve_promote(
    release_id: str = typer.Argument(...),
    human_approved: bool = typer.Option(False, "--human-approved"),
) -> None:
    """Promote only a canary-passed candidate; sensitive artifacts need approval."""
    try:
        item = _evolution_runtime().releases.promote(
            release_id, human_approved=human_approved,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@evolve_app.command("rollback")
def evolve_rollback(
    release_id: str = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Restore the prior active artifact and retain a rollback card."""
    try:
        item = _evolution_runtime().releases.rollback(release_id, reason=reason)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@knowledge_app.command("source-add")
def knowledge_source_add(
    path: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    source_key: Optional[str] = typer.Option(None, "--source-key"),
    uri: Optional[str] = typer.Option(None, "--uri"),
    scope: str = typer.Option("organization", "--scope"),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """Register an immutable source artifact without proposing facts."""
    source_path = Path(path)
    if not source_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    service = _knowledge_runtime()
    artifact = service.sources.ingest(
        tenant_id=tenant, source_key=source_key or str(source_path.resolve()),
        uri=uri or str(source_path.resolve()),
        content=source_path.read_text(encoding="utf-8"), scope=scope,
        acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
    )
    console.print_json(data=artifact.as_dict())


@knowledge_app.command("claim-propose")
def knowledge_claim_propose(
    subject: str = typer.Argument(...),
    predicate: str = typer.Argument(...),
    value: str = typer.Argument(..., help="Literal value; JSON is decoded when valid."),
    source_id: str = typer.Option(..., "--source-id"),
    quote: str = typer.Option(..., "--quote", help="Exact verbatim source quote."),
    tenant: str = typer.Option("default", "--tenant"),
    valid_from: Optional[str] = typer.Option(None, "--valid-from"),
    valid_to: Optional[str] = typer.Option(None, "--valid-to"),
    qualifiers: str = typer.Option("{}", "--qualifiers", help="JSON qualifiers."),
) -> None:
    """Create a candidate claim; it cannot answer queries before approval."""
    import json

    try:
        literal = json.loads(value)
    except json.JSONDecodeError:
        literal = value
    try:
        qualifier_data = json.loads(qualifiers)
        claim = _knowledge_runtime().propose_claim(
            tenant_id=tenant, subject=subject, predicate=predicate,
            literal_value=literal, source_id=source_id, quote=quote,
            qualifiers=qualifier_data, valid_from=valid_from, valid_to=valid_to,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=claim.as_dict())


@knowledge_app.command("claim-approve")
def knowledge_claim_approve(
    claim_id: str = typer.Argument(...),
    reviewer: str = typer.Option("human", "--reviewer"),
) -> None:
    """Activate a candidate only after schema and source-span validation."""
    try:
        claim = _knowledge_runtime().claims.activate(claim_id, actor=reviewer)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=claim.as_dict())


@knowledge_app.command("claim-reject")
def knowledge_claim_reject(
    claim_id: str = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Reject a candidate while retaining its audit trail."""
    try:
        claim = _knowledge_runtime().claims.reject(claim_id, reason=reason)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=claim.as_dict())


@knowledge_app.command("query")
def knowledge_query(
    subject: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    predicate: Optional[str] = typer.Option(None, "--predicate"),
    valid_at: Optional[str] = typer.Option(None, "--at"),
    recorded_at: Optional[str] = typer.Option(None, "--known-at"),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """Query claims by valid time and optional system-known time."""
    from datetime import UTC, datetime

    service = _knowledge_runtime()
    entity = service.entities.resolve(subject, tenant_id=tenant)
    claims = service.claims.query_at(
        tenant_id=tenant, subject_entity_id=entity.entity_id, predicate=predicate,
        valid_at=valid_at or datetime.now(UTC).isoformat(), recorded_at=recorded_at,
        acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
    )
    console.print_json(data=[claim.as_dict() for claim in claims])


@knowledge_app.command("publish")
def knowledge_publish(
    tenant: str = typer.Option("default", "--tenant"),
) -> None:
    """Build staging Wiki/Graph projections and atomically promote on full integrity."""
    try:
        snapshot = _knowledge_runtime().publish(tenant)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=snapshot.__dict__)


@knowledge_app.command("rollback")
def knowledge_rollback(
    snapshot_id: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
) -> None:
    """Atomically point projections back to a previously validated snapshot."""
    try:
        snapshot = _knowledge_runtime().projections.rollback(tenant, snapshot_id)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=snapshot.__dict__)


@knowledge_app.command("graph")
def knowledge_graph(
    entity: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    limit: int = typer.Option(20, "--limit", min=1),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """Inspect an entity's outgoing edges in the active temporal graph."""
    service = _knowledge_runtime()
    snapshot = service.projections.active(tenant)
    if snapshot is None:
        console.print("[red]No active projection snapshot. Run knowledge publish.[/red]")
        raise typer.Exit(1)
    resolved = service.entities.resolve(entity, tenant_id=tenant)
    rows = service.graph.neighbors(
        f"entity:{resolved.entity_id}", tenant_id=tenant,
        snapshot_id=snapshot.snapshot_id,
        acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
        limit=limit,
    )
    console.print_json(data=rows)


@wiki_app.command("list")
def wiki_list(
    tenant: str = typer.Option("default", "--tenant"),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """List non-stale pages in the active projection snapshot."""
    service = _knowledge_runtime()
    snapshot = service.projections.active(tenant)
    pages = service.wiki.list(
        tenant, snapshot.snapshot_id,
        tuple(item.strip() for item in acl.split(",") if item.strip()),
    ) if snapshot else []
    console.print_json(data=[page.as_dict() for page in pages])


@wiki_app.command("show")
def wiki_show(
    slug: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    acl: str = typer.Option("public", "--acl"),
) -> None:
    """Render a cited Wiki page from the active snapshot."""
    from rich.markdown import Markdown

    service = _knowledge_runtime()
    snapshot = service.projections.active(tenant)
    page = service.wiki.get(
        tenant, snapshot.snapshot_id, slug,
        tuple(item.strip() for item in acl.split(",") if item.strip()),
    ) if snapshot else None
    if page is None:
        console.print("[red]Wiki page not found in the active snapshot.[/red]")
        raise typer.Exit(1)
    console.print(Markdown(page.markdown))


@wiki_app.command("approve")
def wiki_approve(
    page_id: str = typer.Argument(...),
    reviewer: str = typer.Option("human", "--reviewer"),
) -> None:
    """Create an approved page version without changing its sourced claims."""
    try:
        page = _knowledge_runtime().wiki.approve(page_id, reviewer=reviewer)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=page.as_dict())


@memory_app.command("list")
def memory_list(
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
) -> None:
    """List current memory versions in this tenant/user namespace."""
    service, context = _memory_runtime(tenant, user)
    console.print_json(data=service.export(context=context, include_deleted=include_deleted))


@memory_app.command("remember")
def memory_remember(
    statement: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Store an explicit user preference through the promotion policy."""
    service, context = _memory_runtime(tenant, user)
    console.print_json(data=service.remember_explicit(statement, context=context).as_dict())


@memory_app.command("edit")
def memory_edit(
    memory_id: str = typer.Argument(...),
    statement: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Create a newer version and supersede an active preference."""
    service, context = _memory_runtime(tenant, user)
    try:
        item = service.edit(memory_id, statement, context=context)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print_json(data=item.as_dict())


@memory_app.command("forget")
def memory_forget(
    identifier: str = typer.Argument(..., help="Memory id, matching text, or 'all'."),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Tombstone matching memories and purge their recoverable payload."""
    service, context = _memory_runtime(tenant, user)
    deleted = service.forget(identifier, context=context)
    console.print_json(data={"deleted": [item.memory_id for item in deleted], "count": len(deleted)})


@memory_app.command("why")
def memory_why(
    memory_id: str = typer.Argument(...),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Show provenance episodes and every promotion/state event."""
    service, context = _memory_runtime(tenant, user)
    item = service.ledger.current(memory_id)
    if item is None or item.tenant_id != context.tenant_id or item.user_id != context.user_id:
        console.print("[red]Memory not found in this scope.[/red]")
        raise typer.Exit(1)
    episodes = [service.ledger.get_episode(source) for source in item.source_episode_ids]
    console.print_json(data={
        "memory": item.as_dict(),
        "episodes": [episode.__dict__ for episode in episodes if episode is not None],
        "events": service.ledger.events(memory_id),
    })


@memory_app.command("export")
def memory_export(
    output: Optional[str] = typer.Option(None, "--output"),
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Export the current namespace as auditable JSON."""
    import json

    service, context = _memory_runtime(tenant, user)
    payload = service.export(context=context, include_deleted=True)
    if output:
        Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"exported {len(payload)} memories → {output}")
    else:
        console.print_json(data=payload)


@memory_app.command("erase-user")
def memory_erase_user(
    tenant: str = typer.Option("default", "--tenant"),
    user: str = typer.Option("anonymous", "--user"),
) -> None:
    """Apply user-scope tombstones and redact raw episode payloads."""
    service, context = _memory_runtime(tenant, user)
    console.print_json(data=service.ledger.erase_user(context.tenant_id, context.user_id))


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
        policy = state.get("policy") or {}
        console.print(
            f"[dim][policy][/dim] {state.get('policy_path') or policy.get('path')} — "
            f"{policy.get('reason', state.get('route_reason', ''))}"
        )
        console.print(f"[dim][router][/dim] {state.get('route')} — {state.get('route_reason', '')}")
        if state.get("rewritten_query"):
            console.print(f"[dim][rewrite][/dim] {state['rewritten_query']}")
        console.print(f"[dim][evidence][/dim] {len(state.get('evidence') or [])} chunk(s); "
                      f"verified={state.get('verified')}")
        for issue in state.get("verify_issues") or []:
            console.print(f"[yellow][verify-issue][/yellow] {issue}")
        cost = state.get("cost") or {}
        if cost:
            console.print(
                f"[dim][cost][/dim] calls={cost.get('llm_calls', 0)} "
                f"tokens={cost.get('total_tokens', 0)} latency_ms={cost.get('latency_ms', 0)}"
            )

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
    tenant: str = typer.Option("default", "--tenant", help="Tenant for source provenance."),
    acl: str = typer.Option("public", "--acl", help="Comma-separated source ACL scopes."),
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
    stats = Indexer(settings).ingest_document(
        path, kb_obj, tenant_id=tenant,
        acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
    )
    console.print(f"🧩 parents={stats['parents']} / children={stats['children']}")
    console.print(f"🔖 snapshot={stats['snapshot_id']}")
    console.print(f"📜 source_artifact={stats['source_artifact_id']}")
    console.print(f"📦 indexed into local Qdrant at {settings.qdrant_path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    debug: bool = typer.Option(False, "--debug", help="Print routing/retrieval decisions."),
    mode: str = typer.Option("auto", "--mode", help="Execution mode: auto|fast|balanced|deep."),
    tenant: str = typer.Option("default", "--tenant", help="Tenant isolation key."),
    user: str = typer.Option("anonymous", "--user", help="User scope for caches and memory."),
    acl: str = typer.Option("public", "--acl", help="Comma-separated authorization scopes."),
    deadline_ms: Optional[int] = typer.Option(None, "--deadline-ms", min=1, help="Request deadline in milliseconds."),
) -> None:
    """Answer a single question through the full Agentic RAG graph."""
    import asyncio

    from linki.graph.workflow import answer_question_async

    try:
        settings, model, judge, retrieve_fn = _build_runtime(debug)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    try:
        from linki.core.context import RequestContext

        state = asyncio.run(answer_question_async(
            question, model=model, judge=judge, settings=settings, retrieve_fn=retrieve_fn,
            execution_mode=mode, deadline_ms=deadline_ms,
            request_context=RequestContext(
                tenant_id=tenant,
                user_id=user,
                acl=tuple(item.strip() for item in acl.split(",") if item.strip()),
            ),
        ))
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
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
    console.print(
        f"✅ indexed documents={stats['documents']} parents={stats['parents']} "
        f"children={stats['children']} snapshot={stats['snapshot_id']}"
    )


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
                      retrieve_fn=retrieve_fn, strict=strict, progress=console.print)
    console.print(format_report(report))

    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"💾 saved report → {out_path}")


def _ensure_benchmark_snapshot(settings, dest: Path) -> str:
    import json

    from linki.ingestion.indexer import VectorStoreManager
    from linki.knowledge.snapshots import SnapshotManifest, file_digest, index_version

    manifest = SnapshotManifest(settings.snapshot_manifest_path)
    active = manifest.active_id(settings.default_kb.name)
    if active:
        return active
    points = VectorStoreManager(settings).collection_points(settings.default_kb.collection)
    if points is None:
        raise ValueError("benchmark index is missing; run linki bench-ingest")
    corpus_path = dest / "corpus.json"
    documents = len(json.loads(corpus_path.read_text(encoding="utf-8")))
    questions = sum(1 for line in (dest / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    return manifest.promote(
        settings.default_kb.name,
        artifact_digest=file_digest(corpus_path),
        index_version=index_version(settings),
        stats={"documents": documents, "questions": questions, "children": int(points)},
        metadata={"registered_existing_index": True, "benchmark": "multihop-rag"},
    ).snapshot_id


@app.command("bench-retrieval")
def bench_retrieval(
    benchmark: str = typer.Option("multihop-rag", "--benchmark"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1),
    seed: int = typer.Option(0, "--seed"),
    variant: str = typer.Option("all", "--variant", help="legacy_k5|rerank_pack|rerank_ppr|all"),
) -> None:
    """Run checkpointed full retrieval ablations without LLM or gold leakage."""
    import json

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
    variants = ["legacy_k5", "rerank_pack", "rerank_ppr"] if variant == "all" else [variant]
    if any(name not in {"legacy_k5", "rerank_pack", "rerank_ppr"} for name in variants):
        console.print("[red]variant must be legacy_k5, rerank_pack, rerank_ppr, or all.[/red]")
        raise typer.Exit(1)
    corpus = dest / "corpus.json"
    try:
        snapshot_id = _ensure_benchmark_snapshot(settings, dest)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    graph = multihop_rag.CorpusPPRGraphRetriever(corpus) if "rerank_ppr" in variants else None
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, object] = {
        "protocol": {
            "benchmark": benchmark, "rows": len(rows), "seed": seed,
            "gold_used_by_retriever": False, "warmup_excluded": True,
            "corpus_snapshot_id": snapshot_id,
        },
        "variants": {},
    }
    for name in variants:
        console.print(f"🔎 retrieval variant={name} rows={len(rows)}")
        variant_protocol = {
            **matrix["protocol"], "variant": name,
            "candidate_k": settings.candidate_k if name != "legacy_k5" else settings.retrieval_k,
            "rerank_k": settings.rerank_k if name != "legacy_k5" else None,
            "ppr": name == "rerank_ppr", "reranker_score_cache": False,
        }
        retrieve_fn = make_retrieve_fn(
            settings, variant=name,
            graph_retriever=graph if name == "rerank_ppr" else None,
            cache_rerank_scores=False,
        )
        # Load embeddings/reranker before timing benchmark rows.
        retrieve_fn("Linki benchmark warmup query", settings.default_kb.tool_name)
        suffix = f"n{len(rows)}-seed{seed}"
        checkpoint = out_dir / f"retrieval-{name}-{suffix}.json"
        report = run_retrieval_eval(
            rows, retrieve_fn, settings.default_kb.tool_name,
            progress=console.print, checkpoint_path=checkpoint,
            protocol=variant_protocol,
        )
        checkpoint.write_text(json.dumps(report, indent=2), encoding="utf-8")
        matrix["variants"][name] = report["aggregate"]
        console.print(json.dumps(report["aggregate"], indent=2))
    matrix_path = out_dir / f"retrieval-matrix-n{len(rows)}-seed{seed}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    console.print(f"💾 saved matrix → {matrix_path}")


@app.command("bench-fair")
def bench_fair(
    benchmark: str = typer.Option("multihop-rag", "--benchmark"),
    fold_size: int = typer.Option(40, "--fold-size", min=1),
) -> None:
    """Run four counterbalanced E2E systems on three disjoint fixed-seed folds."""
    if benchmark != "multihop-rag":
        console.print("[red]Supported benchmark: multihop-rag.[/red]")
        raise typer.Exit(1)
    from linki.eval.benchmarks import multihop_rag
    from linki.eval.fair_eval import format_fair_report, run_fair_eval
    from linki.eval.run_eval import load_dataset
    from linki.tools.retrieve import Retriever

    base = load_settings()
    dest = base.data_dir / "benchmarks" / benchmark
    settings = multihop_rag.benchmark_settings(base, dest)
    dataset_path, corpus_path = dest / "dataset.jsonl", dest / "corpus.json"
    if not dataset_path.exists() or not corpus_path.exists():
        console.print("[red]Benchmark is not prepared. Run linki bench-prep and bench-ingest.[/red]")
        raise typer.Exit(1)
    rows = multihop_rag.select_disjoint_stratified_folds(
        load_dataset(dataset_path), fold_size=fold_size, seeds=(42, 123, 2026),
    )
    try:
        snapshot_id = _ensure_benchmark_snapshot(settings, dest)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    try:
        settings, model, judge, _ = _build_runtime(settings=settings)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    fair_retriever = Retriever(settings, cache_rerank_scores=False)
    retrieval = {
        "legacy_k5": fair_retriever.retrieve_legacy,
        "rerank_pack": fair_retriever.retrieve,
    }
    for retrieve_fn in retrieval.values():
        retrieve_fn("Linki E2E warmup query", settings.default_kb.tool_name)
    out_dir = settings.data_dir / "eval"
    checkpoint = out_dir / f"fair-adaptive-{fold_size * 3}.json"
    console.print(
        f"🧪 E2E rows={len(rows)} · folds=42/123/2026 · systems=4 · checkpoint={checkpoint}"
    )
    report = run_fair_eval(
        rows, model=model, judge=judge, settings=settings,
        retrieval=retrieval, checkpoint_path=checkpoint,
        corpus_snapshot_id=snapshot_id, progress=console.print,
    )
    console.print(format_fair_report(report))
    console.print(f"💾 saved fair report → {checkpoint}")


@app.command("bench-memory")
def bench_memory() -> None:
    """Run the project-owned governed-memory regression suite."""
    import json

    from linki.eval.memory_eval import run_memory_eval

    settings = load_settings()
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_memory_eval()
    path = out_dir / "memory-governance.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print_json(data=report["metrics"])
    console.print(f"💾 saved memory report → {path}")


@app.command("bench-stability")
def bench_stability(
    benchmark: str = typer.Option("multihop-rag", "--benchmark"),
    limit: int = typer.Option(120, "--limit", min=1),
    seed: int = typer.Option(42, "--seed"),
    repeats: int = typer.Option(3, "--repeats", min=2),
    with_answers: bool = typer.Option(
        False, "--with-answers", help="Also repeat a temperature-0 evidence-grounded answer call.",
    ),
) -> None:
    """Repeat retrieval to measure top-k stability, latency variance, and errors."""
    import json

    if benchmark != "multihop-rag":
        console.print("[red]Supported benchmark: multihop-rag.[/red]")
        raise typer.Exit(1)
    from linki.eval.benchmarks import multihop_rag
    from linki.eval.run_eval import load_dataset
    from linki.eval.stability_eval import run_stability_eval
    from linki.tools.retrieve import make_retrieve_fn

    base = load_settings()
    dest = base.data_dir / "benchmarks" / benchmark
    settings = multihop_rag.benchmark_settings(base, dest)
    rows = multihop_rag.select_stratified(
        load_dataset(dest / "dataset.jsonl"), limit, seed=seed,
    )
    retrieve_fn = make_retrieve_fn(
        settings, variant="rerank_pack", cache_rerank_scores=False,
    )
    retrieve_fn("Linki stability warmup query", settings.default_kb.tool_name)
    answer_fn = None
    if with_answers:
        from langchain_core.messages import HumanMessage, SystemMessage

        from linki.core.providers import create_main_model
        from linki.eval.naive import NAIVE_PROMPT

        model = create_main_model(settings, temperature=0.0)

        def answer_fn(question, evidence):
            rendered = "\n\n".join(
                f"[{index}] ({hit.get('source', '?')}) {hit.get('text', '').strip()}"
                for index, hit in enumerate(evidence, start=1)
            ) or "(no evidence retrieved)"
            return model.invoke([
                SystemMessage(content=NAIVE_PROMPT),
                HumanMessage(content=f"Question: {question}\n\nEvidence:\n{rendered}"),
            ]).content
    report = run_stability_eval(
        rows, retrieve_fn, settings.default_kb.tool_name,
        repeats=repeats, answer_fn=answer_fn, progress=console.print,
    )
    out_dir = settings.data_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "answers" if with_answers else "retrieval"
    path = out_dir / f"stability-{suffix}-n{len(rows)}-seed{seed}-r{repeats}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print_json(data=report["aggregate"])
    console.print(f"💾 saved stability report → {path}")


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
