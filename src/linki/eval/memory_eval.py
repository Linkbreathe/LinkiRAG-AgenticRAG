"""Deterministic project memory-governance regression suite."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from linki.core.context import RequestContext
from linki.memory.retriever import MemoryRetriever, render_memory_context
from linki.memory.service import MemoryService


def default_memory_suite() -> list[dict[str, Any]]:
    """Project-owned cases; intentionally not labelled as LongMemEval."""
    return [
        {
            "id": "explicit-preference", "operation": "remember",
            "statement": "code examples should use Python", "expected_status": "ACTIVE",
            "recall_query": "show a code example", "expected_recall": "Python",
        },
        {
            "id": "inferred-preference", "operation": "infer",
            "statement": "Show Python code examples", "expected_status": "PROPOSED",
        },
        {
            "id": "sensitive-preference", "operation": "infer",
            "statement": "Remember that email me at private@example.com", "expected_status": "REVIEW_REQUIRED",
        },
        {
            "id": "preference-update", "operation": "update",
            "statement": "code examples should use Python",
            "updated_statement": "code examples should use TypeScript",
            "expected_status": "ACTIVE", "expected_recall": "TypeScript",
        },
        {
            "id": "preference-delete", "operation": "delete",
            "statement": "reply concisely", "expected_status": "DELETED",
        },
    ]


def run_memory_eval(
    cases: list[dict[str, Any]] | None = None,
    *,
    path: str | Path | None = None,
    token_budget: int = 300,
) -> dict[str, Any]:
    cases = cases or default_memory_suite()
    temporary = tempfile.TemporaryDirectory() if path is None else None
    ledger_path = Path(path) if path else Path(temporary.name) / "memory.sqlite3"
    service = MemoryService(ledger_path)
    items: list[dict[str, Any]] = []
    active_expected = active_actual = active_true_positive = 0
    update_correct = update_total = 0
    stale_recalled = recalled_total = 0
    deletion_complete = deletion_total = 0
    useful_tokens = injected_tokens = 0

    for index, case in enumerate(cases):
        context = RequestContext("memory-eval", f"case-{index}", ("public",))
        operation = case["operation"]
        if operation == "remember":
            item = service.remember_explicit(case["statement"], context=context)
        elif operation == "infer":
            episode = service.ledger.append_episode(
                tenant_id=context.tenant_id, user_id=context.user_id,
                user_input=case["statement"], idempotency_key=case["id"],
            )
            extracted = service.process_episode(episode)
            item = extracted[0] if extracted else None
        elif operation == "update":
            original = service.remember_explicit(case["statement"], context=context)
            item = service.edit(original.memory_id, case["updated_statement"], context=context)
            update_total += 1
            active = service.ledger.list_current(
                tenant_id=context.tenant_id, user_id=context.user_id, statuses=("ACTIVE",),
            )
            if len(active) == 1 and case["expected_recall"] in str(active[0].content):
                update_correct += 1
        elif operation == "delete":
            original = service.remember_explicit(case["statement"], context=context)
            deleted = service.forget(original.memory_id, context=context)
            item = deleted[0]
            deletion_total += 1
            source = service.ledger.get_episode(original.source_episode_ids[0])
            current = service.ledger.current(original.memory_id)
            if current and current.content == {} and source and source.redacted_at and not source.user_input:
                deletion_complete += 1
        else:
            raise ValueError(f"unknown memory eval operation: {operation}")

        actual_status = item.status if item else "NO_CANDIDATE"
        expected_status = case["expected_status"]
        expected_active = expected_status == "ACTIVE"
        actual_active = actual_status == "ACTIVE"
        active_expected += int(expected_active)
        active_actual += int(actual_active)
        active_true_positive += int(expected_active and actual_active)

        recalled = MemoryRetriever(service.ledger).retrieve(
            case.get("recall_query", case.get("updated_statement", case["statement"])),
            tenant_id=context.tenant_id, user_id=context.user_id,
        )
        stale = [entry for entry in recalled if entry["status"] != "ACTIVE"]
        stale_recalled += len(stale)
        recalled_total += len(recalled)
        rendered = render_memory_context(recalled, token_budget)
        injected_tokens += max(0, (len(rendered) + 3) // 4)
        expected_recall = case.get("expected_recall")
        recall_correct = expected_recall is None or expected_recall in rendered
        if expected_recall and recall_correct:
            useful_tokens += max(0, (len(rendered) + 3) // 4)
        items.append({
            "id": case["id"], "expected_status": expected_status,
            "actual_status": actual_status, "status_correct": actual_status == expected_status,
            "recall_correct": recall_correct, "recalled_ids": [entry["memory_id"] for entry in recalled],
        })

    write_precision = active_true_positive / active_actual if active_actual else 1.0
    write_recall = active_true_positive / active_expected if active_expected else 1.0
    report = {
        "protocol": {
            "name": "linki-memory-governance-v1",
            "cases": len(cases), "token_budget": token_budget,
            "external_benchmark_claim": None,
        },
        "metrics": {
            "status_accuracy": sum(item["status_correct"] for item in items) / len(items) if items else 1.0,
            "write_precision": write_precision,
            "write_recall": write_recall,
            "update_correctness": update_correct / update_total if update_total else 1.0,
            "stale_memory_rate": stale_recalled / recalled_total if recalled_total else 0.0,
            "deletion_completeness": deletion_complete / deletion_total if deletion_total else 1.0,
            "useful_memory_token_ratio": useful_tokens / injected_tokens if injected_tokens else 1.0,
        },
        "items": items,
    }
    if temporary is not None:
        temporary.cleanup()
    return report
