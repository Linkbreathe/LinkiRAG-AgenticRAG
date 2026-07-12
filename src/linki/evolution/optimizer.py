"""Offline-only candidate generation and shadow replay adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class OfflineOptimizer(Protocol):
    def propose(self, train: list[dict], dev: list[dict], metric: Callable) -> dict[str, Any]: ...


class DSPyOptimizerAdapter:
    """Optional adapter. It returns a candidate artifact and never promotes it."""

    def __init__(self, optimizer_name: str = "MIPROv2"):
        self.optimizer_name = optimizer_name

    def propose(self, train: list[dict], dev: list[dict], metric: Callable) -> dict[str, Any]:
        try:
            import dspy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("DSPy is optional; install it only for offline optimization") from exc
        # Programs/signatures are project-specific. Keeping this adapter narrow
        # prevents pretending a generic prompt was actually optimized.
        return {
            "optimizer": self.optimizer_name,
            "status": "candidate_requires_project_program",
            "train_rows": len(train),
            "dev_rows": len(dev),
            "auto_deploy": False,
        }


@dataclass(frozen=True)
class ShadowResult:
    cases: int
    active_outputs: tuple[Any, ...]
    candidate_outputs: tuple[Any, ...]
    metrics: dict[str, Any]


def shadow_replay(
    cases: list[dict[str, Any]],
    *,
    active: Callable[[dict[str, Any]], Any],
    candidate: Callable[[dict[str, Any]], Any],
    score: Callable[[list[Any], list[Any], list[dict[str, Any]]], dict[str, Any]],
) -> ShadowResult:
    """Candidate outputs are observed and scored but never returned to users."""
    active_outputs = tuple(active(case) for case in cases)
    candidate_outputs = tuple(candidate(case) for case in cases)
    return ShadowResult(
        cases=len(cases), active_outputs=active_outputs,
        candidate_outputs=candidate_outputs,
        metrics=score(list(active_outputs), list(candidate_outputs), cases),
    )
