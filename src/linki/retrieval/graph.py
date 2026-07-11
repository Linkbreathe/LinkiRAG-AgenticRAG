"""Graph retrieval protocol and a bounded pure-Python PPR pilot."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Protocol

from linki.retrieval.candidates import Candidate


class GraphRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        seed_entities: list[str],
        snapshot_id: str,
        limit: int,
    ) -> list[Candidate]: ...


class PersonalizedPageRankRetriever:
    """Small deterministic pilot; production graph stores implement the protocol."""

    def __init__(
        self,
        adjacency: dict[str, set[str]],
        candidates: dict[str, Candidate],
        *,
        damping: float = 0.85,
        iterations: int = 20,
        max_hops: int = 2,
    ):
        self.adjacency = adjacency
        self.candidates = candidates
        self.damping = damping
        self.iterations = iterations
        self.max_hops = max_hops

    def _bounded_nodes(self, seeds: list[str]) -> set[str]:
        visited = set(seeds)
        queue = deque((seed, 0) for seed in seeds)
        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_hops:
                continue
            for neighbor in self.adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    def retrieve(self, query: str, seed_entities: list[str], snapshot_id: str, limit: int) -> list[Candidate]:
        seeds = [seed for seed in seed_entities if seed in self.adjacency or seed in self.candidates]
        if not seeds or limit <= 0:
            return []
        nodes = self._bounded_nodes(seeds)
        teleport = {node: (1.0 / len(seeds) if node in seeds else 0.0) for node in nodes}
        rank = dict(teleport)
        for _ in range(self.iterations):
            incoming: dict[str, float] = defaultdict(float)
            leaked = 0.0
            for node in nodes:
                neighbors = self.adjacency.get(node, set()) & nodes
                if not neighbors:
                    leaked += rank.get(node, 0.0)
                    continue
                share = rank.get(node, 0.0) / len(neighbors)
                for neighbor in neighbors:
                    incoming[neighbor] += share
            rank = {
                node: (1 - self.damping) * teleport[node]
                + self.damping * (incoming[node] + leaked * teleport[node])
                for node in nodes
            }
        hits = [
            replace(self.candidates[node], retrieval_score=score, channel="graph-ppr")
            for node, score in rank.items()
            if node in self.candidates
        ]
        return sorted(hits, key=lambda item: item.retrieval_score, reverse=True)[:limit]
