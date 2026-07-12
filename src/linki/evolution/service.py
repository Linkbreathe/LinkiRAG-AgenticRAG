"""Evolution subsystem facade."""

from __future__ import annotations

import threading
from pathlib import Path

from linki.evolution.feedback import FeedbackLedger
from linki.evolution.gap_miner import GapMiner
from linki.evolution.release import ReleaseManager
from linki.evolution.store import get_evolution_database


class EvolutionService:
    def __init__(self, path: str | Path):
        self.db = get_evolution_database(path)
        self.feedback = FeedbackLedger(self.db)
        self.gaps = GapMiner(self.db, self.feedback)
        self.releases = ReleaseManager(self.db)


_SERVICES: dict[str, EvolutionService] = {}
_LOCK = threading.Lock()


def get_evolution_service(settings) -> EvolutionService:
    path = str(Path(settings.evolution_path).resolve())
    with _LOCK:
        service = _SERVICES.get(path)
        if service is None:
            service = EvolutionService(path)
            _SERVICES[path] = service
        return service
