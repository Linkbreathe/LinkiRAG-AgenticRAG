"""Shared SQLite store for feedback, gaps and version registries."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class EvolutionDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback_versions (
                feedback_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                run_id TEXT,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                source TEXT NOT NULL,
                acl_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                PRIMARY KEY(feedback_id, version)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_idempotency
                ON feedback_versions(idempotency_key, version);

            CREATE TABLE IF NOT EXISTS gap_versions (
                gap_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                cluster_label TEXT NOT NULL,
                example_questions_json TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                affected_users INTEGER NOT NULL,
                business_impact TEXT NOT NULL,
                missing_entities_json TEXT NOT NULL,
                nearest_sources_json TEXT NOT NULL,
                suggested_source_to_add TEXT NOT NULL,
                status TEXT NOT NULL,
                feedback_ids_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(gap_id, version)
            );

            CREATE TABLE IF NOT EXISTS registry_versions (
                release_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_version TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                parent_release_id TEXT,
                status TEXT NOT NULL,
                train_hash TEXT NOT NULL,
                dev_hash TEXT NOT NULL,
                test_hash TEXT NOT NULL,
                evaluations_json TEXT NOT NULL,
                change_card_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(release_id, version)
            );
            CREATE INDEX IF NOT EXISTS idx_registry_artifact
                ON registry_versions(tenant_id, artifact_type, artifact_version);
            CREATE TABLE IF NOT EXISTS registry_active (
                tenant_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                release_id TEXT NOT NULL,
                PRIMARY KEY(tenant_id, artifact_type)
            );
            """
        )
        self.conn.commit()


_DATABASES: dict[str, EvolutionDatabase] = {}
_LOCK = threading.Lock()


def get_evolution_database(path: str | Path) -> EvolutionDatabase:
    key = str(Path(path).resolve())
    with _LOCK:
        database = _DATABASES.get(key)
        if database is None:
            database = EvolutionDatabase(key)
            _DATABASES[key] = database
        return database
