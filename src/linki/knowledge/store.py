"""Shared SQLite schema/connection for the knowledge truth ledger and projections."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class KnowledgeDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_artifacts (
                source_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                uri TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content_text TEXT NOT NULL,
                scope TEXT NOT NULL,
                acl_json TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(tenant_id, source_key, content_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_source_key
                ON source_artifacts(tenant_id, source_key, version);

            CREATE TABLE IF NOT EXISTS knowledge_episodes (
                episode_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                event_time TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                source_span_json TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                FOREIGN KEY(source_id) REFERENCES source_artifacts(source_id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(tenant_id, normalized_name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS claim_versions (
                claim_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id TEXT,
                literal_json TEXT,
                qualifiers_json TEXT NOT NULL,
                status TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                recorded_at TEXT NOT NULL,
                source_spans_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                supersedes TEXT,
                schema_version TEXT NOT NULL,
                episode_id TEXT,
                PRIMARY KEY(claim_id, version),
                FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY(object_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY(episode_id) REFERENCES knowledge_episodes(episode_id)
            );
            CREATE INDEX IF NOT EXISTS idx_claim_subject
                ON claim_versions(tenant_id, subject_entity_id, predicate, status);
            CREATE TABLE IF NOT EXISTS claim_fingerprints (
                fingerprint TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claim_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wiki_pages (
                page_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                slug TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                status TEXT NOT NULL,
                markdown TEXT NOT NULL,
                claim_ids_json TEXT NOT NULL,
                source_spans_json TEXT NOT NULL,
                last_verified_at TEXT NOT NULL,
                stale_reason TEXT,
                PRIMARY KEY(page_id, version)
            );
            CREATE INDEX IF NOT EXISTS idx_wiki_snapshot
                ON wiki_pages(tenant_id, snapshot_id, slug);

            CREATE TABLE IF NOT EXISTS graph_nodes (
                snapshot_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, node_id)
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                snapshot_id TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                from_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                to_id TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                recorded_at TEXT NOT NULL,
                source_spans_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, edge_id)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_from
                ON graph_edges(snapshot_id, tenant_id, from_id, relation);

            CREATE TABLE IF NOT EXISTS projection_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                status TEXT NOT NULL,
                parent_snapshot_id TEXT,
                claim_snapshot_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                promoted_at TEXT,
                validation_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projection_active (
                tenant_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL
            );
            """
        )
        self.conn.commit()


_DATABASES: dict[str, KnowledgeDatabase] = {}
_DATABASES_LOCK = threading.Lock()


def get_knowledge_database(path: str | Path) -> KnowledgeDatabase:
    key = str(Path(path).resolve())
    with _DATABASES_LOCK:
        database = _DATABASES.get(key)
        if database is None:
            database = KnowledgeDatabase(key)
            _DATABASES[key] = database
        return database
