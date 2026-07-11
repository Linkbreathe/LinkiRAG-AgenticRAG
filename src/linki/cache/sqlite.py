"""Small durable development cache. Values are JSON, never pickle."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_CACHES: dict[str, "SQLiteCache"] = {}
_CACHES_LOCK = threading.Lock()


class SQLiteCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (namespace, cache_key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(expires_at)"
        )
        self._conn.commit()

    def get(self, namespace: str, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json, expires_at FROM cache_entries WHERE namespace=? AND cache_key=?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            if float(row[1]) <= now:
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace=? AND cache_key=?",
                    (namespace, key),
                )
                self._conn.commit()
                return None
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace=? AND cache_key=?",
                    (namespace, key),
                )
                self._conn.commit()
                return None

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = time.time()
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cache_entries(namespace, cache_key, value_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    value_json=excluded.value_json,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at
                """,
                (namespace, key, encoded, now, now + ttl_seconds),
            )
            self._conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM cache_entries WHERE namespace=? AND cache_key=?",
                (namespace, key),
            )
            self._conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE expires_at <= ?", (time.time(),)
            )
            self._conn.commit()
            return max(0, int(cursor.rowcount))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def get_sqlite_cache(path: str | Path) -> SQLiteCache:
    key = str(Path(path).resolve())
    with _CACHES_LOCK:
        cache = _CACHES.get(key)
        if cache is None:
            cache = SQLiteCache(key)
            _CACHES[key] = cache
        return cache
