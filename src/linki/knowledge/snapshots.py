"""Immutable knowledge-base snapshot chain used for cache invalidation."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    kb: str
    parent_snapshot_id: str | None
    artifact_digest: str
    index_version: str
    created_at: str
    stats: dict[str, int]
    metadata: dict[str, Any]


class SnapshotManifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _path_lock(self.path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "active": {}, "snapshots": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(f"Invalid snapshot manifest: {self.path}") from None
        if data.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported snapshot manifest schema: {data.get('schema_version')}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def promote(
        self,
        kb: str,
        *,
        artifact_digest: str,
        index_version: str,
        stats: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        if not kb or not artifact_digest or not index_version:
            raise ValueError("kb, artifact_digest and index_version are required")
        with self._lock:
            data = self._read()
            parent = data["active"].get(kb)
            created_at = datetime.now(UTC).isoformat()
            identity = json.dumps(
                {
                    "kb": kb,
                    "parent": parent,
                    "artifact": artifact_digest,
                    "index": index_version,
                    "stats": stats,
                    "nonce": time.time_ns(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            snapshot_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            record = SnapshotRecord(
                snapshot_id=snapshot_id,
                kb=kb,
                parent_snapshot_id=parent,
                artifact_digest=artifact_digest,
                index_version=index_version,
                created_at=created_at,
                stats={key: int(value) for key, value in stats.items()},
                metadata=metadata or {},
            )
            data["snapshots"].append(asdict(record))
            data["active"][kb] = snapshot_id
            self._write(data)
            return record

    def active_id(self, kb: str) -> str | None:
        with self._lock:
            return self._read()["active"].get(kb)

    def active_ids(self) -> dict[str, str]:
        with self._lock:
            return dict(self._read()["active"])

    def get(self, snapshot_id: str) -> SnapshotRecord | None:
        with self._lock:
            for row in self._read()["snapshots"]:
                if row["snapshot_id"] == snapshot_id:
                    return SnapshotRecord(**row)
        return None

    def rollback(self, kb: str, snapshot_id: str) -> SnapshotRecord:
        """Move the active pointer; immutable snapshot rows are never edited."""
        with self._lock:
            data = self._read()
            row = next(
                (item for item in data["snapshots"] if item["snapshot_id"] == snapshot_id and item["kb"] == kb),
                None,
            )
            if row is None:
                raise KeyError(f"snapshot {snapshot_id!r} does not belong to KB {kb!r}")
            data["active"][kb] = snapshot_id
            self._write(data)
            return SnapshotRecord(**row)

    def combined_id(self, kb_names: list[str]) -> str:
        active = self.active_ids()
        payload = [(name, active.get(name, "unversioned")) for name in sorted(kb_names)]
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:24]


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_version(settings: Any) -> str:
    payload = {
        "dense": settings.dense_model,
        "sparse": settings.sparse_model,
        "chunk": [settings.child_chunk_size, settings.child_chunk_overlap],
        "parents": [settings.min_parent_size, settings.max_parent_size],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
