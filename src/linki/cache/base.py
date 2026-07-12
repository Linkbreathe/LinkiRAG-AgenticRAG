"""Cache protocol and canonical, security-scoped key construction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol


class CacheBackend(Protocol):
    def get(self, namespace: str, key: str) -> Any | None: ...
    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: int) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...
    def purge_expired(self) -> int: ...


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def make_cache_key(namespace: str, **dimensions: Any) -> str:
    """Hash a canonical map; callers must pass all authorization/version axes."""
    payload = json.dumps(
        {"namespace": namespace, **dimensions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NullCache:
    def get(self, namespace: str, key: str) -> Any | None:
        return None

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: int) -> None:
        return None

    def delete(self, namespace: str, key: str) -> None:
        return None

    def purge_expired(self) -> int:
        return 0
