"""Optional production cache adapter; Redis is not required for local Linki."""

from __future__ import annotations

import json
from typing import Any


class RedisCache:
    def __init__(self, url: str | None = None, *, client: Any = None, prefix: str = "linki"):
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("Install Linki's 'cache' extra to use RedisCache") from exc
            client = redis.Redis.from_url(url or "redis://localhost:6379/0")
        self.client = client
        self.prefix = prefix

    def _key(self, namespace: str, key: str) -> str:
        return f"{self.prefix}:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Any | None:
        raw = self.client.get(self._key(namespace, key))
        return json.loads(raw) if raw is not None else None

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: int) -> None:
        self.client.setex(
            self._key(namespace, key), ttl_seconds,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        )

    def delete(self, namespace: str, key: str) -> None:
        self.client.delete(self._key(namespace, key))

    def purge_expired(self) -> int:
        # Redis expires keys itself; no key scan is intentionally performed.
        return 0
