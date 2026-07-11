"""Security and version context carried by every request."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str = "default"
    user_id: str = "anonymous"
    acl: tuple[str, ...] = ("public",)

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must not be empty")
        if not self.user_id.strip():
            raise ValueError("user_id must not be empty")
        normalized = tuple(sorted(set(item.strip() for item in self.acl if item.strip())))
        if not normalized:
            raise ValueError("acl must contain at least one scope")
        object.__setattr__(self, "acl", normalized)

    @property
    def acl_hash(self) -> str:
        return hashlib.sha256("\0".join(self.acl).encode("utf-8")).hexdigest()[:20]

    @property
    def user_scope(self) -> str:
        return f"{self.tenant_id}:{self.user_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "acl": list(self.acl),
            "acl_hash": self.acl_hash,
        }
