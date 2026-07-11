"""Deterministic human-readable projection over active, sourced claims."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from linki.knowledge.claims import ClaimLedger, ClaimVersion
from linki.knowledge.entities import EntityResolver
from linki.knowledge.sources import SourceRepository
from linki.knowledge.store import KnowledgeDatabase


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(text: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.casefold()).strip("-")
    return value or hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class WikiPage:
    page_id: str
    version: int
    tenant_id: str
    slug: str
    title: str
    domain: str
    topic: str
    snapshot_id: str
    status: str
    markdown: str
    claim_ids: tuple[str, ...]
    source_spans: tuple[dict[str, Any], ...]
    last_verified_at: str
    stale_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikiProjection:
    def __init__(
        self,
        database: KnowledgeDatabase,
        claims: ClaimLedger,
        entities: EntityResolver,
        sources: SourceRepository,
    ):
        self.db = database
        self.claims = claims
        self.entities = entities
        self.sources = sources

    @staticmethod
    def _page(row) -> WikiPage:
        return WikiPage(
            page_id=row["page_id"], version=int(row["version"]),
            tenant_id=row["tenant_id"], slug=row["slug"], title=row["title"],
            domain=row["domain"], topic=row["topic"], snapshot_id=row["snapshot_id"],
            status=row["status"], markdown=row["markdown"],
            claim_ids=tuple(json.loads(row["claim_ids_json"])),
            source_spans=tuple(json.loads(row["source_spans_json"])),
            last_verified_at=row["last_verified_at"], stale_reason=row["stale_reason"],
        )

    def _value(self, claim: ClaimVersion) -> str:
        if claim.object_entity_id:
            entity = self.entities.get(claim.object_entity_id)
            return entity.canonical_name if entity else claim.object_entity_id
        return str(claim.literal_value)

    def _render(self, entity_name: str, claims: list[ClaimVersion], history: list[ClaimVersion]) -> str:
        domain = str(claims[0].qualifiers.get("domain", "Knowledge"))
        topic = str(claims[0].qualifiers.get("topic", entity_name))
        lines = [
            f"# {entity_name}", "", f"_Domain: {domain} · Topic: {topic}_", "",
            "## TL;DR", "",
        ]
        for claim in claims:
            span = claim.source_spans[0]
            source = self.sources.get(str(span["source_id"]))
            location = f"{source.uri}#{span['char_start']}-{span['char_end']}" if source else span["source_id"]
            lines.append(
                f"- **{claim.predicate}**: {self._value(claim)} "
                f"`[claim:{claim.claim_id}]` `[source:{location}]`"
            )
        lines.extend(["", "## Current Decision / How It Works", ""])
        lines.extend(
            f"- {entity_name} {claim.predicate} {self._value(claim)}. `[claim:{claim.claim_id}]`"
            for claim in claims
        )
        lines.extend(["", "## Constraints / Exceptions", ""])
        constraints = [claim for claim in claims if claim.qualifiers.get("constraint")]
        lines.extend(
            f"- {claim.qualifiers['constraint']} `[claim:{claim.claim_id}]`" for claim in constraints
        )
        if not constraints:
            lines.append("- No sourced constraints are currently recorded.")
        lines.extend(["", "## Change History", ""])
        for claim in history:
            lines.append(
                f"- {claim.predicate}: {self._value(claim)} "
                f"({claim.valid_from or '?'} → {claim.valid_to or '?'}) `[claim:{claim.claim_id}]`"
            )
        if not history:
            lines.append("- No superseded claim versions.")
        lines.extend(["", "## Open Questions", "", "- None recorded as sourced claims.", "", "## Sources", ""])
        seen: set[tuple[str, int, int]] = set()
        for claim in claims:
            for span in claim.source_spans:
                key = (str(span["source_id"]), int(span["char_start"]), int(span["char_end"]))
                if key in seen:
                    continue
                seen.add(key)
                source = self.sources.get(key[0])
                lines.append(
                    f"- {source.uri if source else key[0]} · chars {key[1]}–{key[2]} "
                    f"— “{span['quote']}”"
                )
        lines.extend(["", "## Backlinks / Related Entities", "", "- Generated from the graph projection.", ""])
        return "\n".join(lines)

    def build(self, tenant_id: str, snapshot_id: str) -> list[WikiPage]:
        active = self.claims.active_claims(tenant_id=tenant_id)
        grouped: dict[str, list[ClaimVersion]] = {}
        for claim in active:
            grouped.setdefault(claim.subject_entity_id, []).append(claim)
        current_all = self.claims._latest_rows(tenant_id)
        pages: list[WikiPage] = []
        with self.db.lock:
            for subject_id, subject_claims in grouped.items():
                entity = self.entities.get(subject_id)
                if entity is None:
                    raise ValueError(f"missing entity for claim subject: {subject_id}")
                history = [
                    claim for claim in current_all
                    if claim.subject_entity_id == subject_id and claim.status == "superseded"
                ]
                slug = _slug(entity.canonical_name)
                page_id = hashlib.sha256(f"{tenant_id}:{subject_id}".encode("utf-8")).hexdigest()[:24]
                row = self.db.conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS version FROM wiki_pages WHERE page_id=?",
                    (page_id,),
                ).fetchone()
                qualifiers = subject_claims[0].qualifiers
                spans = tuple(span for claim in subject_claims for span in claim.source_spans)
                page = WikiPage(
                    page_id=page_id, version=int(row["version"]) + 1,
                    tenant_id=tenant_id, slug=slug, title=entity.canonical_name,
                    domain=str(qualifiers.get("domain", "Knowledge")),
                    topic=str(qualifiers.get("topic", entity.canonical_name)),
                    snapshot_id=snapshot_id, status="generated",
                    markdown=self._render(entity.canonical_name, subject_claims, history),
                    claim_ids=tuple(sorted(claim.claim_id for claim in subject_claims)),
                    source_spans=spans, last_verified_at=_now(),
                )
                self._insert(page)
                pages.append(page)
            self.db.conn.commit()
        return pages

    def _insert(self, page: WikiPage) -> None:
        self.db.conn.execute(
            "INSERT INTO wiki_pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                page.page_id, page.version, page.tenant_id, page.slug, page.title,
                page.domain, page.topic, page.snapshot_id, page.status, page.markdown,
                json.dumps(page.claim_ids), json.dumps(page.source_spans, ensure_ascii=False),
                page.last_verified_at, page.stale_reason,
            ),
        )

    def approve(self, page_id: str, *, reviewer: str) -> WikiPage:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM wiki_pages WHERE page_id=? ORDER BY version DESC LIMIT 1", (page_id,)
            ).fetchone()
            if row is None:
                raise KeyError(page_id)
            current = self._page(row)
            approved = replace(
                current, version=current.version + 1, status="approved",
                last_verified_at=_now(), stale_reason=None,
            )
            self._insert(approved)
            self.db.conn.commit()
            return approved

    def list(
        self, tenant_id: str, snapshot_id: str,
        acl: tuple[str, ...] = ("public",),
    ) -> list[WikiPage]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM wiki_pages WHERE tenant_id=? AND snapshot_id=? ORDER BY slug, version",
                (tenant_id, snapshot_id),
            ).fetchall()
        latest: dict[str, WikiPage] = {}
        for row in rows:
            page = self._page(row)
            latest[page.page_id] = page
        allowed = set(acl) | {"public"}
        visible: list[WikiPage] = []
        for page in latest.values():
            sources = [self.sources.get(str(span.get("source_id"))) for span in page.source_spans]
            if page.status == "stale" or not sources:
                continue
            if all(source is not None and set(source.acl) & allowed for source in sources):
                visible.append(page)
        return visible

    def get(
        self, tenant_id: str, snapshot_id: str, slug: str,
        acl: tuple[str, ...] = ("public",),
    ) -> WikiPage | None:
        pages = [page for page in self.list(tenant_id, snapshot_id, acl) if page.slug == slug]
        return pages[0] if pages else None

    def validate(self, tenant_id: str, snapshot_id: str) -> dict[str, Any]:
        # Validation is an internal full-tenant operation, not a user view.
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM wiki_pages WHERE tenant_id=? AND snapshot_id=? ORDER BY slug, version",
                (tenant_id, snapshot_id),
            ).fetchall()
        latest = {}
        for row in rows:
            page = self._page(row)
            latest[page.page_id] = page
        pages = [page for page in latest.values() if page.status != "stale"]
        active = self.claims.active_claims(tenant_id=tenant_id)
        active_ids = {claim.claim_id for claim in active}
        page_ids = {claim_id for page in pages for claim_id in page.claim_ids}
        spans = [span for page in pages for span in page.source_spans]
        return {
            "pages": len(pages),
            "claim_coverage": len(active_ids & page_ids) / len(active_ids) if active_ids else 1.0,
            "source_span_validity": (
                sum(self.sources.validate_span(span, tenant_id=tenant_id) for span in spans) / len(spans)
                if spans else (1.0 if not active_ids else 0.0)
            ),
        }
