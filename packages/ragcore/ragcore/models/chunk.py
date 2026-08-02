"""The Qdrant payload for a document chunk.

:class:`ChunkPayload` is deliberately flat: Qdrant cannot build a payload index on a
nested object, and every ACL field must be indexable for tenant-scoped filtering to
stay fast. Use :meth:`ChunkPayload.from_access_control` to build one from an
:class:`~ragcore.models.acl.AccessControl` — no other module should flatten an ACL
by hand, or the flat fields and the nested source will drift.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragcore.models.acl import AccessControl, Classification

__all__ = ["ChunkPayload", "SourceType"]


class SourceType(StrEnum):
    """Where a document came from.

    Mirrors the ``source_type`` payload field and the ingestion connector set.
    """

    BLOB = "blob"
    SHAREPOINT = "sharepoint"
    HTTP = "http"
    SQL = "sql"
    UPLOAD = "upload"
    LOCAL = "local"


#: Payload keys holding RFC 3339 timestamps. Qdrant indexes these as datetimes;
#: they are serialised as ISO 8601 strings and parsed back on read.
_DATETIME_FIELDS: tuple[str, ...] = (
    "source_modified_at",
    "effective_from",
    "effective_to",
    "created_at",
    "updated_at",
)


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class ChunkPayload(BaseModel):
    """Flat Qdrant payload for one chunk of one document.

    Every field here is either filterable, needed to render a citation, or needed to
    resolve contradictions by recency and authority. Text lives in ``text``; the
    ``contextual_header`` is prepended at embed time but stored separately so the
    rendered prompt and the citation span both stay verbatim.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # ---- identity
    chunk_id: str = Field(description="Stable chunk id; also the Qdrant point id.")
    document_id: str = Field(description="Owning document id.")
    chunk_index: int = Field(
        ge=0, description="Zero-based position within the document."
    )

    # ---- access control (flattened; see from_access_control)
    tenant_id: str = Field(description="Owning tenant id. Indexed with is_tenant=True.")
    allowed_roles: list[str] = Field(
        default_factory=list,
        description="Permitted app roles; empty means unrestricted.",
    )
    allowed_groups: list[str] = Field(
        default_factory=list, description="Permitted Entra group object IDs."
    )
    allowed_users: list[str] = Field(
        default_factory=list, description="Permitted Entra user object IDs."
    )
    denied_users: list[str] = Field(
        default_factory=list, description="Explicitly denied users; deny always wins."
    )
    classification: str = Field(
        default=Classification.INTERNAL.value,
        description="Classification value, e.g. 'internal'.",
    )
    classification_rank: int = Field(
        default=Classification.INTERNAL.rank,
        ge=0,
        description="Denormalised classification rank for integer range filtering.",
    )

    # ---- provenance
    source_type: str = Field(
        description="blob | sharepoint | http | sql | upload | local."
    )
    source_id: str = Field(description="Source config id that ingested this chunk.")
    source_uri: str = Field(description="Canonical URI of the source document.")

    # ---- citation surface
    title: str = Field(default="", description="Document title shown in citations.")
    section_path: list[str] = Field(
        default_factory=list,
        description="Heading breadcrumb from document root to this chunk.",
    )
    page: int | None = Field(
        default=None, description="1-based page number for paginated sources."
    )

    # ---- content
    text: str = Field(
        description="Verbatim chunk text. Citation spans must match this."
    )
    contextual_header: str = Field(
        default="",
        description=(
            "Title/section context prepended at embed time and stored separately so "
            "it never pollutes a quoted citation span."
        ),
    )
    summary: str | None = Field(
        default=None, description="Optional MODEL_FAST chunk summary."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Extracted keywords for lexical recall."
    )

    # ---- metadata facets
    doc_type: str = Field(
        default="document",
        description="Document class used for filtering and contradiction authority.",
    )
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    author: str | None = Field(default=None, description="Document author, when known.")
    language: str = Field(default="en", description="ISO 639-1 language code.")

    # ---- dedupe
    content_sha256: str = Field(
        default="", description="Hex SHA-256 of the normalised chunk text."
    )
    simhash: str = Field(
        default="",
        description="16 hex characters: unsigned 64-bit simhash of the chunk text.",
    )
    token_count: int = Field(
        default=0, ge=0, description="Token count of text plus contextual header."
    )

    # ---- recency / contradiction resolution
    source_modified_at: datetime | None = Field(
        default=None, description="Last modification time reported by the source."
    )
    effective_from: datetime | None = Field(
        default=None, description="Start of the period this content is valid for."
    )
    effective_to: datetime | None = Field(
        default=None, description="End of the period this content is valid for."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="When this chunk was first indexed."
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="When this chunk was last rewritten."
    )

    # ---- lifecycle
    version: int = Field(default=1, ge=1, description="Monotonic chunk version.")
    is_deleted: bool = Field(
        default=False,
        description="Soft-delete marker. Retrieval filters this out by default.",
    )

    # ---- pii
    pii_types: list[str] = Field(
        default_factory=list, description="PII entity types detected in this chunk."
    )
    pii_redacted: bool = Field(
        default=False, description="Whether text has been through PII redaction."
    )

    # ---- lineage
    ingest_run_id: str = Field(
        default="", description="Ingestion run that produced this chunk version."
    )

    @field_validator("simhash")
    @classmethod
    def _validate_simhash(cls, value: str) -> str:
        """Require an empty string or exactly 16 lowercase hex characters.

        Args:
            value: Candidate simhash.

        Returns:
            The normalised lowercase value.

        Raises:
            ValueError: If the value is neither empty nor 16 hex characters.
        """
        if value == "":
            return value
        normalised = value.lower()
        if len(normalised) != 16:
            msg = f"simhash must be 16 hex characters, got {len(normalised)}"
            raise ValueError(msg)
        try:
            int(normalised, 16)
        except ValueError as exc:
            msg = f"simhash must be hexadecimal, got {value!r}"
            raise ValueError(msg) from exc
        return normalised

    @field_validator("section_path", "tags", "keywords", "pii_types")
    @classmethod
    def _clean_list(cls, value: list[str]) -> list[str]:
        """Drop blank entries from list facets.

        Args:
            value: Raw list.

        Returns:
            The list without empty or whitespace-only entries.
        """
        return [item for item in (v.strip() for v in value) if item]

    @model_validator(mode="after")
    def _sync_classification(self) -> Self:
        """Keep ``classification`` and ``classification_rank`` consistent.

        The rank is denormalised purely so Qdrant can range-filter it; it must never
        disagree with the label.

        Returns:
            The validated payload.

        Raises:
            ValueError: If ``classification`` is not a known label.
        """
        try:
            level = Classification(self.classification)
        except ValueError as exc:
            msg = f"unknown classification {self.classification!r}"
            raise ValueError(msg) from exc
        if self.classification_rank != level.rank:
            object.__setattr__(self, "classification_rank", level.rank)
        return self

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_access_control(
        cls,
        ac: AccessControl,
        **fields: Any,
    ) -> Self:
        """Build a payload from an :class:`AccessControl` plus the remaining fields.

        This is the single supported way to populate the flat ACL fields. Any
        tenant/ACL keys passed in ``fields`` are ignored in favour of ``ac`` so a
        caller cannot accidentally desynchronise the two.

        Args:
            ac: Access control for the owning document.
            **fields: All other :class:`ChunkPayload` fields (``chunk_id``,
                ``document_id``, ``chunk_index``, ``source_type``, ``source_id``,
                ``source_uri``, ``text`` and friends).

        Returns:
            A fully populated payload whose ACL fields match ``ac`` exactly.

        Example:
            >>> from ragcore.models.acl import AccessControl, Classification
            >>> ac = AccessControl(
            ...     tenant_id="t1",
            ...     allowed_groups=["g-eng"],
            ...     classification=Classification.CONFIDENTIAL,
            ... )
            >>> payload = ChunkPayload.from_access_control(
            ...     ac,
            ...     chunk_id="c1",
            ...     document_id="d1",
            ...     chunk_index=0,
            ...     source_type="blob",
            ...     source_id="s1",
            ...     source_uri="https://example/doc.pdf",
            ...     text="hello",
            ... )
            >>> payload.classification_rank == Classification.CONFIDENTIAL.rank
            True
        """
        merged = dict(fields)
        merged.update(ac.to_flat())
        return cls.model_validate(merged)

    def access_control(self) -> AccessControl:
        """Rebuild the nested ACL from the flat fields.

        Returns:
            The :class:`AccessControl` this payload encodes.
        """
        return AccessControl.from_flat(self.model_dump(mode="python"))

    def with_access_control(self, ac: AccessControl) -> Self:
        """Return a copy whose ACL fields come from ``ac``.

        Used by the ACL-only reindex path, which rewrites payloads without
        re-embedding text.

        Args:
            ac: The new access control.

        Returns:
            A new payload with the flat ACL fields replaced and ``updated_at``
            refreshed.
        """
        return self.model_copy(update={**ac.to_flat(), "updated_at": _utcnow()})

    # ------------------------------------------------------------- serialisation
    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialise to the dict stored as a Qdrant point payload.

        Datetimes become RFC 3339 strings so Qdrant's datetime payload index can
        range-filter them; None values are preserved so filters on missing fields
        behave predictably.

        Returns:
            A JSON-safe mapping ready for ``PointStruct(payload=...)``.
        """
        data = self.model_dump(mode="python")
        for key in _DATETIME_FIELDS:
            value = data.get(key)
            if isinstance(value, datetime):
                moment = value if value.tzinfo else value.replace(tzinfo=UTC)
                data[key] = moment.astimezone(UTC).isoformat()
        return data

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any] | None) -> Self:
        """Parse a Qdrant point payload back into a :class:`ChunkPayload`.

        Args:
            payload: The raw payload from a ``ScoredPoint``/``Record``.

        Returns:
            The parsed payload model.

        Raises:
            ValueError: If ``payload`` is None or empty — a point without a payload
                cannot be ACL-checked, so it must never be silently accepted.
        """
        if not payload:
            msg = "cannot build ChunkPayload from an empty Qdrant payload"
            raise ValueError(msg)
        known = set(cls.model_fields)
        cleaned = {key: value for key, value in payload.items() if key in known}
        return cls.model_validate(cleaned)

    # ------------------------------------------------------------------ helpers
    @property
    def embed_text(self) -> str:
        """Text actually sent to the embedder.

        Returns:
            The contextual header followed by the chunk text, or the bare text when
            no header was generated.
        """
        if not self.contextual_header:
            return self.text
        return f"{self.contextual_header}\n\n{self.text}"

    @property
    def section_label(self) -> str:
        """Human-readable section breadcrumb.

        Returns:
            The section path joined with ' > ', or an empty string.
        """
        return " > ".join(self.section_path)

    @property
    def recency_at(self) -> datetime:
        """Best available timestamp for recency comparisons.

        Contradiction resolution prefers ``effective_from``, falling back to
        ``source_modified_at`` and finally to ``created_at``.

        Returns:
            A timezone-aware datetime.
        """
        for candidate in (
            self.effective_from,
            self.source_modified_at,
            self.created_at,
        ):
            if candidate is not None:
                return candidate if candidate.tzinfo else candidate.replace(tzinfo=UTC)
        return _utcnow()

    def is_effective_at(self, moment: datetime | None = None) -> bool:
        """Whether this chunk's content is valid at a given moment.

        Args:
            moment: Point in time to test. Defaults to now (UTC).

        Returns:
            True when ``moment`` falls within ``[effective_from, effective_to)``.
            An unset bound is treated as open-ended.
        """
        now = moment or _utcnow()
        now = now if now.tzinfo else now.replace(tzinfo=UTC)
        if self.effective_from is not None:
            start = self.effective_from
            start = start if start.tzinfo else start.replace(tzinfo=UTC)
            if now < start:
                return False
        if self.effective_to is not None:
            end = self.effective_to
            end = end if end.tzinfo else end.replace(tzinfo=UTC)
            if now >= end:
                return False
        return True

    @property
    def classification_level(self) -> Classification:
        """Typed view of :attr:`classification`.

        Returns:
            The classification enum member.
        """
        return Classification(self.classification)
