"""Ingestion-side models: sources, fetched documents, parse trees and run manifests.

The ingestion pipeline moves a document through four shapes:

``SourceConfig`` (what to fetch, and with which ACL)
    -> ``SourceDocument`` (raw bytes/text plus source metadata and delta tokens)
    -> ``ParsedDocument`` (ordered ``ParsedBlock`` list with heading structure)
    -> ``ChunkPayload`` (in :mod:`ragcore.models.chunk`, ready for Qdrant).

``IngestManifest`` is the per-source delta state that makes nightly refreshes cheap:
it records the content hash, ETag and modification time of every document seen, so a
run can decide create/update/delete/skip without re-embedding unchanged content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragcore.models.acl import AccessControl, Classification
from ragcore.models.chunk import SourceType

__all__ = [
    "BlockKind",
    "IngestAction",
    "IngestManifest",
    "IngestManifestEntry",
    "IngestRunSummary",
    "IngestStatus",
    "IngestTrigger",
    "ParsedBlock",
    "ParsedDocument",
    "SourceConfig",
    "SourceDocument",
]


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class BlockKind(StrEnum):
    """Structural role of a parsed block."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    CAPTION = "caption"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    OTHER = "other"


class IngestAction(StrEnum):
    """What a run decided to do with one document."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SKIP = "skip"
    ACL_ONLY = "acl_only"


class IngestStatus(StrEnum):
    """Terminal state of a run or of a single item within a run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class IngestTrigger(StrEnum):
    """What caused an ingestion run to start."""

    TIMER = "timer"
    QUEUE = "queue"
    HTTP = "http"
    MANUAL = "manual"
    REINDEX = "reindex"
    UPLOAD = "upload"


class SourceConfig(BaseModel):
    """A configured ingestion source for one tenant.

    ``options`` holds connector-specific keys so a new connector never needs a schema
    change here. The recognised keys per ``source_type`` are:

    * ``blob``: ``container``, ``prefix``, ``account_url``
    * ``local``: ``root``, ``include_globs``, ``exclude_globs``
    * ``sharepoint``: ``site_id``, ``drive_id``, ``folder_path``
    * ``http``: ``start_urls``, ``sitemap_url``, ``allow_domains``, ``max_depth``
    * ``sql``: ``dsn_secret_ref``, ``query``, ``watermark_column``, ``id_column``,
      ``text_columns``, ``title_column``

    Attributes:
        cursor: Opaque delta token — a Graph ``deltaLink``, an ISO timestamp
            watermark, or a blob marker. Written back at the end of a successful run.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Stable source id, unique within the tenant.")
    tenant_id: str = Field(description="Owning tenant id.")
    source_type: SourceType = Field(description="Which connector handles this source.")
    name: str = Field(description="Human-readable source name shown in admin UI.")
    enabled: bool = Field(default=True, description="Skip this source when false.")

    options: dict[str, Any] = Field(
        default_factory=dict, description="Connector-specific configuration keys."
    )

    default_classification: Classification = Field(
        default=Classification.INTERNAL,
        description="Classification applied when the source provides none.",
    )
    default_allowed_roles: list[str] = Field(
        default_factory=list, description="Roles applied to every document by default."
    )
    default_allowed_groups: list[str] = Field(
        default_factory=list, description="Groups applied to every document by default."
    )
    default_allowed_users: list[str] = Field(
        default_factory=list, description="Users applied to every document by default."
    )
    default_denied_users: list[str] = Field(
        default_factory=list, description="Users denied on every document by default."
    )
    inherit_source_permissions: bool = Field(
        default=True,
        description=(
            "For SharePoint/OneDrive, map the item's own permissions onto the chunk "
            "ACL and merge them with the defaults above."
        ),
    )

    doc_type: str = Field(
        default="document", description="Default doc_type for this source."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags stamped on every chunk from this source.",
    )
    language: str = Field(default="en", description="Default ISO 639-1 language.")

    cron_override: str | None = Field(
        default=None,
        description="Per-source six-field NCRONTAB overriding settings.ingest_cron.",
    )
    timezone_override: str | None = Field(
        default=None, description="Per-source IANA timezone for the schedule guard."
    )

    cursor: str | None = Field(
        default=None, description="Opaque delta token from the last successful run."
    )
    cursor_updated_at: datetime | None = Field(
        default=None, description="When the cursor was last advanced."
    )
    last_run_id: str | None = Field(default=None, description="Most recent run id.")
    last_run_at: datetime | None = Field(
        default=None, description="Most recent run start."
    )
    last_status: IngestStatus | None = Field(
        default=None, description="Outcome of the most recent run."
    )

    created_at: datetime = Field(default_factory=_utcnow, description="Creation time.")
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update time."
    )

    def default_access_control(self) -> AccessControl:
        """Build the baseline ACL every document from this source inherits.

        Returns:
            An :class:`AccessControl` from the ``default_*`` fields.
        """
        return AccessControl(
            tenant_id=self.tenant_id,
            allowed_roles=list(self.default_allowed_roles),
            allowed_groups=list(self.default_allowed_groups),
            allowed_users=list(self.default_allowed_users),
            denied_users=list(self.default_denied_users),
            classification=self.default_classification,
        )

    def option(self, key: str, default: Any = None) -> Any:
        """Read a connector-specific option.

        Args:
            key: Option name.
            default: Value returned when the option is absent.

        Returns:
            The option value, or ``default``.
        """
        return self.options.get(key, default)

    def require_option(self, key: str) -> Any:
        """Read a connector-specific option that must be present.

        Args:
            key: Option name.

        Returns:
            The option value.

        Raises:
            ValueError: If the option is missing or None.
        """
        value = self.options.get(key)
        if value is None:
            msg = (
                f"source {self.source_id!r} ({self.source_type}) requires "
                f"options[{key!r}]"
            )
            raise ValueError(msg)
        return value


class SourceDocument(BaseModel):
    """A document as fetched from a connector, before parsing.

    Exactly one of ``content_bytes`` or ``content_text`` is normally populated:
    binary sources (PDF, DOCX) carry bytes, textual sources (HTML, SQL rows, Markdown)
    carry text. ``blob_url`` points at the archived raw copy when ingestion stored one.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    document_id: str = Field(
        description="Deterministic document id, derived from tenant + source_uri."
    )
    tenant_id: str = Field(description="Owning tenant id.")
    source_id: str = Field(description="Source config that produced this document.")
    source_type: SourceType = Field(description="Connector that fetched it.")
    source_uri: str = Field(description="Canonical URI, unique within the tenant.")

    title: str = Field(default="", description="Document title.")
    content_bytes: bytes | None = Field(
        default=None, description="Raw bytes for binary formats."
    )
    content_text: str | None = Field(
        default=None, description="Decoded text for textual formats."
    )
    media_type: str = Field(
        default="application/octet-stream", description="MIME type of the payload."
    )
    filename: str | None = Field(default=None, description="Original file name.")
    size_bytes: int = Field(default=0, ge=0, description="Payload size in bytes.")

    etag: str | None = Field(
        default=None,
        description="Source ETag, used for conditional GET / change checks.",
    )
    content_sha256: str = Field(
        default="", description="Hex SHA-256 of the raw payload."
    )
    source_modified_at: datetime | None = Field(
        default=None, description="Last modification time reported by the source."
    )
    effective_from: datetime | None = Field(
        default=None, description="Validity start, when the source expresses one."
    )
    effective_to: datetime | None = Field(
        default=None, description="Validity end, when the source expresses one."
    )

    access_control: AccessControl = Field(
        description="Resolved ACL for this document (source defaults merged with "
        "any per-item permissions)."
    )
    doc_type: str = Field(default="document", description="Document class.")
    tags: list[str] = Field(default_factory=list, description="Tags for this document.")
    author: str | None = Field(default=None, description="Author, when known.")
    language: str | None = Field(
        default=None, description="Language, when the source declares one."
    )

    blob_url: str | None = Field(
        default=None, description="URL of the archived raw copy in Blob Storage."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Connector-specific extra metadata."
    )
    deleted: bool = Field(
        default=False,
        description="True when the connector reports the item was removed at source.",
    )
    fetched_at: datetime = Field(
        default_factory=_utcnow, description="When the fetch completed."
    )

    @property
    def has_content(self) -> bool:
        """Whether any payload is present.

        Returns:
            True when either bytes or text were fetched.
        """
        return bool(self.content_bytes) or bool(self.content_text)

    def text_or_empty(self) -> str:
        """Best-effort text view of the payload.

        Returns:
            ``content_text`` when present, otherwise a UTF-8 decode of
            ``content_bytes`` with undecodable bytes replaced, otherwise "".
        """
        if self.content_text is not None:
            return self.content_text
        if self.content_bytes is not None:
            return self.content_bytes.decode("utf-8", errors="replace")
        return ""


class ParsedBlock(BaseModel):
    """One structural element of a parsed document, in reading order."""

    model_config = ConfigDict(extra="forbid")

    kind: BlockKind = Field(description="Structural role of this block.")
    text: str = Field(description="Block text, already normalised whitespace.")
    order: int = Field(ge=0, description="Zero-based position in reading order.")
    level: int | None = Field(
        default=None,
        ge=1,
        description="Heading depth (1 = top level). Only set for HEADING blocks.",
    )
    page: int | None = Field(
        default=None, ge=1, description="1-based page number for paginated sources."
    )
    section_path: list[str] = Field(
        default_factory=list,
        description="Heading breadcrumb in effect at this block, root first.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser extras, e.g. table shape or code language.",
    )

    @property
    def token_estimate(self) -> int:
        """Cheap pre-tokenisation size estimate for chunk packing.

        Real token counts come from ``LLMClient.count_tokens``; this is only used to
        decide block grouping before any API call.

        Returns:
            Roughly one token per four characters, at minimum 1.
        """
        return max(1, len(self.text) // 4)


class ParsedDocument(BaseModel):
    """A document after parsing, ready for chunking."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Document id, carried over from the source.")
    tenant_id: str = Field(description="Owning tenant id.")
    source_id: str = Field(description="Source config id.")
    source_type: SourceType = Field(description="Connector that fetched the document.")
    source_uri: str = Field(description="Canonical source URI.")

    title: str = Field(default="", description="Resolved document title.")
    blocks: list[ParsedBlock] = Field(
        default_factory=list, description="Parsed blocks in reading order."
    )
    doc_type: str = Field(default="document", description="Document class.")
    language: str = Field(default="en", description="Detected or declared language.")
    author: str | None = Field(default=None, description="Author, when known.")
    tags: list[str] = Field(default_factory=list, description="Document tags.")
    keywords: list[str] = Field(
        default_factory=list, description="Document-level extracted keywords."
    )
    summary: str | None = Field(
        default=None, description="Optional document-level summary."
    )

    page_count: int | None = Field(
        default=None, ge=0, description="Page count for paginated sources."
    )
    access_control: AccessControl = Field(description="Resolved ACL for this document.")
    content_sha256: str = Field(default="", description="Hash of the raw payload.")
    source_modified_at: datetime | None = Field(
        default=None, description="Source modification time."
    )
    effective_from: datetime | None = Field(default=None, description="Validity start.")
    effective_to: datetime | None = Field(default=None, description="Validity end.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Parser and connector extras."
    )
    parsed_at: datetime = Field(
        default_factory=_utcnow, description="When parsing completed."
    )

    @field_validator("blocks")
    @classmethod
    def _ordered(cls, value: list[ParsedBlock]) -> list[ParsedBlock]:
        """Sort blocks by their declared reading order.

        Args:
            value: Parsed blocks, possibly out of order.

        Returns:
            The blocks sorted ascending by ``order``.
        """
        return sorted(value, key=lambda block: block.order)

    @property
    def full_text(self) -> str:
        """Whole-document text, blocks joined by blank lines.

        Returns:
            The concatenated block text.
        """
        return "\n\n".join(block.text for block in self.blocks if block.text)

    def blocks_for_page(self, page: int) -> list[ParsedBlock]:
        """Select blocks belonging to one page.

        Args:
            page: 1-based page number.

        Returns:
            The blocks whose ``page`` matches.
        """
        return [block for block in self.blocks if block.page == page]


class IngestManifestEntry(BaseModel):
    """Per-document delta state recorded by the last successful run."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Document id.")
    source_uri: str = Field(description="Canonical source URI.")
    content_sha256: str = Field(description="Hash of the raw payload last indexed.")
    etag: str | None = Field(default=None, description="Source ETag last seen.")
    source_modified_at: datetime | None = Field(
        default=None, description="Source modification time last seen."
    )
    acl_fingerprint: str = Field(
        default="",
        description=(
            "Hash of the flattened ACL. A change here alone triggers an ACL-only "
            "reindex that rewrites payloads without re-embedding."
        ),
    )
    version: int = Field(default=1, ge=1, description="Document version last indexed.")
    chunk_count: int = Field(default=0, ge=0, description="Chunks written last time.")
    token_count: int = Field(default=0, ge=0, description="Tokens indexed last time.")
    last_run_id: str = Field(default="", description="Run that produced this entry.")
    last_seen_at: datetime = Field(
        default_factory=_utcnow, description="When the document was last observed."
    )
    is_deleted: bool = Field(
        default=False, description="Whether the document is soft-deleted."
    )

    def decide(
        self,
        *,
        content_sha256: str,
        etag: str | None = None,
        acl_fingerprint: str = "",
    ) -> IngestAction:
        """Compare against a freshly fetched document and pick an action.

        Args:
            content_sha256: Hash of the newly fetched payload.
            etag: Newly observed ETag, when the source provides one.
            acl_fingerprint: Fingerprint of the newly resolved ACL.

        Returns:
            ``UPDATE`` when the content changed, ``ACL_ONLY`` when only the ACL
            changed, otherwise ``SKIP``.
        """
        if content_sha256 and content_sha256 != self.content_sha256:
            return IngestAction.UPDATE
        if etag and self.etag and etag != self.etag:
            return IngestAction.UPDATE
        if acl_fingerprint and acl_fingerprint != self.acl_fingerprint:
            return IngestAction.ACL_ONLY
        return IngestAction.SKIP


class IngestManifest(BaseModel):
    """Delta state for one source: what was indexed, and how far we got.

    Persisted per ``(tenant_id, source_id)`` in Blob Storage so a serverless run can
    resume and so deletions can be detected by set difference.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Owning tenant id.")
    source_id: str = Field(description="Source config id.")
    entries: dict[str, IngestManifestEntry] = Field(
        default_factory=dict, description="Manifest entries keyed by document_id."
    )
    cursor: str | None = Field(
        default=None, description="Opaque delta token for the next run."
    )
    last_run_id: str | None = Field(
        default=None, description="Run that last wrote this."
    )
    last_full_scan_at: datetime | None = Field(
        default=None,
        description="Last time every document was enumerated (not a delta pass).",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Manifest write time."
    )
    version: int = Field(default=1, ge=1, description="Manifest schema version.")

    def get(self, document_id: str) -> IngestManifestEntry | None:
        """Look up an entry.

        Args:
            document_id: Document id to find.

        Returns:
            The entry, or None when this document has never been indexed.
        """
        return self.entries.get(document_id)

    def upsert(self, entry: IngestManifestEntry) -> Self:
        """Insert or replace an entry in place.

        Args:
            entry: The entry to store.

        Returns:
            This manifest, for chaining.
        """
        self.entries[entry.document_id] = entry
        self.updated_at = _utcnow()
        return self

    def missing_document_ids(self, seen: set[str]) -> list[str]:
        """Find live documents that a full scan did not observe.

        Args:
            seen: Document ids enumerated by the current run.

        Returns:
            Sorted ids of non-deleted entries absent from ``seen``; these are the
            deletions a full scan should soft-delete.
        """
        return sorted(
            doc_id
            for doc_id, entry in self.entries.items()
            if doc_id not in seen and not entry.is_deleted
        )

    def mark_deleted(self, document_id: str, run_id: str) -> None:
        """Record that a document was soft-deleted.

        Args:
            document_id: Document that disappeared at source.
            run_id: Run performing the deletion.
        """
        entry = self.entries.get(document_id)
        if entry is None:
            return
        entry.is_deleted = True
        entry.last_run_id = run_id
        entry.last_seen_at = _utcnow()
        self.updated_at = _utcnow()

    @property
    def live_count(self) -> int:
        """Number of non-deleted documents tracked.

        Returns:
            The count of live entries.
        """
        return sum(1 for entry in self.entries.values() if not entry.is_deleted)


class IngestRunSummary(BaseModel):
    """Outcome of one ingestion run, mirrored into the ``ingest_runs`` table."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique run id.")
    tenant_id: str = Field(description="Tenant this run processed.")
    source_id: str | None = Field(
        default=None, description="Source processed, or None for an all-sources run."
    )
    trigger: IngestTrigger = Field(
        default=IngestTrigger.TIMER, description="What started the run."
    )
    status: IngestStatus = Field(
        default=IngestStatus.RUNNING, description="Current or final status."
    )

    started_at: datetime = Field(default_factory=_utcnow, description="Run start time.")
    finished_at: datetime | None = Field(default=None, description="Run end time.")

    documents_seen: int = Field(default=0, ge=0, description="Documents enumerated.")
    documents_created: int = Field(
        default=0, ge=0, description="New documents indexed."
    )
    documents_updated: int = Field(default=0, ge=0, description="Documents re-indexed.")
    documents_deleted: int = Field(
        default=0, ge=0, description="Documents soft-deleted."
    )
    documents_skipped: int = Field(default=0, ge=0, description="Unchanged documents.")
    documents_failed: int = Field(
        default=0, ge=0, description="Documents that errored."
    )
    chunks_upserted: int = Field(default=0, ge=0, description="Chunk points written.")
    chunks_deleted: int = Field(default=0, ge=0, description="Chunk points removed.")
    tokens_embedded: int = Field(
        default=0, ge=0, description="Tokens sent to the embedder."
    )
    duplicates_dropped: int = Field(
        default=0, ge=0, description="Chunks dropped by exact-hash or simhash dedupe."
    )
    pii_documents: int = Field(
        default=0, ge=0, description="Documents where PII was detected and redacted."
    )

    forced: bool = Field(
        default=False,
        description="Whether ingest_force overrode the working-hours guard.",
    )
    within_working_hours: bool = Field(
        default=False, description="Whether the run started inside working hours."
    )
    skip_reason: str | None = Field(
        default=None,
        description="Why the run did not proceed: 'disabled' or 'working_hours'.",
    )
    error_message: str | None = Field(
        default=None, description="Redacted failure description, when status is FAILED."
    )
    errors: list[str] = Field(
        default_factory=list, description="Per-document redacted error messages."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Stage timings and derived rates."
    )

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock run duration.

        Returns:
            Seconds between start and finish, or None while still running.
        """
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def finish(self, status: IngestStatus | None = None) -> Self:
        """Stamp the finish time and resolve the final status.

        Args:
            status: Explicit final status. When omitted it is derived from the
                failure counters: any failure with some success is ``PARTIAL``,
                all-failed is ``FAILED``, otherwise ``SUCCEEDED``.

        Returns:
            This summary, for chaining.
        """
        self.finished_at = _utcnow()
        if status is not None:
            self.status = status
        elif self.documents_failed == 0:
            self.status = IngestStatus.SUCCEEDED
        elif self.documents_created or self.documents_updated or self.documents_skipped:
            self.status = IngestStatus.PARTIAL
        else:
            self.status = IngestStatus.FAILED
        return self
