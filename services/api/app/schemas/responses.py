"""Response bodies for the HTTP surface.

Same rule as :mod:`app.schemas.requests`: a shape the contract already models is
re-used, and only what Addendum W leaves open is declared here. Every model sets
``extra="ignore"`` on construction from an ORM row via an explicit ``from_row``
classmethod rather than ``from_attributes``, so adding a column to a table cannot
accidentally start leaking it over HTTP.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.chat import ContextStats, GuardrailEvent, Message
from ragcore.models.memory import UserProfile
from ragcore.models.retrieval import RetrievalResult

__all__ = [
    "ChatResponse",
    "CompactionResponse",
    "DocumentSummary",
    "HealthResponse",
    "LineageResponse",
    "ProblemDetail",
    "ReadinessResponse",
    "ScheduleResponse",
    "SessionSummary",
    "SourceSummary",
    "TenantSummary",
    "UsagePayload",
    "UserProfile",
]


class ProblemDetail(BaseModel):
    """RFC 7807 ``application/problem+json`` body.

    Every error the API returns has this shape, so a client has exactly one error
    parser. ``code`` is :attr:`ragcore.errors.RagError.code`; ``detail`` carries
    the structured context the error attached, which is contractually free of
    unredacted user content and secrets.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="Stable URI identifying the problem type.")
    title: str = Field(description="Short, human-readable summary of the type.")
    status: int = Field(description="HTTP status code.")
    detail: str = Field(default="", description="Explanation of this occurrence.")
    instance: str | None = Field(
        default=None, description="Path the problem occurred at."
    )
    code: str = Field(default="rag_error", description="Machine-readable error code.")
    request_id: str | None = Field(
        default=None, description="Correlation id echoed from the request."
    )
    trace_id: str | None = Field(
        default=None, description="Langfuse trace id, when a trace was open."
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Field-level validation problems, for a 422.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Structured context from the error."
    )


class UsagePayload(BaseModel):
    """Token accounting and cost for one turn — the ``usage`` SSE event."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model that actually served the turn.")
    input_tokens: int = Field(default=0, ge=0, description="Uncached input tokens.")
    output_tokens: int = Field(default=0, ge=0, description="Generated tokens.")
    cache_read_tokens: int = Field(default=0, ge=0, description="Cache-read tokens.")
    cache_write_tokens: int = Field(default=0, ge=0, description="Cache-write tokens.")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Cost in USD.")


class ChatResponse(BaseModel):
    """Body of ``POST /chat`` when ``stream=false``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session the turn belongs to.")
    message: Message = Field(description="The persisted assistant turn.")
    retrieval: RetrievalResult | None = Field(
        default=None, description="Stage 5's audited result, text stripped."
    )
    context_stats: ContextStats | None = Field(
        default=None, description="What stage 9 packed and suppressed."
    )
    guardrails: list[GuardrailEvent] = Field(
        default_factory=list, description="Every guardrail decision, in order."
    )
    usage: UsagePayload | None = Field(
        default=None, description="Token accounting and cost."
    )
    trace_id: str | None = Field(default=None, description="Langfuse trace id.")


class SessionSummary(BaseModel):
    """One row of ``GET /sessions``, mirroring ``chat_sessions``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tenant_id: str
    user_id: str
    title: str = ""
    is_archived: bool = False
    rolling_summary: str = ""
    summary_tokens: int = 0
    compaction_events: int = 0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    last_message_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Any) -> SessionSummary:
        """Project a ``chat_sessions`` row.

        Args:
            row: A :class:`ragcore.db.models.ChatSession`.

        Returns:
            The summary.
        """
        return cls(
            session_id=row.session_id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            title=row.title or "",
            is_archived=bool(row.is_archived),
            rolling_summary=row.rolling_summary or "",
            summary_tokens=int(row.summary_tokens or 0),
            compaction_events=int(row.compaction_events or 0),
            message_count=int(row.message_count or 0),
            total_input_tokens=int(row.total_input_tokens or 0),
            total_output_tokens=int(row.total_output_tokens or 0),
            total_cache_read_tokens=int(row.total_cache_read_tokens or 0),
            total_cost_usd=float(row.total_cost_usd or 0.0),
            last_message_at=row.last_message_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CompactionResponse(BaseModel):
    """Body of ``POST /sessions/{id}/compact``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages_suppressed: int = 0
    compaction_events: int = 0
    summary_tokens: int = 0
    context_stats: ContextStats | None = None


class DocumentSummary(BaseModel):
    """One row of ``GET /documents``, mirroring ``documents``."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    tenant_id: str
    source_id: str | None = None
    source_type: str
    source_uri: str
    title: str = ""
    doc_type: str = "document"
    language: str = "en"
    author: str | None = None
    classification: str = "internal"
    classification_rank: int = 1
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    chunk_count: int = 0
    token_count: int = 0
    page_count: int | None = None
    size_bytes: int = 0
    version: int = 1
    content_sha256: str = ""
    is_deleted: bool = False
    pii_redacted: bool = False
    pii_types: list[str] = Field(default_factory=list)
    source_modified_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    blob_url: str | None = None
    ingest_run_id: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> DocumentSummary:
        """Project a ``documents`` row.

        ACL lists are deliberately **not** projected: a caller who can see a
        document does not thereby get to enumerate which groups can.

        Args:
            row: A :class:`ragcore.db.models.Document`.

        Returns:
            The summary.
        """
        return cls(
            document_id=row.document_id,
            tenant_id=row.tenant_id,
            source_id=row.source_id,
            source_type=row.source_type,
            source_uri=row.source_uri,
            title=row.title or "",
            doc_type=row.doc_type,
            language=row.language,
            author=row.author,
            classification=row.classification,
            classification_rank=int(row.classification_rank or 0),
            tags=list(row.tags or []),
            keywords=list(row.keywords or []),
            summary=row.summary,
            chunk_count=int(row.chunk_count or 0),
            token_count=int(row.token_count or 0),
            page_count=row.page_count,
            size_bytes=int(row.size_bytes or 0),
            version=int(row.version or 1),
            content_sha256=row.content_sha256 or "",
            is_deleted=bool(row.is_deleted),
            pii_redacted=bool(row.pii_redacted),
            pii_types=list(row.pii_types or []),
            source_modified_at=row.source_modified_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            blob_url=row.blob_url,
            ingest_run_id=row.ingest_run_id,
        )


class LineageResponse(BaseModel):
    """Body of ``GET /documents/{id}/lineage``.

    The payload is whatever
    :func:`ragcore.observability.lineage.document_provenance` returned, which is
    already JSON-serialisable and tenant-scoped; this model exists so the endpoint
    has a named schema in the OpenAPI document.
    """

    model_config = ConfigDict(extra="allow")

    document_id: str
    tenant_id: str
    found: bool = False


class SourceSummary(BaseModel):
    """One row of ``GET /admin/sources``, mirroring ``source_configs``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    tenant_id: str
    source_type: str
    name: str
    enabled: bool = True
    doc_type: str = "document"
    tags: list[str] = Field(default_factory=list)
    language: str = "en"
    default_classification: str = "internal"
    cron_override: str | None = None
    timezone_override: str | None = None
    cursor_updated_at: datetime | None = None
    last_run_id: str | None = None
    last_run_at: datetime | None = None
    last_status: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Any) -> SourceSummary:
        """Project a ``source_configs`` row.

        ``options`` and ``cursor`` are omitted: options can hold a secret
        reference and a cursor can hold a deltaLink with an embedded token.

        Args:
            row: A :class:`ragcore.db.models.SourceConfigRow`.

        Returns:
            The summary.
        """
        return cls(
            source_id=row.source_id,
            tenant_id=row.tenant_id,
            source_type=row.source_type,
            name=row.name,
            enabled=bool(row.enabled),
            doc_type=row.doc_type,
            tags=list(row.tags or []),
            language=row.language,
            default_classification=row.default_classification,
            cron_override=row.cron_override,
            timezone_override=row.timezone_override,
            cursor_updated_at=row.cursor_updated_at,
            last_run_id=row.last_run_id,
            last_run_at=row.last_run_at,
            last_status=row.last_status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class TenantSummary(BaseModel):
    """One row of ``GET /admin/tenants``."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    name: str
    entra_tenant_id: str | None = None
    default_classification: str = "internal"
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Any) -> TenantSummary:
        """Project a ``tenants`` row.

        Args:
            row: A :class:`ragcore.db.models.Tenant`.

        Returns:
            The summary.
        """
        return cls(
            tenant_id=row.tenant_id,
            name=row.name,
            entra_tenant_id=row.entra_tenant_id,
            default_classification=row.default_classification,
            is_active=bool(row.is_active),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ScheduleResponse(BaseModel):
    """Body of ``GET /admin/schedule``."""

    model_config = ConfigDict(extra="forbid")

    ingest_cron: str
    ingest_timezone: str
    ingest_enabled: bool
    ingest_working_hours_start: int
    ingest_working_hours_end: int
    within_working_hours: bool
    may_start: bool
    reason: str
    next_run_at: datetime | None = None


class HealthResponse(BaseModel):
    """Body of ``GET /health`` — liveness only, no dependency probes."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    service: str
    version: str
    env: str


class ReadinessResponse(BaseModel):
    """Body of ``GET /readyz`` — one boolean per dependency."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: dict[str, str] = Field(default_factory=dict)
