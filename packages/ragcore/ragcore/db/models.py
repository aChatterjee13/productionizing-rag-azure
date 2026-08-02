"""SQLAlchemy 2.0 ORM models: the 17 relational tables behind the platform.

Every tenant-scoped table carries a ``tenant_id`` column, a plain index on it, and a
composite index on ``(tenant_id, <natural key>)`` — the composite is what makes a
tenant-scoped lookup an index seek rather than a scan, and it is the shape every
repository query in :mod:`ragcore.db.repositories` is written against.

Free-text columns that can contain user or document content (``chat_messages.content``,
``feedback.comment``, ``tool_invocations.arguments``) are only ever written after PII
redaction; the ``pii_redacted`` flag records that the redaction pass actually ran.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ragcore.db.base import JSON_TYPE, Base
from ragcore.models.acl import Classification

__all__ = [
    "AuditLog",
    "ChatMessage",
    "ChatSession",
    "Document",
    "EvalResultRow",
    "EvalRunRow",
    "Feedback",
    "IngestItem",
    "IngestRun",
    "LineageRecordRow",
    "SemanticCacheMeta",
    "SourceConfigRow",
    "Tenant",
    "ToolInvocation",
    "User",
    "UserMemory",
    "UserProfileRow",
]

_ID = String(64)
_SHORT = String(255)
_URI = String(2048)


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class _TimestampMixin:
    """Adds server-defaulted ``created_at`` and ``updated_at`` columns."""

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
        nullable=False,
    )


# ---------------------------------------------------------------- 1. tenants
class Tenant(_TimestampMixin, Base):
    """A customer tenant. The root of every access-control decision."""

    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    name: Mapped[str] = mapped_column(_SHORT, nullable=False)
    entra_tenant_id: Mapped[str | None] = mapped_column(_ID)
    display_domain: Mapped[str | None] = mapped_column(_SHORT)
    default_classification: Mapped[str] = mapped_column(
        String(32), default=Classification.INTERNAL.value, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint("entra_tenant_id", name="uq_tenants_entra_tenant_id"),
        Index("ix_tenants_is_active", "is_active"),
    )


# ------------------------------------------------------------------- 2. users
class User(_TimestampMixin, Base):
    """A principal we have seen, cached from Entra ID token claims."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str | None] = mapped_column(_SHORT)
    display_name: Mapped[str | None] = mapped_column(_SHORT)
    roles: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    groups: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    max_classification: Mapped[str] = mapped_column(
        String(32), default=Classification.INTERNAL.value, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("ix_users_tenant_id", "tenant_id"),
        Index("ix_users_tenant_id_email", "tenant_id", "email"),
        Index("ix_users_tenant_id_last_seen_at", "tenant_id", "last_seen_at"),
    )


# ---------------------------------------------------------- 3. source_configs
class SourceConfigRow(_TimestampMixin, Base):
    """A configured ingestion source, plus its delta cursor and last-run state.

    Mirrors :class:`ragcore.models.document.SourceConfig`.
    """

    __tablename__ = "source_configs"

    source_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(_SHORT, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    options: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    default_classification: Mapped[str] = mapped_column(
        String(32), default=Classification.INTERNAL.value, nullable=False
    )
    default_allowed_roles: Mapped[list[str]] = mapped_column(
        default=list, nullable=False
    )
    default_allowed_groups: Mapped[list[str]] = mapped_column(
        default=list, nullable=False
    )
    default_allowed_users: Mapped[list[str]] = mapped_column(
        default=list, nullable=False
    )
    default_denied_users: Mapped[list[str]] = mapped_column(
        default=list, nullable=False
    )
    inherit_source_permissions: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    doc_type: Mapped[str] = mapped_column(
        String(64), default="document", nullable=False
    )
    tags: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    language: Mapped[str] = mapped_column(String(16), default="en", nullable=False)

    cron_override: Mapped[str | None] = mapped_column(String(128))
    timezone_override: Mapped[str | None] = mapped_column(String(64))

    cursor: Mapped[str | None] = mapped_column(Text)
    cursor_updated_at: Mapped[datetime | None] = mapped_column()
    last_run_id: Mapped[str | None] = mapped_column(_ID)
    last_run_at: Mapped[datetime | None] = mapped_column()
    last_status: Mapped[str | None] = mapped_column(String(32))

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_source_configs_tenant_id_name"),
        Index("ix_source_configs_tenant_id", "tenant_id"),
        Index("ix_source_configs_tenant_id_source_type", "tenant_id", "source_type"),
        Index("ix_source_configs_tenant_id_enabled", "tenant_id", "enabled"),
    )


# --------------------------------------------------------------- 4. documents
class Document(_TimestampMixin, Base):
    """A document that has been ingested, with its resolved ACL and dedupe hashes.

    The relational row is the lineage anchor and the delta-detection record; the
    searchable content lives in Qdrant as ``ChunkPayload`` points.
    """

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str | None] = mapped_column(_ID)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(_URI, nullable=False)

    title: Mapped[str] = mapped_column(_SHORT, default="", nullable=False)
    doc_type: Mapped[str] = mapped_column(
        String(64), default="document", nullable=False
    )
    language: Mapped[str] = mapped_column(String(16), default="en", nullable=False)
    author: Mapped[str | None] = mapped_column(_SHORT)

    content_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    simhash: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    etag: Mapped[str | None] = mapped_column(_SHORT)
    acl_fingerprint: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    classification: Mapped[str] = mapped_column(
        String(32), default=Classification.INTERNAL.value, nullable=False
    )
    classification_rank: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    allowed_roles: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    allowed_groups: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    allowed_users: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    denied_users: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    tags: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)

    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    source_modified_at: Mapped[datetime | None] = mapped_column()
    effective_from: Mapped[datetime | None] = mapped_column()
    effective_to: Mapped[datetime | None] = mapped_column()

    pii_types: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    pii_redacted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column()

    ingest_run_id: Mapped[str | None] = mapped_column(_ID)
    blob_url: Mapped[str | None] = mapped_column(_URI)
    extra: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "source_uri", name="uq_documents_tenant_id_source_uri"
        ),
        Index("ix_documents_tenant_id", "tenant_id"),
        Index("ix_documents_tenant_id_source_id", "tenant_id", "source_id"),
        Index("ix_documents_tenant_id_content_sha256", "tenant_id", "content_sha256"),
        Index("ix_documents_tenant_id_is_deleted", "tenant_id", "is_deleted"),
        Index("ix_documents_tenant_id_doc_type", "tenant_id", "doc_type"),
        Index(
            "ix_documents_tenant_id_source_modified_at",
            "tenant_id",
            "source_modified_at",
        ),
        Index("ix_documents_tenant_id_ingest_run_id", "tenant_id", "ingest_run_id"),
    )


# ------------------------------------------------------------- 5. ingest_runs
class IngestRun(Base):
    """One ingestion run: what it saw, what it changed, and whether it was allowed.

    ``skip_reason`` records a run that the working-hours guard refused, so the audit
    trail distinguishes "nothing changed" from "we declined to run".
    """

    __tablename__ = "ingest_runs"

    run_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str | None] = mapped_column(_ID)
    trigger: Mapped[str] = mapped_column(String(32), default="timer", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)

    started_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column()

    documents_seen: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_upserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_deleted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_embedded: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    duplicates_dropped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pii_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    forced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    within_working_hours: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    skip_reason: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    items: Mapped[list[IngestItem]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_ingest_runs_tenant_id", "tenant_id"),
        Index("ix_ingest_runs_tenant_id_started_at", "tenant_id", "started_at"),
        Index("ix_ingest_runs_tenant_id_source_id", "tenant_id", "source_id"),
        Index("ix_ingest_runs_tenant_id_status", "tenant_id", "status"),
    )


# ------------------------------------------------------------ 6. ingest_items
class IngestItem(Base):
    """One document's fate within one ingestion run."""

    __tablename__ = "ingest_items"

    item_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("ingest_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    document_id: Mapped[str | None] = mapped_column(_ID)
    source_uri: Mapped[str] = mapped_column(_URI, nullable=False)

    action: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(_SHORT)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicates_dropped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    run: Mapped[IngestRun] = relationship(back_populates="items")

    __table_args__ = (
        Index("ix_ingest_items_tenant_id", "tenant_id"),
        Index("ix_ingest_items_tenant_id_run_id", "tenant_id", "run_id"),
        Index("ix_ingest_items_tenant_id_document_id", "tenant_id", "document_id"),
        Index("ix_ingest_items_run_id_status", "run_id", "status"),
    )


# ----------------------------------------------------------- 7. chat_sessions
class ChatSession(_TimestampMixin, Base):
    """A conversation, plus the rolling summary that makes suppression lossless."""

    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(_ID, nullable=False)

    title: Mapped[str] = mapped_column(_SHORT, default="", nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    rolling_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    summary_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compaction_events: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_input_tokens: Mapped[int] = mapped_column(
        BigInteger, default=0, nullable=False
    )
    total_output_tokens: Mapped[int] = mapped_column(
        BigInteger, default=0, nullable=False
    )
    total_cache_read_tokens: Mapped[int] = mapped_column(
        BigInteger, default=0, nullable=False
    )
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_message_at: Mapped[datetime | None] = mapped_column()

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", lazy="noload"
    )

    __table_args__ = (
        Index("ix_chat_sessions_tenant_id", "tenant_id"),
        Index("ix_chat_sessions_tenant_id_user_id", "tenant_id", "user_id"),
        Index(
            "ix_chat_sessions_tenant_id_user_id_last_message_at",
            "tenant_id",
            "user_id",
            "last_message_at",
        ),
    )


# ----------------------------------------------------------- 8. chat_messages
class ChatMessage(Base):
    """A persisted turn. ``content`` is written only after PII redaction."""

    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    session_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(_ID, nullable=False)

    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, default=list, nullable=False
    )
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, default=list, nullable=False
    )
    guardrail_events: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, default=list, nullable=False
    )
    context_stats: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    suppressed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    pii_redacted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    pii_types: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    model: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(_ID)
    stop_reason: Mapped[str | None] = mapped_column(String(32))
    refused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_tenant_id", "tenant_id"),
        Index(
            "ix_chat_messages_tenant_id_session_id_created_at",
            "tenant_id",
            "session_id",
            "created_at",
        ),
        Index("ix_chat_messages_tenant_id_user_id", "tenant_id", "user_id"),
        Index("ix_chat_messages_tenant_id_trace_id", "tenant_id", "trace_id"),
    )


# -------------------------------------------------------- 9. tool_invocations
class ToolInvocation(Base):
    """Every tool call the model made, persisted for audit and lineage.

    ``arguments`` is model-authored and can echo user content, so it goes through the
    same PII redaction as message text before being written.
    """

    __tablename__ = "tool_invocations"

    tool_call_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    session_id: Mapped[str | None] = mapped_column(_ID)
    message_id: Mapped[str | None] = mapped_column(_ID)
    user_id: Mapped[str | None] = mapped_column(_ID)

    tool_name: Mapped[str] = mapped_column(_SHORT, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    result_summary: Mapped[str | None] = mapped_column(Text)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    http_status: Mapped[int | None] = mapped_column(Integer)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(_ID)
    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_tool_invocations_tenant_id", "tenant_id"),
        Index("ix_tool_invocations_tenant_id_session_id", "tenant_id", "session_id"),
        Index("ix_tool_invocations_tenant_id_message_id", "tenant_id", "message_id"),
        Index(
            "ix_tool_invocations_tenant_id_tool_name_created_at",
            "tenant_id",
            "tool_name",
            "created_at",
        ),
    )


# ---------------------------------------------------------- 10. user_profiles
class UserProfileRow(Base):
    """Rolling persona for one user in one tenant.

    Mirrors :class:`ragcore.models.memory.UserProfile`. ``memory_consent`` is the
    user-facing switch that disables stage 13 write-back entirely.
    """

    __tablename__ = "user_profiles"

    tenant_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    user_id: Mapped[str] = mapped_column(_ID, primary_key=True)

    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    preferred_style: Mapped[str | None] = mapped_column(_SHORT)
    preferred_language: Mapped[str | None] = mapped_column(String(16))
    top_topics: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    memory_consent: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    turns_since_refresh: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
        nullable=False,
    )

    __table_args__ = (Index("ix_user_profiles_tenant_id", "tenant_id"),)


# ----------------------------------------------------------- 11. user_memories
class UserMemory(Base):
    """A durable memory. The vector lives in ``rag_memories``; this row records it."""

    __tablename__ = "user_memories"

    memory_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    user_id: Mapped[str] = mapped_column(_ID, nullable=False)

    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    salience: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    source_session_id: Mapped[str | None] = mapped_column(_ID)
    supersedes: Mapped[str | None] = mapped_column(_ID)
    superseded_by: Mapped[str | None] = mapped_column(_ID)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    pii_redacted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    embedded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column()
    expires_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("ix_user_memories_tenant_id", "tenant_id"),
        Index(
            "ix_user_memories_tenant_id_user_id_kind", "tenant_id", "user_id", "kind"
        ),
        Index(
            "ix_user_memories_tenant_id_user_id_salience",
            "tenant_id",
            "user_id",
            "salience",
        ),
        Index("ix_user_memories_tenant_id_expires_at", "tenant_id", "expires_at"),
    )


# ------------------------------------------------------ 12. semantic_cache_meta
class SemanticCacheMeta(Base):
    """Relational mirror of a semantic-cache entry, for admin visibility and TTL sweeps.

    Only the retrieval plan and chunk ids are stored — never a rendered answer — so
    reuse always re-checks ACLs.
    """

    __tablename__ = "semantic_cache_meta"

    cache_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    user_id: Mapped[str | None] = mapped_column(_ID)

    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    transformed_queries: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    chunk_ids: Mapped[list[str]] = mapped_column(default=list, nullable=False)
    filter_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=86_400, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("ix_semantic_cache_meta_tenant_id", "tenant_id"),
        Index(
            "ix_semantic_cache_meta_tenant_id_filter_fingerprint",
            "tenant_id",
            "filter_fingerprint",
        ),
        Index("ix_semantic_cache_meta_tenant_id_user_id", "tenant_id", "user_id"),
        Index("ix_semantic_cache_meta_tenant_id_expires_at", "tenant_id", "expires_at"),
    )


# ------------------------------------------------------- 13. lineage_records
class LineageRecordRow(Base):
    """Requirement #9: complete traceability from answer back to source URI.

    ``parents`` is the upstream edge list, so a chunk points at its document, which
    points at its source URI and ingest run.
    """

    __tablename__ = "lineage_records"

    lineage_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    user_id: Mapped[str | None] = mapped_column(_ID)
    session_id: Mapped[str | None] = mapped_column(_ID)
    trace_id: Mapped[str | None] = mapped_column(_ID)

    subject_id: Mapped[str] = mapped_column(_ID, nullable=False)
    parents: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    operation: Mapped[str] = mapped_column(_SHORT, nullable=False)
    actor: Mapped[str] = mapped_column(_SHORT, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    outputs: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_lineage_records_tenant_id", "tenant_id"),
        Index("ix_lineage_records_tenant_id_subject_id", "tenant_id", "subject_id"),
        Index(
            "ix_lineage_records_tenant_id_kind_created_at",
            "tenant_id",
            "kind",
            "created_at",
        ),
        Index("ix_lineage_records_tenant_id_trace_id", "tenant_id", "trace_id"),
    )


# ------------------------------------------------------------- 14. eval_runs
class EvalRunRow(Base):
    """One evaluation run over the golden set, and whether it cleared the CI gate."""

    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)

    started_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column()

    git_sha: Mapped[str | None] = mapped_column(String(64))
    config_fingerprint: Mapped[str] = mapped_column(
        String(64), default="", nullable=False
    )
    golden_set_path: Mapped[str | None] = mapped_column(_URI)

    aggregate: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    gate_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    results: Mapped[list[EvalResultRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_eval_runs_tenant_id", "tenant_id"),
        Index("ix_eval_runs_tenant_id_started_at", "tenant_id", "started_at"),
        Index("ix_eval_runs_tenant_id_gate_passed", "tenant_id", "gate_passed"),
    )


# ---------------------------------------------------------- 15. eval_results
class EvalResultRow(Base):
    """Per-item evaluation outcome, with the scores that fed the gate."""

    __tablename__ = "eval_results"

    result_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        _ID, ForeignKey("eval_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)

    item_id: Mapped[str] = mapped_column(_ID, nullable=False)
    category: Mapped[str | None] = mapped_column(String(32))
    question: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    scores: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failures: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(_ID)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    run: Mapped[EvalRunRow] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("run_id", "item_id", name="uq_eval_results_run_id_item_id"),
        Index("ix_eval_results_tenant_id", "tenant_id"),
        Index("ix_eval_results_tenant_id_run_id", "tenant_id", "run_id"),
        Index("ix_eval_results_tenant_id_item_id", "tenant_id", "item_id"),
        Index("ix_eval_results_run_id_passed", "run_id", "passed"),
    )


# -------------------------------------------------------------- 16. feedback
class Feedback(Base):
    """Thumbs plus optional comment, mirrored into Langfuse as a score."""

    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    user_id: Mapped[str] = mapped_column(_ID, nullable=False)
    session_id: Mapped[str | None] = mapped_column(_ID)
    message_id: Mapped[str | None] = mapped_column(_ID)

    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(default=list, nullable=False)

    trace_id: Mapped[str | None] = mapped_column(_ID)
    langfuse_score_id: Mapped[str | None] = mapped_column(_ID)
    pii_redacted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_feedback_tenant_id", "tenant_id"),
        Index("ix_feedback_tenant_id_message_id", "tenant_id", "message_id"),
        Index("ix_feedback_tenant_id_session_id", "tenant_id", "session_id"),
        Index("ix_feedback_tenant_id_created_at", "tenant_id", "created_at"),
    )


# ------------------------------------------------------------- 17. audit_log
class AuditLog(Base):
    """Security-relevant events: auth decisions, ACL denials, admin actions.

    Written on every :class:`~ragcore.errors.TenantMismatchError` and on every admin
    mutation. ``detail`` is redacted before it is written.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[str] = mapped_column(_ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(_ID, nullable=False)
    user_id: Mapped[str | None] = mapped_column(_ID)

    action: Mapped[str] = mapped_column(_SHORT, nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(_ID)
    outcome: Mapped[str] = mapped_column(String(32), default="allow", nullable=False)

    detail: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(_SHORT)
    trace_id: Mapped[str | None] = mapped_column(_ID)

    created_at: Mapped[datetime] = mapped_column(
        default=_utcnow, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_log_tenant_id", "tenant_id"),
        Index("ix_audit_log_tenant_id_created_at", "tenant_id", "created_at"),
        Index("ix_audit_log_tenant_id_action", "tenant_id", "action"),
        Index("ix_audit_log_tenant_id_user_id", "tenant_id", "user_id"),
        Index("ix_audit_log_tenant_id_outcome", "tenant_id", "outcome"),
    )
