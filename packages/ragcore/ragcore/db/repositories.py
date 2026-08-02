"""Async repository helpers shared by the API and the ingestion service.

Every function takes ``tenant_id`` explicitly and filters on it. That is not a
convenience: multi-tenancy is a security boundary, so a query that could reach
another tenant's row must be impossible to write by accident here. Functions that
accept an id *and* a tenant use both in the WHERE clause, and raise
:class:`~ragcore.errors.TenantMismatchError` when a row exists but belongs elsewhere,
so a cross-tenant probe is auditable rather than silently returning None.

Callers own the transaction. Each function flushes so generated defaults are visible,
but commits are left to the caller (a request handler commits once at the end; an
ingestion activity commits per batch).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.models import (
    AuditLog,
    ChatMessage,
    ChatSession,
    Document,
    EvalResultRow,
    EvalRunRow,
    Feedback,
    IngestItem,
    IngestRun,
    LineageRecordRow,
    SemanticCacheMeta,
    SourceConfigRow,
    Tenant,
    ToolInvocation,
    User,
    UserMemory,
    UserProfileRow,
)
from ragcore.errors import TenantMismatchError
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chat import Message, Role
from ragcore.models.document import IngestRunSummary
from ragcore.models.memory import LongTermMemory, UserProfile

__all__ = [
    "append_message",
    "count_session_messages",
    "delete_memory",
    "delete_session",
    "get_or_create_profile",
    "get_or_create_session",
    "get_session_row",
    "list_memories",
    "list_session_messages",
    "list_sessions",
    "list_source_configs",
    "mark_documents_deleted",
    "new_id",
    "record_ingest_item",
    "record_ingest_run",
    "record_lineage",
    "set_memory_consent",
    "suppress_messages",
    "upsert_document",
    "upsert_memory",
    "upsert_tenant",
    "upsert_user",
    "write_audit",
    "write_eval_result",
    "write_eval_run",
    "write_feedback",
    "write_tool_invocation",
]


def new_id() -> str:
    """Generate an opaque 32-character identifier.

    Returns:
        A hex UUID4 with dashes stripped, suitable for every ``*_id`` column and for
        Qdrant point ids.
    """
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def _guard_tenant(
    row_tenant_id: str, tenant_id: str, *, resource: str, resource_id: str
) -> None:
    """Raise when a fetched row belongs to a different tenant.

    Args:
        row_tenant_id: Tenant id on the fetched row.
        tenant_id: Tenant id the caller is scoped to.
        resource: Table or resource name, for the audit record.
        resource_id: Primary key of the row, for the audit record.

    Raises:
        TenantMismatchError: If the two tenant ids differ.
    """
    if row_tenant_id != tenant_id:
        raise TenantMismatchError(
            f"{resource} {resource_id} does not belong to tenant {tenant_id}",
            expected=tenant_id,
            actual=row_tenant_id,
            resource=f"{resource}:{resource_id}",
        )


# ------------------------------------------------------------------- tenants
async def upsert_tenant(
    session: AsyncSession,
    *,
    tenant_id: str,
    name: str,
    entra_tenant_id: str | None = None,
    default_classification: Classification = Classification.INTERNAL,
    settings_blob: dict[str, Any] | None = None,
) -> Tenant:
    """Create or update a tenant.

    Args:
        session: Active database session.
        tenant_id: Tenant id (primary key).
        name: Human-readable tenant name.
        entra_tenant_id: Entra directory GUID this tenant maps to.
        default_classification: Default classification for new sources.
        settings_blob: Per-tenant overrides.

    Returns:
        The persisted :class:`Tenant`.
    """
    row = await session.get(Tenant, tenant_id)
    if row is None:
        row = Tenant(tenant_id=tenant_id, name=name)
        session.add(row)
    row.name = name
    if entra_tenant_id is not None:
        row.entra_tenant_id = entra_tenant_id
    row.default_classification = default_classification.value
    if settings_blob is not None:
        row.settings = settings_blob
    await session.flush()
    return row


async def upsert_user(session: AsyncSession, principal: Principal) -> User:
    """Cache a principal's identity after successful token validation.

    Args:
        session: Active database session.
        principal: The authenticated caller, resolved from token claims.

    Returns:
        The persisted :class:`User`.

    Raises:
        TenantMismatchError: If a user row with this id already exists under a
            different tenant — a user object id must never move between tenants.
    """
    row = await session.get(User, principal.user_id)
    if row is None:
        row = User(user_id=principal.user_id, tenant_id=principal.tenant_id)
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id,
            principal.tenant_id,
            resource="users",
            resource_id=principal.user_id,
        )
    row.email = principal.email
    row.display_name = principal.display_name
    row.roles = list(principal.roles)
    row.groups = list(principal.groups)
    row.max_classification = principal.max_classification.value
    row.last_seen_at = _utcnow()
    await session.flush()
    return row


# ----------------------------------------------------------------- documents
async def upsert_document(
    session: AsyncSession,
    *,
    tenant_id: str,
    document_id: str,
    source_uri: str,
    source_type: str,
    access_control: AccessControl,
    source_id: str | None = None,
    title: str = "",
    doc_type: str = "document",
    language: str = "en",
    author: str | None = None,
    content_sha256: str = "",
    simhash: str = "",
    etag: str | None = None,
    acl_fingerprint: str = "",
    tags: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    summary: str | None = None,
    page_count: int | None = None,
    chunk_count: int = 0,
    token_count: int = 0,
    size_bytes: int = 0,
    source_modified_at: datetime | None = None,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    pii_types: Sequence[str] | None = None,
    pii_redacted: bool = False,
    ingest_run_id: str | None = None,
    blob_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Document:
    """Create or update the relational record for an ingested document.

    Bumps ``version`` whenever the content hash changes, so lineage can distinguish
    a re-index of identical content from a genuine revision. Re-appearing documents
    are automatically un-deleted.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; enforced against any existing row.
        document_id: Document id (primary key).
        source_uri: Canonical source URI, unique within the tenant.
        source_type: Connector that produced the document.
        access_control: Resolved ACL, flattened into the row's ACL columns.
        source_id: Source config id.
        title: Document title.
        doc_type: Document class.
        language: ISO 639-1 language code.
        author: Author, when known.
        content_sha256: Hash of the raw payload.
        simhash: Hex simhash of the document text.
        etag: Source ETag.
        acl_fingerprint: Hash of the flattened ACL, for ACL-only reindex detection.
        tags: Document tags.
        keywords: Extracted keywords.
        summary: Document summary.
        page_count: Page count for paginated sources.
        chunk_count: Chunks written for this version.
        token_count: Tokens indexed for this version.
        size_bytes: Raw payload size.
        source_modified_at: Source modification time.
        effective_from: Validity start.
        effective_to: Validity end.
        pii_types: PII entity types detected.
        pii_redacted: Whether redaction ran.
        ingest_run_id: Run that produced this version.
        blob_url: Archived raw copy URL.
        extra: Connector-specific extras.

    Returns:
        The persisted :class:`Document`.

    Raises:
        TenantMismatchError: If the document exists under a different tenant.
        ValueError: If ``access_control`` belongs to a different tenant than
            ``tenant_id``.
    """
    if access_control.tenant_id != tenant_id:
        msg = (
            "access_control.tenant_id "
            f"{access_control.tenant_id!r} does not match tenant_id {tenant_id!r}"
        )
        raise ValueError(msg)

    row = await session.get(Document, document_id)
    if row is None:
        row = Document(
            document_id=document_id, tenant_id=tenant_id, source_uri=source_uri
        )
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id, tenant_id, resource="documents", resource_id=document_id
        )
        if (
            content_sha256
            and row.content_sha256
            and content_sha256 != row.content_sha256
        ):
            row.version = row.version + 1

    row.source_uri = source_uri
    row.source_type = source_type
    row.source_id = source_id
    row.title = title
    row.doc_type = doc_type
    row.language = language
    row.author = author
    row.content_sha256 = content_sha256
    row.simhash = simhash
    row.etag = etag
    row.acl_fingerprint = acl_fingerprint

    flat = access_control.to_flat()
    row.classification = flat["classification"]
    row.classification_rank = flat["classification_rank"]
    row.allowed_roles = flat["allowed_roles"]
    row.allowed_groups = flat["allowed_groups"]
    row.allowed_users = flat["allowed_users"]
    row.denied_users = flat["denied_users"]

    row.tags = list(tags or [])
    row.keywords = list(keywords or [])
    row.summary = summary
    row.page_count = page_count
    row.chunk_count = chunk_count
    row.token_count = token_count
    row.size_bytes = size_bytes
    row.source_modified_at = source_modified_at
    row.effective_from = effective_from
    row.effective_to = effective_to
    row.pii_types = list(pii_types or [])
    row.pii_redacted = pii_redacted
    row.ingest_run_id = ingest_run_id
    row.blob_url = blob_url
    if extra is not None:
        row.extra = extra
    # A document that reappeared at source is live again.
    row.is_deleted = False
    row.deleted_at = None

    await session.flush()
    return row


async def mark_documents_deleted(
    session: AsyncSession, *, tenant_id: str, document_ids: Sequence[str]
) -> int:
    """Soft-delete documents that vanished from their source.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        document_ids: Documents to soft-delete.

    Returns:
        The number of rows updated.
    """
    if not document_ids:
        return 0
    result = await session.execute(
        update(Document)
        .where(
            Document.tenant_id == tenant_id,
            Document.document_id.in_(list(document_ids)),
            Document.is_deleted.is_(False),
        )
        .values(is_deleted=True, deleted_at=_utcnow())
    )
    await session.flush()
    return int(result.rowcount or 0)


async def list_source_configs(
    session: AsyncSession, *, tenant_id: str, enabled_only: bool = True
) -> list[SourceConfigRow]:
    """List a tenant's ingestion sources.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        enabled_only: Restrict to enabled sources.

    Returns:
        The matching source config rows, ordered by name.
    """
    stmt = select(SourceConfigRow).where(SourceConfigRow.tenant_id == tenant_id)
    if enabled_only:
        stmt = stmt.where(SourceConfigRow.enabled.is_(True))
    result = await session.execute(stmt.order_by(SourceConfigRow.name))
    return list(result.scalars())


# --------------------------------------------------------------- ingest runs
async def record_ingest_run(
    session: AsyncSession, summary: IngestRunSummary
) -> IngestRun:
    """Create or update the row for an ingestion run.

    Called once when a run starts (status ``running``) and again when it finishes, so
    an interrupted serverless invocation still leaves an auditable record.

    Args:
        session: Active database session.
        summary: The run summary to persist.

    Returns:
        The persisted :class:`IngestRun`.

    Raises:
        TenantMismatchError: If the run exists under a different tenant.
    """
    row = await session.get(IngestRun, summary.run_id)
    if row is None:
        row = IngestRun(run_id=summary.run_id, tenant_id=summary.tenant_id)
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id,
            summary.tenant_id,
            resource="ingest_runs",
            resource_id=summary.run_id,
        )

    row.source_id = summary.source_id
    row.trigger = summary.trigger.value
    row.status = summary.status.value
    row.started_at = summary.started_at
    row.finished_at = summary.finished_at
    row.documents_seen = summary.documents_seen
    row.documents_created = summary.documents_created
    row.documents_updated = summary.documents_updated
    row.documents_deleted = summary.documents_deleted
    row.documents_skipped = summary.documents_skipped
    row.documents_failed = summary.documents_failed
    row.chunks_upserted = summary.chunks_upserted
    row.chunks_deleted = summary.chunks_deleted
    row.tokens_embedded = summary.tokens_embedded
    row.duplicates_dropped = summary.duplicates_dropped
    row.pii_documents = summary.pii_documents
    row.forced = summary.forced
    row.within_working_hours = summary.within_working_hours
    row.skip_reason = summary.skip_reason
    row.error_message = summary.error_message
    row.metrics = dict(summary.metrics)

    await session.flush()
    return row


async def record_ingest_item(
    session: AsyncSession,
    *,
    tenant_id: str,
    run_id: str,
    source_uri: str,
    action: str,
    status: str,
    document_id: str | None = None,
    reason: str | None = None,
    chunk_count: int = 0,
    token_count: int = 0,
    duplicates_dropped: int = 0,
    duration_ms: float = 0.0,
    error_message: str | None = None,
) -> IngestItem:
    """Record one document's outcome within a run.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        run_id: Parent run id.
        source_uri: Document's canonical source URI.
        action: create | update | delete | skip | acl_only.
        status: succeeded | failed | skipped.
        document_id: Document id, when one was resolved.
        reason: Short machine-readable reason, e.g. ``"unchanged"``.
        chunk_count: Chunks written.
        token_count: Tokens embedded.
        duplicates_dropped: Chunks dropped by dedupe.
        duration_ms: Processing time.
        error_message: Redacted failure description.

    Returns:
        The persisted :class:`IngestItem`.
    """
    row = IngestItem(
        item_id=new_id(),
        run_id=run_id,
        tenant_id=tenant_id,
        document_id=document_id,
        source_uri=source_uri,
        action=action,
        status=status,
        reason=reason,
        chunk_count=chunk_count,
        token_count=token_count,
        duplicates_dropped=duplicates_dropped,
        duration_ms=duration_ms,
        error_message=error_message,
    )
    session.add(row)
    await session.flush()
    return row


# ------------------------------------------------------------------ sessions
async def get_or_create_session(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    title: str = "",
) -> ChatSession:
    """Fetch an existing chat session or start a new one.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id.
        session_id: Existing session to continue. A new session is created when this
            is None or names a session that does not exist.
        title: Title for a newly created session.

    Returns:
        The persisted :class:`ChatSession`.

    Raises:
        TenantMismatchError: If the session exists under a different tenant, or
            belongs to a different user within this tenant.
    """
    if session_id:
        row = await session.get(ChatSession, session_id)
        if row is not None:
            _guard_tenant(
                row.tenant_id,
                tenant_id,
                resource="chat_sessions",
                resource_id=session_id,
            )
            if row.user_id != user_id:
                raise TenantMismatchError(
                    f"chat session {session_id} belongs to another user",
                    expected=tenant_id,
                    actual=row.tenant_id,
                    resource=f"chat_sessions:{session_id}",
                    detail={"reason": "user_mismatch"},
                )
            return row

    row = ChatSession(
        session_id=session_id or new_id(),
        tenant_id=tenant_id,
        user_id=user_id,
        title=title,
    )
    session.add(row)
    await session.flush()
    return row


async def get_session_row(
    session: AsyncSession, *, tenant_id: str, user_id: str | None, session_id: str
) -> ChatSession | None:
    """Fetch one chat session, scoped to tenant and optionally to user.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id, or None to allow any user in the tenant (admin).
        session_id: Session to fetch.

    Returns:
        The session, or None when it does not exist.
    """
    stmt = select(ChatSession).where(
        ChatSession.tenant_id == tenant_id, ChatSession.session_id == session_id
    )
    if user_id is not None:
        stmt = stmt.where(ChatSession.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_sessions(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
) -> list[ChatSession]:
    """List a user's chat sessions, most recently active first.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id.
        limit: Maximum sessions to return.
        offset: Rows to skip, for paging.
        include_archived: Include archived sessions.

    Returns:
        The matching sessions.
    """
    stmt = select(ChatSession).where(
        ChatSession.tenant_id == tenant_id, ChatSession.user_id == user_id
    )
    if not include_archived:
        stmt = stmt.where(ChatSession.is_archived.is_(False))
    stmt = (
        stmt.order_by(ChatSession.last_message_at.desc().nullslast())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return list(result.scalars())


async def delete_session(
    session: AsyncSession, *, tenant_id: str, user_id: str, session_id: str
) -> bool:
    """Delete a chat session and its messages.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        user_id: Owning user id; part of the WHERE clause.
        session_id: Session to delete.

    Returns:
        True when a row was deleted.
    """
    result = await session.execute(
        delete(ChatSession).where(
            ChatSession.tenant_id == tenant_id,
            ChatSession.user_id == user_id,
            ChatSession.session_id == session_id,
        )
    )
    await session.flush()
    return bool(result.rowcount)


# ------------------------------------------------------------------ messages
async def append_message(
    session: AsyncSession,
    *,
    tenant_id: str,
    session_id: str,
    user_id: str,
    role: Role,
    content: str,
    pii_redacted: bool,
    message_id: str | None = None,
    token_count: int = 0,
    citations: Sequence[dict[str, Any]] | None = None,
    tool_calls: Sequence[dict[str, Any]] | None = None,
    guardrail_events: Sequence[dict[str, Any]] | None = None,
    context_stats: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    pii_types: Sequence[str] | None = None,
    model: str | None = None,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    trace_id: str | None = None,
    stop_reason: str | None = None,
    refused: bool = False,
    pinned: bool = False,
) -> ChatMessage:
    """Append a turn to a session and roll up the session counters.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; enforced against the session row.
        session_id: Session to append to.
        user_id: Author (for a user turn) or owner (for an assistant turn).
        role: user | assistant.
        content: Turn text, already PII-redacted.
        pii_redacted: Must be True — asserts the redaction pass actually ran before
            this content was handed to the database.
        message_id: Explicit message id; generated when omitted.
        token_count: Tokens this turn occupies.
        citations: Serialised :class:`~ragcore.models.retrieval.Citation` list.
        tool_calls: Serialised :class:`~ragcore.models.chat.ToolCall` list.
        guardrail_events: Serialised guardrail events for this turn.
        context_stats: Serialised :class:`~ragcore.models.chat.ContextStats`.
        usage: Token usage and cache counters for this turn.
        pii_types: PII entity types found and redacted.
        model: Model that produced an assistant turn.
        latency_ms: End-to-end turn latency.
        cost_usd: Anthropic spend for this turn.
        trace_id: Langfuse trace id.
        stop_reason: Anthropic stop reason.
        refused: True when the model declined (``stop_reason == "refusal"``).
        pinned: Mark this turn as never-suppressible.

    Returns:
        The persisted :class:`ChatMessage`.

    Raises:
        ValueError: If ``pii_redacted`` is False. Raw user content must never reach
            persistent storage.
        TenantMismatchError: If the session belongs to a different tenant.
    """
    if not pii_redacted:
        msg = (
            "refusing to persist message content that has not passed PII redaction: "
            "pass pii_redacted=True only after running the redaction pass"
        )
        raise ValueError(msg)

    parent = await session.get(ChatSession, session_id)
    if parent is None:
        msg = f"chat session {session_id} does not exist"
        raise ValueError(msg)
    _guard_tenant(
        parent.tenant_id, tenant_id, resource="chat_sessions", resource_id=session_id
    )

    now = _utcnow()
    row = ChatMessage(
        message_id=message_id or new_id(),
        tenant_id=tenant_id,
        session_id=session_id,
        user_id=user_id,
        role=role.value,
        content=content,
        token_count=token_count,
        citations=list(citations or []),
        tool_calls=list(tool_calls or []),
        guardrail_events=list(guardrail_events or []),
        context_stats=dict(context_stats or {}),
        usage=dict(usage or {}),
        pii_redacted=True,
        pii_types=list(pii_types or []),
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        trace_id=trace_id,
        stop_reason=stop_reason,
        refused=refused,
        pinned=pinned,
        created_at=now,
    )
    session.add(row)

    parent.message_count = parent.message_count + 1
    parent.last_message_at = now
    parent.total_cost_usd = parent.total_cost_usd + cost_usd
    if usage:
        parent.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        parent.total_output_tokens += int(usage.get("output_tokens", 0) or 0)
        parent.total_cache_read_tokens += int(usage.get("cache_read_tokens", 0) or 0)
    if not parent.title and role is Role.USER:
        parent.title = content[:120]

    await session.flush()
    return row


async def list_session_messages(
    session: AsyncSession,
    *,
    tenant_id: str,
    session_id: str,
    limit: int | None = None,
    include_suppressed: bool = True,
    ascending: bool = True,
) -> list[Message]:
    """Load a session's turns as domain models, including suppression flags.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        session_id: Session to read.
        limit: Maximum turns to return. When set with ``ascending=True`` the most
            recent ``limit`` turns are selected and then returned oldest-first, which
            is what context assembly wants.
        include_suppressed: Include turns folded into the rolling summary. The
            ``GET /sessions/{id}/messages`` endpoint keeps them so the UI can show
            what was suppressed; context assembly excludes them.
        ascending: Return oldest-first. False returns newest-first.

    Returns:
        The turns as :class:`~ragcore.models.chat.Message` instances.
    """
    stmt = select(ChatMessage).where(
        ChatMessage.tenant_id == tenant_id, ChatMessage.session_id == session_id
    )
    if not include_suppressed:
        stmt = stmt.where(
            (ChatMessage.suppressed.is_(False)) | (ChatMessage.pinned.is_(True))
        )

    if limit is not None:
        stmt = stmt.order_by(ChatMessage.created_at.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = list(result.scalars())
        if ascending:
            rows.reverse()
    else:
        order = (
            ChatMessage.created_at.asc() if ascending else ChatMessage.created_at.desc()
        )
        result = await session.execute(stmt.order_by(order))
        rows = list(result.scalars())

    return [_to_message(row) for row in rows]


def _to_message(row: ChatMessage) -> Message:
    """Convert a database row into the domain message model.

    Args:
        row: The persisted row.

    Returns:
        The equivalent :class:`~ragcore.models.chat.Message`.
    """
    return Message.model_validate(
        {
            "message_id": row.message_id,
            "session_id": row.session_id,
            "role": row.role,
            "content": row.content,
            "citations": row.citations or [],
            "tool_calls": row.tool_calls or [],
            "token_count": row.token_count,
            "created_at": row.created_at,
            "suppressed": row.suppressed,
            "pinned": row.pinned,
        }
    )


async def count_session_messages(
    session: AsyncSession, *, tenant_id: str, session_id: str
) -> int:
    """Count turns in a session.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        session_id: Session to count.

    Returns:
        The number of stored turns.
    """
    result = await session.execute(
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.tenant_id == tenant_id, ChatMessage.session_id == session_id)
    )
    return int(result.scalar_one())


async def suppress_messages(
    session: AsyncSession,
    *,
    tenant_id: str,
    session_id: str,
    message_ids: Sequence[str],
    rolling_summary: str | None = None,
    summary_tokens: int | None = None,
) -> int:
    """Mark turns as suppressed and update the rolling summary that replaces them.

    Pinned turns are never suppressed, so they are excluded from the update even if
    the caller passes their ids.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        session_id: Session being compacted.
        message_ids: Turns to suppress.
        rolling_summary: New summary text folding in the suppressed turns.
        summary_tokens: Token count of the new summary.

    Returns:
        The number of turns suppressed.
    """
    suppressed = 0
    if message_ids:
        result = await session.execute(
            update(ChatMessage)
            .where(
                ChatMessage.tenant_id == tenant_id,
                ChatMessage.session_id == session_id,
                ChatMessage.message_id.in_(list(message_ids)),
                ChatMessage.pinned.is_(False),
            )
            .values(suppressed=True)
        )
        suppressed = int(result.rowcount or 0)

    if rolling_summary is not None:
        parent = await get_session_row(
            session, tenant_id=tenant_id, user_id=None, session_id=session_id
        )
        if parent is not None:
            parent.rolling_summary = rolling_summary
            if summary_tokens is not None:
                parent.summary_tokens = summary_tokens
            parent.compaction_events = parent.compaction_events + 1

    await session.flush()
    return suppressed


async def write_tool_invocation(
    session: AsyncSession,
    *,
    tenant_id: str,
    tool_call_id: str,
    tool_name: str,
    kind: str,
    arguments: dict[str, Any],
    session_id: str | None = None,
    message_id: str | None = None,
    user_id: str | None = None,
    result_summary: str | None = None,
    is_error: bool = False,
    error_message: str | None = None,
    http_status: int | None = None,
    truncated: bool = False,
    latency_ms: float = 0.0,
    trace_id: str | None = None,
) -> ToolInvocation:
    """Persist one tool invocation.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        tool_call_id: Matches the model's tool_use block id.
        tool_name: Invoked tool name.
        kind: rest | mcp | retrieval.
        arguments: Redacted arguments the model supplied.
        session_id: Session the call belongs to.
        message_id: Assistant turn the call belongs to.
        user_id: Caller.
        result_summary: Short redacted result preview.
        is_error: Whether the call failed.
        error_message: Redacted failure description.
        http_status: Upstream HTTP status, for REST tools.
        truncated: Whether the result was truncated.
        latency_ms: Invocation latency.
        trace_id: Langfuse trace id.

    Returns:
        The persisted :class:`ToolInvocation`.
    """
    row = ToolInvocation(
        tool_call_id=tool_call_id,
        tenant_id=tenant_id,
        session_id=session_id,
        message_id=message_id,
        user_id=user_id,
        tool_name=tool_name,
        kind=kind,
        arguments=arguments,
        result_summary=result_summary,
        is_error=is_error,
        error_message=error_message,
        http_status=http_status,
        truncated=truncated,
        latency_ms=latency_ms,
        trace_id=trace_id,
    )
    session.add(row)
    await session.flush()
    return row


# -------------------------------------------------------------------- memory
async def get_or_create_profile(
    session: AsyncSession, *, tenant_id: str, user_id: str
) -> UserProfile:
    """Fetch a user's profile, creating a blank one on first contact.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id.

    Returns:
        The profile as a :class:`~ragcore.models.memory.UserProfile`.
    """
    row = await session.get(UserProfileRow, (tenant_id, user_id))
    if row is None:
        row = UserProfileRow(tenant_id=tenant_id, user_id=user_id, summary="")
        session.add(row)
        await session.flush()
    return UserProfile(
        user_id=row.user_id,
        tenant_id=row.tenant_id,
        summary=row.summary,
        preferred_style=row.preferred_style,
        preferred_language=row.preferred_language,
        top_topics=list(row.top_topics or []),
        memory_consent=row.memory_consent,
        updated_at=row.updated_at,
    )


async def save_profile(session: AsyncSession, profile: UserProfile) -> UserProfileRow:
    """Persist an updated profile.

    Args:
        session: Active database session.
        profile: The profile to write.

    Returns:
        The persisted :class:`UserProfileRow`.
    """
    row = await session.get(UserProfileRow, (profile.tenant_id, profile.user_id))
    if row is None:
        row = UserProfileRow(tenant_id=profile.tenant_id, user_id=profile.user_id)
        session.add(row)
    row.summary = profile.summary
    row.preferred_style = profile.preferred_style
    row.preferred_language = profile.preferred_language
    row.top_topics = list(profile.top_topics)
    row.memory_consent = profile.memory_consent
    await session.flush()
    return row


async def set_memory_consent(
    session: AsyncSession, *, tenant_id: str, user_id: str, consent: bool
) -> UserProfileRow:
    """Toggle long-term memory for a user.

    Revoking consent also soft-deletes their stored memories, so "switch it off"
    genuinely means "stop remembering me" rather than "stop reading what you stored".

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id.
        consent: New consent value.

    Returns:
        The updated :class:`UserProfileRow`.
    """
    row = await session.get(UserProfileRow, (tenant_id, user_id))
    if row is None:
        row = UserProfileRow(tenant_id=tenant_id, user_id=user_id)
        session.add(row)
    row.memory_consent = consent
    if not consent:
        await session.execute(
            update(UserMemory)
            .where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == user_id,
                UserMemory.is_deleted.is_(False),
            )
            .values(is_deleted=True)
        )
    await session.flush()
    return row


async def upsert_memory(
    session: AsyncSession, memory: LongTermMemory, *, tenant_id: str
) -> UserMemory:
    """Create or update a long-term memory, applying ``supersedes`` if set.

    Args:
        session: Active database session.
        memory: The memory to persist.
        tenant_id: Tenant the caller is scoped to; must match ``memory.tenant_id``.

    Returns:
        The persisted :class:`UserMemory`.

    Raises:
        TenantMismatchError: If the memory belongs to a different tenant than the
            caller, or an existing row with this id belongs elsewhere.
    """
    if memory.tenant_id != tenant_id:
        raise TenantMismatchError(
            "memory tenant does not match the caller's tenant",
            expected=tenant_id,
            actual=memory.tenant_id,
            resource=f"user_memories:{memory.memory_id}",
        )

    row = await session.get(UserMemory, memory.memory_id)
    if row is None:
        row = UserMemory(memory_id=memory.memory_id, tenant_id=tenant_id)
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id,
            tenant_id,
            resource="user_memories",
            resource_id=memory.memory_id,
        )

    row.user_id = memory.user_id
    row.kind = memory.kind.value
    row.text = memory.text
    row.salience = memory.salience
    row.source_session_id = memory.source_session_id
    row.supersedes = memory.supersedes
    row.hit_count = memory.hit_count
    row.pii_redacted = memory.pii_redacted
    row.last_used_at = memory.last_used_at
    row.expires_at = memory.expires_at
    row.is_deleted = False

    if memory.supersedes:
        await session.execute(
            update(UserMemory)
            .where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == memory.user_id,
                UserMemory.memory_id == memory.supersedes,
            )
            .values(is_deleted=True, superseded_by=memory.memory_id)
        )

    await session.flush()
    return row


async def list_memories(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    kinds: Sequence[str] | None = None,
    limit: int = 100,
    include_deleted: bool = False,
) -> list[UserMemory]:
    """List a user's memories, most salient first.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Owning user id.
        kinds: Restrict to these memory kinds.
        limit: Maximum memories to return.
        include_deleted: Include soft-deleted and superseded memories.

    Returns:
        The matching memory rows.
    """
    stmt = select(UserMemory).where(
        UserMemory.tenant_id == tenant_id, UserMemory.user_id == user_id
    )
    if not include_deleted:
        stmt = stmt.where(UserMemory.is_deleted.is_(False))
    if kinds:
        stmt = stmt.where(UserMemory.kind.in_(list(kinds)))
    result = await session.execute(
        stmt.order_by(UserMemory.salience.desc(), UserMemory.created_at.desc()).limit(
            limit
        )
    )
    return list(result.scalars())


async def delete_memory(
    session: AsyncSession, *, tenant_id: str, user_id: str, memory_id: str
) -> bool:
    """Soft-delete one memory.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id; part of the WHERE clause.
        user_id: Owning user id; part of the WHERE clause.
        memory_id: Memory to delete.

    Returns:
        True when a row was updated.
    """
    result = await session.execute(
        update(UserMemory)
        .where(
            UserMemory.tenant_id == tenant_id,
            UserMemory.user_id == user_id,
            UserMemory.memory_id == memory_id,
        )
        .values(is_deleted=True)
    )
    await session.flush()
    return bool(result.rowcount)


async def upsert_cache_meta(
    session: AsyncSession,
    *,
    tenant_id: str,
    cache_id: str,
    normalized_query: str,
    filter_fingerprint: str,
    chunk_ids: Sequence[str],
    transformed_queries: Sequence[str] | None = None,
    user_id: str | None = None,
    ttl_seconds: int = 86_400,
    expires_at: datetime | None = None,
) -> SemanticCacheMeta:
    """Create or refresh the relational mirror of a semantic-cache entry.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        cache_id: Cache entry id.
        normalized_query: Canonicalised query text.
        filter_fingerprint: Fingerprint that must match exactly for reuse.
        chunk_ids: Chunk ids in the cached plan.
        transformed_queries: Queries the transformer produced.
        user_id: Owning user, or None for a tenant-wide entry.
        ttl_seconds: Entry lifetime.
        expires_at: Explicit expiry, for the TTL sweep.

    Returns:
        The persisted :class:`SemanticCacheMeta`.

    Raises:
        TenantMismatchError: If an existing entry belongs to a different tenant.
    """
    row = await session.get(SemanticCacheMeta, cache_id)
    if row is None:
        row = SemanticCacheMeta(cache_id=cache_id, tenant_id=tenant_id)
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id,
            tenant_id,
            resource="semantic_cache_meta",
            resource_id=cache_id,
        )
        row.hit_count = row.hit_count + 1
    row.user_id = user_id
    row.normalized_query = normalized_query
    row.transformed_queries = list(transformed_queries or [])
    row.chunk_ids = list(chunk_ids)
    row.filter_fingerprint = filter_fingerprint
    row.ttl_seconds = ttl_seconds
    row.last_used_at = _utcnow()
    row.expires_at = expires_at
    await session.flush()
    return row


# ------------------------------------------------------------------- lineage
async def record_lineage(
    session: AsyncSession,
    *,
    tenant_id: str,
    kind: str,
    subject_id: str,
    operation: str,
    actor: str,
    parents: Sequence[str] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    lineage_id: str | None = None,
) -> LineageRecordRow:
    """Write one lineage edge.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        kind: ingest | retrieval | generation | tool | eval.
        subject_id: Subject of the record (document, chunk or message id).
        operation: Operation performed, e.g. ``"chunk"`` or ``"generate"``.
        actor: Who performed it: a user id, ``"system"``, or a model id.
        parents: Upstream ids this subject derives from.
        user_id: Caller, when the operation was user-initiated.
        session_id: Session, for chat-time operations.
        trace_id: Langfuse trace id.
        inputs: Redacted operation inputs.
        outputs: Redacted operation outputs.
        metrics: Numeric measurements for this operation.
        lineage_id: Explicit record id; generated when omitted.

    Returns:
        The persisted :class:`LineageRecordRow`.
    """
    row = LineageRecordRow(
        lineage_id=lineage_id or new_id(),
        kind=kind,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        subject_id=subject_id,
        parents=list(parents or []),
        operation=operation,
        actor=actor,
        inputs=dict(inputs or {}),
        outputs=dict(outputs or {}),
        metrics=dict(metrics or {}),
    )
    session.add(row)
    await session.flush()
    return row


# ---------------------------------------------------------------- evaluation
async def write_eval_run(
    session: AsyncSession,
    *,
    tenant_id: str,
    run_id: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    git_sha: str | None = None,
    config_fingerprint: str = "",
    golden_set_path: str | None = None,
    aggregate: dict[str, Any] | None = None,
    gate_passed: bool = False,
    item_count: int = 0,
    passed_count: int = 0,
    failed_count: int = 0,
    total_cost_usd: float = 0.0,
    notes: str | None = None,
) -> EvalRunRow:
    """Create or update an evaluation run row.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        run_id: Run id.
        started_at: Run start time.
        finished_at: Run end time.
        git_sha: Commit under test.
        config_fingerprint: Hash of the settings that shaped the run.
        golden_set_path: Golden file used.
        aggregate: Aggregate metric mapping.
        gate_passed: Whether the CI gate passed.
        item_count: Items evaluated.
        passed_count: Items that passed.
        failed_count: Items that failed.
        total_cost_usd: Total spend.
        notes: Free-form notes.

    Returns:
        The persisted :class:`EvalRunRow`.

    Raises:
        TenantMismatchError: If the run exists under a different tenant.
    """
    row = await session.get(EvalRunRow, run_id)
    if row is None:
        row = EvalRunRow(run_id=run_id, tenant_id=tenant_id, started_at=started_at)
        session.add(row)
    else:
        _guard_tenant(
            row.tenant_id, tenant_id, resource="eval_runs", resource_id=run_id
        )
    row.finished_at = finished_at
    row.git_sha = git_sha
    row.config_fingerprint = config_fingerprint
    row.golden_set_path = golden_set_path
    row.aggregate = dict(aggregate or {})
    row.gate_passed = gate_passed
    row.item_count = item_count
    row.passed_count = passed_count
    row.failed_count = failed_count
    row.total_cost_usd = total_cost_usd
    row.notes = notes
    await session.flush()
    return row


async def write_eval_result(
    session: AsyncSession,
    *,
    tenant_id: str,
    run_id: str,
    item_id: str,
    answer: str,
    scores: dict[str, Any],
    passed: bool,
    failures: Sequence[str] | None = None,
    retrieved_chunk_ids: Sequence[str] | None = None,
    category: str | None = None,
    question: str | None = None,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    trace_id: str | None = None,
) -> EvalResultRow:
    """Persist one per-item evaluation result.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        run_id: Parent run id.
        item_id: Golden item id.
        answer: Answer the pipeline produced.
        scores: Serialised :class:`~ragcore.models.eval.MetricScores`.
        passed: Whether every assertion held.
        failures: Reasons the item failed.
        retrieved_chunk_ids: Chunk ids retrieved.
        category: Golden item category.
        question: The question asked.
        latency_ms: Item latency.
        cost_usd: Item spend.
        trace_id: Langfuse trace id.

    Returns:
        The persisted :class:`EvalResultRow`.
    """
    row = EvalResultRow(
        result_id=new_id(),
        run_id=run_id,
        tenant_id=tenant_id,
        item_id=item_id,
        category=category,
        question=question,
        answer=answer,
        retrieved_chunk_ids=list(retrieved_chunk_ids or []),
        scores=scores,
        passed=passed,
        failures=list(failures or []),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        trace_id=trace_id,
    )
    session.add(row)
    await session.flush()
    return row


# ------------------------------------------------------------------ feedback
async def write_feedback(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    rating: int,
    pii_redacted: bool,
    session_id: str | None = None,
    message_id: str | None = None,
    comment: str | None = None,
    tags: Sequence[str] | None = None,
    trace_id: str | None = None,
    langfuse_score_id: str | None = None,
) -> Feedback:
    """Record thumbs-up/down feedback plus an optional comment.

    Args:
        session: Active database session.
        tenant_id: Owning tenant id.
        user_id: Caller.
        rating: 1 for thumbs up, -1 for thumbs down.
        pii_redacted: Must be True when ``comment`` is set — the comment is free-form
            user text and goes through redaction first.
        session_id: Session the feedback refers to.
        message_id: Message the feedback refers to.
        comment: Redacted free-text comment.
        tags: Structured feedback tags.
        trace_id: Langfuse trace id the score is attached to.
        langfuse_score_id: Id of the created Langfuse score.

    Returns:
        The persisted :class:`Feedback`.

    Raises:
        ValueError: If ``rating`` is not -1 or 1, or if a comment is supplied without
            ``pii_redacted=True``.
    """
    if rating not in {-1, 1}:
        msg = f"rating must be -1 or 1, got {rating}"
        raise ValueError(msg)
    if comment and not pii_redacted:
        msg = (
            "refusing to persist a feedback comment that has not passed PII "
            "redaction: pass pii_redacted=True only after redaction"
        )
        raise ValueError(msg)

    row = Feedback(
        feedback_id=new_id(),
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        comment=comment,
        tags=list(tags or []),
        trace_id=trace_id,
        langfuse_score_id=langfuse_score_id,
        pii_redacted=pii_redacted,
    )
    session.add(row)
    await session.flush()
    return row


# --------------------------------------------------------------------- audit
async def write_audit(
    session: AsyncSession,
    *,
    tenant_id: str,
    action: str,
    outcome: str = "allow",
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    trace_id: str | None = None,
) -> AuditLog:
    """Write a security-relevant audit record.

    Called on auth decisions, ACL denials, admin mutations and every
    :class:`~ragcore.errors.TenantMismatchError`.

    Args:
        session: Active database session.
        tenant_id: Tenant the action was scoped to.
        action: Action name, e.g. ``"chat.completed"`` or ``"acl.denied"``.
        outcome: allow | deny | error.
        user_id: Caller, when authenticated.
        resource_type: Resource kind touched.
        resource_id: Resource id touched.
        detail: Redacted structured context. Never raw user content or secrets.
        ip_address: Client address.
        user_agent: Client user agent.
        trace_id: Langfuse trace id.

    Returns:
        The persisted :class:`AuditLog`.
    """
    row = AuditLog(
        audit_id=new_id(),
        tenant_id=tenant_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        detail=dict(detail or {}),
        ip_address=ip_address,
        user_agent=user_agent,
        trace_id=trace_id,
    )
    session.add(row)
    await session.flush()
    return row
