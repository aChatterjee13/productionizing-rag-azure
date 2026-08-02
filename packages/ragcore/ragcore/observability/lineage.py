"""Lineage records: requirement #9, complete traceability answer to source URI.

A lineage record is one edge in a provenance graph. ``subject_id`` names the
thing that was produced (a document, a chunk, a message, an evaluation result)
and ``parents`` names what it derives from, so the graph reads
``message -> chunk -> document -> source_uri``.

:func:`document_provenance` walks that graph **upwards** from a document and
returns the whole chain plus the ingest run and per-document ingest items that
produced it.

Two rules callers must respect:

* ``inputs``, ``outputs`` and ``metrics`` are persisted and returned by the API.
  They must contain redacted or structural values only, never raw user or
  document text that has not passed :mod:`ragcore.pii`.
* every read and write is scoped by ``tenant_id``. :func:`document_provenance`
  therefore requires the caller's tenant, and a document belonging to another
  tenant reads as absent rather than as someone else's provenance.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.models import Document, IngestItem, IngestRun, LineageRecordRow
from ragcore.db.repositories import record_lineage as _persist_lineage
from ragcore.logging import get_logger
from ragcore.observability.langfuse import get_current_trace_id

__all__ = [
    "DEFAULT_PROVENANCE_DEPTH",
    "LineageKind",
    "LineageRecord",
    "document_provenance",
    "record_lineage",
    "subject_provenance",
]

_log = get_logger(__name__)

#: How many parent generations :func:`subject_provenance` follows by default.
#: A chunk chain is normally three hops, so eight is generous and still bounds a
#: cyclic or mis-linked graph.
DEFAULT_PROVENANCE_DEPTH = 8


class LineageKind:
    """Allowed values for :attr:`LineageRecord.kind`.

    A plain constant holder rather than an enum, because the contract types
    ``kind`` as ``str`` and the database column is a free-form string.
    """

    INGEST = "ingest"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    TOOL = "tool"
    EVAL = "eval"

    ALL = ("ingest", "retrieval", "generation", "tool", "eval")


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate an opaque 32-character identifier.

    Returns:
        A hex UUID4 with dashes stripped.
    """
    return uuid.uuid4().hex


class LineageRecord(BaseModel):
    """One provenance edge.

    Attributes:
        lineage_id: Record id; generated when omitted.
        kind: ``ingest``, ``retrieval``, ``generation``, ``tool`` or ``eval``.
        tenant_id: Owning tenant. Every query filters on it.
        user_id: Caller, when the operation was user-initiated.
        session_id: Chat session, for chat-time operations.
        trace_id: Langfuse trace id; filled from the ambient trace when omitted.
        subject_id: The artefact produced (document, chunk, message, item).
        parents: Upstream ids this subject derives from.
        operation: What was done, e.g. ``"chunk"``, ``"rerank"``, ``"generate"``.
        actor: Who did it: a user id, ``"system"``, or a model id.
        inputs: Redacted operation inputs.
        outputs: Redacted operation outputs.
        metrics: Numeric measurements (tokens, scores, latency, cost).
        created_at: When the operation happened.
    """

    model_config = ConfigDict(extra="forbid")

    lineage_id: str = Field(default_factory=_new_id)
    kind: str
    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None
    subject_id: str
    parents: list[str] = Field(default_factory=list)
    operation: str
    actor: str = "system"
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _inherit_trace(self) -> LineageRecord:
        """Adopt the ambient Langfuse trace id when the caller did not pass one.

        Returns:
            This record, with ``trace_id`` filled in when a trace is open.
        """
        if self.trace_id is None:
            ambient = get_current_trace_id()
            if ambient:
                self.trace_id = ambient
        return self


def _row_to_record(row: LineageRecordRow) -> LineageRecord:
    """Convert a persisted row into a :class:`LineageRecord`.

    Args:
        row: The ORM row.

    Returns:
        The pydantic record.
    """
    return LineageRecord(
        lineage_id=row.lineage_id,
        kind=row.kind,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        session_id=row.session_id,
        trace_id=row.trace_id,
        subject_id=row.subject_id,
        parents=list(row.parents or []),
        operation=row.operation,
        actor=row.actor,
        inputs=dict(row.inputs or {}),
        outputs=dict(row.outputs or {}),
        metrics=dict(row.metrics or {}),
        created_at=row.created_at,
    )


async def record_lineage(session: AsyncSession, record: LineageRecord) -> None:
    """Persist one lineage edge.

    The row is flushed, not committed: the caller owns the transaction.

    Args:
        session: Active database session.
        record: The edge to write. ``inputs``/``outputs`` must already be
            redacted.
    """
    await _persist_lineage(
        session,
        tenant_id=record.tenant_id,
        kind=record.kind,
        subject_id=record.subject_id,
        operation=record.operation,
        actor=record.actor,
        parents=record.parents,
        user_id=record.user_id,
        session_id=record.session_id,
        trace_id=record.trace_id,
        inputs=record.inputs,
        outputs=record.outputs,
        metrics=record.metrics,
        lineage_id=record.lineage_id,
    )


async def subject_provenance(
    session: AsyncSession,
    *,
    tenant_id: str,
    subject_id: str,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Walk the provenance graph upwards from any subject.

    Args:
        session: Active database session.
        tenant_id: Caller's tenant. Records from other tenants are invisible.
        subject_id: Starting artefact (message, chunk, document, item).
        max_depth: Parent generations to follow. Defaults to
            :data:`DEFAULT_PROVENANCE_DEPTH`.

    Returns:
        A mapping with ``subject_id``, ``records`` (every edge found, oldest
        first), ``chain`` (the same edges annotated with their distance from the
        subject), ``ancestors`` (unique upstream ids), ``depth`` and
        ``truncated``.
    """
    limit = max_depth if max_depth is not None else DEFAULT_PROVENANCE_DEPTH
    seen: set[str] = {subject_id}
    frontier: list[str] = [subject_id]
    chain: list[dict[str, Any]] = []
    records: list[LineageRecord] = []
    depth_reached = 0
    truncated = False

    for depth in range(max(limit, 0)):
        if not frontier:
            break
        depth_reached = depth
        stmt = (
            select(LineageRecordRow)
            .where(
                LineageRecordRow.tenant_id == tenant_id,
                LineageRecordRow.subject_id.in_(frontier),
            )
            .order_by(LineageRecordRow.created_at)
        )
        rows = (await session.execute(stmt)).scalars().all()
        next_frontier: list[str] = []
        for row in rows:
            record = _row_to_record(row)
            records.append(record)
            chain.append(
                {
                    "depth": depth,
                    "lineage_id": record.lineage_id,
                    "kind": record.kind,
                    "subject_id": record.subject_id,
                    "parents": record.parents,
                    "operation": record.operation,
                    "actor": record.actor,
                    "trace_id": record.trace_id,
                    "created_at": record.created_at.isoformat(),
                }
            )
            for parent in record.parents:
                if parent not in seen:
                    seen.add(parent)
                    next_frontier.append(parent)
        frontier = next_frontier
    else:
        truncated = bool(frontier)

    if truncated:
        _log.warning(
            "provenance_truncated",
            tenant_id=tenant_id,
            subject_id=subject_id,
            max_depth=limit,
        )

    return {
        "subject_id": subject_id,
        "tenant_id": tenant_id,
        "records": [r.model_dump(mode="json") for r in records],
        "chain": chain,
        "ancestors": sorted(seen - {subject_id}),
        "depth": depth_reached,
        "truncated": truncated,
    }


async def document_provenance(
    session: AsyncSession,
    document_id: str,
    *,
    tenant_id: str,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Return the full upstream chain for one document.

    Args:
        session: Active database session.
        document_id: Document to trace.
        tenant_id: Caller's tenant. A document under a different tenant reads as
            absent, so a cross-tenant probe learns nothing.
        max_depth: Parent generations to follow. Defaults to
            :data:`DEFAULT_PROVENANCE_DEPTH`.

    Returns:
        A mapping with ``found``, the ``document`` summary (source uri, hashes,
        version, ACL shape, counts), its ``ingest_run`` and ``ingest_items``, and
        the lineage ``records``/``chain``/``ancestors`` from
        :func:`subject_provenance`. Serve a ``found=False`` result as 404.
    """
    doc_stmt = select(Document).where(
        Document.tenant_id == tenant_id,
        Document.document_id == document_id,
    )
    document = (await session.execute(doc_stmt)).scalar_one_or_none()

    provenance = await subject_provenance(
        session,
        tenant_id=tenant_id,
        subject_id=document_id,
        max_depth=max_depth,
    )
    payload: dict[str, Any] = {
        "document_id": document_id,
        "tenant_id": tenant_id,
        "found": document is not None,
        "generated_at": _utcnow().isoformat(),
        **provenance,
    }
    if document is None:
        payload["document"] = None
        payload["ingest_run"] = None
        payload["ingest_items"] = []
        return payload

    payload["document"] = {
        "document_id": document.document_id,
        "source_id": document.source_id,
        "source_type": document.source_type,
        "source_uri": document.source_uri,
        "title": document.title,
        "doc_type": document.doc_type,
        "language": document.language,
        "author": document.author,
        "content_sha256": document.content_sha256,
        "simhash": document.simhash,
        "etag": document.etag,
        "acl_fingerprint": document.acl_fingerprint,
        "version": document.version,
        "classification": document.classification,
        "classification_rank": document.classification_rank,
        "chunk_count": document.chunk_count,
        "token_count": document.token_count,
        "size_bytes": document.size_bytes,
        "pii_types": list(document.pii_types or []),
        "pii_redacted": document.pii_redacted,
        "is_deleted": document.is_deleted,
        "deleted_at": _iso(document.deleted_at),
        "source_modified_at": _iso(document.source_modified_at),
        "effective_from": _iso(document.effective_from),
        "effective_to": _iso(document.effective_to),
        "blob_url": document.blob_url,
        "ingest_run_id": document.ingest_run_id,
        "created_at": _iso(document.created_at),
        "updated_at": _iso(document.updated_at),
    }

    item_stmt = (
        select(IngestItem)
        .where(
            IngestItem.tenant_id == tenant_id,
            IngestItem.document_id == document_id,
        )
        .order_by(IngestItem.created_at.desc())
    )
    items = (await session.execute(item_stmt)).scalars().all()
    payload["ingest_items"] = [
        {
            "item_id": item.item_id,
            "run_id": item.run_id,
            "source_uri": item.source_uri,
            "action": item.action,
            "status": item.status,
            "reason": item.reason,
            "chunk_count": item.chunk_count,
            "token_count": item.token_count,
            "duplicates_dropped": item.duplicates_dropped,
            "duration_ms": item.duration_ms,
            "created_at": _iso(item.created_at),
        }
        for item in items
    ]

    payload["ingest_run"] = None
    if document.ingest_run_id:
        run_stmt = select(IngestRun).where(
            IngestRun.tenant_id == tenant_id,
            IngestRun.run_id == document.ingest_run_id,
        )
        run = (await session.execute(run_stmt)).scalar_one_or_none()
        if run is not None:
            payload["ingest_run"] = {
                "run_id": run.run_id,
                "source_id": run.source_id,
                "trigger": run.trigger,
                "status": run.status,
                "started_at": _iso(run.started_at),
                "finished_at": _iso(run.finished_at),
            }
    return payload


def _iso(value: datetime | None) -> str | None:
    """Render a datetime as RFC 3339, tolerating None.

    Args:
        value: The datetime, or None.

    Returns:
        An ISO 8601 string, or None.
    """
    return value.isoformat() if value is not None else None
