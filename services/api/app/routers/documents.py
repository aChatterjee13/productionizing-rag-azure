"""Corpus management: list, upload, delete, reindex and provenance.

Two things are worth knowing about this router.

**Listing is ACL-filtered, not just tenant-filtered.** The SQL predicate narrows
by tenant, tombstone and clearance rank — the parts a relational index can serve
portably — and :meth:`~ragcore.models.acl.AccessControl.permits` then applies the
role/group/deny rules in process. Doing the membership tests in SQL would need
JSON containment operators that differ between PostgreSQL and sqlite, and
:meth:`permits` is the same rule
:func:`~ragcore.vectorstore.filters.build_acl_filter` encodes, so the two agree by
construction. The query over-fetches and truncates after filtering so a page is
never short because rows were removed.

**Ingestion is a separate deployable.** ``services/ingestion`` runs on Azure
Functions and is not a dependency of this service, so the upload and reindex paths
import it lazily and answer 503 with an actionable code when it is not installed.
In the compose stack and in a workspace checkout it is installed and the path is
fully live.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, Path, Query, UploadFile, status
from sqlalchemy import select

from app.deps import (
    CurrentPrincipal,
    DbSession,
    PageLimit,
    RateLimit,
    SettingsDep,
    TenantAdmin,
)
from app.schemas.responses import DocumentSummary, LineageResponse
from ragcore.db import repositories as repo
from ragcore.db.models import Document
from ragcore.errors import ConfigError, RagError
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.document import IngestTrigger
from ragcore.observability.lineage import document_provenance
from ragcore.vectorstore.client import get_client
from ragcore.vectorstore.writer import soft_delete_document

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_DOCUMENT_ID = Path(description="Document id, scoped to the caller's tenant.")

#: Source id uploads are attributed to. Uploads have no connector, so they share
#: one logical source per tenant rather than inventing one per file.
UPLOAD_SOURCE_ID = "upload"

#: Over-fetch factor for the ACL-filtered listing. Rows removed in process must
#: not shorten a page, and a page four times the requested size is a bounded read.
_OVERFETCH = 4


class DocumentNotFoundError(RagError):
    """The document does not exist for this caller."""

    status_code = 404
    code = "document_not_found"


class IngestionUnavailableError(RagError):
    """The ingestion package is not installed in this deployment."""

    status_code = 503
    code = "ingestion_unavailable"


def _ingestion_pipeline() -> Any:
    """Import the ingestion pipeline lazily.

    Returns:
        The :mod:`ingestion.pipeline` module.

    Raises:
        IngestionUnavailableError: When ``rag-ingestion`` is not installed. It is
            a separate deployable (Azure Functions), so its absence is a
            deployment shape rather than a bug — the message says which.
    """
    try:
        from ingestion import pipeline
    except ImportError as exc:
        _log.warning("ingestion_package_missing")
        raise IngestionUnavailableError(
            "document ingestion runs in the ingestion service, which is not "
            "installed in this deployment; upload through it or install the "
            "'rag-ingestion' workspace package"
        ) from exc
    return pipeline


@router.get("", response_model=list[DocumentSummary], summary="List documents")
async def list_documents(
    principal: CurrentPrincipal,
    session: DbSession,
    limit: PageLimit,
    q: str | None = Query(default=None, description="Case-insensitive title match."),
    source_type: str | None = Query(default=None, description="Filter by source."),
    doc_type: str | None = Query(default=None, description="Filter by document type."),
    include_deleted: bool = Query(
        default=False, description="Include tombstoned documents."
    ),
) -> list[DocumentSummary]:
    """List documents the caller is cleared to see.

    Args:
        principal: The authenticated caller.
        session: Database session.
        limit: Page size.
        q: Optional case-insensitive title substring.
        source_type: Optional source-type filter.
        doc_type: Optional document-type filter.
        include_deleted: Include tombstoned rows.

    Returns:
        Document summaries, newest first.
    """
    statement = (
        select(Document)
        .where(Document.tenant_id == principal.tenant_id)
        .where(Document.classification_rank <= principal.clearance_rank())
        .order_by(Document.updated_at.desc())
        .limit(limit * _OVERFETCH)
    )
    if not include_deleted:
        statement = statement.where(Document.is_deleted.is_(False))
    if source_type:
        statement = statement.where(Document.source_type == source_type)
    if doc_type:
        statement = statement.where(Document.doc_type == doc_type)
    if q:
        statement = statement.where(Document.title.ilike(f"%{q}%"))

    rows = (await session.execute(statement)).scalars().all()
    visible = [row for row in rows if _acl_of(row).permits(principal)]
    return [DocumentSummary.from_row(row) for row in visible[:limit]]


@router.post(
    "",
    response_model=DocumentSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    principal: CurrentPrincipal,
    session: DbSession,
    settings: SettingsDep,
    _limited: RateLimit,
    file: UploadFile = File(description="The document to ingest."),
    title: str = Form(default="", description="Title; defaults to the file name."),
    doc_type: str = Form(default="document", description="Document class."),
    language: str = Form(default="en", description="Declared language."),
    classification: Classification = Form(
        default=Classification.INTERNAL, description="Sensitivity label."
    ),
    tags: list[str] = Form(default_factory=list, description="Tags to stamp."),
    allowed_roles: list[str] = Form(
        default_factory=list, description="Roles permitted to read it."
    ),
    allowed_groups: list[str] = Form(
        default_factory=list, description="Entra groups permitted to read it."
    ),
) -> DocumentSummary:
    """Ingest an uploaded file through the ordinary pipeline.

    The ACL comes from the uploading principal, never from the body: the tenant is
    the caller's, and an uploader cannot label a document above their own
    clearance, which would let them create material they are then not allowed to
    read back.

    Args:
        principal: The authenticated caller.
        session: Database session.
        settings: Active settings.
        _limited: Rate-limit gate — ingestion embeds, which is expensive.
        file: The uploaded file.
        title: Title; the file name is used when empty.
        doc_type: Document class.
        language: Declared language.
        classification: Requested sensitivity, clamped to the caller's clearance.
        tags: Tags to stamp on every chunk.
        allowed_roles: Roles permitted to read it.
        allowed_groups: Entra group object ids permitted to read it.

    Returns:
        The created document row.

    Raises:
        RagError: 413 when the file exceeds ``api_max_upload_bytes``, 400 when it
            is empty, 502 when ingestion failed.
        IngestionUnavailableError: When the ingestion package is absent.
    """
    pipeline = _ingestion_pipeline()
    payload = await file.read()
    if not payload:
        raise RagError("uploaded file is empty", code="empty_upload", status_code=400)
    if len(payload) > settings.api_max_upload_bytes:
        raise RagError(
            "uploaded file exceeds the configured maximum",
            code="upload_too_large",
            status_code=413,
            detail={"max_bytes": settings.api_max_upload_bytes},
        )

    requested = Classification(classification)
    effective = min(requested, principal.max_classification)
    if effective != requested:
        _log.warning(
            "upload_classification_clamped",
            tenant_id=principal.tenant_id,
            requested=requested.value,
            effective=effective.value,
        )
    access = AccessControl(
        tenant_id=principal.tenant_id,
        allowed_roles=list(allowed_roles),
        allowed_groups=list(allowed_groups),
        allowed_users=[],
        denied_users=[],
        classification=effective,
    )

    filename = file.filename or "upload.bin"
    try:
        outcome = await pipeline.ingest_uploaded_document(
            tenant_id=principal.tenant_id,
            source_id=UPLOAD_SOURCE_ID,
            filename=filename,
            payload=payload,
            access_control=access,
            media_type=file.content_type,
            title=title or filename,
            doc_type=doc_type,
            tags=list(tags),
            author=principal.display_name or principal.email,
            language=language,
            settings=settings,
        )
    except ConfigError:
        raise
    except Exception as exc:
        _log.exception("upload_ingest_failed", tenant_id=principal.tenant_id)
        raise RagError(
            "the document could not be ingested",
            code="ingest_failed",
            status_code=502,
            detail={"error": type(exc).__name__},
        ) from exc

    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="document.upload",
        resource_type="document",
        resource_id=outcome.document_id,
        detail={
            "chunks": outcome.chunk_count,
            "classification": effective.value,
            "size_bytes": len(payload),
        },
    )
    await session.commit()

    row = await _fetch_document(session, principal, outcome.document_id)
    _log.info(
        "document_uploaded",
        tenant_id=principal.tenant_id,
        document_id=outcome.document_id,
        chunks=outcome.chunk_count,
    )
    return DocumentSummary.from_row(row)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document",
)
async def delete_document(
    principal: TenantAdmin,
    session: DbSession,
    settings: SettingsDep,
    document_id: str = _DOCUMENT_ID,
) -> None:
    """Tombstone a document and its chunks.

    Soft delete, not purge: ``is_deleted=True`` keeps the lineage chain intact so
    ``GET /documents/{id}/lineage`` still answers for an audit after the content
    has stopped being retrievable.

    Args:
        principal: An administrator of the owning tenant.
        session: Database session.
        settings: Active settings.
        document_id: Document to remove.

    Raises:
        DocumentNotFoundError: When the caller's tenant owns no such document.
    """
    await _fetch_document(session, principal, document_id)
    client = await get_client(settings)
    chunks = await soft_delete_document(
        client,
        collection=settings.qdrant_chunks_collection,
        tenant_id=principal.tenant_id,
        document_id=document_id,
    )
    removed = await repo.mark_documents_deleted(
        session, tenant_id=principal.tenant_id, document_ids=[document_id]
    )
    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="document.delete",
        resource_type="document",
        resource_id=document_id,
        detail={"chunks_tombstoned": chunks, "rows": removed},
    )
    await session.commit()
    _log.info(
        "document_deleted",
        tenant_id=principal.tenant_id,
        document_id=document_id,
        chunks=chunks,
    )


@router.post("/{document_id}/reindex", summary="Re-ingest one document")
async def reindex_document(
    principal: TenantAdmin,
    session: DbSession,
    settings: SettingsDep,
    _limited: RateLimit,
    document_id: str = _DOCUMENT_ID,
) -> Any:
    """Re-fetch and re-index one document from its source.

    Args:
        principal: An administrator of the owning tenant.
        session: Database session.
        settings: Active settings.
        _limited: Rate-limit gate.
        document_id: Document to re-ingest.

    Returns:
        The :class:`~ragcore.models.document.IngestRunSummary` for the run.

    Raises:
        DocumentNotFoundError: When the caller's tenant owns no such document.
        RagError: 502 when the re-ingest failed.
        IngestionUnavailableError: When the ingestion package is absent.
    """
    pipeline = _ingestion_pipeline()
    row = await _fetch_document(session, principal, document_id)
    try:
        summary = await pipeline.ingest_single_document(
            tenant_id=principal.tenant_id,
            source_id=row.source_id or UPLOAD_SOURCE_ID,
            source_uri=row.source_uri,
            trigger=IngestTrigger.REINDEX,
            settings=settings,
            forced=True,
        )
    except Exception as exc:
        _log.exception("reindex_failed", document_id=document_id)
        raise RagError(
            "the document could not be re-indexed",
            code="reindex_failed",
            status_code=502,
            detail={"error": type(exc).__name__},
        ) from exc

    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="document.reindex",
        resource_type="document",
        resource_id=document_id,
        detail={"run_id": summary.run_id, "status": str(summary.status)},
    )
    await session.commit()
    return summary


@router.get(
    "/{document_id}/lineage",
    response_model=LineageResponse,
    summary="Full provenance chain",
)
async def get_lineage(
    principal: CurrentPrincipal,
    session: DbSession,
    document_id: str = _DOCUMENT_ID,
) -> LineageResponse:
    """Walk a document's upstream provenance.

    Args:
        principal: The authenticated caller.
        session: Database session.
        document_id: Document to trace.

    Returns:
        The provenance payload: the document row, its ingest run and items, the
        lineage records and the ancestor chain.

    Raises:
        DocumentNotFoundError: When the caller cannot see the document. The ACL
            check runs first, so lineage cannot be used to enumerate documents a
            caller is not cleared for.
    """
    await _fetch_document(session, principal, document_id)
    payload = await document_provenance(
        session, document_id, tenant_id=principal.tenant_id
    )
    if not payload.get("found"):
        raise DocumentNotFoundError("no such document")
    return LineageResponse(**payload)


# ------------------------------------------------------------------- helpers


def _acl_of(row: Document) -> AccessControl:
    """Rebuild a document's ACL from its flat columns.

    Args:
        row: A ``documents`` row.

    Returns:
        The reconstructed :class:`~ragcore.models.acl.AccessControl`.
    """
    return AccessControl.from_flat(
        {
            "tenant_id": row.tenant_id,
            "allowed_roles": list(row.allowed_roles or []),
            "allowed_groups": list(row.allowed_groups or []),
            "allowed_users": list(row.allowed_users or []),
            "denied_users": list(row.denied_users or []),
            "classification": row.classification,
        }
    )


async def _fetch_document(
    session: Any, principal: Principal, document_id: str
) -> Document:
    """Load one document, enforcing tenancy and the ACL.

    Args:
        session: Database session.
        principal: The caller.
        document_id: Document to load.

    Returns:
        The row.

    Raises:
        DocumentNotFoundError: When the row is absent, belongs to another tenant,
            or the caller is not permitted to read it. All three answer the same
            way on purpose: distinguishing them would confirm the existence of a
            document to someone who may not know it exists.
    """
    statement = (
        select(Document)
        .where(Document.document_id == document_id)
        .where(Document.tenant_id == principal.tenant_id)
    )
    row = (await session.execute(statement)).scalars().first()
    if row is None or not _acl_of(row).permits(principal):
        _log.info(
            "document_access_denied",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            document_id=document_id,
            existed=row is not None,
        )
        raise DocumentNotFoundError("no such document")
    return row
