"""Writes into ``rag_chunks``: batched upsert, soft delete, hard delete, tombstoning.

Deletion is soft by default. A tombstone (``is_deleted=True``) is invisible to
retrieval because :func:`ragcore.vectorstore.filters.build_acl_filter` excludes it,
but it stays queryable for lineage and for the "why did this answer change?" question
that follows every content removal. Hard delete exists for genuine purges (GDPR
erasure, a mis-ingested source) and is always tenant-scoped.

Every filter used here comes from :mod:`ragcore.vectorstore.filters` — ingestion never
builds a filter inline, even though it is not acting on behalf of a principal.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from ragcore.embeddings.base import SparseVec
from ragcore.errors import TenantMismatchError
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import ChunkPayload
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore.collections import DENSE, SPARSE, point_id_for_chunk
from ragcore.vectorstore.filters import build_tenant_filter

__all__ = [
    "ChunkPoint",
    "count_chunks",
    "hard_delete_by_filter",
    "hard_delete_document",
    "soft_delete_document",
    "soft_delete_documents",
    "tombstone_missing",
    "update_access_control",
    "upsert_chunks",
    "upsert_points",
]

_log = structlog.get_logger(__name__)


def _utcnow_iso() -> str:
    """Current time as an RFC 3339 string.

    Returns:
        The current UTC moment, formatted the way Qdrant's datetime index expects.
    """
    return datetime.now(UTC).isoformat()


class ChunkPoint(BaseModel):
    """A chunk payload paired with the vectors that index it.

    Attributes:
        payload: The flat Qdrant payload.
        dense: Dense embedding of :attr:`ChunkPayload.embed_text`.
        sparse: BM25 sparse embedding, or None for a payload-only write.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    payload: ChunkPayload = Field(description="Flat payload stored on the point.")
    dense: list[float] = Field(description="Dense embedding of the chunk's embed_text.")
    sparse: SparseVec | None = Field(
        default=None, description="BM25 sparse embedding of the chunk's embed_text."
    )

    @model_validator(mode="after")
    def _require_dense(self) -> ChunkPoint:
        """Reject a point with no dense vector.

        Returns:
            The validated point.

        Raises:
            ValueError: If ``dense`` is empty — an unvectorised point would be
                unreachable by search while still counting as indexed.
        """
        if not self.dense:
            msg = f"chunk {self.payload.chunk_id!r} has an empty dense vector"
            raise ValueError(msg)
        return self

    @property
    def point_id(self) -> str:
        """Deterministic Qdrant point id for this chunk.

        Returns:
            The UUID derived from ``payload.chunk_id``.
        """
        return point_id_for_chunk(self.payload.chunk_id)

    def to_point_struct(self) -> qm.PointStruct:
        """Render as a Qdrant point.

        Returns:
            A :class:`qdrant_client.models.PointStruct` with the named ``dense``
            vector, the named ``sparse`` vector when present, and the serialised
            payload.
        """
        vectors: dict[str, list[float] | qm.SparseVector] = {DENSE: list(self.dense)}
        if self.sparse is not None and self.sparse.indices:
            vectors[SPARSE] = qm.SparseVector(
                indices=list(self.sparse.indices),
                values=list(self.sparse.values),
            )
        return qm.PointStruct(
            id=self.point_id,
            vector=vectors,
            payload=self.payload.to_qdrant_payload(),
        )


def _batched(
    items: Sequence[qm.PointStruct], size: int
) -> Iterator[list[qm.PointStruct]]:
    """Split a point sequence into fixed-size batches.

    Args:
        items: Points to batch.
        size: Maximum points per batch.

    Yields:
        Successive batches, the last possibly shorter.
    """
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


async def upsert_points(
    client: AsyncQdrantClient,
    *,
    collection: str,
    points: Sequence[qm.PointStruct],
    batch_size: int | None = None,
    wait: bool = True,
    settings: Settings | None = None,
) -> int:
    """Upsert arbitrary points in batches.

    Batching matters at ingest scale: a single request carrying thousands of 1024-d
    vectors will hit the body-size limit or time out, and a failure then costs the
    whole document.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        points: Points to write. Ids must already be UUIDs or unsigned integers.
        batch_size: Points per request. Defaults to
            ``settings.qdrant_upsert_batch_size``.
        wait: Block until the write is applied. Ingestion wants True so a subsequent
            tombstone pass observes its own writes.
        settings: Settings supplying the default batch size.

    Returns:
        The number of points written.
    """
    if not points:
        return 0
    cfg = settings or get_settings()
    size = batch_size or cfg.qdrant_upsert_batch_size
    written = 0
    for batch in _batched(points, size):
        await client.upsert(collection_name=collection, points=batch, wait=wait)
        written += len(batch)
    _log.info(
        "qdrant.upsert",
        collection=collection,
        points=written,
        batch_size=size,
    )
    return written


async def upsert_chunks(
    client: AsyncQdrantClient,
    *,
    collection: str,
    chunks: Sequence[ChunkPoint],
    batch_size: int | None = None,
    wait: bool = True,
    settings: Settings | None = None,
) -> int:
    """Upsert chunk payloads together with their dense and sparse vectors.

    All chunks in one call must belong to one tenant; mixing tenants in a batch is
    rejected rather than written, because a batch is the unit that gets retried and a
    partially-correct retry is how cross-tenant rows appear.

    Args:
        client: Async Qdrant client.
        collection: Target collection, normally
            ``settings.qdrant_chunks_collection``.
        chunks: Chunk points to write.
        batch_size: Points per request. Defaults to
            ``settings.qdrant_upsert_batch_size``.
        wait: Block until the write is applied.
        settings: Settings supplying the default batch size.

    Returns:
        The number of chunks written.

    Raises:
        TenantMismatchError: If the batch spans more than one tenant.
    """
    if not chunks:
        return 0
    tenants = {chunk.payload.tenant_id for chunk in chunks}
    if len(tenants) > 1:
        raise TenantMismatchError(
            "refusing to upsert a batch spanning multiple tenants",
            resource=collection,
            detail={"tenant_count": len(tenants)},
        )
    return await upsert_points(
        client,
        collection=collection,
        points=[chunk.to_point_struct() for chunk in chunks],
        batch_size=batch_size,
        wait=wait,
        settings=settings,
    )


async def count_chunks(
    client: AsyncQdrantClient,
    *,
    collection: str,
    qfilter: qm.Filter,
    exact: bool = True,
) -> int:
    """Count points matching a filter.

    Args:
        client: Async Qdrant client.
        collection: Collection to count in.
        qfilter: Filter restricting the count. Always tenant-scoped in this module.
        exact: Request an exact count rather than an estimate.

    Returns:
        The number of matching points.
    """
    result = await client.count(
        collection_name=collection, count_filter=qfilter, exact=exact
    )
    return int(result.count)


async def soft_delete_documents(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    document_ids: Sequence[str],
    run_id: str | None = None,
    wait: bool = True,
) -> int:
    """Tombstone every live chunk of the given documents.

    Sets ``is_deleted=True`` (plus ``updated_at`` and, when supplied,
    ``ingest_run_id``) via a filtered ``set_payload``, so the points, their vectors
    and their lineage survive while retrieval stops returning them.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        tenant_id: Tenant the documents belong to. Always applied to the filter.
        document_ids: Documents to tombstone.
        run_id: Ingest run to attribute the tombstone to.
        wait: Block until the write is applied.

    Returns:
        The number of chunks tombstoned. Zero when the documents were already
        tombstoned or never existed.
    """
    if not document_ids:
        return 0
    qfilter = build_tenant_filter(
        tenant_id,
        document_ids=list(document_ids),
        include_deleted=False,
    )
    affected = await count_chunks(client, collection=collection, qfilter=qfilter)
    if affected == 0:
        _log.info(
            "qdrant.soft_delete.noop",
            collection=collection,
            tenant_id=tenant_id,
            documents=len(document_ids),
        )
        return 0

    payload: dict[str, object] = {"is_deleted": True, "updated_at": _utcnow_iso()}
    if run_id is not None:
        payload["ingest_run_id"] = run_id
    await client.set_payload(
        collection_name=collection,
        payload=payload,
        points=qm.FilterSelector(filter=qfilter),
        wait=wait,
    )
    _log.info(
        "qdrant.soft_delete",
        collection=collection,
        tenant_id=tenant_id,
        documents=len(document_ids),
        chunks=affected,
        run_id=run_id,
    )
    return affected


async def soft_delete_document(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    document_id: str,
    run_id: str | None = None,
    wait: bool = True,
) -> int:
    """Tombstone every live chunk of one document.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        tenant_id: Tenant the document belongs to.
        document_id: Document to tombstone.
        run_id: Ingest run to attribute the tombstone to.
        wait: Block until the write is applied.

    Returns:
        The number of chunks tombstoned.
    """
    return await soft_delete_documents(
        client,
        collection=collection,
        tenant_id=tenant_id,
        document_ids=[document_id],
        run_id=run_id,
        wait=wait,
    )


async def hard_delete_document(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    document_id: str,
    wait: bool = True,
) -> int:
    """Permanently remove every chunk of one document.

    Irreversible. Use :func:`soft_delete_document` for ordinary content removal;
    this is for erasure requests and for purging a mis-ingested source.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        tenant_id: Tenant the document belongs to. Always applied to the filter, so a
            wrong ``document_id`` can never delete another tenant's data.
        document_id: Document to purge.
        wait: Block until the write is applied.

    Returns:
        The number of chunks removed.
    """
    qfilter = build_tenant_filter(tenant_id, document_ids=[document_id])
    affected = await count_chunks(client, collection=collection, qfilter=qfilter)
    if affected == 0:
        return 0
    await hard_delete_by_filter(
        client, collection=collection, qfilter=qfilter, wait=wait
    )
    _log.warning(
        "qdrant.hard_delete.document",
        collection=collection,
        tenant_id=tenant_id,
        document_id=document_id,
        chunks=affected,
    )
    return affected


async def hard_delete_by_filter(
    client: AsyncQdrantClient,
    *,
    collection: str,
    qfilter: qm.Filter,
    wait: bool = True,
) -> None:
    """Permanently remove every point matching a filter.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        qfilter: Filter selecting the points to purge. Build it with
            :func:`ragcore.vectorstore.filters.build_tenant_filter` so it is
            tenant-scoped; an unscoped filter here would delete across tenants.
        wait: Block until the write is applied.

    Raises:
        ValueError: If the filter has no ``must`` clause, which would make it match
            the whole collection.
    """
    if not qfilter.must:
        msg = (
            "refusing a hard delete with no `must` clause: build the filter with "
            "build_tenant_filter so the tenant boundary is always present"
        )
        raise ValueError(msg)
    await client.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(filter=qfilter),
        wait=wait,
    )


async def update_access_control(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    document_id: str,
    access_control: AccessControl,
    run_id: str | None = None,
    wait: bool = True,
) -> int:
    """Rewrite only the ACL fields of a document's chunks.

    The ``ACL_ONLY`` reindex path: when a SharePoint item's permissions change but its
    bytes do not, re-embedding is pure waste. Writes the six flat ACL fields (plus
    ``classification_rank``) produced by
    :meth:`ragcore.models.acl.AccessControl.to_flat`, which is the only supported way
    to flatten an ACL — so the flat fields cannot drift from their nested source.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        tenant_id: Tenant the document belongs to.
        document_id: Document whose chunks are being re-permissioned.
        access_control: The new access control.
        run_id: Ingest run to attribute the rewrite to.
        wait: Block until the write is applied.

    Returns:
        The number of chunks updated.

    Raises:
        TenantMismatchError: If ``access_control.tenant_id`` disagrees with
            ``tenant_id``.
    """
    if access_control.tenant_id != tenant_id:
        raise TenantMismatchError(
            "access_control tenant disagrees with the requested tenant",
            expected=tenant_id,
            actual=access_control.tenant_id,
            resource=f"{collection}/{document_id}",
        )
    qfilter = build_tenant_filter(tenant_id, document_ids=[document_id])
    affected = await count_chunks(client, collection=collection, qfilter=qfilter)
    if affected == 0:
        return 0
    payload: dict[str, object] = {
        **access_control.to_flat(),
        "updated_at": _utcnow_iso(),
    }
    if run_id is not None:
        payload["ingest_run_id"] = run_id
    await client.set_payload(
        collection_name=collection,
        payload=payload,
        points=qm.FilterSelector(filter=qfilter),
        wait=wait,
    )
    _log.info(
        "qdrant.acl_rewrite",
        collection=collection,
        tenant_id=tenant_id,
        document_id=document_id,
        chunks=affected,
        classification=access_control.classification.value,
        run_id=run_id,
    )
    return affected


async def _live_document_ids(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    source_id: str | None,
    page_size: int,
) -> set[str]:
    """Scroll the collection for the distinct document ids currently indexed.

    Args:
        client: Async Qdrant client.
        collection: Collection to scan.
        tenant_id: Tenant to scope the scan to.
        source_id: Optional source config to scope the scan to.
        page_size: Points per scroll page.

    Returns:
        The set of ``document_id`` values with at least one live chunk.
    """
    qfilter = build_tenant_filter(tenant_id, source_id=source_id, include_deleted=False)
    seen: set[str] = set()
    offset: object | None = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=qfilter,
            limit=page_size,
            offset=offset,  # type: ignore[arg-type]
            with_payload=["document_id"],
            with_vectors=False,
        )
        for record in records:
            document_id = (record.payload or {}).get("document_id")
            if isinstance(document_id, str) and document_id:
                seen.add(document_id)
        if offset is None or not records:
            break
    return seen


async def tombstone_missing(
    client: AsyncQdrantClient,
    *,
    collection: str,
    tenant_id: str,
    manifest_document_ids: Sequence[str],
    source_id: str | None = None,
    run_id: str | None = None,
    page_size: int | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Soft-delete chunks of documents that the current manifest no longer lists.

    This is how a deletion at the source becomes a deletion in the index. A delta
    connector reports what it *saw*; nothing reports what vanished. So the run
    compares the manifest it just built against what is actually indexed for that
    tenant and source, and tombstones the difference.

    Discovery is a scroll-and-diff rather than a ``MatchExcept`` on the whole manifest:
    the filter form degrades badly at tens of thousands of documents, and the diff
    lets the run report exactly which documents disappeared.

    A guard worth knowing about: an **empty** ``manifest_document_ids`` tombstones
    everything indexed for that tenant/source. That is correct when a source really
    was emptied, but it is also what a broken connector looks like, so the call is
    logged at warning level with the count.

    Args:
        client: Async Qdrant client.
        collection: Target collection.
        tenant_id: Tenant the run belongs to. Always applied.
        manifest_document_ids: Every ``document_id`` the current manifest still lists
            as live (``IngestManifest.entries`` minus its tombstones).
        source_id: Restrict to one source config. Strongly recommended: without it a
            per-source run would tombstone documents owned by the tenant's *other*
            sources.
        run_id: Ingest run to attribute the tombstones to.
        page_size: Points per scroll page. Defaults to ``settings.ingest_batch_size``.
        settings: Settings supplying the default page size.

    Returns:
        The document ids that were tombstoned, sorted.
    """
    cfg = settings or get_settings()
    indexed = await _live_document_ids(
        client,
        collection=collection,
        tenant_id=tenant_id,
        source_id=source_id,
        page_size=page_size or cfg.ingest_batch_size,
    )
    missing = sorted(indexed - set(manifest_document_ids))
    if not missing:
        _log.info(
            "qdrant.tombstone_missing.none",
            collection=collection,
            tenant_id=tenant_id,
            source_id=source_id,
            indexed_documents=len(indexed),
        )
        return []

    if not manifest_document_ids:
        _log.warning(
            "qdrant.tombstone_missing.empty_manifest",
            collection=collection,
            tenant_id=tenant_id,
            source_id=source_id,
            documents=len(missing),
            hint="manifest listed no live documents; verify the connector ran",
        )

    chunks = await soft_delete_documents(
        client,
        collection=collection,
        tenant_id=tenant_id,
        document_ids=missing,
        run_id=run_id,
    )
    _log.info(
        "qdrant.tombstone_missing",
        collection=collection,
        tenant_id=tenant_id,
        source_id=source_id,
        documents=len(missing),
        chunks=chunks,
        run_id=run_id,
    )
    return missing
