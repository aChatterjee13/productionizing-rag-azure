"""Embed, dedupe, write to Qdrant, and record lineage.

This is the write end of the pipeline and the place where idempotency is actually
enforced:

* **Deterministic, readable chunk ids.** A chunk is
  ``f"{document_id}::{index:04d}"`` (Addendum I), and the UUID Qdrant needs is
  derived from it by :class:`ragcore.vectorstore.writer.ChunkPoint` via
  ``point_id_for_chunk``. Re-ingesting the same document therefore overwrites the
  same points instead of accumulating duplicates, and a seeded fixture and a real
  ingestion run address the same points.
* **Stale-chunk pruning.** After upserting a document's chunks the writer purges any
  remaining points for that document that are not in the new id set, so a document
  that shrank from 12 chunks to 7 does not leave 5 orphans behind.
* **Dedupe with audited drops.** Exact hashes and simhashes are compared across the
  whole document set of a run, and across the tenant's existing corpus via the
  ``content_sha256`` payload index. Every drop is returned with a reason — nothing is
  discarded silently.
* **Lineage.** One record per chunk (``chunk -> document -> source_uri``) plus one per
  document, which is what makes ``GET /documents/{id}/lineage`` able to walk the chain.

Every Qdrant write goes through :mod:`ragcore.vectorstore.writer`, and every filter
comes from :mod:`ragcore.vectorstore.filters` — ingestion is not acting on behalf of a
principal, so it uses ``build_tenant_filter`` rather than ``build_acl_filter``, but it
never hand-rolls the tenant boundary. The one composed filter in this module (the
content-hash dedupe probe) starts from ``build_tenant_filter`` and only adds a
non-security clause on top of it.

ACL-only reindex never re-embeds: :meth:`RunUpserter.acl_only_reindex` delegates to
``writer.update_access_control``, which rewrites the flat ACL payload fields of a
document's existing points in a single ``set_payload`` call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models as qm

from ingestion.chunk import ChunkDraft
from ragcore.dedupe import content_sha256, hamming64, simhash_hex
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import ChunkPayload
from ragcore.models.document import ParsedDocument
from ragcore.settings import Settings
from ragcore.vectorstore import writer as vector_writer
from ragcore.vectorstore.filters import build_tenant_filter
from ragcore.vectorstore.writer import ChunkPoint

__all__ = [
    "ChunkWriter",
    "DocumentUpsertResult",
    "NullChunkWriter",
    "QdrantChunkWriter",
    "RunUpserter",
    "UpsertError",
    "chunk_id_for",
    "get_chunk_writer",
]

_log = get_logger(__name__)

#: Drop reasons reported in :attr:`DocumentUpsertResult.drop_reasons`.
DROP_EXACT_RUN = "duplicate_exact_run"
DROP_EXACT_CORPUS = "duplicate_exact_corpus"
DROP_NEAR = "duplicate_simhash"
DROP_EMPTY = "empty_after_normalisation"

#: Hex characters per simhash band. Four bands of four characters give cheap LSH
#: candidate generation, so run-level near-duplicate detection is not quadratic.
_BAND_WIDTH = 4


class UpsertError(RagError):
    """Writing chunks to the vector store failed."""

    status_code = 502
    code = "upsert_error"


def chunk_id_for(document_id: str, chunk_index: int) -> str:
    """Derive the logical chunk id for one chunk position.

    Chunk ids are deliberately opaque-but-readable strings so a golden eval item can
    name one. The UUID Qdrant requires is derived from this value by
    :attr:`ragcore.vectorstore.writer.ChunkPoint.point_id`; nothing outside the vector
    store ever handles a point id.

    Args:
        document_id: Owning document id.
        chunk_index: Zero-based chunk position after dedupe.

    Returns:
        ``"<document_id>::<index zero-padded to four digits>"``.
    """
    return f"{document_id}::{chunk_index:04d}"


class DocumentUpsertResult(BaseModel):
    """What writing one document's chunks actually did."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Document that was written.")
    chunks_written: int = Field(default=0, ge=0, description="Points upserted.")
    chunks_deleted: int = Field(
        default=0, ge=0, description="Stale points pruned for this document."
    )
    tokens_embedded: int = Field(
        default=0, ge=0, description="Estimated tokens sent to the embedder."
    )
    duplicates_dropped: int = Field(
        default=0, ge=0, description="Chunks dropped by dedupe."
    )
    drop_reasons: dict[str, int] = Field(
        default_factory=dict, description="Drop reason -> count, for audit."
    )
    chunk_ids: list[str] = Field(
        default_factory=list, description="Chunk ids written, in order."
    )


@runtime_checkable
class ChunkWriter(Protocol):
    """The vector-store operations ingestion needs.

    Every method is tenant-scoped by construction: ``tenant_id`` is a required
    argument and is always part of the Qdrant filter, so no ingestion write can reach
    another tenant's points.
    """

    async def upsert_chunks(
        self, payloads: Sequence[ChunkPayload], vectors: Sequence[Any]
    ) -> int:
        """Upsert chunk points with their dense and sparse vectors.

        Args:
            payloads: Chunk payloads, one per vector.
            vectors: ``ragcore.embeddings.Embedded`` values aligned with ``payloads``.

        Returns:
            The number of points written.
        """
        ...

    async def prune_document(
        self, *, tenant_id: str, document_id: str, keep_chunk_ids: Collection[str]
    ) -> int:
        """Delete a document's points that are not in the new id set.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document whose points to prune.
            keep_chunk_ids: Chunk ids that must survive.

        Returns:
            The number of pruned points.
        """
        ...

    async def soft_delete_document(
        self, *, tenant_id: str, document_id: str, run_id: str
    ) -> int:
        """Tombstone every point of a document.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to tombstone.
            run_id: Run performing the deletion.

        Returns:
            The number of points marked deleted.
        """
        ...

    async def rewrite_access_control(
        self,
        *,
        tenant_id: str,
        document_id: str,
        access_control: AccessControl,
        run_id: str | None = None,
    ) -> int:
        """Rewrite only the flat ACL payload fields of a document's points.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to rewrite.
            access_control: The new ACL.
            run_id: Run to attribute the rewrite to.

        Returns:
            The number of points affected.
        """
        ...

    async def find_by_content_hash(
        self, *, tenant_id: str, hashes: Sequence[str]
    ) -> dict[str, str]:
        """Find existing chunks whose text hash matches any of ``hashes``.

        Args:
            tenant_id: Owning tenant id.
            hashes: Chunk text hashes to look for.

        Returns:
            A mapping of ``content_sha256 -> document_id`` for hashes already indexed.
        """
        ...

    async def close(self) -> None:
        """Release the underlying client."""
        ...


class QdrantChunkWriter:
    """The :class:`ChunkWriter` implementation, delegating to ragcore.

    Every write is a call into :mod:`ragcore.vectorstore.writer` so the guarantees
    documented there — a batch spanning tenants is refused, soft delete is the
    default, an unscoped hard delete is impossible — hold for ingestion too.
    """

    def __init__(
        self,
        client: Any,
        settings: Settings,
        *,
        collection: str,
        owns_client: bool = False,
    ) -> None:
        """Bind the writer to a Qdrant client.

        Args:
            client: An ``AsyncQdrantClient``.
            settings: Process settings supplying batch and page sizes.
            collection: Chunk collection name, from
                ``settings.qdrant_chunks_collection``.
            owns_client: Whether :meth:`close` should close the client. The shared
                client from ``ragcore.vectorstore.get_client`` is cached per endpoint
                and closed at process shutdown, so this is normally False.
        """
        self._client = client
        self._settings = settings
        self._collection = collection
        self._owns_client = owns_client

    async def upsert_chunks(
        self, payloads: Sequence[ChunkPayload], vectors: Sequence[Any]
    ) -> int:
        """Upsert chunk points in batches.

        Args:
            payloads: Chunk payloads.
            vectors: ``Embedded`` vectors aligned with ``payloads``.

        Returns:
            The number of points written.

        Raises:
            UpsertError: If the payload and vector counts disagree, or the vector
                store rejects the write.
        """
        if len(payloads) != len(vectors):
            msg = "payload and vector counts must match"
            raise UpsertError(
                msg, detail={"payloads": len(payloads), "vectors": len(vectors)}
            )
        if not payloads:
            return 0

        try:
            points = [
                ChunkPoint(
                    payload=payload,
                    dense=list(vector.dense),
                    sparse=getattr(vector, "sparse", None),
                )
                for payload, vector in zip(payloads, vectors, strict=True)
            ]
            return await vector_writer.upsert_chunks(
                self._client,
                collection=self._collection,
                chunks=points,
                settings=self._settings,
            )
        except Exception as exc:
            msg = "vector-store upsert failed"
            raise UpsertError(
                msg,
                detail={"collection": self._collection, "error": type(exc).__name__},
            ) from exc

    async def prune_document(
        self, *, tenant_id: str, document_id: str, keep_chunk_ids: Collection[str]
    ) -> int:
        """Purge a document's points outside the surviving id set.

        The survivors were just overwritten in place, so what is left over is a chunk
        *position* the document no longer has. That is not "content deleted at
        source" — it is a stale row of a document being replaced — so it is hard
        deleted rather than tombstoned. Document-level removal still goes through
        :meth:`soft_delete_document`.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to prune.
            keep_chunk_ids: Chunk ids that must survive.

        Returns:
            The number of points deleted.
        """
        indexed = await self._indexed_chunk_ids(tenant_id, document_id)
        stale = sorted(indexed - set(keep_chunk_ids))
        if not stale:
            return 0
        await vector_writer.hard_delete_by_filter(
            self._client,
            collection=self._collection,
            qfilter=build_tenant_filter(
                tenant_id, document_ids=[document_id], chunk_ids=stale
            ),
        )
        _log.info(
            "upsert.pruned",
            tenant_id=tenant_id,
            document_id=document_id,
            points=len(stale),
        )
        return len(stale)

    async def soft_delete_document(
        self, *, tenant_id: str, document_id: str, run_id: str
    ) -> int:
        """Tombstone a document's points rather than purging them.

        Soft deletion keeps the audit trail: the chunk is still there for lineage and
        forensics, but ``build_acl_filter`` excludes ``is_deleted == True`` so it can
        never be retrieved.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to tombstone.
            run_id: Run performing the deletion.

        Returns:
            The number of points affected.
        """
        return await vector_writer.soft_delete_document(
            self._client,
            collection=self._collection,
            tenant_id=tenant_id,
            document_id=document_id,
            run_id=run_id,
        )

    async def rewrite_access_control(
        self,
        *,
        tenant_id: str,
        document_id: str,
        access_control: AccessControl,
        run_id: str | None = None,
    ) -> int:
        """Rewrite the flat ACL fields of a document's points.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to rewrite.
            access_control: The new ACL.
            run_id: Run to attribute the rewrite to.

        Returns:
            The number of points affected.

        Raises:
            UpsertError: If the ACL belongs to another tenant.
        """
        try:
            return await vector_writer.update_access_control(
                self._client,
                collection=self._collection,
                tenant_id=tenant_id,
                document_id=document_id,
                access_control=access_control,
                run_id=run_id,
            )
        except RagError:
            raise
        except Exception as exc:
            msg = "ACL rewrite failed"
            raise UpsertError(
                msg,
                detail={"document_id": document_id, "error": type(exc).__name__},
            ) from exc

    async def find_by_content_hash(
        self, *, tenant_id: str, hashes: Sequence[str]
    ) -> dict[str, str]:
        """Look up already-indexed chunk text hashes within one tenant.

        The filter is ``build_tenant_filter`` plus one non-security clause on the
        ``content_sha256`` payload index: the tenant boundary and the tombstone
        exclusion still come from :mod:`ragcore.vectorstore.filters`.

        Args:
            tenant_id: Owning tenant id.
            hashes: Chunk text hashes to probe.

        Returns:
            A mapping of hash to the document that already owns it.
        """
        wanted = [value for value in dict.fromkeys(hashes) if value]
        if not wanted:
            return {}
        base = build_tenant_filter(tenant_id, include_deleted=False)
        qfilter = qm.Filter(
            must=[
                *(base.must or []),
                qm.FieldCondition(key="content_sha256", match=qm.MatchAny(any=wanted)),
            ]
        )
        found: dict[str, str] = {}
        async for payload in self._scroll(qfilter, ["content_sha256", "document_id"]):
            digest = str(payload.get("content_sha256") or "")
            document_id = str(payload.get("document_id") or "")
            if digest and document_id:
                found.setdefault(digest, document_id)
        return found

    async def _indexed_chunk_ids(self, tenant_id: str, document_id: str) -> set[str]:
        """List the chunk ids currently indexed for one document.

        Tombstoned points are included: a position the document no longer has should
        be purged whether or not a previous run already hid it.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document to scan.

        Returns:
            The set of logical chunk ids stored for that document.
        """
        qfilter = build_tenant_filter(tenant_id, document_ids=[document_id])
        found: set[str] = set()
        async for payload in self._scroll(qfilter, ["chunk_id"]):
            chunk_id = str(payload.get("chunk_id") or "")
            if chunk_id:
                found.add(chunk_id)
        return found

    async def _scroll(
        self, qfilter: qm.Filter, fields: Sequence[str]
    ) -> AsyncIterator[dict[str, Any]]:
        """Page through a filtered scroll, yielding payloads.

        Args:
            qfilter: Filter selecting the points, always tenant-scoped.
            fields: Payload keys to fetch.

        Yields:
            One payload mapping per matching point.
        """
        offset: Any = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=self._settings.ingest_batch_size,
                with_payload=list(fields),
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                yield record.payload or {}
            if offset is None or not records:
                break

    async def close(self) -> None:
        """Close the Qdrant client when this writer owns it."""
        if self._owns_client and self._client is not None:
            await self._client.close()


class NullChunkWriter:
    """A :class:`ChunkWriter` that touches nothing.

    Backs ``--dry-run``: the run enumerates, fetches, parses, chunks and dedupes
    exactly as a real one does, but no Qdrant client is ever opened, so a dry run
    works against a machine that has no vector store at all.
    """

    async def upsert_chunks(
        self, payloads: Sequence[ChunkPayload], vectors: Sequence[Any]
    ) -> int:
        """Pretend to write.

        Args:
            payloads: Chunk payloads that would have been written.
            vectors: Their vectors.

        Returns:
            Always 0 — a dry run writes nothing, and reporting otherwise would make
            the run summary lie.
        """
        return 0

    async def prune_document(
        self, *, tenant_id: str, document_id: str, keep_chunk_ids: Collection[str]
    ) -> int:
        """Prune nothing.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document that would have been pruned.
            keep_chunk_ids: Chunk ids that would have survived.

        Returns:
            Always 0.
        """
        return 0

    async def soft_delete_document(
        self, *, tenant_id: str, document_id: str, run_id: str
    ) -> int:
        """Tombstone nothing.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document that would have been tombstoned.
            run_id: Run that would have performed it.

        Returns:
            Always 0.
        """
        return 0

    async def rewrite_access_control(
        self,
        *,
        tenant_id: str,
        document_id: str,
        access_control: AccessControl,
        run_id: str | None = None,
    ) -> int:
        """Rewrite nothing.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document whose ACL would have been rewritten.
            access_control: The ACL that would have been written.
            run_id: Run that would have performed it.

        Returns:
            Always 0.
        """
        return 0

    async def find_by_content_hash(
        self, *, tenant_id: str, hashes: Sequence[str]
    ) -> dict[str, str]:
        """Report no corpus duplicates.

        Args:
            tenant_id: Owning tenant id.
            hashes: Hashes that would have been probed.

        Returns:
            An empty mapping: a dry run must not depend on a live index, so
            cross-document dedupe is limited to the documents of this run.
        """
        return {}

    async def close(self) -> None:
        """Nothing to release."""
        return None


async def get_chunk_writer(settings: Settings) -> ChunkWriter:
    """Build the chunk writer for this process.

    Args:
        settings: Process settings.

    Returns:
        A writer bound to the shared Qdrant client and to
        ``settings.qdrant_chunks_collection``.

    Raises:
        UpsertError: If Qdrant cannot be reached at all.
    """
    from ragcore.vectorstore.collections import get_client

    try:
        client = await get_client(settings)
    except Exception as exc:
        msg = "could not open a Qdrant client"
        raise UpsertError(msg, detail={"error": type(exc).__name__}) from exc

    return QdrantChunkWriter(
        client,
        settings,
        collection=settings.qdrant_chunks_collection,
        owns_client=False,
    )


class RunUpserter:
    """Embeds, dedupes and writes every document of one ingestion run.

    Dedupe state lives on the instance, which is why the class exists: exact hashes
    and simhash bands accumulate across the whole document set of the run, so a
    paragraph repeated in twelve policy PDFs is indexed once and the other eleven
    copies are reported as audited drops.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        writer: ChunkWriter,
        run_id: str,
        embedder: Any | None = None,
        dry_run: bool = False,
    ) -> None:
        """Create an upserter for one run.

        Args:
            settings: Process settings.
            writer: Vector-store writer.
            run_id: Ingestion run id stamped on every chunk.
            embedder: An ``EmbeddingProvider``; resolved lazily when omitted.
            dry_run: Plan and count, but perform no vector-store or database write.
        """
        self.settings = settings
        self.writer = writer
        self.run_id = run_id
        self.dry_run = dry_run
        self._embedder = embedder
        self._seen_hashes: dict[str, str] = {}
        self._bands: dict[str, list[tuple[str, str]]] = {}

    # ---------------------------------------------------------------- embedding
    def _provider(self) -> Any:
        """Resolve the embedding provider once.

        Returns:
            The shared ``EmbeddingProvider``.

        Raises:
            UpsertError: If ``ragcore.embeddings`` is unavailable — chunks cannot be
                indexed without vectors, so this is fatal rather than degradable.
        """
        if self._embedder is None:
            try:
                from ragcore.embeddings import get_embedding_provider
            except ImportError as exc:  # pragma: no cover - hard dependency
                msg = "ragcore.embeddings is required to index chunks"
                raise UpsertError(msg) from exc
            self._embedder = get_embedding_provider(self.settings)
        return self._embedder

    # ------------------------------------------------------------------ dedupe
    def _register(self, digest: str, simhash: str, chunk_id: str) -> None:
        """Remember a kept chunk so later chunks can be compared against it.

        Args:
            digest: Exact content hash of the chunk text.
            simhash: Hex simhash of the chunk text, or "" when too short.
            chunk_id: The chunk id kept.
        """
        self._seen_hashes[digest] = chunk_id
        if not simhash:
            return
        for band in _bands_for(simhash):
            self._bands.setdefault(band, []).append((simhash, chunk_id))

    def _near_duplicate_of(self, simhash: str) -> str | None:
        """Find a kept chunk within the configured Hamming distance.

        Args:
            simhash: Hex simhash of the candidate chunk.

        Returns:
            The id of the near-duplicate chunk, or None.
        """
        if not simhash:
            return None
        limit = self.settings.dedupe_max_distance
        checked: set[str] = set()
        for band in _bands_for(simhash):
            for other, chunk_id in self._bands.get(band, ()):
                if other in checked:
                    continue
                checked.add(other)
                if hamming64(int(simhash, 16), int(other, 16)) <= limit:
                    return chunk_id
        return None

    # ------------------------------------------------------------------- writes
    async def upsert_document(
        self,
        parsed: ParsedDocument,
        drafts: Sequence[ChunkDraft],
        *,
        version: int = 1,
        session: Any | None = None,
    ) -> DocumentUpsertResult:
        """Embed, dedupe and write one document's chunks.

        Args:
            parsed: The enriched, parsed document.
            drafts: Chunks produced by :func:`ingestion.chunk.chunk_document`.
            version: Document version this write represents.
            session: Optional database session; when given, lineage records are
                written through it. The caller owns the transaction.

        Returns:
            Counters and drop reasons for this document.
        """
        result = DocumentUpsertResult(document_id=parsed.document_id)
        kept: list[tuple[ChunkDraft, str, str, str]] = []

        corpus_hashes = await self._corpus_hashes(parsed, drafts)

        for draft in drafts:
            text = draft.text.strip()
            if not text:
                _bump(result, DROP_EMPTY, duplicate=False)
                continue
            digest = content_sha256(text)
            existing_owner = corpus_hashes.get(digest)
            if digest in self._seen_hashes:
                _bump(result, DROP_EXACT_RUN)
                continue
            if existing_owner is not None and existing_owner != parsed.document_id:
                _bump(result, DROP_EXACT_CORPUS)
                continue
            simhash = (
                simhash_hex(text)
                if len(text) >= self.settings.dedupe_min_chunk_chars
                else ""
            )
            if self.settings.dedupe_enabled and self._near_duplicate_of(simhash):
                _bump(result, DROP_NEAR)
                continue
            chunk_id = chunk_id_for(parsed.document_id, len(kept))
            kept.append((draft, chunk_id, digest, simhash))
            self._register(digest, simhash, chunk_id)

        if not kept:
            _log.info(
                "upsert.no_chunks",
                document_id=parsed.document_id,
                tenant_id=parsed.tenant_id,
                drop_reasons=result.drop_reasons,
            )
            if not self.dry_run:
                result.chunks_deleted = await self.writer.prune_document(
                    tenant_id=parsed.tenant_id,
                    document_id=parsed.document_id,
                    keep_chunk_ids=(),
                )
            return result

        payloads = [
            self._payload(parsed, draft, chunk_id, digest, simhash, index, version)
            for index, (draft, chunk_id, digest, simhash) in enumerate(kept)
        ]
        result.chunk_ids = [payload.chunk_id for payload in payloads]
        result.tokens_embedded = sum(payload.token_count for payload in payloads)

        if self.dry_run:
            _log.info(
                "upsert.dry_run",
                document_id=parsed.document_id,
                tenant_id=parsed.tenant_id,
                chunks=len(payloads),
            )
            return result

        texts = [
            payload.embed_text
            if self.settings.retrieval_contextual_header_enabled
            else payload.text
            for payload in payloads
        ]
        provider = self._provider()
        vectors = await provider.embed_documents(texts)

        result.chunks_written = await self.writer.upsert_chunks(payloads, vectors)
        result.chunks_deleted = await self.writer.prune_document(
            tenant_id=parsed.tenant_id,
            document_id=parsed.document_id,
            keep_chunk_ids=result.chunk_ids,
        )

        if session is not None:
            await self._record_lineage(session, parsed, payloads)
        return result

    async def _corpus_hashes(
        self, parsed: ParsedDocument, drafts: Sequence[ChunkDraft]
    ) -> dict[str, str]:
        """Probe the tenant's corpus for chunk texts that are already indexed.

        Within one process the run-level sets catch duplicates; across the fan-out of
        a Durable Functions run they do not, so the ``content_sha256`` payload index
        is the cross-invocation authority.

        Args:
            parsed: The document being written.
            drafts: Its candidate chunks.

        Returns:
            A mapping of hash to owning document id, excluding this document.
        """
        if not self.settings.dedupe_enabled:
            return {}
        digests = [content_sha256(draft.text.strip()) for draft in drafts if draft.text]
        try:
            found = await self.writer.find_by_content_hash(
                tenant_id=parsed.tenant_id, hashes=digests
            )
        except Exception as exc:
            _log.warning(
                "upsert.hash_probe_failed",
                document_id=parsed.document_id,
                error=type(exc).__name__,
            )
            return {}
        return {
            digest: owner
            for digest, owner in found.items()
            if owner != parsed.document_id
        }

    def _payload(
        self,
        parsed: ParsedDocument,
        draft: ChunkDraft,
        chunk_id: str,
        digest: str,
        simhash: str,
        index: int,
        version: int,
    ) -> ChunkPayload:
        """Build the Qdrant payload for one chunk.

        Args:
            parsed: The enriched document.
            draft: The chunk draft.
            chunk_id: Logical chunk id.
            digest: Exact content hash of the chunk text.
            simhash: Hex simhash, or "".
            index: Final chunk index after dedupe.
            version: Document version.

        Returns:
            A :class:`ChunkPayload` whose ACL fields come from the document's ACL via
            ``from_access_control`` — the only supported way to populate them.
        """
        pii_types = list(parsed.metadata.get("pii_types") or [])
        return ChunkPayload.from_access_control(
            parsed.access_control,
            chunk_id=chunk_id,
            document_id=parsed.document_id,
            chunk_index=index,
            source_type=parsed.source_type.value,
            source_id=parsed.source_id,
            source_uri=parsed.source_uri,
            title=parsed.title,
            section_path=list(draft.section_path),
            page=draft.page,
            text=draft.text,
            contextual_header=draft.contextual_header,
            summary=parsed.summary,
            keywords=list(parsed.keywords),
            doc_type=parsed.doc_type,
            tags=list(parsed.tags),
            author=parsed.author,
            language=parsed.language,
            content_sha256=digest,
            simhash=simhash,
            token_count=draft.token_count,
            source_modified_at=parsed.source_modified_at,
            effective_from=parsed.effective_from,
            effective_to=parsed.effective_to,
            version=version,
            is_deleted=False,
            pii_types=pii_types,
            pii_redacted=bool(parsed.metadata.get("pii_redacted")),
            ingest_run_id=self.run_id,
        )

    async def _record_lineage(
        self,
        session: Any,
        parsed: ParsedDocument,
        payloads: Sequence[ChunkPayload],
    ) -> None:
        """Write the ``chunk -> document -> source_uri`` lineage chain.

        Args:
            session: Database session; the caller commits.
            parsed: The document written.
            payloads: The chunk payloads written.
        """
        from ragcore.db.repositories import record_lineage

        await record_lineage(
            session,
            tenant_id=parsed.tenant_id,
            kind="ingest",
            subject_id=parsed.document_id,
            operation="ingest_document",
            actor="ingestion",
            parents=[parsed.source_uri],
            inputs={
                "source_id": parsed.source_id,
                "source_type": parsed.source_type.value,
                "content_sha256": parsed.content_sha256,
            },
            outputs={
                "chunk_count": len(payloads),
                "doc_type": parsed.doc_type,
                "language": parsed.language,
                "pii_types": list(parsed.metadata.get("pii_types") or []),
            },
            metrics={
                "tokens": float(sum(p.token_count for p in payloads)),
                "chunks": float(len(payloads)),
            },
        )
        for payload in payloads:
            await record_lineage(
                session,
                tenant_id=parsed.tenant_id,
                kind="ingest",
                subject_id=payload.chunk_id,
                operation="chunk",
                actor="ingestion",
                parents=[parsed.document_id, parsed.source_uri],
                inputs={"chunk_index": payload.chunk_index},
                outputs={
                    "section_path": payload.section_path,
                    "page": payload.page,
                    "content_sha256": payload.content_sha256,
                    "simhash": payload.simhash,
                },
                metrics={"tokens": float(payload.token_count)},
            )

    async def acl_only_reindex(
        self,
        *,
        tenant_id: str,
        document_id: str,
        access_control: AccessControl,
    ) -> int:
        """Rewrite a document's ACL payload fields without re-embedding.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document whose ACL changed.
            access_control: The new ACL.

        Returns:
            The number of points rewritten.
        """
        if self.dry_run:
            return 0
        affected = await self.writer.rewrite_access_control(
            tenant_id=tenant_id,
            document_id=document_id,
            access_control=access_control,
            run_id=self.run_id,
        )
        _log.info(
            "upsert.acl_only_reindex",
            tenant_id=tenant_id,
            document_id=document_id,
            points=affected,
        )
        return affected

    async def tombstone(self, *, tenant_id: str, document_id: str) -> int:
        """Soft-delete every chunk of a document.

        Args:
            tenant_id: Owning tenant id.
            document_id: Document that disappeared at source.

        Returns:
            The number of points tombstoned.
        """
        if self.dry_run:
            return 0
        return await self.writer.soft_delete_document(
            tenant_id=tenant_id, document_id=document_id, run_id=self.run_id
        )

    async def close(self) -> None:
        """Release the writer."""
        await self.writer.close()


def _bands_for(simhash: str) -> Iterable[str]:
    """Split a hex simhash into LSH bands.

    Args:
        simhash: 16-character hex simhash.

    Returns:
        Band keys of the form ``"<position>:<hex>"``.
    """
    return [
        f"{index}:{simhash[index : index + _BAND_WIDTH]}"
        for index in range(0, len(simhash), _BAND_WIDTH)
    ]


def _bump(result: DocumentUpsertResult, reason: str, *, duplicate: bool = True) -> None:
    """Record one audited drop.

    Args:
        result: The result being accumulated.
        reason: Machine-readable drop reason.
        duplicate: Whether the drop counts towards ``duplicates_dropped``. An empty
            chunk is dropped but is not a duplicate, and conflating the two would
            misreport the run summary.
    """
    if duplicate:
        result.duplicates_dropped += 1
    result.drop_reasons[reason] = result.drop_reasons.get(reason, 0) + 1
