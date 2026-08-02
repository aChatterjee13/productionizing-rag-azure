"""Semantic retrieval cache — pipeline stage 4, requirement #2.

Faster retrieval for similar queries, without ever becoming a leak.

The cache stores the **retrieval plan** — the normalised query, the transformed
queries and the chunk ids — and never the rendered answer. On a hit the cached chunk
ids are re-fetched through the *live* ACL filter
(:func:`ragcore.vectorstore.filters.build_acl_filter_for_chunk_ids`), so a principal
whose access was revoked between the write and the read silently gets fewer chunks
rather than stale content they may no longer see. That is the whole reason the answer
is not cached: an answer cannot be re-authorised, a chunk id can.

A hit requires **both** conditions from ``docs/CONTRACTS.md``:

* cosine similarity ≥ ``memory_cache_threshold`` (default 0.94), enforced by Qdrant
  itself via ``score_threshold`` and re-checked in
  :meth:`~ragcore.models.memory.SemanticCacheEntry.matches`; and
* an **exact** ``filter_fingerprint`` match, which is part of the Qdrant filter.

Entries are tenant-scoped by construction: the cache id is derived from the tenant, so
an identical question in two tenants can never share an entry, and every probe goes
through :func:`ragcore.vectorstore.filters.build_cache_filter`.

Lifetime is TTL plus LRU-ish eviction: within one ``(tenant, filter_fingerprint)``
bucket, entries past their TTL go first and the rest are ordered by ``hit_count`` then
``last_used_at``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models as qm

from ragcore.models.acl import Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.memory import SemanticCacheEntry, normalize_query
from ragcore.models.retrieval import MetadataFilter, RetrievalStage, RetrievedChunk
from ragcore.observability import observe_cache_lookup
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore import (
    DENSE,
    build_acl_filter_for_chunk_ids,
    build_cache_filter,
    dense_search,
    filter_fingerprint,
    get_client,
    point_id_for_cache,
    upsert_points,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "CacheProbe",
    "ChunkResolver",
    "SemanticCache",
    "get_semantic_cache",
    "reset_semantic_cache",
    "retrieve_by_ids",
]

#: How a cached plan turns back into chunks.
#:
#: The default is :func:`retrieve_by_ids` below, which is the client-level primitive
#: and keeps this module importable without pulling the retriever (and therefore
#: FastEmbed). The orchestrator should inject ``app.rag.retriever.retrieve_by_ids``
#: instead, so a cache hit produces exactly the same ``RetrievedChunk`` shape,
#: ``dropped_reason`` vocabulary and score scale as a live retrieval. Both go through
#: :func:`ragcore.vectorstore.filters.build_acl_filter_for_chunk_ids`, so the security
#: property does not depend on which one is used.
ChunkResolver = Callable[
    [Principal, Sequence[str], "MetadataFilter | None"],
    Awaitable[Sequence["RetrievedChunk"]],
]

_log = structlog.get_logger(__name__)

#: Label used for the cache metric series.
CACHE_LABEL = "semantic"


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


async def retrieve_by_ids(
    client: AsyncQdrantClient,
    *,
    principal: Principal,
    chunk_ids: Sequence[str],
    extra: MetadataFilter | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Re-fetch cached chunk ids through the live ACL filter.

    This is the security-critical half of the cache. The ids are treated as untrusted
    hints: they are resolved with ``build_acl_filter_for_chunk_ids``, which composes
    the full ACL filter (tenant, deny list, clearance ceiling, permissive branch) and
    adds the id restriction. A chunk the caller may no longer read simply does not
    come back.

    Args:
        client: Async Qdrant client.
        principal: The caller whose live permissions are applied.
        chunk_ids: Cached chunk ids, in the cached order.
        extra: The request's metadata filter, applied on top of the ACL filter.
        settings: Active settings. Defaults to the process settings.

    Returns:
        The chunks the caller may still read, in the cached order, marked with
        ``retrieval_stage="cache"``. Scores descend with the cached rank, so
        downstream ordering behaves as if the chunks had just been retrieved.
    """
    ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
    if not ids:
        return []
    cfg = settings or get_settings()
    qfilter = build_acl_filter_for_chunk_ids(
        principal,
        ids,
        extra,
        include_deleted=cfg.retrieval_include_deleted,
    )
    records, _ = await client.scroll(
        collection_name=cfg.qdrant_chunks_collection,
        scroll_filter=qfilter,
        limit=len(ids),
        with_payload=True,
        with_vectors=False,
    )

    by_id: dict[str, ChunkPayload] = {}
    for record in records:
        try:
            payload = ChunkPayload.from_qdrant_payload(record.payload)
        except ValueError:
            _log.warning("cache_chunk_payload_missing")
            continue
        by_id[payload.chunk_id] = payload

    total = len(ids)
    chunks: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ids):
        payload = by_id.get(chunk_id)
        if payload is None:
            continue
        chunks.append(
            RetrievedChunk(
                payload=payload,
                fusion_score=(total - rank) / total,
                final_score=(total - rank) / total,
                retrieval_stage=RetrievalStage.CACHE.value,
            )
        )
    if len(chunks) != total:
        _log.info(
            "cache_chunks_filtered_by_acl",
            tenant_id=principal.tenant_id,
            requested=total,
            visible=len(chunks),
        )
    return chunks


class CacheProbe(BaseModel):
    """The outcome of one cache lookup, hit or miss."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    hit: bool = Field(default=False, description="Whether the plan may be reused.")
    similarity: float = Field(
        default=0.0, description="Cosine similarity of the best candidate."
    )
    entry: SemanticCacheEntry | None = Field(
        default=None, description="The matched entry, when there was one."
    )
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Cached chunks the caller may still read, best first.",
    )
    revoked_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Cached ids the live ACL filter withheld. Non-empty means access was "
            "revoked since the entry was written — the point of re-checking."
        ),
    )
    transformed_queries: list[str] = Field(
        default_factory=list, description="Queries the cached plan used."
    )
    fingerprint: str = Field(default="", description="Fingerprint for this request.")
    normalized_query: str = Field(default="", description="Canonicalised query.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Probe latency.")
    reason: str = Field(
        default="miss",
        description=(
            "'hit' | 'miss' | 'disabled' | 'empty_query' | 'below_threshold' | "
            "'expired' | 'fingerprint_mismatch' | 'empty_after_acl' | 'error'."
        ),
    )

    @property
    def usable(self) -> bool:
        """Whether stage 5 may skip candidate generation.

        Returns:
            True only when the probe hit *and* chunks survived the live ACL filter.
        """
        return self.hit and bool(self.chunks)


class SemanticCache:
    """Similar-query retrieval cache over ``rag_semantic_cache``."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: AsyncQdrantClient | None = None,
        embedder: Any | None = None,
        resolver: ChunkResolver | None = None,
    ) -> None:
        """Initialise the cache.

        Args:
            settings: Active settings. Defaults to the process settings.
            client: Qdrant client. Resolved lazily when omitted.
            embedder: Embedding provider. Resolved lazily when omitted.
            resolver: How cached chunk ids become chunks. Defaults to the local
                :func:`retrieve_by_ids`; inject ``app.rag.retriever.retrieve_by_ids``
                so a cache hit is shaped exactly like a live retrieval.
        """
        self._settings = settings or get_settings()
        self._client = client
        self._embedder = embedder
        self._resolver = resolver

    @property
    def settings(self) -> Settings:
        """Settings this cache was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def collection(self) -> str:
        """Qdrant collection holding cache entries.

        Returns:
            ``settings.qdrant_cache_collection``.
        """
        return self._settings.qdrant_cache_collection

    async def _qdrant(self) -> AsyncQdrantClient:
        """Resolve the Qdrant client.

        Returns:
            The shared async client.
        """
        if self._client is None:
            self._client = await get_client(self._settings)
        return self._client

    async def _embed(self, text: str) -> list[float]:
        """Embed the normalised query.

        Args:
            text: Normalised query text.

        Returns:
            The dense vector.
        """
        if self._embedder is None:
            from ragcore.embeddings import get_embedding_provider

            self._embedder = get_embedding_provider(self._settings)
        embedded = await self._embedder.embed_query(text)
        return list(embedded.dense)

    def fingerprint(self, principal: Principal, extra: MetadataFilter | None) -> str:
        """Compute the request's filter fingerprint.

        Args:
            principal: The caller.
            extra: The request's metadata filter.

        Returns:
            The 32-character fingerprint that must match exactly for a hit.
        """
        return filter_fingerprint(principal, extra)

    # ---------------------------------------------------------------------- probe
    async def probe(
        self,
        principal: Principal,
        query: str,
        *,
        extra: MetadataFilter | None = None,
        now: datetime | None = None,
        db_session: AsyncSession | None = None,
        resolver: ChunkResolver | None = None,
    ) -> CacheProbe:
        """Look for a reusable retrieval plan (pipeline stage 4).

        Args:
            principal: The caller. Tenant and clearance are part of the fingerprint;
                the live ACL filter is applied to the cached ids afterwards.
            query: The raw user query.
            extra: The request's metadata filter.
            now: Reference time for the TTL check. Defaults to now (UTC).
            db_session: Optional session for the relational mirror's hit counter.
            resolver: Overrides the instance's chunk resolver for this probe.

        Returns:
            The probe outcome. A probe that hits but whose chunks are all withheld by
            the ACL filter reports ``reason="empty_after_acl"`` and ``usable=False``,
            so the pipeline falls through to a normal retrieval instead of answering
            from nothing.
        """
        started = time.perf_counter()
        if not self._settings.memory_cache_enabled:
            return CacheProbe(reason="disabled")

        normalized = normalize_query(query)
        if not normalized:
            return CacheProbe(reason="empty_query", normalized_query=normalized)

        fingerprint = self.fingerprint(principal, extra)
        reference = now or _utcnow()
        try:
            client = await self._qdrant()
            vector = await self._embed(normalized)
            points = await dense_search(
                client,
                collection=self.collection,
                dense=vector,
                qfilter=build_cache_filter(principal, fingerprint),
                limit=1,
                score_threshold=self._settings.memory_cache_threshold,
            )
        except Exception:
            _log.warning(
                "semantic_cache_probe_failed",
                tenant_id=principal.tenant_id,
                exc_info=True,
            )
            observe_cache_lookup(cache=CACHE_LABEL, hit=False)
            return CacheProbe(
                reason="error",
                fingerprint=fingerprint,
                normalized_query=normalized,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if not points:
            observe_cache_lookup(cache=CACHE_LABEL, hit=False)
            return CacheProbe(
                reason="below_threshold",
                fingerprint=fingerprint,
                normalized_query=normalized,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        point = points[0]
        similarity = float(point.score or 0.0)
        entry = _parse_entry(point.payload)
        if entry is None:
            observe_cache_lookup(cache=CACHE_LABEL, hit=False)
            return CacheProbe(
                reason="error",
                similarity=similarity,
                fingerprint=fingerprint,
                normalized_query=normalized,
            )

        if not entry.matches(
            fingerprint=fingerprint,
            similarity=similarity,
            threshold=self._settings.memory_cache_threshold,
        ):
            reason = (
                "expired"
                if entry.is_expired(reference)
                else "fingerprint_mismatch"
                if entry.filter_fingerprint != fingerprint
                else "below_threshold"
            )
            observe_cache_lookup(cache=CACHE_LABEL, hit=False)
            return CacheProbe(
                reason=reason,
                similarity=similarity,
                entry=entry,
                fingerprint=fingerprint,
                normalized_query=normalized,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        chunks = await self._resolve_chunks(
            client, principal, entry.chunk_ids, extra, resolver or self._resolver
        )
        visible = {chunk.payload.chunk_id for chunk in chunks}
        revoked = [chunk_id for chunk_id in entry.chunk_ids if chunk_id not in visible]
        if revoked:
            _log.info(
                "semantic_cache_acl_downgrade",
                tenant_id=principal.tenant_id,
                cached=len(entry.chunk_ids),
                visible=len(chunks),
            )

        hit = bool(chunks)
        observe_cache_lookup(cache=CACHE_LABEL, hit=hit)
        if hit:
            await self._touch(entry, now=reference, db_session=db_session)

        return CacheProbe(
            hit=hit,
            similarity=similarity,
            entry=entry,
            chunks=chunks,
            revoked_chunk_ids=revoked,
            transformed_queries=list(entry.transformed_queries),
            fingerprint=fingerprint,
            normalized_query=normalized,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reason="hit" if hit else "empty_after_acl",
        )

    async def _resolve_chunks(
        self,
        client: AsyncQdrantClient,
        principal: Principal,
        chunk_ids: Sequence[str],
        extra: MetadataFilter | None,
        resolver: ChunkResolver | None,
    ) -> list[RetrievedChunk]:
        """Turn cached chunk ids back into chunks the caller may read.

        Args:
            client: Async Qdrant client, used by the default resolver.
            principal: The caller whose live permissions are applied.
            chunk_ids: Cached chunk ids, in the cached order.
            extra: The request's metadata filter.
            resolver: Injected resolver, or None for the local default.

        Returns:
            The visible chunks. A resolver failure degrades to an empty list, which
            reads downstream as a cache miss rather than as an answer with no
            sources.
        """
        if resolver is None:
            return await retrieve_by_ids(
                client,
                principal=principal,
                chunk_ids=chunk_ids,
                extra=extra,
                settings=self._settings,
            )
        try:
            return list(await resolver(principal, list(chunk_ids), extra))
        except Exception:
            _log.warning(
                "semantic_cache_resolver_failed",
                tenant_id=principal.tenant_id,
                exc_info=True,
            )
            return []

    # ---------------------------------------------------------------------- store
    async def store(
        self,
        principal: Principal,
        query: str,
        *,
        chunk_ids: Sequence[str],
        transformed_queries: Sequence[str] = (),
        extra: MetadataFilter | None = None,
        now: datetime | None = None,
        db_session: AsyncSession | None = None,
    ) -> SemanticCacheEntry | None:
        """Cache the retrieval plan for a query.

        Only the plan is stored: normalised query, transformed queries and chunk ids.
        The rendered answer is never cached, because an answer cannot be re-checked
        against the reader's permissions the way a chunk id can.

        Args:
            principal: The caller.
            query: The raw user query.
            chunk_ids: Chunk ids the retrieval produced, best first. Truncated to
                ``memory_cache_max_chunk_ids``.
            transformed_queries: Queries the transformer produced.
            extra: The request's metadata filter.
            now: Reference time. Defaults to now (UTC).
            db_session: Optional session for the ``semantic_cache_meta`` mirror.

        Returns:
            The stored entry, or None when caching is disabled or there is nothing
            worth caching.
        """
        if not self._settings.memory_cache_enabled:
            return None
        normalized = normalize_query(query)
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id][
            : self._settings.memory_cache_max_chunk_ids
        ]
        if not normalized or not ids:
            return None

        fingerprint = self.fingerprint(principal, extra)
        reference = now or _utcnow()
        entry = SemanticCacheEntry(
            cache_id=SemanticCacheEntry.make_cache_id(
                tenant_id=principal.tenant_id,
                normalized_query=normalized,
                fingerprint=fingerprint,
            ),
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            normalized_query=normalized,
            transformed_queries=list(transformed_queries),
            chunk_ids=ids,
            filter_fingerprint=fingerprint,
            created_at=reference,
            last_used_at=reference,
            ttl_seconds=self._settings.memory_cache_ttl_seconds,
        )

        try:
            client = await self._qdrant()
            vector = await self._embed(normalized)
            await upsert_points(
                client,
                collection=self.collection,
                points=[
                    qm.PointStruct(
                        id=point_id_for_cache(entry.cache_id),
                        vector={DENSE: vector},
                        payload=entry.model_dump(mode="json"),
                    )
                ],
                settings=self._settings,
            )
        except Exception:
            _log.warning(
                "semantic_cache_store_failed",
                tenant_id=principal.tenant_id,
                exc_info=True,
            )
            return None

        if db_session is not None:
            await self._mirror(entry, db_session=db_session)
        await self.evict(principal, fingerprint=fingerprint, now=reference)
        return entry

    async def _touch(
        self,
        entry: SemanticCacheEntry,
        *,
        now: datetime,
        db_session: AsyncSession | None = None,
    ) -> None:
        """Record a reuse on the entry.

        Args:
            entry: The reused entry.
            now: Reference time.
            db_session: Optional session for the relational mirror.
        """
        touched = entry.touch(now)
        try:
            client = await self._qdrant()
            await client.set_payload(
                collection_name=self.collection,
                payload={
                    "hit_count": touched.hit_count,
                    "last_used_at": touched.last_used_at.isoformat(),
                },
                points=[point_id_for_cache(entry.cache_id)],
                wait=False,
            )
        except Exception:
            _log.warning("semantic_cache_touch_failed", cache_id=entry.cache_id)
        if db_session is not None:
            await self._mirror(touched, db_session=db_session)

    async def _mirror(
        self, entry: SemanticCacheEntry, *, db_session: AsyncSession
    ) -> None:
        """Mirror an entry into ``semantic_cache_meta``.

        Args:
            entry: The entry to write.
            db_session: Active database session. The caller commits.
        """
        from ragcore.db import repositories as repo

        try:
            await repo.upsert_cache_meta(
                db_session,
                tenant_id=entry.tenant_id,
                cache_id=entry.cache_id,
                normalized_query=entry.normalized_query,
                filter_fingerprint=entry.filter_fingerprint,
                chunk_ids=entry.chunk_ids,
                transformed_queries=entry.transformed_queries,
                user_id=entry.user_id,
                ttl_seconds=entry.ttl_seconds,
                expires_at=entry.expires_at,
            )
        except Exception:
            _log.warning("semantic_cache_mirror_failed", cache_id=entry.cache_id)

    # ------------------------------------------------------------------- eviction
    async def evict(
        self,
        principal: Principal,
        *,
        fingerprint: str,
        now: datetime | None = None,
    ) -> int:
        """Apply TTL and LRU-ish eviction to one cache bucket.

        A bucket is one ``(tenant, filter_fingerprint)`` pair — the unit a probe can
        actually hit — and is enumerated through ``build_cache_filter``, so eviction
        can never reach another tenant's entries. Expired entries go first; the
        remainder are ordered by ``hit_count`` then ``last_used_at``, and anything
        beyond ``memory_cache_max_entries`` is removed.

        Args:
            principal: The caller; supplies the tenant scope.
            fingerprint: The bucket to sweep.
            now: Reference time. Defaults to now (UTC).

        Returns:
            The number of entries removed.
        """
        reference = now or _utcnow()
        cap = int(self._settings.memory_cache_max_entries)
        try:
            client = await self._qdrant()
        except Exception:
            _log.warning("semantic_cache_evict_client_failed", exc_info=True)
            return 0
        try:
            records, _ = await client.scroll(
                collection_name=self.collection,
                scroll_filter=build_cache_filter(principal, fingerprint),
                limit=cap * 2,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            _log.warning("semantic_cache_evict_scan_failed", exc_info=True)
            return 0

        live: list[tuple[Any, SemanticCacheEntry]] = []
        victims: list[Any] = []
        for record in records:
            entry = _parse_entry(record.payload)
            if entry is None or entry.is_expired(reference):
                victims.append(record.id)
                continue
            live.append((record.id, entry))

        if len(live) > cap:
            live.sort(key=lambda item: (item[1].hit_count, item[1].last_used_at))
            victims.extend(point_id for point_id, _ in live[: len(live) - cap])

        if not victims:
            return 0
        try:
            await client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(points=victims),
                wait=False,
            )
        except Exception:
            _log.warning("semantic_cache_evict_failed", exc_info=True)
            return 0
        _log.info(
            "semantic_cache_evicted",
            tenant_id=principal.tenant_id,
            removed=len(victims),
        )
        return len(victims)

    async def invalidate(self, principal: Principal, *, fingerprint: str) -> int:
        """Remove every entry in one bucket.

        Used when a document's ACLs or content change in a way that makes cached
        plans untrustworthy. The chunk-id re-fetch already prevents a leak; this
        prevents a *stale* plan from pinning retrieval to deleted chunks.

        Args:
            principal: The caller; supplies the tenant scope.
            fingerprint: The bucket to clear.

        Returns:
            The number of entries removed.
        """
        cap = int(self._settings.memory_cache_max_entries)
        try:
            client = await self._qdrant()
            records, _ = await client.scroll(
                collection_name=self.collection,
                scroll_filter=build_cache_filter(principal, fingerprint),
                limit=cap * 2,
                with_payload=False,
                with_vectors=False,
            )
            if not records:
                return 0
            await client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(
                    points=[record.id for record in records]
                ),
                wait=True,
            )
            return len(records)
        except Exception:
            _log.warning("semantic_cache_invalidate_failed", exc_info=True)
            return 0


def _parse_entry(payload: Mapping[str, Any] | None) -> SemanticCacheEntry | None:
    """Rebuild a cache entry from a Qdrant payload.

    Args:
        payload: Raw point payload.

    Returns:
        The parsed entry, or None when it is missing or malformed. An unparseable
        entry is treated as a miss, never as a hit with unknown contents.
    """
    if not payload:
        return None
    known = set(SemanticCacheEntry.model_fields)
    cleaned = {key: value for key, value in payload.items() if key in known}
    try:
        return SemanticCacheEntry.model_validate(cleaned)
    except ValueError:
        _log.warning("semantic_cache_payload_unparseable")
        return None


_CACHES: dict[str, SemanticCache] = {}


def get_semantic_cache(settings: Settings | None = None) -> SemanticCache:
    """Return the process-wide semantic cache.

    ``Settings`` is unhashable, so the cache key is the Qdrant endpoint plus the
    cache collection and the similarity threshold.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached instance.
    """
    cfg = settings or get_settings()
    key = f"{cfg.qdrant_url}|{cfg.qdrant_cache_collection}|{cfg.memory_cache_threshold}"
    existing = _CACHES.get(key)
    if existing is None:
        existing = SemanticCache(settings=cfg)
        _CACHES[key] = existing
    return existing


def reset_semantic_cache() -> None:
    """Drop the cached instances. Test helper."""
    _CACHES.clear()
