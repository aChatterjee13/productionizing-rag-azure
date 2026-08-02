"""Retrieval — requirement #6, pipeline stage 5.

The full candidate pipeline, in the order `docs/CONTRACTS.md` fixes it:

1. embed every probe once (rewritten query, sub-questions, optional HyDE passage);
2. hybrid dense + sparse search per probe, fused **server-side**, always filtered by
   :func:`ragcore.vectorstore.filters.build_acl_filter`;
3. union across probes keeping the best score per chunk;
4. exact-hash and simhash dedupe;
5. cross-encoder rerank over the top ``rerank_candidate_limit`` candidates;
6. MMR diversification;
7. per-document cap and the final ``retrieval_top_n``.

Two properties everything here is arranged around.

**Every drop is audited.** A chunk that falls out at any stage is returned in
:attr:`~ragcore.models.retrieval.RetrievalResult.dropped` carrying the reason it
fell out, and ``total_candidates`` / ``after_dedupe`` / ``after_rerank`` reconcile
with those drops. Requirement #9 asks for exactly that: a retrieval that silently
loses a chunk is indistinguishable from one that never found it.

**ACL failures are the one exception, deliberately.** A candidate that comes back
from Qdrant but fails the in-process ACL mirror is counted and logged and then
discarded — it is *never* placed in ``dropped``, because ``dropped`` is serialised
to the client by ``RetrievalResult.without_text()`` and would leak the title, URI and
section path of a document the principal may not see. Reaching that branch at all
means the Qdrant filter is broken, so it logs at error level.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from app.rag.mmr import maximal_marginal_relevance
from ragcore.dedupe import dedupe_chunks
from ragcore.embeddings import (
    Embedded,
    EmbeddingProvider,
    get_embedding_provider,
    truncate_for_embedding,
)
from ragcore.errors import RetrievalError
from ragcore.models.acl import Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.retrieval import (
    MetadataFilter,
    RetrievalResult,
    RetrievalStage,
    RetrievedChunk,
)
from ragcore.observability import get_tracer, observe_retrieval_stage
from ragcore.rerank import NoopReranker, Reranker, get_reranker
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore import DENSE, get_client
from ragcore.vectorstore.filters import (
    build_acl_filter,
    build_acl_filter_for_chunk_ids,
    serialise_filter,
)
from ragcore.vectorstore.hybrid import hybrid_search

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qm

__all__ = [
    "DROP_ACL",
    "DROP_CANDIDATE_LIMIT",
    "DROP_DELETED",
    "DROP_MAX_PER_DOCUMENT",
    "DROP_RERANK_MIN_SCORE",
    "DROP_RERANK_TOP_N",
    "DROP_TOP_N",
    "PROBE_HYDE",
    "PROBE_QUERY",
    "retrieve",
    "retrieve_by_ids",
]

_log = structlog.get_logger(__name__)

#: Candidate exceeded ``rerank_candidate_limit`` and was never scored.
DROP_CANDIDATE_LIMIT = "rerank:candidate_limit"
#: Cross-encoder score fell below ``rerank_min_score``.
DROP_RERANK_MIN_SCORE = "rerank:min_score"
#: Survived reranking but ranked below ``rerank_top_n``.
DROP_RERANK_TOP_N = "rerank:top_n"
#: Ranked below ``retrieval_top_n`` after MMR diversification.
DROP_TOP_N = "top_n"
#: Owning document already contributed ``retrieval_max_per_document`` chunks.
DROP_MAX_PER_DOCUMENT = "max_per_document"
#: Tombstoned chunk returned by a search that should have excluded it.
DROP_DELETED = "deleted"
#: Failed the in-process ACL mirror. Counted and logged, never returned.
DROP_ACL = "acl"

#: Probe kinds recorded on ``queries_used``.
PROBE_QUERY = "query"
PROBE_HYDE = "hyde"

#: Latency buckets reported in ``RetrievalResult.latency_ms``.
_STAGES = ("embed", "search", "dedupe", "rerank", "vectors", "mmr", "total")


@dataclass(slots=True)
class _Probe:
    """One text actually sent to the search engine.

    Attributes:
        text: The probe text.
        kind: :data:`PROBE_QUERY` or :data:`PROBE_HYDE`.
    """

    text: str
    kind: str = PROBE_QUERY


@dataclass(slots=True)
class _Ledger:
    """Running record of everything the pipeline discarded.

    Attributes:
        dropped: Candidates safe to return to the caller, each already marked with
            its ``dropped_reason``.
        acl_rejected: Count of candidates discarded by the in-process ACL mirror.
            Deliberately a count, not a list — see the module docstring.
        payload_invalid: Count of points whose payload would not parse.
    """

    dropped: list[RetrievedChunk] = field(default_factory=list)
    acl_rejected: int = 0
    payload_invalid: int = 0

    def add(self, chunks: Sequence[RetrievedChunk], reason: str) -> None:
        """Mark candidates as dropped for one reason.

        Args:
            chunks: The candidates leaving the pipeline.
            reason: Machine-readable cause, one of the ``DROP_*`` constants or a
                ``ragcore.dedupe`` reason.
        """
        self.dropped.extend(chunk.drop(reason) for chunk in chunks)


class _Clock:
    """Accumulates per-stage wall time in milliseconds."""

    def __init__(self) -> None:
        """Start the total-time clock."""
        self._marks: dict[str, float] = {}
        self._started = time.perf_counter()

    def start(self) -> float:
        """Take a reading to hand back to :meth:`stop`.

        Returns:
            A monotonic timestamp.
        """
        return time.perf_counter()

    def stop(
        self, stage: str, started: float, *, candidates: int | None = None
    ) -> None:
        """Record the time a stage took and export it to Prometheus.

        Args:
            stage: Bucket name, one of :data:`_STAGES`.
            started: The value :meth:`start` returned.
            candidates: Number of candidates the stage emitted, when meaningful.
        """
        elapsed = (time.perf_counter() - started) * 1000.0
        self._marks[stage] = self._marks.get(stage, 0.0) + elapsed
        observe_retrieval_stage(stage=stage, latency_ms=elapsed, candidates=candidates)

    def finish(self) -> dict[str, float]:
        """Close the total bucket and return every measurement.

        Returns:
            Stage name to milliseconds, rounded to three decimals, including a
            ``total`` bucket covering the whole call.
        """
        self._marks["total"] = (time.perf_counter() - self._started) * 1000.0
        return {
            stage: round(self._marks[stage], 3)
            for stage in _STAGES
            if stage in self._marks
        }


def _build_probes(
    queries: Sequence[str], hyde_passage: str, *, settings: Settings
) -> list[_Probe]:
    """Normalise the caller's queries into the probes stage 5 will issue.

    Args:
        queries: Rewritten question first, then sub-questions.
        hyde_passage: Hypothetical passage from stage 3, or an empty string.
        settings: Settings supplying ``qt_max_subqueries``, which bounds how many
            probes a runaway decomposition can turn into.

    Returns:
        De-duplicated probes, most important first. The HyDE passage is last: it is
        an extra dense signal, not a replacement for the user's own words.
    """
    probes: list[_Probe] = []
    seen: set[str] = set()
    # The rewritten query plus at most qt_max_subqueries sub-questions.
    limit = settings.qt_max_subqueries + 1
    for text in queries:
        candidate = text.strip()
        if not candidate:
            continue
        marker = candidate.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        probes.append(_Probe(text=candidate))
        if len(probes) >= limit:
            break
    passage = hyde_passage.strip()
    if passage:
        probes.append(
            _Probe(text=passage[: settings.qt_hyde_max_chars], kind=PROBE_HYDE)
        )
    return probes


async def _embed_probes(
    probes: Sequence[_Probe], *, embedder: EmbeddingProvider, settings: Settings
) -> list[Embedded | None]:
    """Embed every probe exactly once, concurrently.

    Args:
        probes: The probes to embed.
        embedder: Provider supplying dense and sparse vectors.
        settings: Settings supplying ``embedding_max_chars`` and the probe
            concurrency bound.

    Returns:
        One :class:`~ragcore.embeddings.Embedded` per probe, in order, or None for a
        probe whose embedding failed. A single failed probe must not cost the whole
        turn its retrieval, so the failure is localised here.
    """
    limit = int(settings.retrieval_query_concurrency)
    gate = asyncio.Semaphore(max(1, limit))

    async def _one(probe: _Probe) -> Embedded | None:
        text = truncate_for_embedding(probe.text, settings.embedding_max_chars)
        async with gate:
            try:
                return await embedder.embed_query(text)
            except Exception as exc:  # one bad probe, not a lost turn
                _log.warning(
                    "retrieval.embed_failed",
                    error=type(exc).__name__,
                    kind=probe.kind,
                    query_chars=len(probe.text),
                )
                return None

    return list(await asyncio.gather(*(_one(probe) for probe in probes)))


async def _search_probes(
    client: AsyncQdrantClient,
    probes: Sequence[_Probe],
    embeddings: Sequence[Embedded | None],
    *,
    qfilter: qm.Filter,
    collection: str,
    settings: Settings,
) -> list[list[qm.ScoredPoint]]:
    """Run one hybrid search per successfully embedded probe.

    Args:
        client: Async Qdrant client.
        probes: The probes, aligned with ``embeddings``.
        embeddings: Embedding per probe; None entries are skipped.
        qfilter: The composed ACL + metadata filter. The same object goes to every
            branch of every probe.
        collection: Chunk collection name.
        settings: Settings supplying the limits and the fusion strategy.

    Returns:
        One list of scored points per probe, empty for a probe that was skipped or
        that failed.

    Raises:
        RetrievalError: If every probe failed against Qdrant. One failing probe is
            a degraded search; all of them failing is an outage the caller must see.
    """
    limit = int(settings.retrieval_query_concurrency)
    gate = asyncio.Semaphore(max(1, limit))
    failures: list[str] = []

    async def _one(probe: _Probe, embedded: Embedded | None) -> list[qm.ScoredPoint]:
        if embedded is None:
            return []
        async with gate:
            try:
                return await hybrid_search(
                    client,
                    collection=collection,
                    query_text=probe.text,
                    dense=embedded.dense,
                    sparse=embedded.sparse,
                    qfilter=qfilter,
                    limit=settings.retrieval_limit,
                    prefetch_limit=settings.retrieval_prefetch_limit,
                    fusion=settings.retrieval_fusion,
                )
            except Exception as exc:  # partial results beat none
                failures.append(type(exc).__name__)
                _log.warning(
                    "retrieval.search_failed",
                    error=type(exc).__name__,
                    kind=probe.kind,
                    query_chars=len(probe.text),
                )
                return []

    results = list(
        await asyncio.gather(
            *(
                _one(probe, embedded)
                for probe, embedded in zip(probes, embeddings, strict=True)
            )
        )
    )
    attempted = sum(1 for embedded in embeddings if embedded is not None)
    if attempted and len(failures) == attempted:
        msg = f"every retrieval probe failed ({failures[0]})"
        raise RetrievalError(msg)
    return results


def _to_candidate(
    point: qm.ScoredPoint, *, principal: Principal, settings: Settings, ledger: _Ledger
) -> RetrievedChunk | None:
    """Turn one scored point into a candidate, enforcing the ACL a second time.

    Qdrant has already filtered by :func:`build_acl_filter`. Re-checking in process
    is defence in depth: the cost is a few comparisons, and the failure it catches is
    a cross-tenant leak.

    Args:
        point: A scored point from fusion.
        principal: The authenticated caller.
        settings: Settings supplying ``retrieval_include_deleted``.
        ledger: Where rejections are counted.

    Returns:
        The candidate, or None when the payload will not parse or the point does not
        actually belong to the principal.
    """
    try:
        payload = ChunkPayload.from_qdrant_payload(point.payload)
    except ValueError:
        ledger.payload_invalid += 1
        _log.warning("retrieval.payload_invalid", point_id=str(point.id))
        return None

    if payload.tenant_id != principal.tenant_id or not payload.access_control().permits(
        principal
    ):
        ledger.acl_rejected += 1
        # An error, not a warning: the Qdrant filter should have made this
        # unreachable. Only structural identifiers are logged.
        _log.error(
            "retrieval.acl_rejected",
            chunk_id=payload.chunk_id,
            document_id=payload.document_id,
            principal_tenant=principal.tenant_id,
            cross_tenant=payload.tenant_id != principal.tenant_id,
        )
        return None

    chunk = RetrievedChunk(
        payload=payload,
        fusion_score=float(point.score or 0.0),
        retrieval_stage=RetrievalStage.FUSION.value,
    )
    if payload.is_deleted and not settings.retrieval_include_deleted:
        ledger.add([chunk], DROP_DELETED)
        return None
    return chunk


def _union(
    batches: Sequence[Sequence[qm.ScoredPoint]],
    *,
    principal: Principal,
    settings: Settings,
    ledger: _Ledger,
) -> tuple[list[RetrievedChunk], int]:
    """Merge per-probe result sets, keeping the best score per chunk.

    A chunk found by three sub-questions is one candidate, not three, and the score
    it carries forward is the best any probe gave it — fusion scores from different
    probes are on the same scale, so the maximum is the right union operator.

    Args:
        batches: Scored points per probe.
        principal: The authenticated caller.
        settings: Active settings.
        ledger: Where rejections and tombstones are recorded.

    Returns:
        A ``(candidates, total_seen)`` pair. ``candidates`` is sorted by descending
        fusion score, which is the order dedupe needs to keep the best copy.
        ``total_seen`` counts every point returned across probes, including the ones
        that never became candidates.
    """
    best: dict[str, RetrievedChunk] = {}
    total = 0
    for points in batches:
        for point in points:
            total += 1
            candidate = _to_candidate(
                point, principal=principal, settings=settings, ledger=ledger
            )
            if candidate is None:
                continue
            existing = best.get(candidate.payload.chunk_id)
            if existing is None or candidate.fusion_score > existing.fusion_score:
                best[candidate.payload.chunk_id] = candidate
    candidates = sorted(best.values(), key=lambda chunk: -chunk.fusion_score)
    return candidates, total


def _dedupe(
    candidates: list[RetrievedChunk], *, settings: Settings, ledger: _Ledger
) -> list[RetrievedChunk]:
    """Collapse exact and near-duplicate chunks, recording every drop.

    Args:
        candidates: Candidates sorted best-first, so the survivor of a duplicate pair
            is the higher-ranked copy.
        settings: Settings supplying ``dedupe_enabled``, ``dedupe_max_distance`` and
            ``dedupe_min_chunk_chars``.
        ledger: Where the duplicates are recorded.

    Returns:
        The surviving candidates, order preserved.
    """
    if not settings.dedupe_enabled or len(candidates) < 2:
        return candidates

    minimum = settings.dedupe_min_chunk_chars

    def _simhash_key(chunk: RetrievedChunk) -> str:
        # Chunks below the minimum length opt out of the near-duplicate layer: a
        # short chunk's simhash collides with anything else short.
        if len(chunk.payload.text) < minimum:
            return ""
        return chunk.payload.simhash

    kept, dropped = dedupe_chunks(
        candidates,
        key=lambda chunk: chunk.payload.content_sha256,
        simhash_key=_simhash_key,
        max_distance=settings.dedupe_max_distance,
    )
    for chunk, reason in dropped:
        ledger.add([chunk], reason)
    return kept


def _rerank_text(chunk: RetrievedChunk, *, settings: Settings) -> str:
    """Render the text a cross-encoder scores for one candidate.

    Args:
        chunk: The candidate.
        settings: Settings supplying ``retrieval_contextual_header_enabled`` and the
            character budget.

    Returns:
        The chunk text, prefixed with its contextual header when headers are enabled
        — the header is what tells the encoder that a bare paragraph belongs to the
        travel policy's meals section — clipped to ``embedding_max_chars``.
    """
    text = (
        chunk.payload.embed_text
        if settings.retrieval_contextual_header_enabled
        else chunk.payload.text
    )
    return truncate_for_embedding(text, settings.embedding_max_chars)


async def _rerank(
    candidates: list[RetrievedChunk],
    *,
    query: str,
    reranker: Reranker,
    settings: Settings,
    ledger: _Ledger,
) -> list[RetrievedChunk]:
    """Score candidates with the cross-encoder and apply the two rerank bounds.

    The reranker itself never drops anything (see Addendum B): the candidate limit
    and the minimum score are enforced here so each exclusion carries a reason.

    Args:
        candidates: Deduplicated candidates, best-first.
        query: The text scored against — the rewritten standalone question, not a
            sub-question, because the user asked that one.
        reranker: The cross-encoder, or the no-op when reranking is disabled.
        settings: Settings supplying the limits.
        ledger: Where excluded candidates are recorded.

    Returns:
        Survivors ordered by descending cross-encoder score. When reranking is
        disabled the fusion order is returned unchanged.
    """
    if not candidates:
        return []

    scored = candidates
    if len(scored) > settings.rerank_candidate_limit:
        ledger.add(scored[settings.rerank_candidate_limit :], DROP_CANDIDATE_LIMIT)
        scored = scored[: settings.rerank_candidate_limit]

    documents = [_rerank_text(chunk, settings=settings) for chunk in scored]
    try:
        # top_n is the full pool: truncation is this function's job, not the
        # reranker's, so that every exclusion below carries a dropped_reason.
        results = await reranker.rerank(query, documents, len(documents))
    except Exception as exc:  # a rerank outage degrades, never fails
        _log.warning(
            "retrieval.rerank_failed", error=type(exc).__name__, candidates=len(scored)
        )
        return scored

    ordered: list[RetrievedChunk] = []
    for result in results:
        if not 0 <= result.index < len(scored):
            continue
        chunk = scored[result.index]
        ordered.append(
            chunk.model_copy(
                update={
                    "rerank_score": float(result.score),
                    "retrieval_stage": RetrievalStage.RERANK.value,
                }
            )
        )
    # A reranker that returned fewer results than it was given must not silently
    # lose candidates; the unscored tail keeps its fusion order behind the scored.
    returned = {result.index for result in results}
    ordered.extend(chunk for index, chunk in enumerate(scored) if index not in returned)

    if settings.rerank_min_score is not None:
        floor = settings.rerank_min_score
        below = [
            chunk
            for chunk in ordered
            if chunk.rerank_score is not None and chunk.rerank_score < floor
        ]
        if below:
            ledger.add(below, DROP_RERANK_MIN_SCORE)
            excluded = {chunk.payload.chunk_id for chunk in below}
            ordered = [
                chunk for chunk in ordered if chunk.payload.chunk_id not in excluded
            ]

    if len(ordered) > settings.rerank_top_n:
        ledger.add(ordered[settings.rerank_top_n :], DROP_RERANK_TOP_N)
        ordered = ordered[: settings.rerank_top_n]
    return ordered


def _relevance(chunk: RetrievedChunk, *, scale: float, calibrated: bool) -> float:
    """Map a candidate's ranking signal onto the ``[0, 1]`` relevance scale.

    ``guardrail_ood_min_score`` and ``eval`` both read ``final_score`` as a
    probability-like relevance, so the raw signal has to be mapped. A
    ``bge-reranker`` logit is trained with binary cross-entropy, which makes its
    logistic transform the model's own estimate of "this chunk answers this query".
    A reciprocal-rank-fusion score carries no such calibration, so it is divided by
    a configured reference instead.

    Args:
        chunk: The candidate.
        scale: ``retrieval_fusion_score_scale``, the fusion reference.
        calibrated: Whether ``rerank_score`` came from a real cross-encoder. The
            no-op reranker's scores encode input order only, so they are ignored.

    Returns:
        A score in ``[0, 1]``.
    """
    if calibrated and chunk.rerank_score is not None:
        # Guard against overflow on a large negative logit.
        logit = max(-60.0, min(60.0, chunk.rerank_score))
        return 1.0 / (1.0 + math.exp(-logit))
    if scale <= 0.0:
        return 0.0
    return max(0.0, min(1.0, chunk.fusion_score / scale))


async def _dense_vectors(
    chunks: Sequence[RetrievedChunk],
    *,
    client: AsyncQdrantClient,
    principal: Principal,
    filters: MetadataFilter | None,
    collection: str,
    settings: Settings,
    embedder: EmbeddingProvider,
) -> list[list[float]]:
    """Fetch the dense vector of each candidate so MMR can measure redundancy.

    Read back from Qdrant first — the vectors are already there and exact — through
    :func:`build_acl_filter_for_chunk_ids`, so even this bookkeeping read is
    tenant-scoped. Anything the read misses is re-embedded locally when
    ``retrieval_mmr_embed_fallback`` allows it.

    Args:
        chunks: The candidates, in current order.
        client: Async Qdrant client.
        principal: The authenticated caller.
        filters: The caller's metadata filter, reapplied for consistency.
        collection: Chunk collection name.
        settings: Active settings.
        embedder: Fallback embedding provider.

    Returns:
        One vector per candidate, in order. An entry is an empty list when the vector
        could not be obtained, which :mod:`app.rag.mmr` treats as "no redundancy
        signal" rather than as an error.
    """
    chunk_ids = [chunk.payload.chunk_id for chunk in chunks]
    found: dict[str, list[float]] = {}
    try:
        response = await client.query_points(
            collection_name=collection,
            query_filter=build_acl_filter_for_chunk_ids(principal, chunk_ids, filters),
            limit=len(chunk_ids),
            with_payload=["chunk_id"],
            with_vectors=True,
        )
        for point in response.points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            vector = _named_vector(point.vector)
            if chunk_id and vector:
                found[chunk_id] = vector
    except Exception as exc:  # diversity is best-effort
        _log.warning("retrieval.vector_fetch_failed", error=type(exc).__name__)

    missing = [chunk for chunk in chunks if chunk.payload.chunk_id not in found]
    if missing and settings.retrieval_mmr_embed_fallback:
        try:
            texts = [
                truncate_for_embedding(
                    chunk.payload.embed_text, settings.embedding_max_chars
                )
                for chunk in missing
            ]
            vectors = await embedder.embed_dense(texts)
            for chunk, vector in zip(missing, vectors, strict=False):
                found[chunk.payload.chunk_id] = list(vector)
        except Exception as exc:  # diversity is best-effort
            _log.warning("retrieval.vector_embed_failed", error=type(exc).__name__)

    return [found.get(chunk.payload.chunk_id, []) for chunk in chunks]


def _named_vector(vector: Any) -> list[float]:
    """Extract the dense vector from a point's vector field.

    Every collection in this platform uses a *named* dense vector, so Qdrant returns
    a mapping. A bare list is accepted too, for a client or fixture that returns the
    unnamed form.

    Args:
        vector: The ``ScoredPoint.vector`` value.

    Returns:
        The dense vector, or an empty list when there is none.
    """
    if isinstance(vector, Mapping):
        candidate = vector.get(DENSE)
    else:
        candidate = vector
    if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
        return [float(component) for component in candidate]
    return []


def _diversify(
    chunks: list[RetrievedChunk],
    vectors: Sequence[Sequence[float]],
    *,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Reorder survivors for coverage with maximal marginal relevance.

    Args:
        chunks: Reranked survivors, best-first.
        vectors: Dense vector per survivor, aligned with ``chunks``.
        settings: Settings supplying ``retrieval_mmr_lambda``.

    Returns:
        The same candidates in diversified order. Nothing is dropped here — the
        per-document cap and ``retrieval_top_n`` do that, with reasons.
    """
    if len(chunks) < 2:
        return chunks
    relevance = [
        chunk.rerank_score if chunk.rerank_score is not None else chunk.fusion_score
        for chunk in chunks
    ]
    selection = maximal_marginal_relevance(
        vectors, relevance, top_n=None, settings=settings
    )
    return [chunks[index] for index in selection.order]


def _finalise(
    chunks: list[RetrievedChunk],
    *,
    top_n: int,
    settings: Settings,
    calibrated: bool,
    ledger: _Ledger,
) -> list[RetrievedChunk]:
    """Apply the per-document cap and the final cut, scoring everything on the way.

    Args:
        chunks: Diversified survivors, best-first.
        top_n: Final number of chunks to keep.
        settings: Settings supplying ``retrieval_max_per_document``.
        calibrated: Whether cross-encoder scores are trustworthy relevance signals.
        ledger: Where the final exclusions are recorded.

    Returns:
        The chunks handed to context assembly, each carrying a ``final_score``.
    """
    scale = float(settings.retrieval_fusion_score_scale)
    scored = [
        chunk.model_copy(
            update={
                "final_score": _relevance(chunk, scale=scale, calibrated=calibrated)
            }
        )
        for chunk in chunks
    ]

    kept: list[RetrievedChunk] = []
    per_document: dict[str, int] = {}
    overflow: list[RetrievedChunk] = []
    tail: list[RetrievedChunk] = []
    for chunk in scored:
        if len(kept) >= top_n:
            tail.append(chunk)
            continue
        document_id = chunk.payload.document_id
        if per_document.get(document_id, 0) >= settings.retrieval_max_per_document:
            overflow.append(chunk)
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        kept.append(chunk)

    ledger.add(overflow, DROP_MAX_PER_DOCUMENT)
    ledger.add(tail, DROP_TOP_N)
    return kept


async def retrieve(
    principal: Principal,
    queries: Sequence[str],
    filters: MetadataFilter | None = None,
    *,
    top_n: int | None = None,
    hyde_passage: str = "",
    rerank_query: str | None = None,
    include_deleted: bool | None = None,
    settings: Settings | None = None,
    client: AsyncQdrantClient | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    collection: str | None = None,
) -> RetrievalResult:
    """Run pipeline stage 5 and return the full audited result.

    Args:
        principal: The authenticated caller. Every Qdrant request this function makes
            is scoped by :func:`build_acl_filter`, which derives from this principal.
        queries: The rewritten standalone question first, then any sub-questions from
            stage 3. Passing the raw user message is valid when transformation
            degraded.
        filters: Facets to narrow by, already merged with anything stage 3 extracted.
        top_n: Final chunk count. Defaults to ``settings.retrieval_top_n``.
        hyde_passage: Stage 3's hypothetical passage, used as an additional dense
            probe. Empty when HyDE did not fire.
        rerank_query: Text the cross-encoder scores against. Defaults to the first
            query, which is the user's actual question.
        include_deleted: Forensic override. Defaults to
            ``settings.retrieval_include_deleted``, which is False and should stay
            that way — soft-deleted chunks are removed content.
        settings: Active settings. Defaults to :func:`ragcore.settings.get_settings`.
        client: Async Qdrant client. Defaults to the shared cached client.
        embedder: Embedding provider. Defaults to the cached FastEmbed provider.
        reranker: Cross-encoder. Defaults to the cached reranker, which is the no-op
            when ``rerank_enabled`` is False.
        collection: Chunk collection. Defaults to
            ``settings.qdrant_chunks_collection``.

    Returns:
        A :class:`~ragcore.models.retrieval.RetrievalResult` with every field
        populated: the ordered chunks, the probes actually issued, the serialised
        filter, the three stage counters, per-stage latencies and every dropped
        candidate with its reason.

    Raises:
        RetrievalError: If every probe failed against Qdrant.
    """
    cfg = settings or get_settings()
    clock = _Clock()
    ledger = _Ledger()

    probes = _build_probes(queries, hyde_passage, settings=cfg)
    limit = cfg.retrieval_top_n if top_n is None else max(0, top_n)
    deleted = (
        cfg.retrieval_include_deleted if include_deleted is None else include_deleted
    )
    chunk_collection = collection or cfg.qdrant_chunks_collection
    qfilter = build_acl_filter(principal, filters, include_deleted=deleted)
    filter_applied = serialise_filter(qfilter)

    if not probes:
        return RetrievalResult(
            queries_used=[],
            filter_applied=filter_applied,
            latency_ms=clock.finish(),
        )

    tracer = get_tracer(cfg)
    async with tracer.span(
        "rag.retrieve",
        metadata={
            "probes": len(probes),
            "tenant_id": principal.tenant_id,
            "top_n": limit,
            "has_filter": filters is not None,
            "hyde": bool(hyde_passage.strip()),
        },
    ):
        provider = embedder or get_embedding_provider(cfg)
        ranker = reranker or get_reranker(cfg)
        qdrant = client or await get_client(cfg)

        mark = clock.start()
        embeddings = await _embed_probes(probes, embedder=provider, settings=cfg)
        clock.stop("embed", mark)
        if all(embedded is None for embedded in embeddings):
            # Returning nothing here would look exactly like an out-of-domain query
            # and the OOD gate would tell the user the corpus does not cover it. An
            # embedder outage is not a coverage answer.
            msg = "every retrieval probe failed to embed"
            raise RetrievalError(msg)

        mark = clock.start()
        batches = await _search_probes(
            qdrant,
            probes,
            embeddings,
            qfilter=qfilter,
            collection=chunk_collection,
            settings=cfg,
        )
        candidates, total_candidates = _union(
            batches, principal=principal, settings=cfg, ledger=ledger
        )
        clock.stop("search", mark, candidates=total_candidates)

        mark = clock.start()
        deduped = _dedupe(candidates, settings=cfg, ledger=ledger)
        clock.stop("dedupe", mark, candidates=len(deduped))

        mark = clock.start()
        survivors = await _rerank(
            deduped,
            query=(rerank_query or probes[0].text),
            reranker=ranker,
            settings=cfg,
            ledger=ledger,
        )
        clock.stop("rerank", mark, candidates=len(survivors))

        ordered = survivors
        if cfg.retrieval_mmr_enabled and len(survivors) > 1:
            mark = clock.start()
            vectors = await _dense_vectors(
                survivors,
                client=qdrant,
                principal=principal,
                filters=filters,
                collection=chunk_collection,
                settings=cfg,
                embedder=provider,
            )
            clock.stop("vectors", mark)
            mark = clock.start()
            ordered = _diversify(survivors, vectors, settings=cfg)
            clock.stop("mmr", mark, candidates=len(ordered))

        calibrated = cfg.rerank_enabled and not isinstance(ranker, NoopReranker)
        kept = _finalise(
            ordered, top_n=limit, settings=cfg, calibrated=calibrated, ledger=ledger
        )

    result = RetrievalResult(
        chunks=kept,
        queries_used=[probe.text for probe in probes],
        filter_applied=filter_applied,
        total_candidates=total_candidates,
        after_dedupe=len(deduped),
        after_rerank=len(survivors),
        latency_ms=clock.finish(),
        cache_hit=False,
        dropped=ledger.dropped,
    )
    _log.info(
        "retrieval.done",
        tenant_id=principal.tenant_id,
        probes=len(probes),
        total_candidates=total_candidates,
        after_dedupe=result.after_dedupe,
        after_rerank=result.after_rerank,
        kept=len(result.chunks),
        dropped=len(result.dropped),
        acl_rejected=ledger.acl_rejected,
        payload_invalid=ledger.payload_invalid,
        max_score=result.max_score,
        latency_ms=result.latency_ms.get("total"),
    )
    return result


async def retrieve_by_ids(
    principal: Principal,
    chunk_ids: Sequence[str],
    filters: MetadataFilter | None = None,
    *,
    queries: Sequence[str] = (),
    top_n: int | None = None,
    include_deleted: bool | None = None,
    settings: Settings | None = None,
    client: AsyncQdrantClient | None = None,
    collection: str | None = None,
) -> RetrievalResult:
    """Re-fetch a cached chunk set through the live ACL filter (stage 4's hit path).

    The semantic cache stores the retrieval *plan* — the chunk ids — never the
    rendered answer, precisely so that reuse can be re-authorised. This function is
    that re-authorisation: the ids are looked up through
    :func:`build_acl_filter_for_chunk_ids`, so a principal who has since lost access
    to a document simply gets fewer chunks instead of a stale leak, and a principal
    from another tenant gets none at all.

    Args:
        principal: The authenticated caller.
        chunk_ids: Cached chunk ids, in the order the original retrieval ranked them.
        filters: The caller's metadata filter, reapplied so a cache hit obeys a
            narrower filter than the one that populated it.
        queries: The transformed queries the cache entry recorded, echoed back in
            ``queries_used`` so the trace still shows what was searched.
        top_n: Final chunk count. Defaults to ``settings.retrieval_top_n``.
        include_deleted: Forensic override; defaults to the settings value.
        settings: Active settings.
        client: Async Qdrant client. Defaults to the shared cached client.
        collection: Chunk collection. Defaults to
            ``settings.qdrant_chunks_collection``.

    Returns:
        A :class:`~ragcore.models.retrieval.RetrievalResult` with ``cache_hit=True``,
        the surviving chunks in their cached order, and ``total_candidates`` equal to
        the number of ids asked for — so the difference between that and
        ``len(chunks)`` is exactly what the live ACL filter removed.
    """
    cfg = settings or get_settings()
    clock = _Clock()
    ledger = _Ledger()

    limit = cfg.retrieval_top_n if top_n is None else max(0, top_n)
    deleted = (
        cfg.retrieval_include_deleted if include_deleted is None else include_deleted
    )
    chunk_collection = collection or cfg.qdrant_chunks_collection
    wanted = [chunk_id for chunk_id in chunk_ids if chunk_id]
    qfilter = build_acl_filter_for_chunk_ids(
        principal, wanted, filters, include_deleted=deleted
    )
    filter_applied = serialise_filter(qfilter)

    if not wanted:
        _log.warning("retrieval.cache_empty", tenant_id=principal.tenant_id)
        return RetrievalResult(
            queries_used=list(queries),
            filter_applied=filter_applied,
            latency_ms=clock.finish(),
            cache_hit=True,
        )

    qdrant = client or await get_client(cfg)
    mark = clock.start()
    response = await qdrant.query_points(
        collection_name=chunk_collection,
        query_filter=qfilter,
        limit=len(wanted),
        with_payload=True,
        with_vectors=False,
    )
    clock.stop("search", mark, candidates=len(response.points))

    by_id: dict[str, RetrievedChunk] = {}
    for point in response.points:
        candidate = _to_candidate(
            point, principal=principal, settings=cfg, ledger=ledger
        )
        if candidate is None:
            continue
        by_id[candidate.payload.chunk_id] = candidate

    # Cached order is the ranking the original retrieval produced; there is nothing
    # to rerank, so the score is derived from that rank and stays in [0, 1] for the
    # out-of-domain gate.
    total = len(wanted)
    ordered: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(wanted):
        candidate = by_id.get(chunk_id)
        if candidate is None:
            continue
        ordered.append(
            candidate.model_copy(
                update={
                    "retrieval_stage": RetrievalStage.CACHE.value,
                    "final_score": (total - rank) / total,
                }
            )
        )

    kept = ordered[:limit]
    ledger.add(ordered[limit:], DROP_TOP_N)
    result = RetrievalResult(
        chunks=kept,
        queries_used=list(queries),
        filter_applied=filter_applied,
        total_candidates=total,
        after_dedupe=len(ordered),
        after_rerank=len(ordered),
        latency_ms=clock.finish(),
        cache_hit=True,
        dropped=ledger.dropped,
    )
    _log.info(
        "retrieval.cache_hit",
        tenant_id=principal.tenant_id,
        requested=total,
        resolved=len(ordered),
        kept=len(kept),
        acl_rejected=ledger.acl_rejected,
        missing=total - len(ordered) - ledger.acl_rejected,
        latency_ms=result.latency_ms.get("total"),
    )
    return result
