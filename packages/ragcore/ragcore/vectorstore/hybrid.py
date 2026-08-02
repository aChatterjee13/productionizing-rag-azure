"""Hybrid dense + BM25 retrieval with **server-side** fusion (requirement #6).

One request to the Qdrant Query API carries two ``Prefetch`` branches — dense
(``using="dense"``) and sparse (``using="sparse"``) — each with the *same* ACL filter,
and a top-level ``FusionQuery`` that fuses them inside Qdrant.

Fusing in Python would be a correctness bug, not just a slow path: it would mean
pulling two candidate lists over the wire, and every intermediate list would be one
more place where the ACL filter has to be remembered. Keeping filtering, scoring and
fusion in Qdrant means the filter is applied once, by the engine, to both branches —
and the same filter is repeated on the outer query as defence in depth, so a future
edit that dropped it from a prefetch branch still cannot leak.

Two fusion strategies, chosen by ``settings.retrieval_fusion``:

* ``rrf`` — Reciprocal Rank Fusion. Rank-based, so the wildly different score scales
  of cosine similarity and BM25 never need normalising. The default.
* ``dbsf`` — Distribution-Based Score Fusion. Normalises each branch's score
  distribution before combining; better when one branch is much stronger.
"""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from ragcore.embeddings.base import SparseVec
from ragcore.vectorstore.collections import DENSE, SPARSE

__all__ = ["FUSIONS", "dense_search", "hybrid_search", "resolve_fusion"]

_log = structlog.get_logger(__name__)

#: Accepted ``fusion`` values mapped to the Qdrant enum.
FUSIONS: dict[str, qm.Fusion] = {
    "rrf": qm.Fusion.RRF,
    "dbsf": qm.Fusion.DBSF,
}


def resolve_fusion(fusion: str) -> qm.Fusion:
    """Map a settings string onto the Qdrant fusion enum.

    Args:
        fusion: ``"rrf"`` or ``"dbsf"`` (case-insensitive).

    Returns:
        The corresponding :class:`qdrant_client.models.Fusion` member.

    Raises:
        ValueError: If the strategy is unknown. ``settings.retrieval_fusion`` is a
            ``Literal``, so reaching this means a call site passed a raw string.
    """
    try:
        return FUSIONS[fusion.lower()]
    except KeyError as exc:
        msg = f"unknown fusion strategy {fusion!r}; expected one of {sorted(FUSIONS)}"
        raise ValueError(msg) from exc


def _sparse_query(sparse: SparseVec | None) -> qm.SparseVector | None:
    """Convert a provider sparse vector into the Qdrant wire form.

    Args:
        sparse: Sparse vector from the embedding provider, or None.

    Returns:
        The Qdrant sparse vector, or None when there are no terms — an empty sparse
        query matches nothing, so the branch is dropped instead of sent.
    """
    if sparse is None or not sparse.indices:
        return None
    return qm.SparseVector(indices=list(sparse.indices), values=list(sparse.values))


async def hybrid_search(
    client: AsyncQdrantClient,
    *,
    collection: str,
    query_text: str,
    dense: Sequence[float],
    sparse: SparseVec | None,
    qfilter: qm.Filter,
    limit: int,
    prefetch_limit: int,
    fusion: str = "rrf",
) -> list[qm.ScoredPoint]:
    """Run one hybrid query and return the fused, raw scored points.

    Args:
        client: Async Qdrant client.
        collection: Collection to query, normally ``settings.qdrant_chunks_collection``.
        query_text: The query as text. Not sent to Qdrant — both vectors are already
            computed — but carried for tracing. Only its length is ever logged: it is
            raw user content and has not necessarily passed PII redaction.
        dense: Dense embedding of the query (1024-d for ``BAAI/bge-m3``).
        sparse: BM25 sparse embedding of the query. May be None or empty (a query of
            pure stopwords), in which case the sparse branch is skipped.
        qfilter: The filter from :func:`ragcore.vectorstore.filters.build_acl_filter`.
            Applied to both prefetch branches and to the outer query.
        limit: Number of fused results to return.
        prefetch_limit: Candidates each branch contributes before fusion. Should
            exceed ``limit`` — that headroom is what fusion has to work with.
        fusion: ``"rrf"`` or ``"dbsf"``.

    Returns:
        The fused :class:`qdrant_client.models.ScoredPoint` list, payloads included
        and vectors excluded. Raw points: mapping them onto
        :class:`~ragcore.models.retrieval.RetrievedChunk` is the retriever's job.

    Raises:
        ValueError: If ``fusion`` is not a known strategy.
    """
    strategy = resolve_fusion(fusion)

    prefetch: list[qm.Prefetch] = [
        qm.Prefetch(
            query=list(dense),
            using=DENSE,
            filter=qfilter,
            limit=prefetch_limit,
        )
    ]
    sparse_query = _sparse_query(sparse)
    if sparse_query is not None:
        prefetch.append(
            qm.Prefetch(
                query=sparse_query,
                using=SPARSE,
                filter=qfilter,
                limit=prefetch_limit,
            )
        )

    response = await client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=strategy),
        # Same filter on the fused result set. Redundant with the prefetch filters by
        # design: the ACL boundary should survive an edit to either layer.
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points = list(response.points)
    _log.debug(
        "qdrant.hybrid_search",
        collection=collection,
        fusion=strategy.value,
        branches=len(prefetch),
        sparse_terms=len(sparse.indices) if sparse is not None else 0,
        query_chars=len(query_text),
        limit=limit,
        prefetch_limit=prefetch_limit,
        returned=len(points),
    )
    return points


async def dense_search(
    client: AsyncQdrantClient,
    *,
    collection: str,
    dense: Sequence[float],
    qfilter: qm.Filter,
    limit: int,
    score_threshold: float | None = None,
    using: str = DENSE,
) -> list[qm.ScoredPoint]:
    """Run a dense-only query against a single-vector collection.

    Used for ``rag_memories`` (long-term memory recall) and ``rag_semantic_cache``
    (the similar-query probe), which carry no sparse vector. Kept here so those call
    sites do not hand-roll a Query API request either.

    Args:
        client: Async Qdrant client.
        collection: Collection to query.
        dense: Dense embedding of the query.
        qfilter: Filter from :func:`ragcore.vectorstore.filters.build_memory_filter`
            or :func:`~ragcore.vectorstore.filters.build_cache_filter`.
        limit: Maximum points to return.
        score_threshold: Drop points scoring below this. For the semantic cache pass
            ``settings.memory_cache_threshold`` so the engine enforces the similarity
            floor rather than the caller.
        using: Named vector to search. Every collection in this platform names its
            dense vector :data:`~ragcore.vectorstore.collections.DENSE`.

    Returns:
        The scored points, payloads included and vectors excluded.
    """
    response = await client.query_points(
        collection_name=collection,
        query=list(dense),
        using=using,
        query_filter=qfilter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )
    points = list(response.points)
    _log.debug(
        "qdrant.dense_search",
        collection=collection,
        limit=limit,
        score_threshold=score_threshold,
        returned=len(points),
    )
    return points
