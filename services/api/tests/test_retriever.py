"""Stage 5 against a mocked Qdrant: ordering, dedupe and drop accounting.

The invariants pinned here are the ones requirement #6 and requirement #9 are made
of. Every candidate Qdrant returned is either in ``chunks`` or in ``dropped`` with a
reason — with one deliberate exception, the ACL rejections, which must not be
returned at all because ``dropped`` is serialised to the client. And every read is
scoped by a filter that came from ``ragcore.vectorstore.filters``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models as qm

from app.rag.retriever import (
    DROP_CANDIDATE_LIMIT,
    DROP_MAX_PER_DOCUMENT,
    DROP_RERANK_MIN_SCORE,
    DROP_RERANK_TOP_N,
    DROP_TOP_N,
    retrieve,
    retrieve_by_ids,
)
from ragcore.dedupe import content_sha256, simhash_hex
from ragcore.embeddings import Embedded, SparseVec
from ragcore.errors import RetrievalError
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.retrieval import MetadataFilter
from ragcore.rerank.base import RerankResult
from ragcore.settings import Settings
from ragcore.vectorstore.filters import build_acl_filter

TENANT = "tenant-acme"
OTHER_TENANT = "tenant-globex"


def make_settings(**overrides: Any) -> Settings:
    """Settings isolated from the developer's own .env, with MMR off by default."""
    base: dict[str, Any] = {
        "rerank_enabled": True,
        "retrieval_mmr_enabled": False,
        "langfuse_enabled": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def principal(tenant: str = TENANT, **overrides: Any) -> Principal:
    return Principal(
        user_id="user-1",
        tenant_id=tenant,
        roles=["rag.user"],
        groups=["g-acme-engineering"],
        max_classification=Classification.CONFIDENTIAL,
        **overrides,
    )


def payload(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    tenant_id: str = TENANT,
    classification: Classification = Classification.PUBLIC,
    allowed_groups: list[str] | None = None,
    is_deleted: bool = False,
) -> ChunkPayload:
    access = AccessControl(
        tenant_id=tenant_id,
        allowed_groups=allowed_groups or [],
        classification=classification,
    )
    now = datetime.now(UTC)
    return ChunkPayload.from_access_control(
        access,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        source_type="blob",
        source_id="src-1",
        source_uri=f"https://example.invalid/{document_id}",
        title="Travel Policy",
        section_path=["Meals"],
        page=1,
        text=text,
        # The chunk id rides in the header (not the text) so FakeReranker can score
        # by id while content_sha256 and simhash still see only the chunk's words.
        contextual_header=f"Travel Policy > Meals ({chunk_id})",
        summary=None,
        keywords=[],
        doc_type="policy",
        tags=[],
        author=None,
        language="en",
        content_sha256=content_sha256(text),
        simhash=simhash_hex(text),
        token_count=max(1, len(text) // 4),
        source_modified_at=now,
        effective_from=None,
        effective_to=None,
        created_at=now,
        updated_at=now,
        version=1,
        is_deleted=is_deleted,
        pii_types=[],
        pii_redacted=True,
        ingest_run_id="run-1",
    )


def point(chunk: ChunkPayload, score: float) -> qm.ScoredPoint:
    return qm.ScoredPoint(
        id="00000000-0000-0000-0000-000000000000",
        version=1,
        score=score,
        payload=chunk.to_qdrant_payload(),
    )


class FakeEmbedder:
    """Deterministic embedder: each distinct text gets its own basis vector."""

    dim = 8

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        """Record the texts seen so a vector can be reversed back to its probe."""
        self.texts: list[str] = []
        self.fail_on = fail_on or set()

    def index_of(self, text: str) -> int:
        if text not in self.texts:
            self.texts.append(text)
        return self.texts.index(text)

    def _dense(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        vector[self.index_of(text) % self.dim] = 1.0
        return vector

    async def embed_query(self, text: str) -> Embedded:
        if text in self.fail_on:
            raise RuntimeError("embedding backend unavailable")
        return Embedded(
            dense=self._dense(text), sparse=SparseVec(indices=[1], values=[1.0])
        )

    async def embed_documents(self, texts: Any) -> list[Embedded]:
        return [await self.embed_query(text) for text in texts]

    async def embed_dense(self, texts: Any) -> list[list[float]]:
        return [self._dense(text) for text in texts]


class FakeQdrant:
    """Answers the three shapes of request the retriever makes."""

    def __init__(
        self,
        embedder: FakeEmbedder,
        *,
        results: dict[str, list[qm.ScoredPoint]] | None = None,
        by_id: list[qm.ScoredPoint] | None = None,
        vectors: dict[str, list[float]] | None = None,
        fail_search: bool = False,
    ) -> None:
        """Configure the canned responses for each request shape."""
        self.embedder = embedder
        self.results = results or {}
        self.by_id = by_id or []
        self.vectors = vectors or {}
        self.fail_search = fail_search
        self.searches: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []

    def _probe_for(self, prefetch: list[qm.Prefetch]) -> str:
        dense = list(prefetch[0].query or [])
        index = dense.index(1.0)
        return next(
            text
            for text in self.embedder.texts
            if self.embedder.index_of(text) == index
        )

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("prefetch") is not None:
            if self.fail_search:
                raise RuntimeError("qdrant unreachable")
            self.searches.append(kwargs)
            return SimpleNamespace(
                points=self.results.get(self._probe_for(kwargs["prefetch"]), [])
            )
        if kwargs.get("with_vectors"):
            self.fetches.append(kwargs)
            return SimpleNamespace(
                points=[
                    qm.ScoredPoint(
                        id=chunk_id,
                        version=1,
                        score=0.0,
                        payload={"chunk_id": chunk_id},
                        vector={"dense": vector},
                    )
                    for chunk_id, vector in self.vectors.items()
                ]
            )
        self.fetches.append(kwargs)
        return SimpleNamespace(points=self.by_id)


class FakeReranker:
    """Returns configured logits by chunk id, defaulting to fusion order."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        """Map chunk id to the logit the fake cross-encoder will return."""
        self.scores = scores or {}
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self, query: str, documents: Any, top_n: int
    ) -> list[RerankResult]:
        docs = list(documents)
        self.calls.append((query, len(docs)))
        results = [
            RerankResult(
                index=index,
                score=next(
                    (
                        score
                        for chunk_id, score in self.scores.items()
                        if chunk_id in document
                    ),
                    float(len(docs) - index),
                ),
            )
            for index, document in enumerate(docs)
        ]
        results.sort(key=lambda item: -item.score)
        return results[:top_n]


# ------------------------------------------------------------------ the pipeline
async def test_every_candidate_is_either_kept_or_dropped_with_a_reason() -> None:
    cfg = make_settings(retrieval_top_n=2, retrieval_max_per_document=1)
    embedder = FakeEmbedder()
    keep = payload(
        "c-keep", "The daily meal allowance is EUR 60 per day.", document_id="d1"
    )
    same = payload(
        "c-dup", "The daily meal allowance is EUR 60 per day.", document_id="d2"
    )
    other = payload(
        "c-other", "Expenses above 500 EUR require director approval.", document_id="d3"
    )
    third = payload(
        "c-third", "Hotel bookings must go through the travel desk.", document_id="d4"
    )
    client = FakeQdrant(
        embedder,
        results={
            "meal allowance": [
                point(keep, 0.9),
                point(same, 0.8),
                point(other, 0.7),
                point(third, 0.6),
            ]
        },
    )
    reranker = FakeReranker({"c-keep": 4.0, "c-other": 2.0, "c-third": 1.0})

    result = await retrieve(
        principal(),
        ["meal allowance"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=reranker,
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-keep", "c-other"]
    assert result.total_candidates == 4
    assert result.after_dedupe == 3  # the exact duplicate is gone
    assert result.after_rerank == 3
    reasons = {chunk.payload.chunk_id: chunk.dropped_reason for chunk in result.dropped}
    assert reasons == {"c-dup": "duplicate:sha256", "c-third": DROP_TOP_N}
    # Nothing vanishes: kept + dropped reconstructs everything Qdrant returned.
    assert len(result.chunks) + len(result.dropped) == result.total_candidates
    assert result.cache_hit is False
    assert result.queries_used == ["meal allowance"]
    assert set(result.latency_ms) >= {"embed", "search", "dedupe", "rerank", "total"}
    assert all(value >= 0.0 for value in result.latency_ms.values())
    assert 0.0 <= result.max_score <= 1.0


async def test_scores_are_ordered_and_normalised_for_the_ood_gate() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    high = payload(
        "c-high", "Meals are reimbursed at EUR 60 per day.", document_id="d1"
    )
    low = payload(
        "c-low", "The office plant rota is published monthly.", document_id="d2"
    )
    client = FakeQdrant(embedder, results={"q": [point(high, 0.5), point(low, 0.4)]})
    reranker = FakeReranker({"c-high": 5.0, "c-low": -6.0})

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=reranker,
    )

    scores = [chunk.final_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.99  # sigmoid(5.0)
    assert scores[1] < 0.01  # sigmoid(-6.0)
    assert result.chunks[0].rerank_score == 5.0


async def test_the_union_keeps_the_best_score_per_chunk() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    shared = payload("c-shared", "Contractors follow the same expense rules.")
    client = FakeQdrant(
        embedder,
        results={
            "main question": [point(shared, 0.30)],
            "sub question": [point(shared, 0.80)],
        },
    )

    result = await retrieve(
        principal(),
        ["main question", "sub question"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].fusion_score == pytest.approx(0.80)
    assert result.total_candidates == 2  # two points seen, one candidate
    assert result.after_dedupe == 1
    assert len(client.searches) == 2


async def test_a_hyde_passage_becomes_an_extra_probe() -> None:
    cfg = make_settings(qt_hyde_max_chars=200)
    embedder = FakeEmbedder()
    client = FakeQdrant(embedder, results={})

    result = await retrieve(
        principal(),
        ["how do we think about remote work?"],
        hyde_passage="Remote work is permitted where the role allows it.",
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert len(client.searches) == 2
    assert result.queries_used == [
        "how do we think about remote work?",
        "Remote work is permitted where the role allows it.",
    ]


async def test_rerank_bounds_are_enforced_by_the_retriever_not_the_reranker() -> None:
    cfg = make_settings(
        rerank_candidate_limit=3,
        rerank_top_n=2,
        rerank_min_score=0.0,
        retrieval_top_n=8,
    )
    embedder = FakeEmbedder()
    chunks = [
        payload(
            f"c{index}",
            f"Distinct paragraph number {index} about travel.",
            document_id=f"d{index}",
        )
        for index in range(5)
    ]
    client = FakeQdrant(
        embedder,
        results={
            "q": [point(chunk, 1.0 - index / 10) for index, chunk in enumerate(chunks)]
        },
    )
    reranker = FakeReranker({"c0": 3.0, "c1": -1.0, "c2": 2.0})

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=reranker,
    )

    assert reranker.calls[0][1] == 3  # only the candidate limit reached the model
    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c0", "c2"]
    reasons = {chunk.payload.chunk_id: chunk.dropped_reason for chunk in result.dropped}
    assert reasons["c3"] == DROP_CANDIDATE_LIMIT
    assert reasons["c4"] == DROP_CANDIDATE_LIMIT
    assert reasons["c1"] == DROP_RERANK_MIN_SCORE
    assert DROP_RERANK_TOP_N not in reasons.values()


async def test_one_document_cannot_monopolise_the_result() -> None:
    cfg = make_settings(retrieval_max_per_document=2, retrieval_top_n=5)
    embedder = FakeEmbedder()
    hoggers = [
        payload(
            f"h{index}",
            f"Section {index} of the very same handbook document.",
            document_id="d-hog",
        )
        for index in range(4)
    ]
    outsider = payload(
        "c-out", "A paragraph from a different document entirely.", document_id="d-out"
    )
    client = FakeQdrant(
        embedder,
        results={
            "q": [
                point(chunk, 0.9 - index / 100)
                for index, chunk in enumerate([*hoggers, outsider])
            ]
        },
    )

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    kept = [chunk.payload.chunk_id for chunk in result.chunks]
    assert kept == ["h0", "h1", "c-out"]
    reasons = {chunk.payload.chunk_id: chunk.dropped_reason for chunk in result.dropped}
    assert reasons == {"h2": DROP_MAX_PER_DOCUMENT, "h3": DROP_MAX_PER_DOCUMENT}


async def test_mmr_diversifies_a_pool_of_near_identical_chunks() -> None:
    cfg = make_settings(
        retrieval_mmr_enabled=True, retrieval_mmr_lambda=0.5, retrieval_top_n=2
    )
    embedder = FakeEmbedder()
    twins = [
        payload("t1", "Meal allowance guidance, first phrasing.", document_id="d1"),
        payload("t2", "Meal allowance guidance, second phrasing.", document_id="d2"),
    ]
    different = payload(
        "t3", "Visa requirements for business travel.", document_id="d3"
    )
    client = FakeQdrant(
        embedder,
        results={
            "q": [point(twins[0], 0.9), point(twins[1], 0.89), point(different, 0.5)]
        },
        vectors={"t1": [1.0, 0.0], "t2": [1.0, 0.0], "t3": [0.0, 1.0]},
    )
    reranker = FakeReranker({"t1": 3.0, "t2": 2.9, "t3": 1.0})

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=reranker,
    )

    # Without MMR the top two would be the two near-identical chunks.
    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["t1", "t3"]
    assert client.fetches, "MMR must read the dense vectors back through Qdrant"
    assert "mmr" in result.latency_ms


async def test_mmr_falls_back_to_the_embedder_when_qdrant_has_no_vectors() -> None:
    cfg = make_settings(retrieval_mmr_enabled=True, retrieval_top_n=3)
    embedder = FakeEmbedder()
    chunks = [
        payload("v1", "First distinct paragraph.", document_id="d1"),
        payload("v2", "Second distinct paragraph.", document_id="d2"),
    ]
    client = FakeQdrant(
        embedder,
        results={"q": [point(chunks[0], 0.9), point(chunks[1], 0.8)]},
        vectors={},
    )

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert len(result.chunks) == 2


# ------------------------------------------------------------------- ACL and I/O
async def test_the_acl_filter_comes_from_ragcore_and_reaches_every_branch() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    client = FakeQdrant(embedder, results={})
    caller = principal()
    filters = MetadataFilter(doc_types=["policy"])

    result = await retrieve(
        caller,
        ["q"],
        filters,
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    expected = build_acl_filter(caller, filters)
    request = client.searches[0]
    assert request["query_filter"] == expected
    assert all(branch.filter == expected for branch in request["prefetch"])
    assert result.filter_applied
    assert result.filter_applied["must"]


async def test_a_cross_tenant_chunk_is_discarded_and_never_reported() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    mine = payload("c-mine", "A paragraph this tenant owns.")
    theirs = payload(
        "c-theirs", "Another tenant's confidential paragraph.", tenant_id=OTHER_TENANT
    )
    client = FakeQdrant(
        embedder, results={"q": [point(theirs, 0.99), point(mine, 0.5)]}
    )

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-mine"]
    # Not in `dropped` either: that list is serialised to the client and would leak
    # the other tenant's title, URI and section path.
    assert all(chunk.payload.chunk_id != "c-theirs" for chunk in result.dropped)
    assert result.total_candidates == 2
    serialised = result.without_text()
    assert "c-theirs" not in str(serialised)


async def test_a_chunk_above_the_principals_clearance_is_discarded() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    secret = payload(
        "c-secret",
        "Restricted incident detail.",
        classification=Classification.RESTRICTED,
    )
    ok = payload("c-ok", "Public travel guidance.")
    client = FakeQdrant(embedder, results={"q": [point(secret, 0.99), point(ok, 0.1)]})

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-ok"]
    assert all(chunk.payload.chunk_id != "c-secret" for chunk in result.dropped)


async def test_a_tombstoned_chunk_is_dropped_with_a_reason() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    gone = payload("c-gone", "Content removed at source.", is_deleted=True)
    live = payload("c-live", "Current guidance.")
    client = FakeQdrant(embedder, results={"q": [point(gone, 0.9), point(live, 0.8)]})

    result = await retrieve(
        principal(),
        ["q"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-live"]
    assert [chunk.dropped_reason for chunk in result.dropped] == ["deleted"]


async def test_one_failing_probe_degrades_but_all_of_them_raise() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder(fail_on={"broken"})
    survivor = payload("c1", "Still retrievable.")
    client = FakeQdrant(embedder, results={"fine": [point(survivor, 0.7)]})

    partial = await retrieve(
        principal(),
        ["fine", "broken"],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )
    assert [chunk.payload.chunk_id for chunk in partial.chunks] == ["c1"]

    dead = FakeQdrant(FakeEmbedder(), fail_search=True)
    with pytest.raises(RetrievalError):
        await retrieve(
            principal(),
            ["fine"],
            settings=cfg,
            client=dead,
            embedder=dead.embedder,
            reranker=FakeReranker(),
        )


async def test_no_usable_query_returns_an_empty_but_complete_result() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    client = FakeQdrant(embedder)

    result = await retrieve(
        principal(),
        ["  "],
        settings=cfg,
        client=client,
        embedder=embedder,
        reranker=FakeReranker(),
    )

    assert result.chunks == []
    assert result.queries_used == []
    assert result.filter_applied
    assert result.latency_ms["total"] >= 0.0
    assert client.searches == []


# ------------------------------------------------------- the semantic cache path
async def test_retrieve_by_ids_preserves_cached_order_and_reapplies_the_acl() -> None:
    cfg = make_settings(retrieval_top_n=8)
    embedder = FakeEmbedder()
    first = payload("c-1", "First cached chunk.", document_id="d1")
    second = payload("c-2", "Second cached chunk.", document_id="d2")
    # Qdrant returns points in id order, not the cached ranking order.
    client = FakeQdrant(embedder, by_id=[point(second, 0.0), point(first, 0.0)])
    caller = principal()

    result = await retrieve_by_ids(
        caller,
        ["c-1", "c-2", "c-revoked"],
        queries=["the original question"],
        settings=cfg,
        client=client,
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-1", "c-2"]
    assert result.cache_hit is True
    assert result.total_candidates == 3
    assert result.after_dedupe == 2
    assert result.queries_used == ["the original question"]
    assert all(chunk.retrieval_stage == "cache" for chunk in result.chunks)
    scores = [chunk.final_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert result.filter_applied["must"]


async def test_retrieve_by_ids_never_serves_another_tenants_cached_chunk() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder()
    theirs = payload("c-theirs", "Another tenant's paragraph.", tenant_id=OTHER_TENANT)
    client = FakeQdrant(embedder, by_id=[point(theirs, 0.0)])

    result = await retrieve_by_ids(
        principal(), ["c-theirs"], settings=cfg, client=client
    )

    assert result.chunks == []
    assert result.dropped == []
    assert result.total_candidates == 1


async def test_retrieve_by_ids_with_no_ids_does_not_query() -> None:
    cfg = make_settings()
    client = FakeQdrant(FakeEmbedder())

    result = await retrieve_by_ids(principal(), [], settings=cfg, client=client)

    assert result.chunks == []
    assert result.cache_hit is True
    assert client.fetches == []


async def test_retrieve_by_ids_trims_to_top_n_with_an_audited_reason() -> None:
    cfg = make_settings(retrieval_top_n=1)
    embedder = FakeEmbedder()
    first = payload("c-1", "First cached chunk.", document_id="d1")
    second = payload("c-2", "Second cached chunk.", document_id="d2")
    client = FakeQdrant(embedder, by_id=[point(first, 0.0), point(second, 0.0)])

    result = await retrieve_by_ids(
        principal(), ["c-1", "c-2"], settings=cfg, client=client
    )

    assert [chunk.payload.chunk_id for chunk in result.chunks] == ["c-1"]
    assert [chunk.dropped_reason for chunk in result.dropped] == [DROP_TOP_N]


async def test_a_total_embedding_outage_is_an_error_not_an_empty_corpus() -> None:
    cfg = make_settings()
    embedder = FakeEmbedder(fail_on={"q"})
    client = FakeQdrant(embedder)

    with pytest.raises(RetrievalError):
        await retrieve(
            principal(),
            ["q"],
            settings=cfg,
            client=client,
            embedder=embedder,
            reranker=FakeReranker(),
        )
