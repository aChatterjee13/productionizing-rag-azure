"""Tests for server-side hybrid fusion.

No live Qdrant: a fake client records the request and the assertions are about its
*shape*. That is the right level here, because the thing that can silently break is
not "does Qdrant work" but "did we ask it the right question" — two prefetch branches
on the right named vectors, the ACL filter on both of them, and fusion delegated to
the server rather than done in Python.
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import models as qm

from ragcore.embeddings.base import SparseVec
from ragcore.models.acl import Principal
from ragcore.vectorstore.collections import DENSE, SPARSE
from ragcore.vectorstore.filters import build_acl_filter
from ragcore.vectorstore.hybrid import dense_search, hybrid_search, resolve_fusion


class FakeQueryResponse:
    """Mimics the ``QueryResponse`` wrapper the Query API returns."""

    def __init__(self, points: list[qm.ScoredPoint]) -> None:
        """Wrap canned points."""
        self.points = points


class FakeClient:
    """Records every ``query_points`` call and returns canned points."""

    def __init__(self, points: list[qm.ScoredPoint] | None = None) -> None:
        """Start with an empty call log and the points every query will return."""
        self.points = points if points is not None else []
        self.calls: list[dict[str, Any]] = []

    async def query_points(self, **kwargs: Any) -> FakeQueryResponse:
        self.calls.append(kwargs)
        return FakeQueryResponse(self.points)

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "query_points was never called"
        return self.calls[-1]


def scored(point_id: str, score: float, *, chunk_id: str) -> qm.ScoredPoint:
    return qm.ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={"chunk_id": chunk_id, "tenant_id": "tenant-a"},
    )


@pytest.fixture
def principal() -> Principal:
    return Principal(user_id="u1", tenant_id="tenant-a", groups=["g-eng"])


@pytest.fixture
def qfilter(principal: Principal) -> qm.Filter:
    return build_acl_filter(principal)


DENSE_VEC = [0.1, 0.2, 0.3, 0.4]
SPARSE_VEC = SparseVec(indices=[7, 42, 99], values=[1.5, 0.5, 2.0])


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


async def test_hybrid_search_issues_one_request_with_two_prefetch_branches(qfilter):
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="how much annual leave do I get?",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=30,
        prefetch_limit=60,
    )

    # One round trip: fusion happens in Qdrant, not by issuing two searches.
    assert len(client.calls) == 1
    call = client.last
    assert call["collection_name"] == "rag_chunks"
    assert call["limit"] == 30
    assert call["with_payload"] is True
    assert call["with_vectors"] is False

    prefetch = call["prefetch"]
    assert isinstance(prefetch, list)
    assert len(prefetch) == 2
    assert all(isinstance(branch, qm.Prefetch) for branch in prefetch)

    dense_branch, sparse_branch = prefetch
    assert dense_branch.using == DENSE
    assert dense_branch.query == DENSE_VEC
    assert dense_branch.limit == 60

    assert sparse_branch.using == SPARSE
    assert isinstance(sparse_branch.query, qm.SparseVector)
    assert sparse_branch.query.indices == SPARSE_VEC.indices
    assert sparse_branch.query.values == SPARSE_VEC.values
    assert sparse_branch.limit == 60


async def test_both_branches_carry_the_identical_acl_filter(qfilter):
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="q",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
    )
    call = client.last
    for branch in call["prefetch"]:
        assert branch.filter is qfilter
    # And on the outer query too: defence in depth, so dropping it from one prefetch
    # branch in a future edit still cannot leak across the tenant boundary.
    assert call["query_filter"] is qfilter


async def test_the_filter_actually_carries_the_tenant_boundary(principal, qfilter):
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="q",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
    )
    sent = client.last["prefetch"][0].filter
    tenant = [
        c
        for c in sent.must
        if isinstance(c, qm.FieldCondition) and c.key == "tenant_id"
    ]
    assert len(tenant) == 1
    assert tenant[0].match.value == principal.tenant_id
    assert sent.must_not[0].key == "denied_users"
    assert sent.min_should.min_count == 1


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("rrf", qm.Fusion.RRF), ("dbsf", qm.Fusion.DBSF), ("RRF", qm.Fusion.RRF)],
)
def test_resolve_fusion(name, expected):
    assert resolve_fusion(name) is expected


def test_resolve_fusion_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="unknown fusion strategy"):
        resolve_fusion("weighted")


async def test_fusion_is_delegated_to_qdrant_as_a_fusion_query(qfilter):
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="q",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
    )
    query = client.last["query"]
    assert isinstance(query, qm.FusionQuery)
    assert query.fusion is qm.Fusion.RRF


async def test_dbsf_is_passed_through(qfilter):
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="q",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
        fusion="dbsf",
    )
    assert client.last["query"].fusion is qm.Fusion.DBSF


async def test_unknown_fusion_fails_before_any_request(qfilter):
    client = FakeClient()
    with pytest.raises(ValueError, match="unknown fusion strategy"):
        await hybrid_search(
            client,
            collection="rag_chunks",
            query_text="q",
            dense=DENSE_VEC,
            sparse=SPARSE_VEC,
            qfilter=qfilter,
            limit=10,
            prefetch_limit=20,
            fusion="magic",
        )
    assert client.calls == []


# ---------------------------------------------------------------------------
# Degenerate sparse vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse", [None, SparseVec.empty()], ids=["none", "empty"])
async def test_empty_sparse_vector_drops_the_branch_rather_than_matching_nothing(
    qfilter, sparse
):
    """A stopword-only query has no BM25 terms; sending it would match nothing."""
    client = FakeClient()
    await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="the a of",
        dense=DENSE_VEC,
        sparse=sparse,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
    )
    prefetch = client.last["prefetch"]
    assert len(prefetch) == 1
    assert prefetch[0].using == DENSE
    assert prefetch[0].filter is qfilter
    # Still a fusion query, so the response shape does not change with the branch count.
    assert isinstance(client.last["query"], qm.FusionQuery)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


async def test_raw_scored_points_are_returned_in_server_order(qfilter):
    points = [
        scored("11111111-1111-1111-1111-111111111111", 0.9, chunk_id="c1"),
        scored("22222222-2222-2222-2222-222222222222", 0.4, chunk_id="c2"),
    ]
    client = FakeClient(points)
    result = await hybrid_search(
        client,
        collection="rag_chunks",
        query_text="q",
        dense=DENSE_VEC,
        sparse=SPARSE_VEC,
        qfilter=qfilter,
        limit=10,
        prefetch_limit=20,
    )
    assert result == points
    assert all(isinstance(point, qm.ScoredPoint) for point in result)
    # No re-scoring, no re-sorting: ordering is the server's fused ranking.
    assert [point.score for point in result] == [0.9, 0.4]


async def test_empty_result_is_an_empty_list_not_an_error(qfilter):
    client = FakeClient([])
    assert (
        await hybrid_search(
            client,
            collection="rag_chunks",
            query_text="q",
            dense=DENSE_VEC,
            sparse=SPARSE_VEC,
            qfilter=qfilter,
            limit=10,
            prefetch_limit=20,
        )
        == []
    )


# ---------------------------------------------------------------------------
# dense_search (memories and the semantic cache)
# ---------------------------------------------------------------------------


async def test_dense_search_uses_the_named_dense_vector_and_no_prefetch():
    principal = Principal(user_id="u1", tenant_id="tenant-a")
    from ragcore.vectorstore.filters import build_memory_filter

    memory_filter = build_memory_filter(principal)
    client = FakeClient()
    await dense_search(
        client,
        collection="rag_memories",
        dense=DENSE_VEC,
        qfilter=memory_filter,
        limit=6,
    )
    call = client.last
    assert call["collection_name"] == "rag_memories"
    assert call["using"] == DENSE
    assert call["query"] == DENSE_VEC
    assert call["query_filter"] is memory_filter
    assert call["limit"] == 6
    assert call["score_threshold"] is None
    assert "prefetch" not in call


async def test_dense_search_pushes_the_similarity_floor_to_the_server():
    principal = Principal(user_id="u1", tenant_id="tenant-a")
    from ragcore.vectorstore.filters import build_cache_filter, filter_fingerprint

    fingerprint = filter_fingerprint(principal, None)
    client = FakeClient()
    await dense_search(
        client,
        collection="rag_semantic_cache",
        dense=DENSE_VEC,
        qfilter=build_cache_filter(principal, fingerprint),
        limit=1,
        score_threshold=0.94,
    )
    assert client.last["score_threshold"] == 0.94
