"""Semantic cache: a hit is a plan to re-authorise, never content to replay.

The central test is :func:`test_hit_with_downgraded_principal_returns_fewer_chunks`.
A user loses a group between the write and the read; the fingerprint is unchanged (by
design — group membership deliberately does not contribute to it), so the entry still
matches, and the *only* thing standing between the cached ids and a leak is the live
ACL re-fetch. The fake Qdrant here evaluates the real
:meth:`ragcore.models.acl.AccessControl.permits` semantics, and the tests assert the
filter handed to it really is the one ``build_acl_filter_for_chunk_ids`` produces.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from qdrant_client import models as qm

from app.rag.memory.semantic_cache import CacheProbe, SemanticCache, retrieve_by_ids
from ragcore.embeddings import cosine_similarity
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.memory import SemanticCacheEntry, normalize_query
from ragcore.models.retrieval import MetadataFilter
from ragcore.settings import Settings
from ragcore.vectorstore import DENSE, point_id_for_cache, point_id_for_chunk

TENANT = "tenant-acme"
OTHER_TENANT = "tenant-globex"
USER = "user-1"
ENG_GROUP = "g-acme-engineering"
CANARY = "CANARY-ACME-VPN-5A17"

VOCAB = [
    "vpn",
    "access",
    "remote",
    "work",
    "travel",
    "policy",
    "expenses",
    "allowance",
    "onboarding",
    "laptop",
]


def _vector(text: str) -> list[float]:
    words = {token.strip(".,;:?").lower() for token in text.split()}
    vector = [1.0 if token in words else 0.0 for token in VOCAB]
    if not any(vector):
        vector = [1e-6] * len(VOCAB)
    return vector


class FakeEmbedder:
    """Bag-of-words embedder so cosine similarity is meaningful and deterministic."""

    dim = len(VOCAB)

    async def embed_query(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(dense=_vector(text))


def make_chunk(
    chunk_id: str,
    *,
    text: str,
    classification: Classification = Classification.INTERNAL,
    allowed_groups: list[str] | None = None,
    tenant: str = TENANT,
) -> ChunkPayload:
    return ChunkPayload.from_access_control(
        AccessControl(
            tenant_id=tenant,
            classification=classification,
            allowed_groups=allowed_groups or [],
        ),
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        chunk_index=0,
        source_type="blob",
        source_id="src-1",
        source_uri=f"https://example.invalid/{chunk_id}",
        title=chunk_id,
        text=text,
    )


CHUNKS: dict[str, ChunkPayload] = {
    "doc-travel::0000": make_chunk(
        "doc-travel::0000",
        text="The travel allowance is EUR 60 per day.",
        classification=Classification.PUBLIC,
    ),
    "doc-remote::0000": make_chunk(
        "doc-remote::0000",
        text="Remote work is allowed two days a week.",
    ),
    "doc-vpn::0000": make_chunk(
        "doc-vpn::0000",
        text=f"VPN access runbook. {CANARY}",
        allowed_groups=[ENG_GROUP],
    ),
}
CACHED_IDS = list(CHUNKS)


def _must_values(qfilter: qm.Filter) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for condition in qfilter.must or []:
        key = getattr(condition, "key", None)
        match = getattr(condition, "match", None)
        if key and match is not None and hasattr(match, "value"):
            values[key] = match.value
    return values


def _has_ids(qfilter: qm.Filter) -> list[str]:
    for condition in qfilter.must or []:
        ids = getattr(condition, "has_id", None)
        if ids:
            return [str(value) for value in ids]
    return []


class FakeQdrant:
    """Cache collection plus a chunk store that enforces the real ACL semantics."""

    def __init__(self) -> None:
        """Start with an empty cache collection."""
        self.cache: dict[str, dict[str, Any]] = {}
        self.principal: Principal | None = None
        self.acl_filters: list[qm.Filter] = []

    # ------------------------------------------------------------- cache points
    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        using: str,
        query_filter: qm.Filter,
        limit: int,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> SimpleNamespace:
        wanted = _must_values(query_filter)
        points = []
        for point_id, record in self.cache.items():
            payload = record["payload"]
            if any(payload.get(key) != value for key, value in wanted.items()):
                continue
            score = cosine_similarity(query, record["vector"][DENSE])
            if score_threshold is not None and score < score_threshold:
                continue
            points.append(SimpleNamespace(id=point_id, score=score, payload=payload))
        points.sort(key=lambda point: point.score, reverse=True)
        return SimpleNamespace(points=points[:limit])

    async def upsert(
        self, *, collection_name: str, points: list[Any], wait: bool = True
    ) -> None:
        for point in points:
            self.cache[str(point.id)] = {
                "payload": dict(point.payload),
                "vector": dict(point.vector),
            }

    async def set_payload(
        self,
        *,
        collection_name: str,
        payload: dict[str, Any],
        points: list[str],
        wait: bool = True,
    ) -> None:
        for point_id in points:
            if point_id in self.cache:
                self.cache[point_id]["payload"].update(payload)

    async def delete(
        self, *, collection_name: str, points_selector: Any, wait: bool = True
    ) -> None:
        for point_id in getattr(points_selector, "points", []):
            self.cache.pop(str(point_id), None)

    # ------------------------------------------------------------ chunk lookups
    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: qm.Filter,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> tuple[list[SimpleNamespace], None]:
        if collection_name.endswith("semantic_cache"):
            records = [
                SimpleNamespace(id=point_id, payload=record["payload"])
                for point_id, record in self.cache.items()
                if all(
                    record["payload"].get(key) == value
                    for key, value in _must_values(scroll_filter).items()
                )
            ]
            return records[:limit], None

        # Chunk collection: this must be the composed ACL filter, not a bare id list.
        assert scroll_filter.must_not, "ACL filter must carry the deny list"
        assert scroll_filter.min_should is not None, "ACL filter must carry min_should"
        self.acl_filters.append(scroll_filter)

        requested = set(_has_ids(scroll_filter))
        principal = self.principal
        assert principal is not None
        records = []
        for chunk_id, payload in CHUNKS.items():
            if point_id_for_chunk(chunk_id) not in requested:
                continue
            if not payload.access_control().permits(principal):
                continue
            records.append(
                SimpleNamespace(
                    id=point_id_for_chunk(chunk_id),
                    payload=payload.to_qdrant_payload(),
                )
            )
        return records[:limit], None


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "local",
        "redis_enabled": False,
        "langfuse_enabled": False,
        "memory_cache_enabled": True,
        "memory_cache_threshold": 0.9,
        "memory_cache_ttl_seconds": 3_600,
        "memory_cache_max_chunk_ids": 32,
    }
    base.update(overrides)
    return Settings(**base)


def engineer(**overrides: Any) -> Principal:
    fields: dict[str, Any] = {
        "user_id": USER,
        "tenant_id": TENANT,
        "roles": ["rag.user"],
        "groups": [ENG_GROUP],
        "max_classification": Classification.CONFIDENTIAL,
    }
    fields.update(overrides)
    return Principal(**fields)


def build_cache(settings: Settings, client: FakeQdrant) -> SemanticCache:
    return SemanticCache(settings=settings, client=client, embedder=FakeEmbedder())


QUESTION = "how do I get vpn access for remote work"


async def seed_entry(
    cache: SemanticCache, client: FakeQdrant, principal: Principal
) -> SemanticCacheEntry:
    client.principal = principal
    entry = await cache.store(
        principal,
        QUESTION,
        chunk_ids=CACHED_IDS,
        transformed_queries=["vpn access remote work"],
    )
    assert entry is not None
    return entry


# ------------------------------------------------------------------------ tests
async def test_store_then_hit_returns_the_cached_plan() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()
    await seed_entry(cache, client, principal)

    client.principal = principal
    probe = await cache.probe(principal, QUESTION)

    assert probe.hit is True
    assert probe.usable is True
    assert probe.reason == "hit"
    assert [chunk.payload.chunk_id for chunk in probe.chunks] == CACHED_IDS
    assert probe.revoked_chunk_ids == []
    assert probe.transformed_queries == ["vpn access remote work"]
    assert all(chunk.retrieval_stage == "cache" for chunk in probe.chunks)
    # The hit was recorded on the entry.
    payload = client.cache[point_id_for_cache(probe.entry.cache_id)]["payload"]
    assert payload["hit_count"] == 1


async def test_hit_with_downgraded_principal_returns_fewer_chunks() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)

    full = engineer()
    await seed_entry(cache, client, full)

    # The user is removed from the engineering group. Group membership deliberately
    # does not contribute to the fingerprint, so the entry still matches and the
    # live ACL re-fetch is the only thing preventing a leak.
    downgraded = engineer(groups=[])
    assert cache.fingerprint(full, None) == cache.fingerprint(downgraded, None)

    client.principal = downgraded
    probe = await cache.probe(downgraded, QUESTION)

    assert probe.hit is True
    visible = [chunk.payload.chunk_id for chunk in probe.chunks]
    assert visible == ["doc-travel::0000", "doc-remote::0000"]
    assert len(visible) < len(CACHED_IDS)
    assert probe.revoked_chunk_ids == ["doc-vpn::0000"]
    # No restricted content survived, not even a canary token.
    assert all(CANARY not in chunk.payload.text for chunk in probe.chunks)
    # And the withheld chunk really was filtered by the composed ACL filter.
    assert client.acl_filters
    assert client.acl_filters[-1].must_not


async def test_all_chunks_revoked_falls_through_to_a_miss() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    full = engineer()

    # A plan whose every chunk is group-restricted.
    client.principal = full
    entry = await cache.store(full, QUESTION, chunk_ids=["doc-vpn::0000"])
    assert entry is not None

    downgraded = engineer(groups=[])
    client.principal = downgraded
    probe = await cache.probe(downgraded, QUESTION)

    # The entry matched, but nothing survived re-authorisation, so the pipeline is
    # told to retrieve normally rather than to answer from an empty plan.
    assert probe.entry is not None
    assert probe.hit is False
    assert probe.usable is False
    assert probe.reason == "empty_after_acl"
    assert probe.chunks == []
    assert probe.revoked_chunk_ids == ["doc-vpn::0000"]


async def test_clearance_downgrade_changes_the_fingerprint_and_misses() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    await seed_entry(cache, client, engineer())

    lowered = engineer(max_classification=Classification.PUBLIC)
    assert cache.fingerprint(engineer(), None) != cache.fingerprint(lowered, None)

    client.principal = lowered
    probe = await cache.probe(lowered, QUESTION)

    assert probe.hit is False
    assert probe.chunks == []


async def test_a_different_metadata_filter_misses() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()
    await seed_entry(cache, client, principal)

    client.principal = principal
    probe = await cache.probe(
        principal, QUESTION, extra=MetadataFilter(doc_types=["runbook"])
    )

    assert probe.hit is False


async def test_cross_tenant_probe_never_matches() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    await seed_entry(cache, client, engineer())

    outsider = Principal(
        user_id=USER,
        tenant_id=OTHER_TENANT,
        roles=["rag.user"],
        max_classification=Classification.CONFIDENTIAL,
    )
    client.principal = outsider
    probe = await cache.probe(outsider, QUESTION)

    assert probe.hit is False
    assert probe.chunks == []


async def test_only_the_retrieval_plan_is_cached() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    entry = await seed_entry(cache, client, engineer())

    payload = client.cache[point_id_for_cache(entry.cache_id)]["payload"]

    assert set(payload) <= set(SemanticCacheEntry.model_fields)
    assert "answer" not in payload
    # The rendered answer text is nowhere in the entry.
    serialised = str(payload)
    assert "EUR 60" not in serialised
    assert CANARY not in serialised
    assert payload["chunk_ids"] == CACHED_IDS
    assert payload["normalized_query"] == normalize_query(QUESTION)


async def test_probe_and_store_are_no_ops_when_disabled() -> None:
    settings = make_settings(memory_cache_enabled=False)
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()

    probe = await cache.probe(principal, QUESTION)
    stored = await cache.store(principal, QUESTION, chunk_ids=CACHED_IDS)

    assert probe.reason == "disabled"
    assert stored is None
    assert client.cache == {}


async def test_chunk_ids_are_capped_at_the_configured_maximum() -> None:
    settings = make_settings(memory_cache_max_chunk_ids=2)
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()
    client.principal = principal

    entry = await cache.store(principal, QUESTION, chunk_ids=CACHED_IDS)

    assert entry is not None
    assert entry.chunk_ids == CACHED_IDS[:2]


async def test_eviction_removes_expired_then_least_used() -> None:
    settings = make_settings(memory_cache_max_entries=2)
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()
    fingerprint = cache.fingerprint(principal, None)
    now = datetime.now(UTC)

    def seed(cache_id: str, *, hits: int, age_seconds: int, ttl: int) -> None:
        entry = SemanticCacheEntry(
            cache_id=cache_id,
            tenant_id=TENANT,
            user_id=USER,
            normalized_query=f"query {cache_id}",
            chunk_ids=["doc-travel::0000"],
            filter_fingerprint=fingerprint,
            hit_count=hits,
            created_at=now - timedelta(seconds=age_seconds),
            last_used_at=now - timedelta(seconds=age_seconds),
            ttl_seconds=ttl,
        )
        client.cache[point_id_for_cache(cache_id)] = {
            "payload": entry.model_dump(mode="json"),
            "vector": {DENSE: _vector(entry.normalized_query)},
        }

    seed("expired", hits=99, age_seconds=10_000, ttl=60)
    seed("cold", hits=0, age_seconds=100, ttl=3_600)
    seed("warm", hits=5, age_seconds=100, ttl=3_600)
    seed("hot", hits=50, age_seconds=100, ttl=3_600)

    removed = await cache.evict(principal, fingerprint=fingerprint, now=now)

    assert removed == 2
    survivors = {record["payload"]["cache_id"] for record in client.cache.values()}
    assert survivors == {"warm", "hot"}


async def test_an_injected_resolver_receives_the_cached_plan() -> None:
    settings = make_settings()
    client = FakeQdrant()
    cache = build_cache(settings, client)
    principal = engineer()
    await seed_entry(cache, client, principal)

    seen: dict[str, Any] = {}

    async def resolver(
        who: Principal, ids: list[str], extra: MetadataFilter | None
    ) -> list[Any]:
        seen["principal"] = who
        seen["ids"] = list(ids)
        seen["extra"] = extra
        return []

    client.principal = principal
    probe = await cache.probe(principal, QUESTION, resolver=resolver)

    assert seen["ids"] == CACHED_IDS
    assert seen["principal"] is principal
    # The orchestrator's retriever answered instead of the local primitive, and an
    # empty result still reads as "retrieve normally".
    assert client.acl_filters == []
    assert probe.reason == "empty_after_acl"
    assert probe.usable is False


async def test_retrieve_by_ids_preserves_the_cached_order() -> None:
    settings = make_settings()
    client = FakeQdrant()
    principal = engineer()
    client.principal = principal
    reversed_ids = list(reversed(CACHED_IDS))

    chunks = await retrieve_by_ids(
        client,
        principal=principal,
        chunk_ids=reversed_ids,
        settings=settings,
    )

    assert [chunk.payload.chunk_id for chunk in chunks] == reversed_ids
    # Scores descend with the cached rank so downstream ordering is preserved.
    assert chunks[0].final_score > chunks[-1].final_score


async def test_retrieve_by_ids_on_an_empty_list_touches_nothing() -> None:
    settings = make_settings()
    client = FakeQdrant()

    chunks = await retrieve_by_ids(
        client, principal=engineer(), chunk_ids=[], settings=settings
    )

    assert chunks == []
    assert client.acl_filters == []


def test_cache_probe_defaults_to_unusable() -> None:
    probe = CacheProbe()

    assert probe.hit is False
    assert probe.usable is False
    assert probe.reason == "miss"
