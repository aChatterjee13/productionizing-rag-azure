"""Qdrant collection layout: named vectors, payload indexes and point ids.

Three collections, all created by :func:`ensure_collections`, which is idempotent
and safe to run on every process start and from concurrent workers:

* ``rag_chunks`` — document chunks. Named vectors ``dense`` (1024-d, cosine) and
  ``sparse`` (BM25 with the **IDF modifier**, which is what makes Qdrant compute
  real BM25 rather than raw term weights).
* ``rag_memories`` — long-term user memories, dense only.
* ``rag_semantic_cache`` — cached retrieval plans, dense only.

All three carry a single **named** dense vector called ``dense`` (never an unnamed
default vector), so every query in the platform passes ``using=DENSE`` regardless of
which collection it targets.

**Multitenancy pattern.** Qdrant's documented approach for a hard tenant boundary is
to disable the global HNSW graph (``m=0``) and let it build per-tenant subgraphs
instead (``payload_m``), then mark the tenant field's keyword index with
``is_tenant=True``. That combination makes a tenant-scoped search traverse only that
tenant's points, which is both faster and structurally aligned with the security
boundary. Both knobs live in settings (``qdrant_hnsw_m``, ``qdrant_hnsw_payload_m``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from ragcore.settings import Settings, get_settings

# Re-exported so `from ragcore.vectorstore.collections import get_client` works, as
# docs/CONTRACTS.md lists get_client alongside ensure_collections in this module.
from ragcore.vectorstore.client import get_client

__all__ = [
    "CACHE_PAYLOAD_INDEXES",
    "CHUNKS",
    "CHUNK_PAYLOAD_INDEXES",
    "DENSE",
    "MEMORIES",
    "MEMORY_PAYLOAD_INDEXES",
    "POINT_ID_NAMESPACE",
    "SEMANTIC_CACHE",
    "SPARSE",
    "CollectionSpec",
    "collection_specs",
    "ensure_collections",
    "get_client",
    "point_id_for_cache",
    "point_id_for_chunk",
    "point_id_for_memory",
    "stable_point_id",
]

_log = structlog.get_logger(__name__)

#: Collection holding document chunks.
CHUNKS = "rag_chunks"
#: Collection holding long-term user memories.
MEMORIES = "rag_memories"
#: Collection holding semantic-cache probes.
SEMANTIC_CACHE = "rag_semantic_cache"

#: Name of the dense named vector, in every collection.
DENSE = "dense"
#: Name of the sparse (BM25) named vector, in ``rag_chunks`` only.
SPARSE = "sparse"

#: UUID namespace used to derive deterministic Qdrant point ids from string ids.
#: Qdrant point ids must be an unsigned integer or a UUID, but ``chunk_id`` and
#: ``memory_id`` are opaque strings, so they are mapped through UUIDv5. The mapping
#: is deterministic, which is what makes re-ingesting a chunk an upsert rather than a
#: duplicate.
POINT_ID_NAMESPACE = uuid.UUID("6f0f4d3e-7a1f-5c2b-9d84-1b2c3d4e5f60")


def _keyword(*, is_tenant: bool = False) -> qm.KeywordIndexParams:
    """Build a keyword payload index schema.

    Args:
        is_tenant: Mark this field as the tenant discriminator. Qdrant uses it to
            co-locate a tenant's points and to build the per-tenant HNSW subgraph.

    Returns:
        The index parameters.
    """
    return qm.KeywordIndexParams(
        type=qm.KeywordIndexType.KEYWORD,
        is_tenant=is_tenant or None,
    )


def _integer(
    *, range_index: bool = True, is_principal: bool = False
) -> qm.IntegerIndexParams:
    """Build an integer payload index schema.

    Args:
        range_index: Support range queries (needed for ``classification_rank <= n``).
        is_principal: Mark as the dominant ordering field. Not used here.

    Returns:
        The index parameters.
    """
    return qm.IntegerIndexParams(
        type=qm.IntegerIndexType.INTEGER,
        lookup=True,
        range=range_index,
        is_principal=is_principal or None,
    )


def _bool() -> qm.BoolIndexParams:
    """Build a boolean payload index schema.

    Returns:
        The index parameters.
    """
    return qm.BoolIndexParams(type=qm.BoolIndexType.BOOL)


def _datetime() -> qm.DatetimeIndexParams:
    """Build a datetime payload index schema.

    Returns:
        The index parameters.
    """
    return qm.DatetimeIndexParams(type=qm.DatetimeIndexType.DATETIME)


#: Payload indexes for ``rag_chunks``, in creation order. Every field the ACL and
#: metadata filters touch is indexed — an unindexed clause forces Qdrant to scan
#: payloads, which is precisely what must not happen on the tenant boundary.
CHUNK_PAYLOAD_INDEXES: tuple[tuple[str, qm.PayloadSchemaParams], ...] = (
    ("tenant_id", _keyword(is_tenant=True)),
    ("allowed_roles", _keyword()),
    ("allowed_groups", _keyword()),
    ("allowed_users", _keyword()),
    ("denied_users", _keyword()),
    ("classification_rank", _integer()),
    ("document_id", _keyword()),
    ("source_type", _keyword()),
    ("doc_type", _keyword()),
    ("tags", _keyword()),
    ("language", _keyword()),
    ("author", _keyword()),
    ("is_deleted", _bool()),
    ("content_sha256", _keyword()),
    ("source_modified_at", _datetime()),
    ("section_path", _keyword()),
    ("pii_types", _keyword()),
)

#: Payload indexes for ``rag_memories``.
MEMORY_PAYLOAD_INDEXES: tuple[tuple[str, qm.PayloadSchemaParams], ...] = (
    ("tenant_id", _keyword(is_tenant=True)),
    ("user_id", _keyword()),
    ("kind", _keyword()),
    ("expires_at", _datetime()),
)

#: Payload indexes for ``rag_semantic_cache``.
CACHE_PAYLOAD_INDEXES: tuple[tuple[str, qm.PayloadSchemaParams], ...] = (
    ("tenant_id", _keyword(is_tenant=True)),
    ("user_id", _keyword()),
    ("filter_fingerprint", _keyword()),
)


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """Declarative description of one collection.

    Attributes:
        name: Collection name as configured in settings.
        with_sparse: Whether the collection carries the BM25 sparse named vector.
        payload_indexes: ``(field_name, field_schema)`` pairs to ensure.
    """

    name: str
    with_sparse: bool
    payload_indexes: tuple[tuple[str, qm.PayloadSchemaParams], ...]


def collection_specs(settings: Settings | None = None) -> tuple[CollectionSpec, ...]:
    """Resolve the three collection specs against settings.

    The names come from settings (``qdrant_chunks_collection`` and friends) whose
    defaults are :data:`CHUNKS`, :data:`MEMORIES` and :data:`SEMANTIC_CACHE`, so a
    deployment can rename a collection without touching code.

    Args:
        settings: Settings to read collection names from. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        Specs for the chunk, memory and semantic-cache collections, in that order.
    """
    cfg = settings or get_settings()
    return (
        CollectionSpec(
            name=cfg.qdrant_chunks_collection,
            with_sparse=True,
            payload_indexes=CHUNK_PAYLOAD_INDEXES,
        ),
        CollectionSpec(
            name=cfg.qdrant_memories_collection,
            with_sparse=False,
            payload_indexes=MEMORY_PAYLOAD_INDEXES,
        ),
        CollectionSpec(
            name=cfg.qdrant_cache_collection,
            with_sparse=False,
            payload_indexes=CACHE_PAYLOAD_INDEXES,
        ),
    )


def stable_point_id(value: str, *, namespace: uuid.UUID = POINT_ID_NAMESPACE) -> str:
    """Map an opaque string id onto a Qdrant-acceptable point id.

    Qdrant accepts only unsigned integers and UUIDs as point ids. A value that
    already parses as a UUID is returned in canonical hyphenated form so ids stay
    recognisable; anything else is hashed with UUIDv5, which is deterministic — the
    same logical id always lands on the same point, making re-ingestion an upsert
    instead of a duplicate.

    Args:
        value: Logical id, e.g. a ``chunk_id``.
        namespace: UUIDv5 namespace. Defaults to :data:`POINT_ID_NAMESPACE`.

    Returns:
        A canonical UUID string.
    """
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(namespace, value))


def point_id_for_chunk(chunk_id: str) -> str:
    """Point id for a chunk in ``rag_chunks``.

    Args:
        chunk_id: The chunk's logical id.

    Returns:
        The deterministic Qdrant point id.
    """
    return stable_point_id(chunk_id)


def point_id_for_memory(memory_id: str) -> str:
    """Point id for a memory in ``rag_memories``.

    Args:
        memory_id: The memory's logical id.

    Returns:
        The deterministic Qdrant point id.
    """
    return stable_point_id(memory_id)


def point_id_for_cache(cache_id: str) -> str:
    """Point id for an entry in ``rag_semantic_cache``.

    Args:
        cache_id: The cache entry's logical id.

    Returns:
        The deterministic Qdrant point id.
    """
    return stable_point_id(cache_id)


def _vectors_config(settings: Settings) -> dict[str, qm.VectorParams]:
    """Named dense vector configuration shared by all three collections.

    Args:
        settings: Settings providing ``embedding_dim``.

    Returns:
        A mapping with the single :data:`DENSE` named vector.
    """
    return {
        DENSE: qm.VectorParams(
            size=settings.embedding_dim,
            distance=qm.Distance.COSINE,
        )
    }


def _sparse_vectors_config() -> dict[str, qm.SparseVectorParams]:
    """Sparse vector configuration for ``rag_chunks``.

    ``Modifier.IDF`` is not optional: ``Qdrant/bm25`` emits raw term frequencies and
    relies on Qdrant to apply the inverse-document-frequency term at query time. A
    collection created without it silently scores term frequency only, which looks
    like a working BM25 branch while ranking badly.

    Returns:
        A mapping with the single :data:`SPARSE` named sparse vector.
    """
    return {
        SPARSE: qm.SparseVectorParams(
            index=qm.SparseIndexParams(on_disk=False),
            modifier=qm.Modifier.IDF,
        )
    }


def _hnsw_config(settings: Settings) -> qm.HnswConfigDiff:
    """HNSW configuration implementing Qdrant's multitenancy pattern.

    Args:
        settings: Settings providing ``qdrant_hnsw_m``, ``qdrant_hnsw_payload_m``
            and ``qdrant_hnsw_ef_construct``.

    Returns:
        A diff with ``m=0`` (global graph disabled) and ``payload_m`` set, so the
        index is built per tenant against the ``is_tenant=True`` keyword index.
    """
    return qm.HnswConfigDiff(
        m=settings.qdrant_hnsw_m,
        payload_m=settings.qdrant_hnsw_payload_m,
        ef_construct=settings.qdrant_hnsw_ef_construct,
    )


def _is_already_exists(exc: UnexpectedResponse) -> bool:
    """Whether an error means the collection was created concurrently.

    Args:
        exc: The Qdrant error.

    Returns:
        True for a 409 conflict, which is what a concurrent create looks like.
    """
    return exc.status_code == 409


async def _create_collection(
    client: AsyncQdrantClient, settings: Settings, spec: CollectionSpec
) -> bool:
    """Create one collection, tolerating a concurrent creator.

    Args:
        client: Async Qdrant client.
        settings: Settings providing vector and HNSW configuration.
        spec: Collection to create.

    Returns:
        True when this call created the collection, False when it already existed.
    """
    try:
        await client.create_collection(
            collection_name=spec.name,
            vectors_config=_vectors_config(settings),
            sparse_vectors_config=(
                _sparse_vectors_config() if spec.with_sparse else None
            ),
            hnsw_config=_hnsw_config(settings),
            shard_number=settings.qdrant_shard_number,
            replication_factor=settings.qdrant_replication_factor,
            on_disk_payload=True,
        )
    except UnexpectedResponse as exc:
        if _is_already_exists(exc):
            _log.info("qdrant.collection.race", collection=spec.name)
            return False
        raise
    except ValueError as exc:
        # The local/in-memory backend raises ValueError for an existing collection.
        if "already exists" not in str(exc):
            raise
        _log.info("qdrant.collection.race", collection=spec.name)
        return False
    return True


async def _existing_indexes(
    client: AsyncQdrantClient, collection: str
) -> dict[str, qm.PayloadIndexInfo]:
    """Read the payload schema already present on a collection.

    Args:
        client: Async Qdrant client.
        collection: Collection name.

    Returns:
        A mapping of field name to index info; empty when the collection reports no
        payload schema.
    """
    info = await client.get_collection(collection)
    return dict(info.payload_schema or {})


def _tenant_index_is_partitioned(info: qm.PayloadIndexInfo) -> bool:
    """Whether an existing tenant index carries ``is_tenant=True``.

    Args:
        info: Index info reported by Qdrant.

    Returns:
        True when the index is marked as the tenant discriminator.
    """
    params = getattr(info, "params", None)
    return bool(getattr(params, "is_tenant", False))


async def _ensure_payload_indexes(
    client: AsyncQdrantClient, spec: CollectionSpec
) -> tuple[list[str], list[str]]:
    """Create any missing payload index on a collection.

    Existing indexes are left alone rather than recreated, so a restart never
    triggers an expensive rebuild. The one exception is a diagnostic: if the tenant
    index exists without ``is_tenant=True`` the collection predates the multitenancy
    pattern and searches will not use the per-tenant subgraph, which is worth a loud
    warning.

    Args:
        client: Async Qdrant client.
        spec: Collection whose indexes should be ensured.

    Returns:
        A ``(created, found)`` pair of field-name lists.
    """
    existing = await _existing_indexes(client, spec.name)
    created: list[str] = []
    found: list[str] = []
    for field_name, field_schema in spec.payload_indexes:
        info = existing.get(field_name)
        if info is not None:
            found.append(field_name)
            if field_name == "tenant_id" and not _tenant_index_is_partitioned(info):
                _log.warning(
                    "qdrant.index.tenant_not_partitioned",
                    collection=spec.name,
                    field=field_name,
                    hint=(
                        "existing tenant_id index lacks is_tenant=True; recreate the "
                        "collection to get per-tenant HNSW subgraphs"
                    ),
                )
            continue
        await client.create_payload_index(
            collection_name=spec.name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        created.append(field_name)
    return created, found


async def ensure_collections(
    client: AsyncQdrantClient,
    settings: Settings | None = None,
) -> None:
    """Create the three collections and every payload index, idempotently.

    Safe to call on every process start and from several workers at once: an
    existing collection is left untouched, a concurrent creator is tolerated, and an
    existing payload index is reported as found rather than rebuilt.

    Args:
        client: Async Qdrant client, usually from :func:`get_client`.
        settings: Settings providing collection names, vector size and HNSW knobs.
            Defaults to :func:`ragcore.settings.get_settings`.
    """
    cfg = settings or get_settings()
    for spec in collection_specs(cfg):
        exists = await client.collection_exists(spec.name)
        if not exists:
            was_created = await _create_collection(client, cfg, spec)
            event = (
                "qdrant.collection.created" if was_created else "qdrant.collection.race"
            )
            _log.info(
                event,
                collection=spec.name,
                dense_vector=DENSE,
                dense_dim=cfg.embedding_dim,
                sparse_vector=SPARSE if spec.with_sparse else None,
                sparse_modifier="idf" if spec.with_sparse else None,
                hnsw_m=cfg.qdrant_hnsw_m,
                hnsw_payload_m=cfg.qdrant_hnsw_payload_m,
            )
        else:
            _log.info("qdrant.collection.found", collection=spec.name)

        created, found = await _ensure_payload_indexes(client, spec)
        _log.info(
            "qdrant.indexes.ensured",
            collection=spec.name,
            created=created,
            found=found,
            created_count=len(created),
            found_count=len(found),
        )
