"""Qdrant access layer: collections, filters, hybrid search and writes.

Importing this package pulls ``qdrant-client``, so keep it out of hot import paths
that only need contracts (``ragcore.models``) or configuration (``ragcore.settings``).

The one rule this package exists to enforce: **every** filter that serves data to a
principal comes from :mod:`ragcore.vectorstore.filters`. No other module in the
repository may construct a :class:`qdrant_client.models.Filter` for a read path.
"""

from ragcore.vectorstore.client import (
    check_qdrant,
    close_all_clients,
    close_client,
    get_client,
)
from ragcore.vectorstore.collections import (
    CACHE_PAYLOAD_INDEXES,
    CHUNK_PAYLOAD_INDEXES,
    CHUNKS,
    DENSE,
    MEMORIES,
    MEMORY_PAYLOAD_INDEXES,
    SEMANTIC_CACHE,
    SPARSE,
    CollectionSpec,
    collection_specs,
    ensure_collections,
    point_id_for_cache,
    point_id_for_chunk,
    point_id_for_memory,
    stable_point_id,
)
from ragcore.vectorstore.filters import (
    build_acl_filter,
    build_acl_filter_for_chunk_ids,
    build_cache_filter,
    build_memory_filter,
    build_tenant_filter,
    classification_ceiling,
    filter_fingerprint,
    serialise_filter,
)
from ragcore.vectorstore.hybrid import dense_search, hybrid_search, resolve_fusion
from ragcore.vectorstore.writer import (
    ChunkPoint,
    count_chunks,
    hard_delete_by_filter,
    hard_delete_document,
    soft_delete_document,
    soft_delete_documents,
    tombstone_missing,
    update_access_control,
    upsert_chunks,
    upsert_points,
)

__all__ = [
    "CACHE_PAYLOAD_INDEXES",
    "CHUNKS",
    "CHUNK_PAYLOAD_INDEXES",
    "DENSE",
    "MEMORIES",
    "MEMORY_PAYLOAD_INDEXES",
    "SEMANTIC_CACHE",
    "SPARSE",
    "ChunkPoint",
    "CollectionSpec",
    "build_acl_filter",
    "build_acl_filter_for_chunk_ids",
    "build_cache_filter",
    "build_memory_filter",
    "build_tenant_filter",
    "check_qdrant",
    "classification_ceiling",
    "close_all_clients",
    "close_client",
    "collection_specs",
    "count_chunks",
    "dense_search",
    "ensure_collections",
    "filter_fingerprint",
    "get_client",
    "hard_delete_by_filter",
    "hard_delete_document",
    "hybrid_search",
    "point_id_for_cache",
    "point_id_for_chunk",
    "point_id_for_memory",
    "resolve_fusion",
    "serialise_filter",
    "soft_delete_document",
    "soft_delete_documents",
    "stable_point_id",
    "tombstone_missing",
    "update_access_control",
    "upsert_chunks",
    "upsert_points",
]
