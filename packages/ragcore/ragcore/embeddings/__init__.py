"""Local embeddings: dense ``BAAI/bge-m3`` and sparse ``Qdrant/bm25``.

Importing this package is cheap — the FastEmbed implementation is only pulled in when
:func:`get_embedding_provider` is first called, so a process that never embeds never
loads onnxruntime.
"""

from ragcore.embeddings.base import (
    Embedded,
    EmbeddingProvider,
    SparseVec,
    cosine_similarity,
    get_embedding_provider,
    reset_embedding_providers,
    truncate_for_embedding,
)

__all__ = [
    "Embedded",
    "EmbeddingProvider",
    "SparseVec",
    "cosine_similarity",
    "get_embedding_provider",
    "reset_embedding_providers",
    "truncate_for_embedding",
]
