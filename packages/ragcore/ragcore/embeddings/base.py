"""Embedding contracts: the vector shapes and the provider protocol.

Deliberately free of heavy imports. Anything that touches FastEmbed lives in
:mod:`ragcore.embeddings.fastembed_provider` and is imported lazily by
:func:`get_embedding_provider`, so ``import ragcore.embeddings`` costs nothing and
cannot fail because an ONNX runtime is missing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragcore.settings import Settings, get_settings

__all__ = [
    "Embedded",
    "EmbeddingProvider",
    "SparseVec",
    "cosine_similarity",
    "get_embedding_provider",
    "reset_embedding_providers",
    "truncate_for_embedding",
]


class SparseVec(BaseModel):
    """A sparse vector as Qdrant stores it: parallel index and value arrays.

    For ``Qdrant/bm25`` the indices are hashed term ids and the values are term
    frequencies. The inverse-document-frequency factor is **not** applied here — the
    collection's ``Modifier.IDF`` makes Qdrant apply it at query time, using corpus
    statistics that only the server has.
    """

    model_config = ConfigDict(extra="forbid")

    indices: list[int] = Field(
        default_factory=list, description="Hashed term ids, parallel to `values`."
    )
    values: list[float] = Field(
        default_factory=list, description="Term weights, parallel to `indices`."
    )

    @model_validator(mode="after")
    def _validate_parallel(self) -> SparseVec:
        """Require the two arrays to be the same length.

        Returns:
            The validated sparse vector.

        Raises:
            ValueError: If the arrays disagree in length, which would silently
                mis-pair terms with weights.
        """
        if len(self.indices) != len(self.values):
            msg = (
                "sparse indices and values must be the same length, got "
                f"{len(self.indices)} and {len(self.values)}"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def empty(cls) -> Self:
        """Build the empty sparse vector.

        Returned for a query with no indexable terms (pure stopwords, punctuation).
        Callers must skip the sparse branch rather than sending it, since an empty
        sparse query matches nothing.

        Returns:
            An empty sparse vector.
        """
        return cls(indices=[], values=[])

    @property
    def is_empty(self) -> bool:
        """Whether this vector carries no terms.

        Returns:
            True when there are no indices.
        """
        return not self.indices

    @property
    def nnz(self) -> int:
        """Number of non-zero entries.

        Returns:
            The term count.
        """
        return len(self.indices)


class Embedded(BaseModel):
    """One text's dense and sparse representations."""

    model_config = ConfigDict(extra="forbid")

    dense: list[float] = Field(description="Dense embedding; 1024-d for BAAI/bge-m3.")
    sparse: SparseVec = Field(
        default_factory=SparseVec, description="BM25 sparse embedding."
    )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What the pipeline needs from an embedder.

    Attributes:
        dim: Dense vector dimensionality. Must equal the ``rag_chunks`` dense vector
            size or every upsert is rejected by Qdrant.
    """

    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedded]:
        """Embed documents for indexing, densely and sparsely.

        Args:
            texts: Texts to embed, normally ``ChunkPayload.embed_text``.

        Returns:
            One :class:`Embedded` per input, in order.
        """
        ...

    async def embed_query(self, text: str) -> Embedded:
        """Embed one query, densely and sparsely.

        Args:
            text: The query.

        Returns:
            The query's dense and sparse vectors.
        """
        ...

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts densely only.

        Used for memories, semantic-cache probes and the evaluation harness's
        semantic-similarity metric, none of which have a sparse vector.

        Args:
            texts: Texts to embed.

        Returns:
            One dense vector per input, in order.
        """
        ...


def truncate_for_embedding(text: str, max_chars: int) -> str:
    """Clip a text to the embedder's input budget.

    ``BAAI/bge-m3`` truncates internally at its context limit; clipping first keeps
    tokenisation cost bounded and makes the truncation point explicit rather than a
    property of the model's tokeniser.

    Args:
        text: Input text.
        max_chars: Character budget (``settings.embedding_max_chars``).

    Returns:
        The text, clipped to ``max_chars``.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors.

    Needed off the Qdrant path: memory write-back compares a candidate memory against
    existing ones (``memory_dedupe_threshold``) and the eval harness scores
    ``semantic_similarity`` against ground truth.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1, 1]``, or 0.0 when either vector is zero-length or
        has zero magnitude.

    Raises:
        ValueError: If the vectors have different dimensionality.
    """
    if len(a) != len(b):
        msg = f"cannot compare vectors of dimension {len(a)} and {len(b)}"
        raise ValueError(msg)
    if not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b, strict=True):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _provider_key(settings: Settings) -> tuple[Any, ...]:
    """Cache key identifying one embedding provider configuration.

    Args:
        settings: Settings carrying the ``embedding_*`` fields.

    Returns:
        A hashable tuple. :class:`Settings` is a pydantic model and unhashable, so it
        can never be an ``lru_cache`` key.
    """
    return (
        settings.embedding_model,
        settings.embedding_sparse_model,
        settings.embedding_dim,
        settings.embedding_batch_size,
        settings.embedding_cache_dir,
        settings.embedding_threads,
        settings.embedding_max_chars,
    )


_PROVIDERS: dict[tuple[Any, ...], EmbeddingProvider] = {}


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the process-wide embedding provider.

    The underlying ONNX models are hundreds of megabytes and take seconds to load, so
    exactly one provider — and one pair of loaded models — exists per configuration
    per process. The provider itself loads the models lazily on first use, so
    constructing it is free.

    Args:
        settings: Settings selecting the models and batch size. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        A cached :class:`FastEmbedProvider` typed as :class:`EmbeddingProvider`.
    """
    cfg = settings or get_settings()
    key = _provider_key(cfg)
    existing = _PROVIDERS.get(key)
    if existing is not None:
        return existing

    # Imported here, not at module scope: fastembed pulls onnxruntime and tokenizers,
    # which must not be a cost (or a failure mode) of importing ragcore.embeddings.
    from ragcore.embeddings.fastembed_provider import FastEmbedProvider

    provider = FastEmbedProvider(cfg)
    _PROVIDERS[key] = provider
    return provider


def reset_embedding_providers() -> None:
    """Drop the cached providers.

    For tests that reconfigure ``embedding_*`` settings. Loaded model weights stay in
    the FastEmbed module-level cache, so this is cheap to call.
    """
    _PROVIDERS.clear()
