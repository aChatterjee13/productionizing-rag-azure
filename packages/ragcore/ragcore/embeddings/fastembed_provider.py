"""Local FastEmbed provider: ``BAAI/bge-m3`` dense + ``Qdrant/bm25`` sparse.

Three properties make this safe to use from an async service:

1. **Loaded once per process.** The ONNX sessions are hundreds of megabytes and take
   seconds to initialise. They live in a module-level cache keyed on the model names,
   guarded by a :class:`threading.Lock` because loading happens inside a worker
   thread and two coroutines can race into it.
2. **Never on the event loop.** FastEmbed is synchronous and CPU-bound. Every call
   goes through ``anyio.to_thread.run_sync``, so a 40 ms batch does not stall every
   other in-flight request. Batches are awaited one at a time, which also gives the
   loop a scheduling point between them.
3. **Correct BM25 usage.** Documents go through ``embed`` and queries through
   ``query_embed``. For ``Qdrant/bm25`` those differ: the document side emits term
   frequencies, the query side emits bare term ids, and the IDF factor is applied by
   Qdrant via ``Modifier.IDF`` on the collection. Embedding a query with ``embed``
   would double-weight terms and quietly degrade ranking.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import anyio
import structlog

from ragcore.embeddings.base import (
    Embedded,
    SparseVec,
    truncate_for_embedding,
)
from ragcore.errors import ConfigError
from ragcore.settings import Settings

__all__ = ["FastEmbedProvider", "reset_loaded_models"]

_log = structlog.get_logger(__name__)


@dataclass(slots=True)
class _LoadedModels:
    """A loaded dense/sparse model pair.

    Attributes:
        dense: FastEmbed ``TextEmbedding`` instance.
        sparse: FastEmbed ``SparseTextEmbedding`` instance.
    """

    dense: Any
    sparse: Any


_MODELS: dict[tuple[Any, ...], _LoadedModels] = {}
_LOAD_LOCK = threading.Lock()


def reset_loaded_models() -> None:
    """Drop the cached FastEmbed models.

    Test-only. Releases the ONNX sessions so a subsequent load picks up different
    model names or a different cache directory.
    """
    with _LOAD_LOCK:
        _MODELS.clear()


def _model_key(settings: Settings) -> tuple[Any, ...]:
    """Cache key for a loaded model pair.

    Args:
        settings: Settings naming the models and their runtime options.

    Returns:
        A hashable tuple.
    """
    return (
        settings.embedding_model,
        settings.embedding_sparse_model,
        settings.embedding_cache_dir,
        settings.embedding_threads,
    )


def _load_models(settings: Settings) -> _LoadedModels:
    """Load (or fetch from cache) the dense and sparse models.

    Runs inside a worker thread. The lock makes the load happen exactly once even when
    several coroutines hit a cold cache simultaneously; without it, two threads would
    each spend seconds building an ONNX session and one would be thrown away.

    Args:
        settings: Settings naming the models and their runtime options.

    Returns:
        The loaded model pair.

    Raises:
        ConfigError: If FastEmbed is not installed. Embeddings are not optional —
            the platform embeds locally by design — so this is a hard failure, but it
            surfaces on first use rather than at import time.
    """
    key = _model_key(settings)
    with _LOAD_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached
        try:
            from fastembed import SparseTextEmbedding, TextEmbedding
        except ImportError as exc:
            msg = (
                "fastembed is required for local embeddings "
                f"({settings.embedding_model} + {settings.embedding_sparse_model}); "
                "install the ragcore dependencies with `uv sync --all-packages`"
            )
            raise ConfigError(msg) from exc

        _log.info(
            "embeddings.loading",
            dense_model=settings.embedding_model,
            sparse_model=settings.embedding_sparse_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )
        loaded = _LoadedModels(
            dense=TextEmbedding(
                model_name=settings.embedding_model,
                cache_dir=settings.embedding_cache_dir,
                threads=settings.embedding_threads,
            ),
            sparse=SparseTextEmbedding(
                model_name=settings.embedding_sparse_model,
                cache_dir=settings.embedding_cache_dir,
                threads=settings.embedding_threads,
            ),
        )
        _MODELS[key] = loaded
        _log.info(
            "embeddings.loaded",
            dense_model=settings.embedding_model,
            sparse_model=settings.embedding_sparse_model,
        )
        return loaded


def _to_float_list(vector: Any) -> list[float]:
    """Convert a FastEmbed dense output to a plain float list.

    FastEmbed yields NumPy arrays; Qdrant's pydantic models want ``list[float]`` and a
    NumPy array would serialise as an object.

    Args:
        vector: A dense embedding, NumPy array or sequence.

    Returns:
        The embedding as a list of Python floats.
    """
    tolist = getattr(vector, "tolist", None)
    if tolist is not None:
        return [float(value) for value in tolist()]
    return [float(value) for value in vector]


def _to_sparse_vec(embedding: Any) -> SparseVec:
    """Convert a FastEmbed ``SparseEmbedding`` to :class:`SparseVec`.

    Args:
        embedding: A FastEmbed sparse embedding with ``indices`` and ``values``.

    Returns:
        The converted sparse vector.
    """
    indices = [int(index) for index in embedding.indices]
    values = [float(value) for value in embedding.values]
    return SparseVec(indices=indices, values=values)


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Split a text sequence into fixed-size batches.

    Args:
        items: Texts to batch.
        size: Maximum texts per batch.

    Yields:
        Successive batches, the last possibly shorter.
    """
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class FastEmbedProvider:
    """:class:`~ragcore.embeddings.base.EmbeddingProvider` backed by FastEmbed.

    Construct through :func:`ragcore.embeddings.base.get_embedding_provider` so the
    instance — and therefore the loaded models — is shared per process.
    """

    def __init__(self, settings: Settings) -> None:
        """Store configuration without loading any model.

        Args:
            settings: Settings naming the models, the batch size and the input cap.
        """
        self._settings = settings
        self.dim = settings.embedding_dim

    # ------------------------------------------------------------------ internals
    def _prepare(self, texts: Sequence[str]) -> list[str]:
        """Clip inputs to the configured character budget.

        Args:
            texts: Raw texts.

        Returns:
            Clipped texts, one per input.
        """
        cap = self._settings.embedding_max_chars
        return [truncate_for_embedding(text, cap) for text in texts]

    def _models(self) -> _LoadedModels:
        """Load the models. Call only from inside a worker thread.

        Returns:
            The loaded model pair.
        """
        return _load_models(self._settings)

    def _embed_documents_sync(self, texts: list[str]) -> list[Embedded]:
        """Embed a document batch densely and sparsely. Runs in a worker thread.

        Args:
            texts: Already-clipped texts.

        Returns:
            One :class:`Embedded` per input, in order.
        """
        models = self._models()
        batch_size = self._settings.embedding_batch_size
        dense = [
            _to_float_list(vector)
            for vector in models.dense.embed(texts, batch_size=batch_size)
        ]
        sparse = [
            _to_sparse_vec(vector)
            for vector in models.sparse.embed(texts, batch_size=batch_size)
        ]
        return [
            Embedded(dense=dense_vector, sparse=sparse_vector)
            for dense_vector, sparse_vector in zip(dense, sparse, strict=True)
        ]

    def _embed_dense_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch densely only. Runs in a worker thread.

        Args:
            texts: Already-clipped texts.

        Returns:
            One dense vector per input, in order.
        """
        models = self._models()
        return [
            _to_float_list(vector)
            for vector in models.dense.embed(
                texts, batch_size=self._settings.embedding_batch_size
            )
        ]

    def _embed_query_sync(self, text: str) -> Embedded:
        """Embed one query. Runs in a worker thread.

        Uses ``query_embed`` on both models: for BM25 the query side must not carry
        term frequencies, because Qdrant applies IDF server-side.

        Args:
            text: Already-clipped query text.

        Returns:
            The query's dense and sparse vectors.
        """
        models = self._models()
        dense = _to_float_list(next(iter(models.dense.query_embed(text))))
        sparse_iter = iter(models.sparse.query_embed(text))
        sparse = _to_sparse_vec(next(sparse_iter))
        return Embedded(dense=dense, sparse=sparse)

    # -------------------------------------------------------------- public surface
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedded]:
        """Embed documents for indexing, densely and sparsely.

        Args:
            texts: Texts to embed, normally ``ChunkPayload.embed_text``.

        Returns:
            One :class:`Embedded` per input, in order. Empty input returns an empty
            list without touching a model.
        """
        prepared = self._prepare(texts)
        if not prepared:
            return []
        out: list[Embedded] = []
        for batch in _batches(prepared, self._settings.embedding_batch_size):
            out.extend(
                await anyio.to_thread.run_sync(self._embed_documents_sync, batch)
            )
        return out

    async def embed_query(self, text: str) -> Embedded:
        """Embed one query, densely and sparsely.

        Args:
            text: The query.

        Returns:
            The query's dense and sparse vectors. A query with no indexable terms
            yields an empty sparse vector, which callers must treat as "skip the
            sparse branch".
        """
        prepared = truncate_for_embedding(text, self._settings.embedding_max_chars)
        return await anyio.to_thread.run_sync(self._embed_query_sync, prepared)

    async def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts densely only.

        Args:
            texts: Texts to embed.

        Returns:
            One dense vector per input, in order.
        """
        prepared = self._prepare(texts)
        if not prepared:
            return []
        out: list[list[float]] = []
        for batch in _batches(prepared, self._settings.embedding_batch_size):
            out.extend(await anyio.to_thread.run_sync(self._embed_dense_sync, batch))
        return out

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        """Embed texts sparsely only.

        Useful on the ACL-only reindex path, which rewrites payloads and can reuse
        stored dense vectors.

        Args:
            texts: Texts to embed.

        Returns:
            One sparse vector per input, in order.
        """
        embedded = await self.embed_documents(texts)
        return [item.sparse for item in embedded]

    async def warm_up(self) -> None:
        """Load the models now instead of on the first request.

        Call from an application startup hook: without it the first user query pays
        several seconds of ONNX session construction and looks like a hang.
        """
        await anyio.to_thread.run_sync(self._models)
