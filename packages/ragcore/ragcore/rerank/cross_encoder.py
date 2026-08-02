"""Cross-encoder reranking via FastEmbed ``TextCrossEncoder``, plus a no-op fallback.

Same three properties as the embedding provider: the ONNX session is loaded once per
process behind a lock, every call is offloaded with ``anyio.to_thread.run_sync`` so the
event loop never blocks, and pairs are scored in batches.

**Neither reranker drops candidates.** ``rerank_min_score`` and
``rerank_candidate_limit`` are enforced by ``app/rag/retriever.py``, which records a
``dropped_reason`` for each excluded chunk. Filtering here would make those drops
invisible, and requirement #9 asks for every drop to be auditable.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import partial
from typing import Any

import anyio
import structlog

from ragcore.errors import ConfigError
from ragcore.rerank.base import RerankResult
from ragcore.settings import Settings

__all__ = ["CrossEncoderReranker", "NoopReranker", "reset_loaded_encoders"]

_log = structlog.get_logger(__name__)

_ENCODERS: dict[tuple[Any, ...], Any] = {}
_LOAD_LOCK = threading.Lock()


def reset_loaded_encoders() -> None:
    """Drop the cached cross-encoder sessions.

    Test-only. Releases the ONNX session so a subsequent load picks up a different
    model name or cache directory.
    """
    with _LOAD_LOCK:
        _ENCODERS.clear()


def _encoder_key(settings: Settings) -> tuple[Any, ...]:
    """Cache key for a loaded cross-encoder.

    Args:
        settings: Settings naming the model and its runtime options.

    Returns:
        A hashable tuple.
    """
    return (
        settings.rerank_model,
        settings.embedding_cache_dir,
        settings.embedding_threads,
    )


def _load_encoder(settings: Settings) -> Any:
    """Load (or fetch from cache) the cross-encoder.

    Runs inside a worker thread; the lock makes a cold-cache race load the model once.

    Args:
        settings: Settings naming the model and its runtime options.

    Returns:
        The loaded ``TextCrossEncoder``.

    Raises:
        ConfigError: If FastEmbed is not installed. Surfaces on first use rather than
            at import time, and only when ``rerank_enabled`` is True — a CI run with
            ``rerank_enabled=false`` never reaches this.
    """
    key = _encoder_key(settings)
    with _LOAD_LOCK:
        cached = _ENCODERS.get(key)
        if cached is not None:
            return cached
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            msg = (
                f"fastembed is required for reranking with {settings.rerank_model}; "
                "install the ragcore dependencies with `uv sync --all-packages`, or "
                "set RAG_RERANK_ENABLED=false to use the no-op reranker"
            )
            raise ConfigError(msg) from exc

        _log.info(
            "rerank.loading",
            model=settings.rerank_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )
        encoder = TextCrossEncoder(
            model_name=settings.rerank_model,
            cache_dir=settings.embedding_cache_dir,
            threads=settings.embedding_threads,
        )
        _ENCODERS[key] = encoder
        _log.info("rerank.loaded", model=settings.rerank_model)
        return encoder


class CrossEncoderReranker:
    """Reranker backed by FastEmbed ``TextCrossEncoder``.

    Construct through :func:`ragcore.rerank.base.get_reranker` so the loaded model is
    shared per process.
    """

    def __init__(self, settings: Settings) -> None:
        """Store configuration without loading the model.

        Args:
            settings: Settings naming the model and the pair batch size.
        """
        self._settings = settings

    def _rerank_sync(self, query: str, documents: list[str]) -> list[float]:
        """Score every (query, document) pair. Runs in a worker thread.

        Args:
            query: The search query.
            documents: Candidate texts.

        Returns:
            One score per document, in input order.
        """
        encoder = _load_encoder(self._settings)
        scores = encoder.rerank(
            query, documents, batch_size=self._settings.rerank_batch_size
        )
        return [float(score) for score in scores]

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankResult]:
        """Score documents against the query and return the best ones.

        Args:
            query: The search query.
            documents: Candidate texts, in retrieval order. The caller is responsible
                for having already capped this at ``rerank_candidate_limit`` with an
                audited ``dropped_reason``.
            top_n: Maximum results to return.

        Returns:
            Up to ``top_n`` results sorted by descending score. Ties keep retrieval
            order, so fusion rank breaks a cross-encoder tie.
        """
        candidates = list(documents)
        if not candidates or top_n <= 0:
            return []
        scores = await anyio.to_thread.run_sync(
            partial(self._rerank_sync, query, candidates)
        )
        results = [
            RerankResult(index=index, score=score) for index, score in enumerate(scores)
        ]
        # Stable sort on the negated score: equal scores retain their retrieval order.
        results.sort(key=lambda item: -item.score)
        _log.debug(
            "rerank.scored",
            model=self._settings.rerank_model,
            candidates=len(candidates),
            returned=min(top_n, len(results)),
            top_score=results[0].score if results else None,
        )
        return results[:top_n]

    async def warm_up(self) -> None:
        """Load the model now instead of on the first request.

        Call from an application startup hook so the first user query does not pay
        ONNX session construction.
        """
        await anyio.to_thread.run_sync(partial(_load_encoder, self._settings))


class NoopReranker:
    """Identity reranker: preserves retrieval order and scores nothing.

    Selected by ``rerank_enabled=False``. Exists so CI, local development and the
    "does the plumbing work" smoke test can exercise the entire pipeline — including
    the rerank stage's bookkeeping — without downloading a cross-encoder.

    The scores it returns decrease monotonically, so downstream ordering, MMR and
    ``final_score`` arithmetic behave exactly as they do with a real model.
    """

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankResult]:
        """Return the first ``top_n`` documents unchanged.

        Args:
            query: Unused; accepted to satisfy the protocol.
            documents: Candidate texts, in retrieval order.
            top_n: Maximum results to return.

        Returns:
            Up to ``top_n`` results in input order, with scores descending from 1.0
            towards 0.0.
        """
        del query
        total = len(documents)
        if total == 0 or top_n <= 0:
            return []
        keep = min(top_n, total)
        return [
            RerankResult(index=index, score=(total - index) / total)
            for index in range(keep)
        ]

    async def warm_up(self) -> None:
        """No-op, so startup hooks can call it unconditionally."""
        return
