"""Reranking contracts: the result shape, the protocol, and the factory.

A cross-encoder scores the (query, document) pair jointly instead of comparing two
independently-computed vectors, which is why it recovers precision that bi-encoder
retrieval loses. It is also quadratically more expensive, so it only ever sees the
fused top candidates, never the corpus.

Kept free of heavy imports: the FastEmbed cross-encoder is loaded lazily by
:func:`get_reranker`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ragcore.settings import Settings, get_settings

__all__ = [
    "RerankResult",
    "Reranker",
    "get_reranker",
    "reset_rerankers",
]


class RerankResult(BaseModel):
    """One document's cross-encoder score.

    Attributes:
        index: Position of the document in the list passed to
            :meth:`Reranker.rerank`. The reranker never returns the text, so the
            caller keeps ownership of the candidate objects and their payloads.
        score: Cross-encoder relevance. Scale is model-dependent and **not**
            comparable to a cosine or fusion score — only to other scores from the
            same call.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, description="Index into the input document list.")
    score: float = Field(description="Cross-encoder relevance score.")


@runtime_checkable
class Reranker(Protocol):
    """What the retriever needs from a reranker."""

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[RerankResult]:
        """Score documents against a query and return the best ones.

        Args:
            query: The search query.
            documents: Candidate texts, in retrieval order.
            top_n: Maximum results to return.

        Returns:
            Up to ``top_n`` results sorted by descending score.
        """
        ...


def _reranker_key(settings: Settings) -> tuple[Any, ...]:
    """Cache key identifying one reranker configuration.

    Args:
        settings: Settings carrying the ``rerank_*`` fields.

    Returns:
        A hashable tuple. :class:`Settings` is a pydantic model and unhashable, so it
        can never be an ``lru_cache`` key.
    """
    return (
        settings.rerank_enabled,
        settings.rerank_model,
        settings.rerank_batch_size,
        settings.embedding_cache_dir,
        settings.embedding_threads,
    )


_RERANKERS: dict[tuple[Any, ...], Reranker] = {}


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Return the process-wide reranker.

    Args:
        settings: Settings selecting the model. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        A cached ``CrossEncoderReranker``, or a ``NoopReranker`` when
        ``rerank_enabled`` is False — which is how CI runs the whole pipeline without
        downloading a 600 MB cross-encoder.
    """
    cfg = settings or get_settings()
    key = _reranker_key(cfg)
    existing = _RERANKERS.get(key)
    if existing is not None:
        return existing

    # Imported here so that importing ragcore.rerank never pulls onnxruntime.
    from ragcore.rerank.cross_encoder import CrossEncoderReranker, NoopReranker

    reranker: Reranker = (
        CrossEncoderReranker(cfg) if cfg.rerank_enabled else NoopReranker()
    )
    _RERANKERS[key] = reranker
    return reranker


def reset_rerankers() -> None:
    """Drop the cached rerankers.

    For tests that toggle ``rerank_enabled`` or change the model.
    """
    _RERANKERS.clear()
