"""Sentence-embedding similarity between an answer and its ground truth.

Requirement #8 asks for semantic similarity alongside RAGAS. It is computed with
:mod:`ragcore.embeddings` — the same local ``BAAI/bge-m3`` provider the retrieval
path uses — so the harness never introduces a second embedding stack whose drift
would show up as an unexplained score change.

Every value returned is clamped to ``[0, 1]``: cosine over bge-m3 is bounded by
``[-1, 1]``, a negative score means "unrelated" rather than "anti-correlated", and
:class:`~ragcore.models.eval.MetricScores` is a 0..1 scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragcore.embeddings import cosine_similarity, get_embedding_provider
from ragcore.logging import get_logger
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ragcore.embeddings import EmbeddingProvider

__all__ = [
    "cosine_similarity",
    "embed_texts",
    "semantic_similarity",
    "semantic_similarity_batch",
]

_log = get_logger(__name__)


def _clamp(value: float) -> float:
    """Clamp a cosine score onto the 0..1 metric scale.

    Args:
        value: Raw cosine similarity in ``[-1, 1]``.

    Returns:
        The value clamped to ``[0, 1]``.
    """
    return max(0.0, min(1.0, value))


def _warn_on_model_mismatch(settings: Settings) -> None:
    """Log once when the similarity model is not the retrieval embedding model.

    Args:
        settings: Resolved settings.
    """
    configured = str(getattr(settings, "eval_similarity_model", "") or "")
    active = str(settings.embedding_model)
    if configured and configured != active:
        _log.warning(
            "eval_similarity_model_ignored",
            configured=configured,
            active=active,
            reason="scores use ragcore.embeddings; no second embedding stack",
        )


async def embed_texts(
    texts: Sequence[str],
    *,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[list[float]]:
    """Embed a batch of texts with the platform's dense provider.

    Args:
        texts: Texts to embed. Empty strings are embedded as-is; callers filter.
        settings: Resolved settings. Defaults to :func:`ragcore.settings.get_settings`.
        embedder: Provider override, mainly for tests.

    Returns:
        One dense vector per input, in input order.
    """
    cfg = settings or get_settings()
    provider = embedder or get_embedding_provider(cfg)
    _warn_on_model_mismatch(cfg)
    return await provider.embed_dense(list(texts))


async def semantic_similarity(
    answer: str,
    ground_truth: str,
    *,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
) -> float | None:
    """Score how close an answer is to its reference answer.

    Args:
        answer: The answer the pipeline produced.
        ground_truth: The golden item's reference answer.
        settings: Resolved settings.
        embedder: Provider override, mainly for tests.

    Returns:
        Cosine similarity clamped to ``[0, 1]``, or None when either side is empty
        (there is nothing to compare, which is not the same as scoring zero).
    """
    scores = await semantic_similarity_batch(
        [(answer, ground_truth)], settings=settings, embedder=embedder
    )
    return scores[0]


async def semantic_similarity_batch(
    pairs: Sequence[tuple[str, str]],
    *,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[float | None]:
    """Score many answer/ground-truth pairs in one embedding batch.

    FastEmbed is CPU-bound and batches well; scoring a 40-item golden set one pair
    at a time costs far more than one call with 80 texts.

    Args:
        pairs: ``(answer, ground_truth)`` tuples.
        settings: Resolved settings.
        embedder: Provider override, mainly for tests.

    Returns:
        One score per pair, in input order. A pair with an empty side scores None.
    """
    scorable: list[int] = []
    texts: list[str] = []
    for index, (answer, ground_truth) in enumerate(pairs):
        if not answer.strip() or not ground_truth.strip():
            continue
        scorable.append(index)
        texts.append(answer)
        texts.append(ground_truth)

    results: list[float | None] = [None] * len(pairs)
    if not scorable:
        return results

    vectors = await embed_texts(texts, settings=settings, embedder=embedder)
    if len(vectors) != len(texts):  # pragma: no cover - provider contract violation
        _log.error(
            "semantic_similarity_vector_count_mismatch",
            expected=len(texts),
            received=len(vectors),
        )
        return results

    for position, index in enumerate(scorable):
        answer_vector = vectors[position * 2]
        truth_vector = vectors[position * 2 + 1]
        results[index] = _clamp(cosine_similarity(answer_vector, truth_vector))
    return results
