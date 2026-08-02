"""Maximal Marginal Relevance diversification for retrieval stage 5.

Reranking optimises one thing — how well a single chunk answers the query — and it
is very good at it. That is exactly why a reranked top-k is often eight near-copies
of the same paragraph: the second-best chunk is usually the one that says the same
thing as the best one. MMR trades a little relevance for coverage:

``mmr(d) = lambda * rel(d) - (1 - lambda) * max_{s in selected} cos(d, s)``

``lambda`` comes from ``settings.retrieval_mmr_lambda`` (1.0 is pure relevance, 0.0
pure diversity). Relevance is min-max normalised inside the candidate pool, because
cross-encoder logits and RRF scores live on incomparable scales and the subtraction
above only makes sense when both terms are in ``[0, 1]``.

This module returns a **full ordering**, not a truncated one. The retriever needs to
report a ``dropped_reason`` for every candidate it does not keep, and it also applies
``retrieval_max_per_document`` after diversification — both of which need the losers
in rank order rather than discarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragcore.settings import Settings, get_settings

__all__ = [
    "MMRSelection",
    "maximal_marginal_relevance",
    "mmr_select",
    "normalise_scores",
]


@dataclass(frozen=True, slots=True)
class MMRSelection:
    """The diversified ordering of a candidate pool.

    Attributes:
        order: Every input index, in the order MMR selected them. Length always
            equals the number of candidates supplied — nothing is discarded here.
        kept: The first ``top_n`` entries of :attr:`order`.
        dropped: The remainder of :attr:`order`, in selection order, so the caller
            can attribute a drop reason to each.
        scores: MMR score each index earned at the moment it was selected. The first
            selection scores ``lambda * relevance`` (no redundancy penalty applies
            to an empty selection).
        lambda_mult: The trade-off actually used.
    """

    order: list[int] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    lambda_mult: float = 1.0


def normalise_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalise relevance scores into ``[0, 1]``.

    Cross-encoder logits are unbounded and can be negative, RRF scores cluster near
    ``1/60``, and a cosine sits in ``[-1, 1]``. Subtracting a cosine redundancy
    penalty from any of those raw values would let the scale, not the ranking, decide
    the outcome — so the pool is rescaled first.

    Args:
        scores: Relevance scores in candidate order.

    Returns:
        Scores rescaled to ``[0, 1]``, preserving order. When every score is equal
        (or a single candidate was supplied) every entry becomes ``1.0``, which makes
        MMR fall back to pure diversity — the correct behaviour when relevance
        carries no information.
    """
    if not scores:
        return []
    lowest = min(scores)
    highest = max(scores)
    spread = highest - lowest
    if spread <= 0.0:
        return [1.0] * len(scores)
    return [(score - lowest) / spread for score in scores]


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale a vector to unit length so similarity is a plain dot product.

    Args:
        vector: A dense embedding.

    Returns:
        The unit-length vector, or an all-zero vector when the input has zero
        magnitude (an unembeddable candidate is then maximally dissimilar to
        everything, so it is never penalised for redundancy).
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(component / norm for component in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product of two equal-length unit vectors.

    Args:
        left: First unit vector.
        right: Second unit vector.

    Returns:
        The dot product, which for unit vectors is the cosine similarity. Returns
        ``0.0`` when the dimensions disagree rather than raising: a dimension
        mismatch means one candidate could not be embedded, and dropping diversity
        information is far better than failing a chat turn inside stage 5.
    """
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def maximal_marginal_relevance(
    vectors: Sequence[Sequence[float]],
    relevance: Sequence[float],
    *,
    top_n: int | None = None,
    lambda_mult: float | None = None,
    settings: Settings | None = None,
) -> MMRSelection:
    """Order candidates by maximal marginal relevance.

    Args:
        vectors: Dense embedding per candidate, in candidate order. An empty vector
            is allowed and simply contributes no redundancy signal.
        relevance: Relevance score per candidate, in the same order. Any scale;
            :func:`normalise_scores` rescales it.
        top_n: How many candidates count as kept. Defaults to every candidate.
            Values below zero are clamped to zero.
        lambda_mult: Relevance/diversity trade-off in ``[0, 1]``. Defaults to
            ``settings.retrieval_mmr_lambda``. Values outside the range are clamped.
        settings: Settings supplying the default lambda. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        An :class:`MMRSelection` covering every input index.

    Raises:
        ValueError: If ``vectors`` and ``relevance`` have different lengths, which
            would silently pair a candidate with another candidate's score.

    Example:
        >>> selection = maximal_marginal_relevance(
        ...     [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ...     [1.0, 0.9, 0.5],
        ...     top_n=2,
        ...     lambda_mult=0.5,
        ... )
        >>> selection.kept
        [0, 2]
    """
    if len(vectors) != len(relevance):
        msg = (
            "mmr needs one vector per relevance score, got "
            f"{len(vectors)} vectors and {len(relevance)} scores"
        )
        raise ValueError(msg)

    cfg = settings or get_settings()
    lam = cfg.retrieval_mmr_lambda if lambda_mult is None else lambda_mult
    lam = min(1.0, max(0.0, lam))

    total = len(vectors)
    keep = total if top_n is None else max(0, min(top_n, total))
    if total == 0:
        return MMRSelection(lambda_mult=lam)

    units = [_unit(vector) for vector in vectors]
    scaled = normalise_scores(relevance)

    remaining = set(range(total))
    order: list[int] = []
    scores: dict[int, float] = {}
    # Running maximum similarity of each remaining candidate to anything already
    # selected. Updating it incrementally keeps the loop O(n^2) dot products in the
    # worst case rather than O(n^3).
    redundancy = [0.0] * total

    while remaining:
        best_index = -1
        best_score = -math.inf
        for index in sorted(remaining):
            candidate = lam * scaled[index] - (1.0 - lam) * redundancy[index]
            if candidate > best_score:
                best_score = candidate
                best_index = index
        order.append(best_index)
        scores[best_index] = best_score
        remaining.discard(best_index)
        chosen = units[best_index]
        for index in remaining:
            similarity = _dot(units[index], chosen)
            if similarity > redundancy[index]:
                redundancy[index] = similarity

    return MMRSelection(
        order=order,
        kept=order[:keep],
        dropped=order[keep:],
        scores=scores,
        lambda_mult=lam,
    )


def mmr_select(
    vectors: Sequence[Sequence[float]],
    relevance: Sequence[float],
    *,
    top_n: int | None = None,
    lambda_mult: float | None = None,
    settings: Settings | None = None,
) -> list[int]:
    """Convenience wrapper returning only the kept indices.

    Args:
        vectors: Dense embedding per candidate.
        relevance: Relevance score per candidate.
        top_n: How many candidates to keep. Defaults to all of them.
        lambda_mult: Relevance/diversity trade-off. Defaults to
            ``settings.retrieval_mmr_lambda``.
        settings: Settings supplying the default lambda.

    Returns:
        The kept candidate indices, in diversified order.
    """
    return maximal_marginal_relevance(
        vectors,
        relevance,
        top_n=top_n,
        lambda_mult=lambda_mult,
        settings=settings,
    ).kept
