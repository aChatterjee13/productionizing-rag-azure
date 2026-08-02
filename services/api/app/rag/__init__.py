"""The RAG pipeline (``services/api``).

This package holds the ordered stages `docs/CONTRACTS.md` specifies for
``services/api/app/rag``. The modules owned here are the retrieval half of that
pipeline:

* :mod:`app.rag.query_transform` -- stage 3, requirement #10. One ``MODEL_FAST``
  structured call turns the latest user message into a retrieval plan.
* :mod:`app.rag.retriever` -- stage 5, requirement #6. Hybrid search, union across
  sub-questions, dedupe, cross-encoder rerank, MMR diversification and the audited
  drop accounting.
* :mod:`app.rag.mmr` -- the diversification primitive stage 5 uses.
* :mod:`app.rag.citations` -- stage 11, the citation half of requirement #9. The
  numbered source block, ``[n]`` marker parsing, span verification and the
  groundedness verdict.

Sibling packages (:mod:`app.rag.guardrails`, :mod:`app.rag.memory`,
:mod:`app.rag.tools`) and the orchestrator are owned elsewhere and are imported
directly rather than re-exported here, so this module stays importable before they
exist.

Every symbol below is resolved lazily through :func:`__getattr__`: importing
``app.rag`` must stay free of ``qdrant-client`` and ``anthropic``, both of which the
retriever pulls in.

**Tunables.** Every threshold, limit and model name is a ``ragcore.settings`` field.
The fields in :data:`RAG_SETTING_DEFAULTS` were introduced with these modules and are
read through :func:`rag_setting`, which honours the real settings field the moment it
exists and otherwise falls back to the documented default. See "Addendum R" of
`docs/CONTRACTS.md`. Nothing in this package hard-codes a threshold at a call site.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ragcore.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from app.rag.citations import (
        CitationDrop,
        CitationReport,
        CitationVerdict,
        MarkerRef,
        SourceBlock,
        SpanMatch,
        append_uncertainty_notice,
        build_source_block,
        extract_citations,
        parse_markers,
        strip_unresolved_markers,
        verify_span,
    )
    from app.rag.mmr import (
        MMRSelection,
        maximal_marginal_relevance,
        mmr_select,
        normalise_scores,
    )
    from app.rag.query_transform import (
        QueryTransformPayload,
        TransformedQuery,
        fallback_transform,
        is_abstract_query,
        merge_filters,
        transform_query,
    )
    from app.rag.retriever import (
        DROP_ACL,
        DROP_CANDIDATE_LIMIT,
        DROP_DELETED,
        DROP_MAX_PER_DOCUMENT,
        DROP_RERANK_MIN_SCORE,
        DROP_RERANK_TOP_N,
        DROP_TOP_N,
        retrieve,
        retrieve_by_ids,
    )

__all__ = [
    "DROP_ACL",
    "DROP_CANDIDATE_LIMIT",
    "DROP_DELETED",
    "DROP_MAX_PER_DOCUMENT",
    "DROP_RERANK_MIN_SCORE",
    "DROP_RERANK_TOP_N",
    "DROP_TOP_N",
    "RAG_SETTING_DEFAULTS",
    "CitationDrop",
    "CitationReport",
    "CitationVerdict",
    "MMRSelection",
    "MarkerRef",
    "QueryTransformPayload",
    "SourceBlock",
    "SpanMatch",
    "TransformedQuery",
    "append_uncertainty_notice",
    "build_source_block",
    "extract_citations",
    "fallback_transform",
    "is_abstract_query",
    "maximal_marginal_relevance",
    "merge_filters",
    "mmr_select",
    "normalise_scores",
    "parse_markers",
    "rag_setting",
    "retrieve",
    "retrieve_by_ids",
    "strip_unresolved_markers",
    "transform_query",
    "verify_span",
]


#: Tunables these modules need that ``ragcore.settings`` does not declare yet, with
#: the value that applies until it does. Keys are real ``RAG_``-prefixed setting
#: names: exporting ``RAG_CITATION_FUZZY_THRESHOLD`` starts working the moment the
#: field exists on :class:`ragcore.settings.Settings`. Documented in Addendum R of
#: `docs/CONTRACTS.md`.
RAG_SETTING_DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------- stage 3, query transform
    # A query at or below this many words is sparse enough that lexical matching is
    # unreliable, which is the case HyDE exists for.
    "qt_hyde_max_query_words": 12,
    # ------------------------------------------------------- stage 5, retrieval
    # Divisor that maps a server-side fusion score onto the [0, 1] relevance scale
    # ``guardrail_ood_min_score`` is expressed in, used only when no cross-encoder
    # score is available. 0.05 is the reciprocal-rank-fusion reference: a chunk
    # ranked first by both branches scores 2/61 = 0.033, so a scale of 0.05 puts a
    # dual-top-ranked chunk at 0.66. Set it near 2.0 when ``retrieval_fusion`` is
    # ``dbsf``, whose scores are sums of normalised branch scores.
    "retrieval_fusion_score_scale": 0.05,
    # Probes (rewritten query, sub-questions, HyDE passage) searched concurrently.
    "retrieval_query_concurrency": 4,
    # Where MMR gets its dense vectors: read them back from Qdrant (cheap, exact) or
    # re-embed the candidate text locally. Qdrant is tried first either way; this
    # decides whether a miss falls back to the embedder.
    "retrieval_mmr_embed_fallback": True,
    # ------------------------------------------------------- stage 11, citations
    # Similarity a paraphrased sentence must reach against its best-matching window
    # of the cited chunk to count as supported.
    "citation_fuzzy_threshold": 0.55,
    # Similarity an *explicitly quoted* literal must reach. Much stricter: the
    # answer prompt promises verbatim quoting, so a near-miss is a fabrication.
    "citation_quote_min_ratio": 0.9,
    # Fraction of a span's content tokens that must appear in the matched window
    # for token recall to stand in for character similarity. Character similarity
    # punishes a faithful paraphrase for reordering a sentence; recall does not.
    "citation_token_recall_threshold": 0.6,
    # Require every multi-digit number a cited sentence asserts to occur in the
    # cited chunk. A wrong figure is the costliest hallucination and the cheapest
    # to catch. Skipped automatically when a sentence cites several sources.
    "citation_number_check": True,
    # Spans shorter than this carry too little signal to verify on their own and are
    # extended with the preceding sentence first.
    "citation_min_span_chars": 12,
    # Hard cap on the span compared against a chunk, so one runaway sentence cannot
    # dominate verification cost.
    "citation_max_span_chars": 600,
    # Sentences with at least this many words count as factual claims for coverage.
    "citation_min_claim_words": 5,
    # Fraction of claim sentences that must carry a verified citation before the
    # verdict is ``grounded`` rather than ``partial``.
    "citation_min_coverage": 0.5,
    # Rare tokens used to anchor the fuzzy window search inside a chunk.
    "citation_anchor_tokens": 3,
    # Upper bound on candidate windows scored per span. Bounds the O(n*m) work.
    "citation_max_windows": 24,
    # Remove ``[n]`` markers from the answer when no citation survived for them.
    "citation_strip_unresolved_markers": True,
}

#: Where each lazily re-exported symbol lives.
_LAZY_EXPORTS: dict[str, str] = {
    "CitationDrop": "app.rag.citations",
    "CitationReport": "app.rag.citations",
    "CitationVerdict": "app.rag.citations",
    "MarkerRef": "app.rag.citations",
    "SourceBlock": "app.rag.citations",
    "SpanMatch": "app.rag.citations",
    "append_uncertainty_notice": "app.rag.citations",
    "build_source_block": "app.rag.citations",
    "extract_citations": "app.rag.citations",
    "parse_markers": "app.rag.citations",
    "strip_unresolved_markers": "app.rag.citations",
    "verify_span": "app.rag.citations",
    "MMRSelection": "app.rag.mmr",
    "maximal_marginal_relevance": "app.rag.mmr",
    "mmr_select": "app.rag.mmr",
    "normalise_scores": "app.rag.mmr",
    "QueryTransformPayload": "app.rag.query_transform",
    "TransformedQuery": "app.rag.query_transform",
    "fallback_transform": "app.rag.query_transform",
    "is_abstract_query": "app.rag.query_transform",
    "merge_filters": "app.rag.query_transform",
    "transform_query": "app.rag.query_transform",
    "DROP_ACL": "app.rag.retriever",
    "DROP_CANDIDATE_LIMIT": "app.rag.retriever",
    "DROP_DELETED": "app.rag.retriever",
    "DROP_MAX_PER_DOCUMENT": "app.rag.retriever",
    "DROP_RERANK_MIN_SCORE": "app.rag.retriever",
    "DROP_RERANK_TOP_N": "app.rag.retriever",
    "DROP_TOP_N": "app.rag.retriever",
    "retrieve": "app.rag.retriever",
    "retrieve_by_ids": "app.rag.retriever",
}


def rag_setting(settings: Settings, name: str) -> Any:
    """Read a pipeline tunable, preferring the settings field over the default.

    The rule "every tunable is a ``ragcore.settings`` field with a default, never a
    literal at a call site" still holds. This indirection exists only so a knob these
    modules need can be introduced without editing :mod:`ragcore.settings`, which a
    different component owns, and it stops being consulted the moment the field is
    declared there.

    Args:
        settings: Active settings object.
        name: A key of :data:`RAG_SETTING_DEFAULTS`, e.g.
            ``"citation_fuzzy_threshold"``.

    Returns:
        ``getattr(settings, name)`` when the field exists and is set, otherwise the
        documented default.

    Raises:
        KeyError: If ``name`` is not a declared tunable. That is a typo at the call
            site, not a configuration problem, so it fails loudly.
    """
    if name not in RAG_SETTING_DEFAULTS:
        msg = f"unknown rag tunable {name!r}"
        raise KeyError(msg)
    value = getattr(settings, name, None)
    return RAG_SETTING_DEFAULTS[name] if value is None else value


def __getattr__(name: str) -> Any:
    """Resolve a lazily re-exported submodule symbol.

    Args:
        name: Attribute being imported from this package.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public surface, including the lazily loaded names.

    Returns:
        Sorted attribute names.
    """
    return sorted(set(__all__) | set(globals()))
