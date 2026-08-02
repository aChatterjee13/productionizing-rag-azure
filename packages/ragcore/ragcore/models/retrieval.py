"""Retrieval request and result models.

``MetadataFilter`` is the user-facing facet filter; it is always composed with the
ACL filter inside :func:`ragcore.vectorstore.filters.build_acl_filter` and never used
on its own — a metadata-only filter would return cross-tenant points.

``RetrievedChunk`` keeps every score from every stage so a trace can explain why a
chunk made or missed the cut, and ``dropped_reason`` makes every drop auditable
rather than silent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragcore.models.acl import Classification
from ragcore.models.chunk import ChunkPayload

__all__ = [
    "UNSATISFIABLE",
    "Citation",
    "MetadataFilter",
    "RetrievalResult",
    "RetrievalStage",
    "RetrievedChunk",
]

#: Written into a facet list when intersecting two filters leaves it empty.
#:
#: ``MetadataFilter`` deliberately reads an empty list as "no constraint" so a UI
#: sending ``[]`` does not match nothing. That is right for user input and wrong for
#: an intersection, where empty means the two filters share no value and the correct
#: answer is to return nothing. Storing this sentinel keeps the clause present and
#: unsatisfiable; no document_id, tag, author or language can equal it, because the
#: NUL byte cannot appear in any of the sources those values come from.
UNSATISFIABLE = "\x00__unsatisfiable__"


def _narrower_section(mine: str | None, theirs: str | None) -> str | None:
    """Intersect two ``section_prefix`` constraints.

    ``section_path`` is a keyword array holding the breadcrumb, so a prefix names a
    subtree. Two different headings name disjoint subtrees — taking either one would
    widen the filter past the other, so the intersection is empty.

    Args:
        mine: This filter's section prefix, or None.
        theirs: The other filter's section prefix, or None.

    Returns:
        The single constraint when only one side sets it, the shared value when both
        agree, otherwise :data:`UNSATISFIABLE`.
    """
    if mine is None or theirs is None:
        return mine or theirs
    return mine if mine == theirs else UNSATISFIABLE


class RetrievalStage(StrEnum):
    """Which stage produced or last touched a candidate."""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSION = "fusion"
    RERANK = "rerank"
    CACHE = "cache"
    TOOL = "tool"


class MetadataFilter(BaseModel):
    """Facet filter applied alongside the mandatory ACL filter.

    Every list field is an OR within itself and an AND across fields. A None value
    means "no constraint on this facet"; an empty list is treated the same way so a
    UI that sends `[]` does not accidentally match nothing.
    """

    model_config = ConfigDict(extra="forbid")

    doc_types: list[str] | None = Field(
        default=None, description="Match any of these doc_type values."
    )
    source_types: list[str] | None = Field(
        default=None, description="Match any of these source_type values."
    )
    tags: list[str] | None = Field(default=None, description="Match any of these tags.")
    authors: list[str] | None = Field(
        default=None, description="Match any of these authors."
    )
    languages: list[str] | None = Field(
        default=None, description="Match any of these ISO 639-1 language codes."
    )
    document_ids: list[str] | None = Field(
        default=None, description="Restrict to these document ids."
    )
    section_prefix: str | None = Field(
        default=None,
        description="Restrict to chunks whose section_path starts with this heading.",
    )
    date_from: datetime | None = Field(
        default=None, description="Lower bound on source_modified_at, inclusive."
    )
    date_to: datetime | None = Field(
        default=None, description="Upper bound on source_modified_at, inclusive."
    )
    max_classification: Classification | None = Field(
        default=None,
        description=(
            "Extra classification ceiling. Narrows but never widens the principal's "
            "clearance — the effective ceiling is the minimum of the two."
        ),
    )
    exclude_pii: bool = Field(
        default=False, description="Exclude chunks with any detected PII entity."
    )

    @model_validator(mode="after")
    def _normalise(self) -> MetadataFilter:
        """Collapse empty lists to None and order the date range.

        Returns:
            The normalised filter.

        Raises:
            ValueError: If ``date_from`` is later than ``date_to``.
        """
        for name in (
            "doc_types",
            "source_types",
            "tags",
            "authors",
            "languages",
            "document_ids",
        ):
            value = getattr(self, name)
            if value is not None and len(value) == 0:
                object.__setattr__(self, name, None)
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            msg = "date_from must not be later than date_to"
            raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """Whether this filter constrains anything at all.

        Returns:
            True when no facet is set and ``exclude_pii`` is False.
        """
        return not any(
            (
                self.doc_types,
                self.source_types,
                self.tags,
                self.authors,
                self.languages,
                self.document_ids,
                self.section_prefix,
                self.date_from,
                self.date_to,
                self.max_classification,
                self.exclude_pii,
            )
        )

    def merged_with(self, other: MetadataFilter | None) -> MetadataFilter:
        """Intersect this filter with another, taking the narrower constraint.

        Used to combine a filter the user supplied with facets the query transformer
        extracted. Neither side can widen the other: when both sides constrain a
        facet and share no value the result is :data:`UNSATISFIABLE`, so the clause
        survives and matches nothing. Returning an empty list there would be read as
        "no constraint" by :meth:`_normalise` and would widen the filter to every
        document the ACL allows.

        Args:
            other: The filter to merge in, or None.

        Returns:
            A new :class:`MetadataFilter` no wider than either input.
        """
        if other is None:
            return self.model_copy()

        def _intersect(a: list[str] | None, b: list[str] | None) -> list[str] | None:
            if a is None:
                return list(b) if b is not None else None
            if b is None:
                return list(a)
            common = [item for item in a if item in set(b)]
            return common or [UNSATISFIABLE]

        classifications = [
            level
            for level in (self.max_classification, other.max_classification)
            if level is not None
        ]
        return MetadataFilter(
            doc_types=_intersect(self.doc_types, other.doc_types),
            source_types=_intersect(self.source_types, other.source_types),
            tags=_intersect(self.tags, other.tags),
            authors=_intersect(self.authors, other.authors),
            languages=_intersect(self.languages, other.languages),
            document_ids=_intersect(self.document_ids, other.document_ids),
            section_prefix=_narrower_section(self.section_prefix, other.section_prefix),
            date_from=max(
                [d for d in (self.date_from, other.date_from) if d is not None],
                default=None,
            ),
            date_to=min(
                [d for d in (self.date_to, other.date_to) if d is not None],
                default=None,
            ),
            max_classification=min(classifications) if classifications else None,
            exclude_pii=self.exclude_pii or other.exclude_pii,
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        """Canonical, order-stable representation for cache fingerprinting.

        Lists are sorted so two filters that differ only in ordering hash the same.

        Returns:
            A JSON-safe mapping used by
            :func:`ragcore.vectorstore.filters.filter_fingerprint`.
        """
        data = self.model_dump(mode="json", exclude_none=True)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = sorted(value)
        return data


class RetrievedChunk(BaseModel):
    """One candidate chunk with the score it earned at each stage."""

    model_config = ConfigDict(extra="forbid")

    payload: ChunkPayload = Field(description="The chunk's Qdrant payload.")
    dense_score: float | None = Field(
        default=None, description="Cosine score from the dense prefetch branch."
    )
    sparse_score: float | None = Field(
        default=None, description="BM25 score from the sparse prefetch branch."
    )
    fusion_score: float = Field(
        default=0.0, description="Server-side RRF or DBSF fused score."
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder relevance score, when reranked."
    )
    final_score: float = Field(
        default=0.0,
        description="Score used for ordering after rerank and MMR diversification.",
    )
    retrieval_stage: str = Field(
        default=RetrievalStage.FUSION.value,
        description="dense | sparse | fusion | rerank | cache | tool.",
    )
    dropped_reason: str | None = Field(
        default=None,
        description=(
            "Set when this candidate was dropped, e.g. 'duplicate:simhash', "
            "'classification', 'mmr', 'max_per_document'. Drops are audited, "
            "never silent."
        ),
    )

    @property
    def chunk_id(self) -> str:
        """Convenience accessor for the chunk id.

        Returns:
            The payload's ``chunk_id``.
        """
        return self.payload.chunk_id

    @property
    def document_id(self) -> str:
        """Convenience accessor for the owning document id.

        Returns:
            The payload's ``document_id``.
        """
        return self.payload.document_id

    @property
    def is_dropped(self) -> bool:
        """Whether this candidate was excluded from the final set.

        Returns:
            True when ``dropped_reason`` is set.
        """
        return self.dropped_reason is not None

    def drop(self, reason: str) -> RetrievedChunk:
        """Return a copy marked as dropped.

        Args:
            reason: Short machine-readable reason, e.g. ``"duplicate:sha256"``.

        Returns:
            A copy with ``dropped_reason`` set.
        """
        return self.model_copy(update={"dropped_reason": reason})

    def to_citation(self, marker: str, *, confidence: float = 1.0) -> Citation:
        """Build a citation skeleton pointing at this chunk.

        The quoted span is filled in later by citation extraction, which verifies the
        span actually occurs in ``payload.text``.

        Args:
            marker: Rendered marker such as ``"[1]"``.
            confidence: Initial confidence before span verification.

        Returns:
            A :class:`Citation` without a quoted span.
        """
        return Citation(
            marker=marker,
            document_id=self.payload.document_id,
            chunk_id=self.payload.chunk_id,
            title=self.payload.title,
            source_uri=self.payload.source_uri,
            section_path=list(self.payload.section_path),
            page=self.payload.page,
            quoted_span=None,
            char_start=None,
            char_end=None,
            confidence=confidence,
        )


class Citation(BaseModel):
    """A verified pointer from an answer marker back to its supporting chunk."""

    model_config = ConfigDict(extra="forbid")

    marker: str = Field(description="Marker as rendered in the answer, e.g. '[1]'.")
    document_id: str = Field(description="Cited document id.")
    chunk_id: str = Field(description="Cited chunk id.")
    title: str = Field(default="", description="Document title for display.")
    source_uri: str = Field(default="", description="Link target for the citation.")
    section_path: list[str] = Field(
        default_factory=list, description="Heading breadcrumb for display."
    )
    page: int | None = Field(default=None, description="1-based page number, if any.")
    quoted_span: str | None = Field(
        default=None,
        description=(
            "Verbatim supporting text. Verified to occur in the cited chunk under "
            "whitespace-normalised matching before the citation is kept."
        ),
    )
    char_start: int | None = Field(
        default=None,
        ge=0,
        description="Start offset of the span within the chunk text.",
    )
    char_end: int | None = Field(
        default=None, ge=0, description="End offset of the span within the chunk text."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Verification confidence. Unverifiable citations are flagged.",
    )

    @model_validator(mode="after")
    def _validate_span(self) -> Citation:
        """Ensure the character range is coherent when both bounds are present.

        Returns:
            The validated citation.

        Raises:
            ValueError: If ``char_start`` is greater than ``char_end``.
        """
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start > self.char_end
        ):
            msg = "char_start must not exceed char_end"
            raise ValueError(msg)
        return self

    @property
    def is_verified(self) -> bool:
        """Whether this citation carries a located, verbatim span.

        Returns:
            True when a quoted span and its offsets are present.
        """
        return bool(self.quoted_span) and self.char_start is not None


class RetrievalResult(BaseModel):
    """Everything stage 5 produced, including the counters the UI surfaces."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Final ordered chunks handed to generation."
    )
    queries_used: list[str] = Field(
        default_factory=list,
        description="Queries actually issued after transformation and decomposition.",
    )
    filter_applied: dict = Field(
        default_factory=dict,
        description="Serialised view of the composed ACL + metadata filter.",
    )
    total_candidates: int = Field(
        default=0, ge=0, description="Candidates returned by fusion across all queries."
    )
    after_dedupe: int = Field(
        default=0,
        ge=0,
        description="Candidates remaining after hash and simhash dedupe.",
    )
    after_rerank: int = Field(
        default=0, ge=0, description="Candidates remaining after cross-encoder rerank."
    )
    latency_ms: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-stage latency: embed, dense, sparse, fuse, dedupe, rerank, mmr."
        ),
    )
    cache_hit: bool = Field(
        default=False,
        description="True when the semantic cache supplied the candidate chunk ids.",
    )
    dropped: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Audited drops with their reasons. Never discarded silently.",
    )

    @property
    def max_score(self) -> float:
        """Highest final score in the result set.

        The OOD gate compares this against ``guardrail_ood_min_score``.

        Returns:
            The maximum ``final_score``, or 0.0 when nothing was retrieved.
        """
        return max((chunk.final_score for chunk in self.chunks), default=0.0)

    @property
    def chunk_ids(self) -> list[str]:
        """Ids of the retained chunks, in order.

        Returns:
            The ordered chunk ids.
        """
        return [chunk.payload.chunk_id for chunk in self.chunks]

    @property
    def document_ids(self) -> list[str]:
        """Distinct document ids represented in the result, in first-seen order.

        Returns:
            The de-duplicated document ids.
        """
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.payload.document_id, None)
        return list(seen)

    @property
    def total_latency_ms(self) -> float:
        """Sum of all recorded stage latencies.

        Returns:
            Total retrieval latency in milliseconds.
        """
        return sum(self.latency_ms.values())

    def without_text(self) -> dict[str, Any]:
        """Serialise for the SSE ``retrieval`` event, dropping chunk text.

        The web client only needs scores and citation metadata mid-stream; sending
        full chunk text would both bloat the stream and risk emitting content before
        the output guard has run.

        Returns:
            A JSON-safe mapping with ``text``, ``contextual_header`` and ``summary``
            removed from every chunk payload.
        """
        data = self.model_dump(mode="json")
        for bucket in ("chunks", "dropped"):
            for chunk in data.get(bucket, []):
                payload = chunk.get("payload", {})
                payload.pop("text", None)
                payload.pop("contextual_header", None)
                payload.pop("summary", None)
        return data
