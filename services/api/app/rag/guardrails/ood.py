"""Out-of-domain adjudication — pipeline stage 6.

A RAG system's worst failure is not "I don't know". It is a fluent answer
assembled from three chunks that share vocabulary with the question and nothing
else. This stage decides whether the retrieved evidence actually bears on the
question, and when it does not, produces a refusal that is *useful*: it says the
question is outside the indexed corpus **and** names what the corpus does cover,
derived from the tenant's own ``doc_type``/``tags``/title statistics as seen
through the caller's ACL filter.

Three independent sources of evidence are combined:

1. **Retrieval evidence** (:func:`relevance_signals`). Max score below
   ``guardrail_ood_min_score``, mean below ``guardrail_ood_mean_score_min``, or a
   *collapsed* score distribution — the top-k scores all within
   ``guardrail_ood_collapse_spread`` of each other and none above
   ``guardrail_ood_collapse_max_score``. Collapse is the interesting one: a healthy
   retrieval has a clear winner, so "everything is equally mediocre" means the
   ranker found nothing to prefer, which is what a query about an unindexed subject
   looks like even when the absolute scores are unremarkable.
2. **The query transformer's flag** (``TransformedQuery.is_out_of_domain``), used
   only when the transform did not degrade — a fallback plan's flags are defaults,
   not evidence.
3. **An optional ``MODEL_CHEAP`` adjudication** (``guardrail_ood_classifier_enabled``,
   off by default) that runs only when the first two disagree.

A query the evidence calls out-of-domain but that a tool could serve is **not**
refused: it is handed to stage 8 with ``needs_tool`` set. That is the contract's
"and no tool can serve the query" clause, and it is why this gate takes
``tool_available``.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import fmean
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.acl import Principal
from ragcore.models.chat import GuardrailAction, GuardrailEvent, GuardrailKind
from ragcore.models.retrieval import MetadataFilter, RetrievalResult, RetrievedChunk
from ragcore.observability import observe_guardrail
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore.collections import CHUNKS
from ragcore.vectorstore.filters import build_acl_filter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.rag.query_transform import TransformedQuery
    from ragcore.llm.client import LLMClient

__all__ = [
    "CoverageItem",
    "DomainCoverage",
    "OODVerdict",
    "RelevanceSignals",
    "clear_coverage_cache",
    "fallback_refusal",
    "relevance_signals",
    "run_ood_gate",
    "tenant_coverage",
]

_log = structlog.get_logger(__name__)

#: Labels for the optional cheap adjudicator, **safest first**: ``LLMClient.classify``
#: returns ``labels[0]`` on a refusal, a parse failure or an unknown label, and a
#: refusal is the safe failure for this decision.
_OOD_LABELS = ("out_of_domain", "needs_tool", "in_domain")

#: ``(monotonic_deadline, coverage)`` keyed by tenant and effective clearance.
_COVERAGE_CACHE: dict[tuple[Any, ...], tuple[float, DomainCoverage]] = {}


class CoverageItem(BaseModel):
    """One facet value the tenant's indexed corpus contains."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(description="Facet value, e.g. 'policy' or 'travel'.")
    count: int = Field(ge=0, description="Chunks in the sample carrying it.")


class DomainCoverage(BaseModel):
    """What the corpus covers, as far as this principal is allowed to know.

    Built by sampling chunks **through**
    :func:`~ragcore.vectorstore.filters.build_acl_filter`, so a refusal can never
    advertise the existence of a document type or a title the caller is not cleared
    for. That is a real leak vector: "we have no answer, but we do hold Board
    Compensation Minutes" is itself disclosure.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Tenant the sample was taken from.")
    doc_types: list[CoverageItem] = Field(
        default_factory=list, description="Document types, most frequent first."
    )
    tags: list[CoverageItem] = Field(
        default_factory=list, description="Tags, most frequent first."
    )
    titles: list[str] = Field(
        default_factory=list, description="Representative document titles."
    )
    documents_sampled: int = Field(
        default=0, ge=0, description="Distinct documents seen in the sample."
    )
    chunks_sampled: int = Field(
        default=0, ge=0, description="Chunks inspected to build the summary."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the sample was taken; cached for the configured TTL.",
    )

    @property
    def is_empty(self) -> bool:
        """Whether the sample found nothing at all.

        Returns:
            True when the principal can see no indexed chunks.
        """
        return self.chunks_sampled == 0

    def describe(self, *, max_items: int = 8) -> str:
        """Render the coverage as one prose fragment for a refusal.

        Args:
            max_items: Maximum values listed per facet.

        Returns:
            A sentence fragment such as ``"policies, handbooks and runbooks
            (11 documents), on topics including travel, onboarding, vpn"``, or an
            empty string when nothing is indexed for this principal.
        """
        if self.is_empty:
            return ""
        parts: list[str] = []
        if self.doc_types:
            names = [item.value for item in self.doc_types[:max_items]]
            parts.append(_join(names))
        if self.documents_sampled:
            noun = "document" if self.documents_sampled == 1 else "documents"
            suffix = f"{self.documents_sampled} {noun}"
            if parts:
                parts[-1] = f"{parts[-1]} ({suffix})"
            else:
                parts.append(suffix)
        if self.tags:
            topics = [item.value for item in self.tags[:max_items]]
            parts.append(f"on topics including {_join(topics)}")
        elif self.titles:
            parts.append(f"including {_join(self.titles[:3])}")
        return ", ".join(part for part in parts if part)


class RelevanceSignals(BaseModel):
    """The retrieval-side evidence the gate weighs."""

    model_config = ConfigDict(extra="forbid")

    candidate_count: int = Field(default=0, ge=0, description="Candidates considered.")
    max_score: float = Field(default=0.0, description="Best candidate score.")
    mean_score: float = Field(default=0.0, description="Mean over the top-k window.")
    top_k_spread: float = Field(
        default=0.0, description="max - min across the top-k window."
    )
    score_source: str = Field(
        default="final",
        description="'rerank' when rerank scores were available, else 'final'.",
    )
    below_min_score: bool = Field(
        default=False, description="max_score < guardrail_ood_min_score."
    )
    below_mean_score: bool = Field(
        default=False, description="mean_score < guardrail_ood_mean_score_min."
    )
    collapsed: bool = Field(
        default=False,
        description="Score distribution collapsed: no candidate stands out.",
    )
    too_few_candidates: bool = Field(
        default=False, description="Fewer than guardrail_ood_min_candidates."
    )

    @property
    def weak(self) -> bool:
        """Whether the retrieval evidence fails to support an answer.

        Returns:
            True when any individual weakness test fired.
        """
        return (
            self.too_few_candidates
            or self.below_min_score
            or self.below_mean_score
            or self.collapsed
        )

    @property
    def reason(self) -> str:
        """Machine-readable name of the strongest weakness.

        Returns:
            One of ``no_candidates``, ``low_max_score``, ``collapsed_distribution``,
            ``low_mean_score`` or ``sufficient_evidence``.
        """
        if self.too_few_candidates:
            return "no_candidates"
        if self.below_min_score:
            return "low_max_score"
        if self.collapsed:
            return "collapsed_distribution"
        if self.below_mean_score:
            return "low_mean_score"
        return "sufficient_evidence"


class OODVerdict(BaseModel):
    """Stage 6's decision."""

    model_config = ConfigDict(extra="forbid")

    is_out_of_domain: bool = Field(
        default=False, description="True when the turn must be refused."
    )
    needs_tool: bool = Field(
        default=False,
        description=(
            "True when the corpus cannot answer but a registered tool could. The "
            "orchestrator routes to stage 8 instead of refusing."
        ),
    )
    reason: str = Field(
        default="in_domain", description="Machine-readable cause of the decision."
    )
    detail: str = Field(default="", description="Redacted explanation for the trace.")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the decision."
    )
    signals: RelevanceSignals = Field(
        default_factory=RelevanceSignals, description="Retrieval evidence."
    )
    coverage: DomainCoverage | None = Field(
        default=None, description="Corpus coverage used to compose the refusal."
    )
    refusal: str = Field(
        default="",
        description="User-facing refusal. Empty unless is_out_of_domain is set.",
    )
    classifier_label: str | None = Field(
        default=None, description="Label from the optional cheap adjudicator."
    )
    events: list[GuardrailEvent] = Field(
        default_factory=list, description="Events to stream and persist."
    )


def _join(values: Sequence[str]) -> str:
    """Join names into an English list.

    Args:
        values: Names to join.

    Returns:
        ``"a"``, ``"a and b"`` or ``"a, b and c"``. Empty input yields "".
    """
    items = [value for value in values if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _chunks_of(
    source: RetrievalResult | Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Accept either a result object or a bare candidate list.

    Args:
        source: A :class:`~ragcore.models.retrieval.RetrievalResult` or a sequence of
            chunks.

    Returns:
        The retained chunks.
    """
    if isinstance(source, RetrievalResult):
        return list(source.chunks)
    return list(source)


def _score_of(chunk: RetrievedChunk, *, use_rerank: bool) -> float:
    """Pick the score the gate should judge a candidate by.

    Cross-encoder scores are the honest relevance signal; a fusion score is a rank
    artefact and says little about whether the passage answers anything. When the
    reranker is disabled the final score is all there is.

    Args:
        chunk: The candidate.
        use_rerank: Whether rerank scores are present across the set.

    Returns:
        The score to use.
    """
    if use_rerank and chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    return float(chunk.final_score)


def relevance_signals(
    source: RetrievalResult | Sequence[RetrievedChunk],
    *,
    settings: Settings | None = None,
) -> RelevanceSignals:
    """Summarise the retrieval evidence for one turn.

    Args:
        source: The stage-5 result, or its chunks.
        settings: Process settings.

    Returns:
        A :class:`RelevanceSignals` with every weakness test already evaluated.
    """
    resolved = settings or get_settings()
    chunks = _chunks_of(source)
    min_candidates = int(resolved.guardrail_ood_min_candidates)

    if not chunks:
        return RelevanceSignals(too_few_candidates=True)

    use_rerank = any(chunk.rerank_score is not None for chunk in chunks)
    scores = sorted(
        (_score_of(chunk, use_rerank=use_rerank) for chunk in chunks), reverse=True
    )
    top_k = max(2, int(resolved.guardrail_ood_collapse_top_k))
    window = scores[:top_k]

    signals = RelevanceSignals(
        candidate_count=len(chunks),
        max_score=round(scores[0], 4),
        mean_score=round(fmean(window), 4),
        top_k_spread=round(window[0] - window[-1], 4),
        score_source="rerank" if use_rerank else "final",
        too_few_candidates=len(chunks) < min_candidates,
    )
    signals.below_min_score = signals.max_score < resolved.guardrail_ood_min_score
    signals.below_mean_score = signals.mean_score < float(
        resolved.guardrail_ood_mean_score_min
    )

    collapse_enabled = bool(resolved.guardrail_ood_collapse_enabled)
    if collapse_enabled and len(window) >= 3:
        spread_limit = float(resolved.guardrail_ood_collapse_spread)
        ceiling = float(resolved.guardrail_ood_collapse_max_score)
        signals.collapsed = (
            signals.top_k_spread <= spread_limit and signals.max_score <= ceiling
        )
    return signals


def clear_coverage_cache() -> None:
    """Drop every cached :class:`DomainCoverage`.

    Called by tests and by the admin reindex path: coverage is derived from the
    index, so a bulk ingest invalidates it before its TTL expires.
    """
    _COVERAGE_CACHE.clear()


def _coverage_key(
    principal: Principal, extra: MetadataFilter | None
) -> tuple[Any, ...]:
    """Build the cache key for a principal's view of the corpus.

    Roles and groups are part of the key because they change *what the sample can
    see*: two users of one tenant with different group memberships must not share a
    coverage summary.

    Args:
        principal: The caller.
        extra: Additional metadata filter applied to the sample.

    Returns:
        A hashable key.
    """
    return (
        principal.tenant_id,
        principal.clearance_rank(),
        tuple(sorted(principal.roles)),
        tuple(sorted(principal.groups)),
        None if extra is None else repr(sorted(extra.fingerprint_payload().items())),
    )


async def tenant_coverage(
    principal: Principal,
    *,
    settings: Settings | None = None,
    client: Any = None,
    extra: MetadataFilter | None = None,
) -> DomainCoverage:
    """Sample what this principal can see, for the refusal to describe.

    Reads a bounded sample (``guardrail_ood_coverage_sample`` chunks) through the
    ACL filter and tallies ``doc_type``, ``tags`` and titles. Cached for
    ``guardrail_ood_coverage_ttl_seconds`` per (tenant, clearance, roles, groups).

    Every failure mode returns an empty coverage and logs: a refusal that cannot
    list what *is* covered is worse than a refusal, but a turn that dies because
    Qdrant hiccuped while composing an apology is worse still.

    Args:
        principal: The caller. Supplies the tenant and the ACL filter.
        settings: Process settings.
        client: Qdrant client override. Defaults to the shared cached client.
        extra: Optional metadata filter, so a filtered search can describe the
            filtered corpus rather than the whole one.

    Returns:
        A :class:`DomainCoverage`, possibly empty.
    """
    resolved = settings or get_settings()
    key = _coverage_key(principal, extra)
    now = time.monotonic()
    cached = _COVERAGE_CACHE.get(key)
    if cached and cached[0] > now:
        return cached[1]

    coverage = DomainCoverage(tenant_id=principal.tenant_id)
    sample_size = int(resolved.guardrail_ood_coverage_sample)
    max_items = int(resolved.guardrail_ood_coverage_max_items)

    try:
        qdrant = client
        if qdrant is None:
            from ragcore.vectorstore.client import get_client

            qdrant = await get_client(resolved)
        points, _ = await qdrant.scroll(
            collection_name=CHUNKS,
            scroll_filter=build_acl_filter(principal, extra),
            limit=sample_size,
            with_payload=["doc_type", "tags", "title", "document_id"],
            with_vectors=False,
        )
    except Exception:
        _log.warning(
            "ood_coverage_unavailable",
            tenant_id=principal.tenant_id,
            exc_info=True,
        )
        return coverage

    doc_types: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    titles: dict[str, None] = {}
    documents: set[str] = set()

    for point in points:
        payload = getattr(point, "payload", None) or {}
        doc_type = str(payload.get("doc_type") or "").strip()
        if doc_type:
            doc_types[doc_type] += 1
        for tag in payload.get("tags") or []:
            cleaned = str(tag).strip()
            if cleaned:
                tags[cleaned] += 1
        title = str(payload.get("title") or "").strip()
        if title:
            titles.setdefault(title, None)
        document_id = str(payload.get("document_id") or "").strip()
        if document_id:
            documents.add(document_id)

    coverage.doc_types = [
        CoverageItem(value=value, count=count)
        for value, count in doc_types.most_common(max_items)
    ]
    coverage.tags = [
        CoverageItem(value=value, count=count)
        for value, count in tags.most_common(max_items)
    ]
    coverage.titles = list(titles)[:max_items]
    coverage.documents_sampled = len(documents)
    coverage.chunks_sampled = len(points)

    ttl = float(resolved.guardrail_ood_coverage_ttl_seconds)
    _COVERAGE_CACHE[key] = (now + ttl, coverage)
    return coverage


def fallback_refusal(
    coverage: DomainCoverage | None,
    *,
    question: str = "",
    needs_tool: bool = False,
    settings: Settings | None = None,
) -> str:
    """Compose the deterministic out-of-domain refusal.

    Deliberately not a model call by default: the one thing this text must never do
    is answer the question, and a template cannot. It states the boundary, names
    what *is* indexed, and gives the reader somewhere to go — the three things the
    contract requires and that a bare "I don't know" fails to provide.

    Args:
        coverage: Corpus summary for this principal. ``None`` or empty produces the
            "nothing indexed / nothing visible" variant.
        question: The user's question. Only its presence is used; it is never echoed
            back, because it is unredacted user content.
        needs_tool: True when the answer would require live data no document holds.
        settings: Process settings. Reserved for the coverage item cap.

    Returns:
        The refusal text.
    """
    resolved = settings or get_settings()
    max_items = int(resolved.guardrail_ood_coverage_max_items)
    lines = [
        "That question is outside the material indexed for you: nothing in the "
        "corpus addresses it, so answering would mean making it up."
    ]

    described = coverage.describe(max_items=max_items) if coverage else ""
    if described:
        lines.append(f"What is indexed here: {described}.")
    elif coverage is not None and coverage.is_empty:
        lines.append(
            "No documents are indexed for your account yet, so there is nothing "
            "for me to search."
        )

    if needs_tool:
        lines.append(
            "This looks like live or per-record data rather than documentation — "
            "it would come from the owning system, not from the document corpus."
        )
    elif described:
        lines.append(
            "Try narrowing the question to one of those areas, or point me at the "
            "document you have in mind."
        )
    else:
        lines.append(
            "Ask an administrator to index the relevant documents, or ask the "
            "system that owns this data."
        )
    return " ".join(lines)


async def _phrase_refusal(
    *,
    question: str,
    coverage: DomainCoverage,
    settings: Settings,
    llm: LLMClient | None,
    needs_tool: bool,
) -> str:
    """Ask ``MODEL_FAST`` to phrase the refusal, falling back to the template.

    Args:
        question: The user's question.
        coverage: Corpus summary.
        settings: Process settings.
        llm: Client override.
        needs_tool: Whether a tool could serve the request.

    Returns:
        The refusal text; the deterministic template on any failure or refusal.
    """
    from ragcore.llm.prompts import OOD_REFUSAL_SYSTEM, prompt_metadata

    template = fallback_refusal(
        coverage, question=question, needs_tool=needs_tool, settings=settings
    )
    client = llm
    if client is None:
        from ragcore.llm.client import get_llm_client

        client = get_llm_client(settings)

    max_items = int(settings.guardrail_ood_coverage_max_items)
    user_turn = (
        f"<question>{question}</question>\n"
        f"<coverage>{coverage.describe(max_items=max_items) or 'nothing indexed'}"
        "</coverage>"
    )
    try:
        response = await client.complete(
            system=OOD_REFUSAL_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
            model=settings.anthropic_model_fast,
            effort=settings.anthropic_effort_fast,
            thinking=False,
            name="guardrail.ood_refusal",
            metadata=prompt_metadata("ood_refusal"),
        )
    except Exception:
        _log.warning("ood_refusal_generation_failed", exc_info=True)
        return template
    if response.refused or not response.text.strip():
        return template
    return response.text.strip()


async def _adjudicate(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
    llm: LLMClient | None,
) -> str | None:
    """Ask ``MODEL_CHEAP`` whether the candidates actually bear on the question.

    Args:
        question: The rewritten question.
        chunks: Best candidates, already truncated by the caller.
        settings: Process settings.
        llm: Client override.

    Returns:
        One of :data:`_OOD_LABELS`, or None when the classifier did not run.
    """
    from app.rag.guardrails.injection import wrap_untrusted
    from ragcore.llm.prompts import OOD_ADJUDICATION_SYSTEM, prompt_metadata

    client = llm
    if client is None:
        from ragcore.llm.client import get_llm_client

        client = get_llm_client(settings)

    snippet_chars = int(settings.guardrail_contradiction_snippet_chars)
    rendered = [f"<question>{question}</question>", ""]
    for index, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        rendered.append(
            wrap_untrusted(
                payload.text[:snippet_chars],
                label=f"candidate {index} (score {chunk.final_score:.3f})",
                marker=f"[{index}]",
                include_preamble=index == 1,
            )
        )
    try:
        return await client.classify(
            system=OOD_ADJUDICATION_SYSTEM,
            text="\n".join(rendered),
            labels=list(_OOD_LABELS),
            name="guardrail.ood",
            metadata=prompt_metadata("ood_adjudication"),
        )
    except Exception:
        _log.warning("ood_classifier_failed", exc_info=True)
        return None


async def run_ood_gate(
    *,
    question: str,
    result: RetrievalResult | Sequence[RetrievedChunk],
    principal: Principal,
    transformed: TransformedQuery | None = None,
    tool_available: bool = False,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    client: Any = None,
) -> OODVerdict:
    """Decide whether this turn can be answered from the corpus.

    Args:
        question: The rewritten, standalone question. Used for the optional
            adjudication call only.
        result: Stage 5's output, or its chunks.
        principal: The caller, for the ACL-scoped coverage sample.
        transformed: Stage 3's plan. Its ``is_out_of_domain`` flag counts as evidence
            only when the transform did not degrade.
        tool_available: Whether the tool loop has at least one tool this principal
            may call. A tool-servable question is routed, not refused.
        settings: Process settings.
        llm: LLM client override.
        client: Qdrant client override for the coverage sample.

    Returns:
        An :class:`OODVerdict`. When ``is_out_of_domain`` is set, ``refusal`` is a
        complete user-facing answer and the caller must not run generation.
    """
    resolved = settings or get_settings()
    chunks = _chunks_of(result)
    signals = relevance_signals(chunks, settings=resolved)
    verdict = OODVerdict(signals=signals)

    if not resolved.guardrail_ood_enabled:
        verdict.reason = "disabled"
        verdict.detail = "out-of-domain gate disabled by configuration"
        return verdict

    transformer_flag = bool(
        transformed is not None
        and transformed.is_out_of_domain
        and not transformed.degraded
    )
    wants_tool = bool(transformed is not None and transformed.needs_tools)

    decided = signals.weak
    reason = signals.reason if decided else "sufficient_evidence"
    confidence = 0.6 if decided else 0.4

    if transformer_flag and decided:
        confidence = 0.9
        reason = f"{reason}+transformer_flag"
    elif transformer_flag and not decided:
        # The transformer says out-of-domain but retrieval found something solid.
        # That disagreement is exactly the ambiguous case worth a cheap opinion.
        label = await _maybe_classify(
            question=question,
            chunks=chunks,
            settings=resolved,
            llm=llm,
        )
        verdict.classifier_label = label
        if label == "out_of_domain":
            decided, reason, confidence = True, "classifier", 0.7
        elif label == "needs_tool":
            decided, reason, confidence = True, "classifier_needs_tool", 0.7
            wants_tool = True
        elif label is None:
            decided, reason, confidence = False, "transformer_flag_overruled", 0.5
    elif decided and not transformer_flag and not signals.too_few_candidates:
        label = await _maybe_classify(
            question=question,
            chunks=chunks,
            settings=resolved,
            llm=llm,
        )
        verdict.classifier_label = label
        if label == "in_domain":
            decided, reason, confidence = False, "classifier_in_domain", 0.6
        elif label == "needs_tool":
            wants_tool = True
            reason = f"{reason}+classifier_needs_tool"
            confidence = 0.8
        elif label == "out_of_domain":
            confidence = 0.9

    if not decided:
        verdict.reason = reason
        verdict.detail = (
            f"evidence sufficient: max={signals.max_score:.3f} "
            f"mean={signals.mean_score:.3f} spread={signals.top_k_spread:.3f} "
            f"({signals.score_source})"
        )
        verdict.confidence = confidence
        verdict.events.append(
            GuardrailEvent(
                stage="retrieval",
                kind=GuardrailKind.OOD.value,
                action=GuardrailAction.ALLOW.value,
                detail=verdict.detail,
                score=signals.max_score,
            )
        )
        return verdict

    if tool_available and wants_tool:
        verdict.needs_tool = True
        verdict.reason = "tool_can_serve"
        verdict.confidence = confidence
        verdict.detail = (
            "corpus evidence is weak but a registered tool can serve this request; "
            "routing to the tool loop instead of refusing"
        )
        event = GuardrailEvent(
            stage="retrieval",
            kind=GuardrailKind.OOD.value,
            action=GuardrailAction.WARN.value,
            detail=verdict.detail,
            score=signals.max_score,
        )
        verdict.events.append(event)
        observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)
        return verdict

    coverage = await tenant_coverage(principal, settings=resolved, client=client)
    verdict.coverage = coverage
    verdict.is_out_of_domain = True
    verdict.needs_tool = wants_tool
    verdict.reason = reason
    verdict.confidence = confidence
    verdict.detail = (
        f"refused as out of domain ({reason}): candidates={signals.candidate_count} "
        f"max={signals.max_score:.3f} mean={signals.mean_score:.3f} "
        f"spread={signals.top_k_spread:.3f} collapsed={signals.collapsed}"
    )

    if resolved.guardrail_ood_llm_refusal:
        verdict.refusal = await _phrase_refusal(
            question=question,
            coverage=coverage,
            settings=resolved,
            llm=llm,
            needs_tool=wants_tool,
        )
    else:
        verdict.refusal = fallback_refusal(
            coverage, question=question, needs_tool=wants_tool, settings=resolved
        )

    event = GuardrailEvent(
        stage="retrieval",
        kind=GuardrailKind.OOD.value,
        action=GuardrailAction.BLOCK.value,
        detail=verdict.detail,
        entities=[item.value for item in coverage.doc_types[:6]],
        score=signals.max_score,
    )
    verdict.events.append(event)
    observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)
    _log.info(
        "ood_refusal",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        reason=reason,
        candidates=signals.candidate_count,
        max_score=signals.max_score,
        collapsed=signals.collapsed,
        classifier_label=verdict.classifier_label,
    )
    return verdict


async def _maybe_classify(
    *,
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
    llm: LLMClient | None,
) -> str | None:
    """Run the cheap adjudicator when it is enabled and there is anything to judge.

    Args:
        question: The rewritten question.
        chunks: Candidates; only the best few are sent.
        settings: Process settings.
        llm: Client override.

    Returns:
        A label, or None when the classifier is disabled or unavailable.
    """
    if not settings.guardrail_ood_classifier_enabled:
        return None
    if not chunks:
        return None
    top_k = max(1, int(settings.guardrail_ood_collapse_top_k))
    return await _adjudicate(
        question=question,
        chunks=list(chunks)[:top_k],
        settings=settings,
        llm=llm,
    )
