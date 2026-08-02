"""Query transformation — requirement #10, pipeline stage 3.

One structured ``MODEL_FAST`` call turns the user's latest message into a retrieval
plan: a context-resolved standalone question, an optional decomposition, an optional
HyDE probe, the facets the user actually stated, and the routing flags the rest of
the pipeline needs (``needs_retrieval``, ``needs_tools``, ``is_out_of_domain``).

Three properties this module is built around:

* **It can never break a chat turn.** Every failure mode — transformation disabled,
  a refusal, a malformed response, a timeout, an unreachable API — degrades to
  :func:`fallback_transform`, which retrieves on the raw user message. A degraded
  transform is a slightly worse search, not a failed request, so the whole call is
  wrapped and ``degraded`` is reported rather than raised.
* **The wire schema is flat.** :class:`QueryTransformPayload` uses only strings,
  booleans, numbers and string lists. Structured outputs reject much of JSON Schema
  (see ``LLMClient.structured``), and a nested ``MetadataFilter`` with datetimes and
  an enum is exactly the shape most likely to come back unparseable. The flat payload
  is validated and mapped onto :class:`~ragcore.models.retrieval.MetadataFilter`
  here, where a bad facet can be dropped instead of failing the call.
* **An extracted facet never overrides a user-supplied one.** See
  :func:`merge_filters`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ragcore.llm import get_llm_client
from ragcore.llm.client import LLMClient
from ragcore.llm.prompts import (
    QUERY_TRANSFORM_SYSTEM,
    prompt_metadata,
    render_history,
)
from ragcore.models.chat import Message
from ragcore.models.chunk import SourceType
from ragcore.models.retrieval import MetadataFilter
from ragcore.settings import Settings, get_settings

__all__ = [
    "QueryTransformPayload",
    "TransformedQuery",
    "fallback_transform",
    "is_abstract_query",
    "merge_filters",
    "transform_query",
]

_log = structlog.get_logger(__name__)

#: Trace/operation name for the single transform call.
TRANSFORM_CALL_NAME = "rag.query_transform"

#: Tokens that make a query concrete enough that BM25 will find the right document
#: on its own: digits, upper-case codes, quoted literals, dotted identifiers.
_CONCRETE_TOKEN = re.compile(r"\d|[A-Z]{2,}|[\"'`]|\w+[._/-]\w+")

_ISO_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y-%m",
    "%Y",
)


class QueryTransformPayload(BaseModel):
    """Flat wire schema for the single structured transform call.

    Deliberately primitive: structured outputs strip most JSON Schema keywords, so a
    nested model carrying a ``datetime`` and a ``StrEnum`` is the shape most likely
    to come back unusable. Facets arrive as plain strings and are validated into a
    :class:`~ragcore.models.retrieval.MetadataFilter` by :func:`transform_query`.
    """

    model_config = ConfigDict(extra="ignore")

    intent: str = Field(default="", description="Short noun phrase naming the need.")
    rewritten: str = Field(
        default="", description="Standalone, context-resolved question."
    )
    sub_questions: list[str] = Field(
        default_factory=list, description="Independent retrievable sub-questions."
    )
    needs_retrieval: bool = Field(
        default=True, description="False only for chit-chat or meta questions."
    )
    needs_tools: bool = Field(
        default=False, description="True when live or transactional data is required."
    )
    tool_hints: list[str] = Field(
        default_factory=list, description="Kinds of tool required, not tool ids."
    )
    hyde_passage: str = Field(
        default="", description="Hypothetical answering passage, or empty."
    )
    is_out_of_domain: bool = Field(
        default=False, description="True when no enterprise corpus could answer this."
    )
    confidence: float = Field(
        default=0.5, description="Transformer's own confidence in this plan, 0..1."
    )
    filter_doc_types: list[str] = Field(
        default_factory=list, description="Stated document types."
    )
    filter_source_types: list[str] = Field(
        default_factory=list, description="Stated source systems."
    )
    filter_tags: list[str] = Field(default_factory=list, description="Stated tags.")
    filter_authors: list[str] = Field(
        default_factory=list, description="Stated authors."
    )
    filter_languages: list[str] = Field(
        default_factory=list, description="Stated ISO 639-1 language codes."
    )
    filter_section: str = Field(
        default="", description="Stated section or heading to restrict to."
    )
    filter_date_from: str = Field(
        default="", description="Inclusive lower bound, ISO 8601 date or empty."
    )
    filter_date_to: str = Field(
        default="", description="Inclusive upper bound, ISO 8601 date or empty."
    )


class TransformedQuery(BaseModel):
    """The retrieval plan stage 3 hands to stages 4, 5, 6 and 8.

    ``metadata_filter`` holds **only** the facets the transformer extracted. It is
    never pre-merged with the caller's filter — see :func:`merge_filters` for why
    intersecting the two is unsafe — so the orchestrator merges explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str = Field(default="", description="Short noun phrase naming the need.")
    needs_retrieval: bool = Field(
        default=True, description="Whether stage 5 should run at all."
    )
    needs_tools: bool = Field(
        default=False, description="Whether the tool loop is likely to be required."
    )
    tool_hints: list[str] = Field(
        default_factory=list, description="Kinds of tool the transformer expects."
    )
    rewritten: str = Field(
        description="Standalone question with pronouns and ellipsis resolved."
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Decomposition, bounded by settings.qt_max_subqueries.",
    )
    hyde_passage: str = Field(
        default="",
        description=(
            "Hypothetical passage used as an extra dense probe. Empty unless "
            "qt_hyde_enabled and the query is abstract enough to need one."
        ),
    )
    metadata_filter: MetadataFilter | None = Field(
        default=None, description="Facets extracted from the query, or None."
    )
    is_out_of_domain: bool = Field(
        default=False, description="Transformer's out-of-domain flag for stage 6."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in this plan."
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True when the plan is the raw-query fallback rather than a model "
            "result. Stage 6 should not treat a degraded plan's flags as evidence."
        ),
    )
    degraded_reason: str = Field(
        default="",
        description=(
            "Machine-readable cause: 'disabled', 'empty_query', 'llm_error', "
            "'invalid_response'. Empty when the transform succeeded."
        ),
    )

    @property
    def queries(self) -> list[str]:
        """Every distinct query text stage 5 should search for.

        Returns:
            The rewritten question first, then any sub-questions, de-duplicated
            case-insensitively so a sub-question that merely restates the main one
            does not double the retrieval cost.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for text in (self.rewritten, *self.sub_questions):
            candidate = text.strip()
            if not candidate:
                continue
            marker = candidate.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            ordered.append(candidate)
        return ordered


def fallback_transform(
    message: str,
    *,
    reason: str,
    settings: Settings | None = None,
) -> TransformedQuery:
    """Build the safe plan used whenever transformation cannot be trusted.

    Retrieving on the user's literal words is a worse search than a resolved one, but
    it is a working search. The alternative — propagating the failure — turns a
    degraded model call into a failed chat turn, which requirement #10 explicitly
    rules out.

    Args:
        message: The raw user message.
        reason: Machine-readable cause recorded in
            :attr:`TransformedQuery.degraded_reason`.
        settings: Unused today; accepted so the signature stays stable if the
            fallback ever needs a tunable.

    Returns:
        A plan that retrieves on the raw message, extracts no facets, and claims no
        confidence. ``needs_retrieval`` stays True: with no model opinion available,
        searching and finding nothing is recoverable, while skipping retrieval and
        answering from parametric knowledge is not.
    """
    del settings
    text = message.strip()
    return TransformedQuery(
        intent="",
        needs_retrieval=bool(text),
        needs_tools=False,
        tool_hints=[],
        rewritten=text,
        sub_questions=[],
        hyde_passage="",
        metadata_filter=None,
        is_out_of_domain=False,
        confidence=0.0,
        degraded=True,
        degraded_reason=reason,
    )


def is_abstract_query(text: str, *, settings: Settings | None = None) -> bool:
    """Decide whether a query is sparse or abstract enough to justify HyDE.

    HyDE helps exactly when lexical matching cannot: a short, jargon-light question
    shares almost no rare terms with the passage that answers it, so BM25 contributes
    nothing and the dense probe is doing all the work. When the query already carries
    concrete anchors — a number, an upper-case code, a quoted literal, a dotted
    identifier — BM25 will find the document and a hallucinated passage only adds
    drift and a second embedding.

    Args:
        text: The query, normally the rewritten standalone question.
        settings: Settings supplying ``qt_hyde_max_query_words``. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        True when the query is short or carries no concrete anchor.
    """
    cfg = settings or get_settings()
    max_words = int(cfg.qt_hyde_max_query_words)
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped.split()) <= max_words:
        return True
    return _CONCRETE_TOKEN.search(stripped) is None


def merge_filters(
    base: MetadataFilter | None, extracted: MetadataFilter | None
) -> MetadataFilter | None:
    """Combine a caller-supplied filter with transformer-extracted facets.

    **Fill-in, not intersection.** ``MetadataFilter.merged_with`` intersects list
    facets, and its validator collapses an empty list to None — so a user asking for
    ``doc_types=["standard"]`` while the transformer guesses ``["policy"]`` would
    intersect to ``[]`` and normalise to *no constraint at all*, silently widening
    the search the user narrowed. That is the wrong failure direction for a filter,
    so a facet the user set is never touched: extracted facets only fill facets the
    user left unset.

    Classification and ``exclude_pii`` are still combined strictly (the stricter side
    wins), because those narrow rather than redirect. Date bounds take the narrower
    of the two unless doing so would invert the range, in which case the caller's
    range stands.

    Args:
        base: The filter the caller supplied, or None.
        extracted: The filter the transformer produced, or None.

    Returns:
        The merged filter, or None when neither side constrains anything.
    """
    if extracted is None or extracted.is_empty:
        return base
    if base is None or base.is_empty:
        return extracted

    def _fill(left: list[str] | None, right: list[str] | None) -> list[str] | None:
        return left if left else right

    date_from = base.date_from
    date_to = base.date_to
    candidate_from = max(
        [d for d in (base.date_from, extracted.date_from) if d is not None],
        default=None,
    )
    candidate_to = min(
        [d for d in (base.date_to, extracted.date_to) if d is not None],
        default=None,
    )
    if candidate_from is None or candidate_to is None or candidate_from <= candidate_to:
        date_from, date_to = candidate_from, candidate_to

    classifications = [
        level
        for level in (base.max_classification, extracted.max_classification)
        if level is not None
    ]
    return MetadataFilter(
        doc_types=_fill(base.doc_types, extracted.doc_types),
        source_types=_fill(base.source_types, extracted.source_types),
        tags=_fill(base.tags, extracted.tags),
        authors=_fill(base.authors, extracted.authors),
        languages=_fill(base.languages, extracted.languages),
        document_ids=_fill(base.document_ids, extracted.document_ids),
        section_prefix=base.section_prefix or extracted.section_prefix,
        date_from=date_from,
        date_to=date_to,
        max_classification=min(classifications) if classifications else None,
        exclude_pii=base.exclude_pii or extracted.exclude_pii,
    )


def _history_turns(history: Sequence[Any], *, limit: int) -> list[tuple[str, str]]:
    """Normalise the short-term window into ``(role, content)`` pairs.

    Args:
        history: Recent turns as :class:`~ragcore.models.chat.Message` objects,
            ``(role, content)`` tuples, or mappings with ``role``/``content`` keys.
            Oldest first.
        limit: Keep at most this many of the most recent turns
            (``settings.qt_history_turns``).

    Returns:
        The most recent turns as ``(role, content)`` pairs, oldest first. Suppressed
        messages are kept: they are exactly the older context a pronoun may point at,
        and this window is sent to the model rather than persisted.
    """
    if limit <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    for item in history:
        role: str
        content: str
        if isinstance(item, Message):
            role, content = str(item.role), item.content
        elif isinstance(item, Mapping):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
        elif isinstance(item, tuple | list) and len(item) >= 2:
            role, content = str(item[0]), str(item[1])
        else:
            continue
        text = content.strip()
        if text:
            pairs.append((role, text))
    return pairs[-limit:]


def _vocabulary(settings: Settings) -> tuple[list[str], list[str]]:
    """Enumerate the facet vocabularies the transformer may choose from.

    Giving the model the closed sets keeps ``doc_type`` labels identical to the ones
    ingestion assigns (``guardrail_doc_type_authority`` drives both the classifier in
    ``ingestion.enrich`` and the contradiction resolver), so a filter can actually
    match something.

    Args:
        settings: Settings supplying the doc-type authority table.

    Returns:
        A ``(doc_types, source_types)`` pair of sorted labels.
    """
    doc_types = sorted({*settings.guardrail_doc_type_authority, "document"})
    source_types = sorted(member.value for member in SourceType)
    return doc_types, source_types


def _build_user_turn(
    *,
    message: str,
    history: Sequence[tuple[str, str]],
    settings: Settings,
    now: datetime,
) -> str:
    """Assemble the volatile half of the transform prompt.

    Everything here changes per request, so none of it may live in
    :data:`~ragcore.llm.prompts.QUERY_TRANSFORM_SYSTEM` — the system block is the
    cached prefix and must stay byte-stable.

    Args:
        message: The user's latest message, already clipped.
        history: Recent ``(role, content)`` turns for pronoun resolution.
        settings: Settings supplying the vocabularies and sub-question bound.
        now: Reference date for resolving relative expressions such as "last year".

    Returns:
        The user-turn text.
    """
    doc_types, source_types = _vocabulary(settings)
    blocks = [
        render_history(history),
        f"<reference_date>{now.date().isoformat()}</reference_date>",
        (
            "<vocabulary>\n"
            f"doc_type: {', '.join(doc_types)}\n"
            f"source_type: {', '.join(source_types)}\n"
            "</vocabulary>"
        ),
        (
            "<limits>\n"
            f"sub_questions: at most {settings.qt_max_subqueries}\n"
            f"hyde_passage: at most {settings.qt_hyde_max_chars} characters; "
            f"{'allowed' if settings.qt_hyde_enabled else 'must be empty'}\n"
            "</limits>"
        ),
        f"<message>\n{message}\n</message>",
    ]
    return "\n\n".join(blocks)


def _clean_list(values: Sequence[str], *, allowed: set[str] | None = None) -> list[str]:
    """Strip, de-duplicate and optionally vocabulary-check a facet list.

    Args:
        values: Raw strings from the model.
        allowed: Closed vocabulary the values must belong to (case-insensitively),
            or None to accept any non-empty string.

    Returns:
        The cleaned values, order preserved. A value outside ``allowed`` is dropped:
        a filter nothing can match hides the answer, and the prompt is explicit that
        no filter beats a guessed one.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        if allowed is not None:
            if marker not in allowed:
                continue
            text = marker
        seen.add(marker)
        cleaned.append(text)
    return cleaned


def _parse_date(value: str, *, end_of_range: bool) -> datetime | None:
    """Parse a model-supplied date bound.

    Args:
        value: An ISO 8601 date, a ``YYYY-MM`` or ``YYYY`` prefix, or an empty
            string.
        end_of_range: When True a bare year or month resolves to the *last* instant
            it covers, so "documents from 2024" includes December.

    Returns:
        A timezone-aware UTC datetime, or None when the value is empty or
        unparseable. Unparseable is treated as "no bound" rather than as an error:
        a guessed date range silently hides documents.
    """
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass
    for fmt in _ISO_DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
        if not end_of_range:
            return parsed
        if fmt == "%Y":
            return datetime(parsed.year, 12, 31, 23, 59, 59, tzinfo=UTC)
        if fmt == "%Y-%m":
            last = date(parsed.year + parsed.month // 12, parsed.month % 12 + 1, 1)
            return datetime(last.year, last.month, last.day, 0, 0, 0, tzinfo=UTC)
        return parsed.replace(hour=23, minute=59, second=59)
    return None


def _filter_from_payload(
    payload: QueryTransformPayload, *, settings: Settings
) -> MetadataFilter | None:
    """Map the flat facet fields onto a :class:`MetadataFilter`.

    Args:
        payload: The validated model response.
        settings: Settings supplying the closed vocabularies.

    Returns:
        The extracted filter, or None when the transformer extracted nothing usable.
        A filter that fails validation (an inverted date range, say) is discarded
        rather than raised: an over-eager facet must never cost the user their turn.
    """
    doc_types, source_types = _vocabulary(settings)
    date_from = _parse_date(payload.filter_date_from, end_of_range=False)
    date_to = _parse_date(payload.filter_date_to, end_of_range=True)
    if date_from is not None and date_to is not None and date_from > date_to:
        date_from, date_to = None, None
    try:
        extracted = MetadataFilter(
            doc_types=_clean_list(
                payload.filter_doc_types, allowed={t.casefold() for t in doc_types}
            )
            or None,
            source_types=_clean_list(
                payload.filter_source_types,
                allowed={t.casefold() for t in source_types},
            )
            or None,
            tags=_clean_list(payload.filter_tags) or None,
            authors=_clean_list(payload.filter_authors) or None,
            languages=_clean_list(payload.filter_languages) or None,
            section_prefix=payload.filter_section.strip() or None,
            date_from=date_from,
            date_to=date_to,
        )
    except ValueError:
        _log.warning("query_transform.filter_rejected")
        return None
    return None if extracted.is_empty else extracted


def _plan_from_payload(
    payload: QueryTransformPayload,
    *,
    message: str,
    settings: Settings,
) -> TransformedQuery:
    """Validate, bound and sanitise a model response into a retrieval plan.

    Args:
        payload: The validated model response.
        message: The raw user message, used whenever the model returned nothing
            usable for ``rewritten``.
        settings: Settings supplying the bounds and feature switches.

    Returns:
        The sanitised plan.
    """
    rewritten = payload.rewritten.strip() or message.strip()
    if not settings.qt_rewrite_enabled:
        rewritten = message.strip()

    sub_questions: list[str] = []
    seen = {rewritten.casefold()}
    for candidate in payload.sub_questions:
        text = candidate.strip()
        marker = text.casefold()
        if not text or marker in seen:
            continue
        seen.add(marker)
        sub_questions.append(text)
        if len(sub_questions) >= settings.qt_max_subqueries:
            break

    hyde = payload.hyde_passage.strip()
    if not settings.qt_hyde_enabled or not is_abstract_query(
        rewritten, settings=settings
    ):
        hyde = ""
    hyde = hyde[: settings.qt_hyde_max_chars]

    return TransformedQuery(
        intent=payload.intent.strip()[:200],
        needs_retrieval=payload.needs_retrieval,
        needs_tools=payload.needs_tools,
        tool_hints=_clean_list(payload.tool_hints)[: settings.tool_max_iterations],
        rewritten=rewritten,
        sub_questions=sub_questions,
        hyde_passage=hyde,
        metadata_filter=_filter_from_payload(payload, settings=settings),
        is_out_of_domain=payload.is_out_of_domain,
        confidence=min(1.0, max(0.0, payload.confidence)),
        degraded=False,
        degraded_reason="",
    )


async def transform_query(
    message: str,
    *,
    history: Sequence[Any] = (),
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    now: datetime | None = None,
) -> TransformedQuery:
    """Turn one user message into a retrieval plan (pipeline stage 3).

    Exactly one ``MODEL_FAST`` structured call is made. Pronoun and ellipsis
    resolution happens inside it, against the short-term window supplied in
    ``history`` — the model sees the recent turns and returns a standalone question,
    so "and what about contractors?" becomes a query that retrieves.

    Args:
        message: The user's latest message, already through the stage 1 input guard.
        history: The short-term window, oldest first. Accepts
            :class:`~ragcore.models.chat.Message` objects, ``(role, content)`` pairs
            or mappings. Trimmed to ``settings.qt_history_turns``.
        settings: Settings supplying the ``qt_*`` bounds. Defaults to
            :func:`ragcore.settings.get_settings`.
        llm: Client to call. Defaults to
            :func:`ragcore.llm.get_llm_client`. Injectable for tests.
        now: Reference moment for relative date expressions. Defaults to now (UTC).

    Returns:
        A :class:`TransformedQuery`. **Never raises**: any failure returns
        :func:`fallback_transform` with ``degraded=True`` and a machine-readable
        ``degraded_reason``, because a worse search is always preferable to a lost
        turn.
    """
    cfg = settings or get_settings()

    text = message.strip()
    if not text:
        return fallback_transform(message, reason="empty_query", settings=cfg)
    if not cfg.qt_enabled:
        return fallback_transform(message, reason="disabled", settings=cfg)

    clipped = text[: cfg.qt_max_query_chars]
    turns = _history_turns(history, limit=cfg.qt_history_turns)
    user_turn = _build_user_turn(
        message=clipped,
        history=turns,
        settings=cfg,
        now=now or datetime.now(UTC),
    )
    client = llm or get_llm_client(cfg)

    try:
        payload = await client.structured(
            system=QUERY_TRANSFORM_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
            schema=QueryTransformPayload,
            name=TRANSFORM_CALL_NAME,
            metadata=prompt_metadata("query_transform"),
        )
    except Exception as exc:  # stage 3 must never fail a turn
        # Deliberately broad. The failure surface spans anthropic HTTP errors, a
        # refusal (LLMRefusedError), a schema violation (pydantic ValidationError),
        # unparseable JSON (ValueError) and cancellation-adjacent runtime faults.
        # Enumerating them here would be a list that silently rots; the contract is
        # simply that transformation cannot break a chat turn.
        _log.warning(
            "query_transform.failed",
            error=type(exc).__name__,
            query_chars=len(clipped),
            history_turns=len(turns),
        )
        return fallback_transform(message, reason="llm_error", settings=cfg)

    try:
        plan = _plan_from_payload(payload, message=clipped, settings=cfg)
    except Exception as exc:
        # As broad as the call above, and for the same reason: a response that is
        # the wrong *shape* (a bare dict, a missing field, a string where a list
        # belongs) raises AttributeError or TypeError rather than ValueError, and
        # none of those may cost the user their turn.
        _log.warning("query_transform.invalid_plan", error=type(exc).__name__)
        return fallback_transform(message, reason="invalid_response", settings=cfg)

    _log.debug(
        "query_transform.ok",
        query_chars=len(clipped),
        rewritten_chars=len(plan.rewritten),
        sub_questions=len(plan.sub_questions),
        hyde=bool(plan.hyde_passage),
        needs_retrieval=plan.needs_retrieval,
        needs_tools=plan.needs_tools,
        out_of_domain=plan.is_out_of_domain,
        has_filter=plan.metadata_filter is not None,
    )
    return plan
