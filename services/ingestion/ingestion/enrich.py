"""Document-level enrichment: summary, keywords, doc_type, language, PII, validity.

Everything here is computed once per *document*, not per chunk, because the outputs
are document properties and paying an LLM call per chunk would dominate the cost of a
nightly refresh. Results are cached by content hash, so a re-run over unchanged
sources performs no model calls at all.

Order matters: **PII redaction runs before the LLM sees the text.** Ingested corpora
routinely contain personal data, and the platform's rule is that unredacted content is
never logged, persisted or shipped off-box. The summary and keywords are therefore
derived from the redacted text, and ``pii_types`` / ``pii_redacted`` travel onto every
chunk payload so retrieval can filter on them.

Degradation is deliberate and total: if ``ragcore.llm`` cannot be imported or no API
key is configured, enrichment falls back to a lead-sentence summary, frequency-based
keywords and the source's declared ``doc_type``, logs once, and keeps going. Ingestion
never fails because an optional service is down.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragcore.logging import get_logger
from ragcore.models.document import ParsedBlock, ParsedDocument
from ragcore.settings import Settings, get_settings

__all__ = [
    "ENRICH_PROMPT_VERSION",
    "DocumentInsights",
    "EnrichmentCache",
    "detect_language",
    "enrich_document",
    "enrich_documents",
    "extract_effective_dates",
    "get_enrichment_cache",
    "heuristic_insights",
    "scan_and_redact",
]

_log = get_logger(__name__)

#: Bump when the enrichment prompt changes, so cached and traced results stay
#: attributable to the prompt that produced them.
ENRICH_PROMPT_VERSION = "ingest-enrich-v1"

ENRICH_SYSTEM_PROMPT = (
    "You summarise enterprise documents for a retrieval index.\n"
    "Given a document's title and text, produce:\n"
    "- summary: one or two sentences, factual, no preamble, under 320 characters.\n"
    "- keywords: 5-12 lowercase noun phrases a colleague would search for.\n"
    "- doc_type: the single best label from the allowed list.\n"
    "Use only what the text states. Never speculate about missing content. "
    "Placeholders such as <EMAIL_ADDRESS> are redactions; do not comment on them."
)

DOC_TYPE_SYSTEM_PROMPT = (
    "Classify the document type of an enterprise document from its title and "
    "opening text. Answer with exactly one label from the provided list."
)

#: Characters of document text sent to the model. Enrichment only needs the opening
#: of a document to describe it, and a bounded prefix keeps the prompt cacheable.
_ENRICH_PREFIX_CHARS = 6_000

#: Words ignored when deriving fallback keywords.
_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "who",
        "whom",
        "whose",
        "you",
        "your",
        "we",
        "our",
        "not",
        "no",
        "can",
        "may",
        "shall",
        "should",
        "must",
        "also",
        "than",
        "then",
        "when",
        "where",
        "while",
    ]
)

#: Unicode script probes, checked before Latin stopword scoring.
_SCRIPT_RANGES: tuple[tuple[str, str, str], ...] = (
    ("hi", "\u0900", "\u097f"),
    ("ru", "\u0400", "\u04ff"),
    ("el", "\u0370", "\u03ff"),
    ("he", "\u0590", "\u05ff"),
    ("ar", "\u0600", "\u06ff"),
    ("ja", "\u3040", "\u30ff"),
    ("zh", "\u4e00", "\u9fff"),
    ("ko", "\uac00", "\ud7af"),
)

#: Marker words per Latin-script language, used for a cheap frequency vote.
_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "and",
            "of",
            "to",
            "in",
            "that",
            "is",
            "for",
            "with",
            "are",
            "as",
            "be",
            "this",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "nicht",
            "mit",
            "von",
            "für",
            "sich",
            "auch",
            "werden",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "des",
            "et",
            "est",
            "pour",
            "dans",
            "que",
            "une",
            "avec",
            "sur",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "de",
            "que",
            "es",
            "para",
            "con",
            "una",
            "por",
            "no",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "lo",
            "la",
            "di",
            "che",
            "è",
            "per",
            "con",
            "una",
            "del",
            "non",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "a",
            "os",
            "as",
            "de",
            "que",
            "é",
            "para",
            "com",
            "uma",
            "do",
            "não",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "en",
            "van",
            "is",
            "niet",
            "met",
            "voor",
            "ook",
            "worden",
        ]
    ),
}

#: Phrases that introduce a validity start date.
_EFFECTIVE_FROM_RE = re.compile(
    r"(?:effective(?:\s+(?:from|as\s+of|date))?|valid\s+from|in\s+force\s+from|"
    r"with\s+effect\s+from|applies\s+from)\s*[:\-\u2013]?\s*"
    r"([0-9]{1,2}\s+\w+\s+[0-9]{4}|\w+\s+[0-9]{1,2},?\s+[0-9]{4}|"
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/.][0-9]{1,2}[/.][0-9]{2,4})",
    re.IGNORECASE,
)

#: Phrases that introduce a validity end date.
_EFFECTIVE_TO_RE = re.compile(
    r"(?:valid\s+until|expires(?:\s+on)?|until|superseded\s+on|review\s+by)\s*"
    r"[:\-\u2013]?\s*"
    r"([0-9]{1,2}\s+\w+\s+[0-9]{4}|\w+\s+[0-9]{1,2},?\s+[0-9]{4}|"
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/.][0-9]{1,2}[/.][0-9]{2,4})",
    re.IGNORECASE,
)


class DocumentInsights(BaseModel):
    """Structured output of the enrichment call.

    Attributes:
        summary: One or two factual sentences describing the document.
        keywords: Lowercase noun phrases for lexical recall.
        doc_type: Best-fitting document class from the configured label set.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(default="", description="One or two sentence summary.")
    keywords: list[str] = Field(
        default_factory=list, description="Lowercase keyword phrases."
    )
    doc_type: str = Field(default="document", description="Document class label.")


class EnrichmentCache:
    """Process-lifetime cache of insights keyed by document content hash.

    A nightly refresh re-reads documents whose ETag moved but whose bytes did not;
    without this cache those documents would pay for an LLM call to produce a summary
    identical to the one already stored.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        """Create an empty cache.

        Args:
            max_entries: Soft ceiling; the oldest half is dropped when exceeded.
        """
        self._entries: dict[str, DocumentInsights] = {}
        self._max_entries = max_entries

    def get(self, content_sha256: str) -> DocumentInsights | None:
        """Look up cached insights.

        Args:
            content_sha256: Hash of the document payload.

        Returns:
            The cached insights, or None on a miss.
        """
        return self._entries.get(content_sha256) if content_sha256 else None

    def put(self, content_sha256: str, insights: DocumentInsights) -> None:
        """Store insights for a content hash.

        Args:
            content_sha256: Hash of the document payload.
            insights: Insights to remember.
        """
        if not content_sha256:
            return
        if len(self._entries) >= self._max_entries:
            for key in list(self._entries)[: self._max_entries // 2]:
                self._entries.pop(key, None)
        self._entries[content_sha256] = insights

    def clear(self) -> None:
        """Drop every entry."""
        self._entries.clear()


_CACHE = EnrichmentCache()


def get_enrichment_cache() -> EnrichmentCache:
    """Return the process-wide enrichment cache.

    Returns:
        The shared :class:`EnrichmentCache`.
    """
    return _CACHE


# ------------------------------------------------------------------- language
def detect_language(text: str, *, default: str = "en") -> str:
    """Detect a document's language without an external dependency.

    Non-Latin scripts are identified by codepoint range; Latin-script languages by a
    marker-word vote. This is deliberately cheap — language is a filter facet, not a
    correctness-critical value, and a wrong guess degrades recall rather than leaking
    data.

    Args:
        text: Document text.
        default: Language returned when detection is inconclusive.

    Returns:
        An ISO 639-1 code.
    """
    sample = text[:4000]
    if not sample.strip():
        return default
    for code, low, high in _SCRIPT_RANGES:
        hits = sum(1 for char in sample if low <= char <= high)
        if hits >= max(8, len(sample) // 50):
            return code

    words = Counter(re.findall(r"[a-zà-öø-ÿ]+", sample.lower()))
    if not words:
        return default
    scores = {
        code: sum(words[marker] for marker in markers)
        for code, markers in _LANGUAGE_MARKERS.items()
    }
    best = max(scores, key=lambda code: scores[code])
    return best if scores[best] > 0 else default


# -------------------------------------------------------------- effective dates
def extract_effective_dates(
    text: str, *, now: datetime | None = None
) -> tuple[datetime | None, datetime | None]:
    """Extract a document's validity window from its own wording.

    Recency is how contradictions get resolved (pipeline stage 7), so an explicit
    "effective from 1 April 2026" in the text is far more trustworthy than a file's
    modification time.

    Args:
        text: Document text; only the opening is inspected, where such statements
            live.
        now: Reference moment used to reject implausible parses. Defaults to now.

    Returns:
        An ``(effective_from, effective_to)`` pair; either may be None.
    """
    prefix = text[:_ENRICH_PREFIX_CHARS]
    reference = now or datetime.now(UTC)
    start = _first_date(_EFFECTIVE_FROM_RE, prefix, reference)
    end = _first_date(_EFFECTIVE_TO_RE, prefix, reference)
    if start and end and end <= start:
        return start, None
    return start, end


def _first_date(
    pattern: re.Pattern[str], text: str, reference: datetime
) -> datetime | None:
    """Parse the first date matched by a pattern.

    Args:
        pattern: Compiled pattern whose first group is the date text.
        text: Text to search.
        reference: Reference moment for plausibility bounds.

    Returns:
        An aware UTC datetime, or None when nothing plausible was found.
    """
    for match in pattern.finditer(text):
        parsed = _parse_date(match.group(1))
        if parsed is None:
            continue
        if abs((parsed - reference).days) > 365 * 50:
            continue
        return parsed
    return None


def _parse_date(value: str) -> datetime | None:
    """Parse a human-written date.

    Args:
        value: Date text such as ``"1 April 2026"`` or ``"2026-04-01"``.

    Returns:
        An aware UTC datetime, or None when it cannot be parsed.
    """
    cleaned = value.strip().rstrip(".,;")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None
    if parsed is None:
        try:
            from dateutil import parser as date_parser
        except ImportError:  # pragma: no cover - dateutil is a hard dependency
            return None
        try:
            parsed = date_parser.parse(cleaned, dayfirst=True, fuzzy=False)
        except (ValueError, OverflowError):
            return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ------------------------------------------------------------------------- pii
def scan_and_redact(
    parsed: ParsedDocument, settings: Settings, detector: Any | None = None
) -> tuple[ParsedDocument, list[str], bool]:
    """Scan every block for PII and optionally redact it in place.

    Args:
        parsed: The parsed document.
        settings: Process settings; ``pii_enabled`` and ``pii_redaction_mode`` apply.
        detector: A :class:`ragcore.pii.PIIDetector`. Resolved from
            ``ragcore.pii.get_pii_detector`` when omitted.

    Returns:
        A ``(document, pii_types, redacted)`` triple. When redaction is on, the
        returned document's block text is already redacted, so nothing downstream —
        embeddings, payloads, logs, lineage — ever sees the raw values.
    """
    if not settings.pii_enabled:
        return parsed, [], False
    engine = detector if detector is not None else _get_pii_detector(settings)
    if engine is None:
        return parsed, [], False

    entity_types: list[str] = []
    blocks: list[ParsedBlock] = []
    changed = False
    for block in parsed.blocks:
        report = engine.analyze(block.text, language=parsed.language or "en")
        if not report.has_pii:
            blocks.append(block)
            continue
        entity_types.extend(report.entity_types)
        redacted = engine.redact(block.text, report, mode=settings.pii_redaction_mode)
        if redacted != block.text:
            changed = True
            blocks.append(block.model_copy(update={"text": redacted}))
        else:
            blocks.append(block)

    ordered_types = sorted(set(entity_types))
    if not ordered_types:
        return parsed, [], False
    document = parsed.model_copy(update={"blocks": blocks}) if changed else parsed
    _log.info(
        "enrich.pii_detected",
        document_id=parsed.document_id,
        tenant_id=parsed.tenant_id,
        entity_types=ordered_types,
        redacted=changed,
    )
    return document, ordered_types, changed


def _get_pii_detector(settings: Settings) -> Any | None:
    """Resolve the PII detector, tolerating an absent module.

    Args:
        settings: Process settings.

    Returns:
        A detector, or None when ``ragcore.pii`` cannot be imported.
    """
    try:
        from ragcore.pii import get_pii_detector
    except ImportError:
        _log.warning("enrich.pii_unavailable")
        return None
    return get_pii_detector(settings)


# ------------------------------------------------------------------- insights
def heuristic_insights(parsed: ParsedDocument, settings: Settings) -> DocumentInsights:
    """Derive insights without a model call.

    Args:
        parsed: The parsed document.
        settings: Process settings; the ``doc_type`` label set comes from
            ``guardrail_doc_type_authority``.

    Returns:
        A lead-sentence summary, frequency-ranked keywords and a title-matched
        ``doc_type`` (falling back to the source's declared type).
    """
    text = parsed.full_text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:2]).strip()[:320]

    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
        if word not in _STOPWORDS
    ]
    keywords = [word for word, _ in Counter(words).most_common(10)]

    tag_text = " ".join(parsed.tags)
    haystack = f"{parsed.title} {tag_text}".lower()
    doc_type = parsed.doc_type
    for label in allowed_doc_types(settings):
        if label != "document" and label in haystack:
            doc_type = label
            break
    return DocumentInsights(summary=summary, keywords=keywords, doc_type=doc_type)


def allowed_doc_types(settings: Settings) -> list[str]:
    """Label set the classifier may choose from.

    Args:
        settings: Process settings.

    Returns:
        The ``guardrail_doc_type_authority`` keys plus ``"document"``, so the
        classifier's labels and the contradiction resolver's authority table can
        never drift apart.
    """
    labels = sorted(settings.guardrail_doc_type_authority)
    return [*labels, "document"] if "document" not in labels else labels


async def enrich_document(
    parsed: ParsedDocument,
    *,
    settings: Settings | None = None,
    llm: Any | None = None,
    detector: Any | None = None,
    cache: EnrichmentCache | None = None,
) -> ParsedDocument:
    """Enrich one document: PII, language, validity window, summary and keywords.

    Args:
        parsed: The parsed document.
        settings: Process settings; ``get_settings()`` is used when omitted.
        llm: A :class:`ragcore.llm.LLMClient`. Resolved lazily when omitted; when it
            cannot be resolved, heuristics are used.
        detector: A PII detector; resolved lazily when omitted.
        cache: Insight cache; the process-wide cache is used when omitted.

    Returns:
        A new :class:`ParsedDocument` with ``summary``, ``keywords``, ``doc_type``,
        ``language``, ``effective_from``/``effective_to`` and the PII metadata filled
        in. ``metadata["pii_types"]`` and ``metadata["pii_redacted"]`` carry the PII
        outcome to the chunk payloads.
    """
    active = settings or get_settings()
    store = cache or _CACHE

    redacted, pii_types, was_redacted = scan_and_redact(parsed, active, detector)
    text = redacted.full_text
    language = detect_language(text, default=redacted.language or "en")
    effective_from, effective_to = extract_effective_dates(text)

    key = redacted.content_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
    insights = store.get(key)
    if insights is None:
        insights = await _insights_for(redacted, active, llm)
        store.put(key, insights)

    updates: dict[str, Any] = {
        "language": language,
        "summary": insights.summary or redacted.summary,
        "keywords": _clean_keywords(insights.keywords),
        "doc_type": insights.doc_type or redacted.doc_type,
        "metadata": {
            **redacted.metadata,
            "pii_types": pii_types,
            "pii_redacted": was_redacted,
            "enrich_prompt_version": ENRICH_PROMPT_VERSION,
        },
    }
    if effective_from is not None and redacted.effective_from is None:
        updates["effective_from"] = effective_from
    if effective_to is not None and redacted.effective_to is None:
        updates["effective_to"] = effective_to
    return redacted.model_copy(update=updates)


async def enrich_documents(
    documents: Sequence[ParsedDocument],
    *,
    settings: Settings | None = None,
    llm: Any | None = None,
    detector: Any | None = None,
    cache: EnrichmentCache | None = None,
) -> list[ParsedDocument]:
    """Enrich a batch of documents with bounded concurrency.

    Documents sharing a content hash resolve against the same cache entry, so a
    corpus with duplicated files pays for one enrichment.

    Args:
        documents: Parsed documents to enrich.
        settings: Process settings; ``ingest_max_parallel_docs`` bounds concurrency.
        llm: Shared LLM client.
        detector: Shared PII detector.
        cache: Shared insight cache.

    Returns:
        The enriched documents, in input order.
    """
    active = settings or get_settings()
    semaphore = asyncio.Semaphore(active.ingest_max_parallel_docs)

    async def one(document: ParsedDocument) -> ParsedDocument:
        async with semaphore:
            return await enrich_document(
                document,
                settings=active,
                llm=llm,
                detector=detector,
                cache=cache,
            )

    return list(await asyncio.gather(*(one(document) for document in documents)))


async def _insights_for(
    parsed: ParsedDocument, settings: Settings, llm: Any | None
) -> DocumentInsights:
    """Produce insights for one document, preferring the LLM.

    Args:
        parsed: The (already redacted) parsed document.
        settings: Process settings.
        llm: LLM client, or None to resolve one lazily.

    Returns:
        Model-derived insights, or heuristics when the model is unavailable or
        refuses.
    """
    client = llm if llm is not None else _get_llm(settings)
    fallback = heuristic_insights(parsed, settings)
    if client is None:
        return fallback

    labels = allowed_doc_types(settings)
    excerpt = parsed.full_text[:_ENRICH_PREFIX_CHARS]
    user = (
        f"Title: {parsed.title}\n"
        f"Allowed doc_type labels: {', '.join(labels)}\n\n"
        f"Text:\n{excerpt}"
    )
    try:
        insights = await client.structured(
            system=ENRICH_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            schema=DocumentInsights,
            model=settings.anthropic_model_fast,
            effort=settings.anthropic_effort_fast,
        )
    except Exception as exc:
        _log.warning(
            "enrich.llm_failed",
            document_id=parsed.document_id,
            error=type(exc).__name__,
        )
        return fallback

    doc_type = insights.doc_type if insights.doc_type in labels else ""
    if not doc_type:
        doc_type = await _classify_doc_type(parsed, settings, client, labels)
    return DocumentInsights(
        summary=(insights.summary or fallback.summary)[:320],
        keywords=insights.keywords or fallback.keywords,
        doc_type=doc_type or fallback.doc_type,
    )


async def _classify_doc_type(
    parsed: ParsedDocument,
    settings: Settings,
    client: Any,
    labels: Sequence[str],
) -> str:
    """Classify a document's type with the cheap classification model.

    Args:
        parsed: The parsed document.
        settings: Process settings.
        client: LLM client.
        labels: Allowed labels.

    Returns:
        One of ``labels``, or "" when classification failed.
    """
    try:
        label = await client.classify(
            system=DOC_TYPE_SYSTEM_PROMPT,
            text=f"{parsed.title}\n\n{parsed.full_text[:2000]}",
            labels=list(labels),
        )
    except Exception as exc:
        _log.warning(
            "enrich.classify_failed",
            document_id=parsed.document_id,
            error=type(exc).__name__,
        )
        return ""
    return label if label in labels else ""


def _get_llm(settings: Settings) -> Any | None:
    """Resolve the LLM client, tolerating an absent module or missing key.

    Args:
        settings: Process settings.

    Returns:
        An ``LLMClient``, or None when enrichment must fall back to heuristics.
    """
    try:
        from ragcore.llm import get_llm_client
    except ImportError:
        _log.warning("enrich.llm_unavailable", reason="import")
        return None
    try:
        return get_llm_client(settings)
    except Exception as exc:
        _log.warning("enrich.llm_unavailable", reason=type(exc).__name__)
        return None


def _clean_keywords(keywords: Sequence[str]) -> list[str]:
    """Normalise model-produced keywords.

    Args:
        keywords: Raw keyword list.

    Returns:
        Lowercased, de-duplicated, non-empty keywords, at most 12 of them.
    """
    out: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = " ".join(str(keyword).lower().split()).strip(" .,:;")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= 12:
            break
    return out
