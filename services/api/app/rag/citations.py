"""Citations — pipeline stage 11, the citation half of requirement #9.

Four jobs, in the order the pipeline needs them:

1. :func:`build_source_block` renders the retrieved chunks as the numbered sources
   :data:`~ragcore.llm.prompts.ANSWER_SYSTEM` requires, and keeps the marker → chunk
   mapping the rest of this module needs.
2. :func:`parse_markers` pulls the ``[n]`` markers back out of the generated answer.
3. :func:`verify_span` checks that the text the answer attributes to a source really
   occurs in that source, tolerating whitespace, case, Unicode form and punctuation,
   and falling back to a bounded fuzzy window search for a legitimate paraphrase.
4. :func:`extract_citations` ties the three together into a
   :class:`CitationReport`: verified :class:`~ragcore.models.retrieval.Citation`
   objects, an audited list of what was dropped and why, a ``citation_validity``
   score and a groundedness verdict.

Design notes worth knowing before changing anything here.

**A quoted span always comes from the chunk, never from the answer.** When a
paraphrase is accepted, the ``quoted_span`` stored on the citation is the verbatim
window of the *chunk* that supported it, with real character offsets into
``ChunkPayload.text``. That is what makes a citation clickable and checkable rather
than a restatement of the model's own words.

**Drops carry no content.** :class:`CitationDrop` records the marker, the chunk id,
a reason and the numbers behind the decision — never the span itself. Stage 11 runs
*before* the stage 12 PII egress scan, so at this point the answer text has not
passed redaction and must not be logged, traced or persisted.

**Nothing here rewrites the answer in place.** :func:`strip_unresolved_markers`
produces a cleaned copy and :attr:`CitationReport.cleaned_answer` carries it; whether
to use it is the orchestrator's decision.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.rag import rag_setting
from ragcore.llm.prompts import (
    UNCERTAINTY_NOTICE,
    SourceSnippet,
    render_numbered_sources,
)
from ragcore.models.retrieval import Citation, RetrievedChunk
from ragcore.settings import Settings, get_settings

__all__ = [
    "DROP_EMPTY_SPAN",
    "DROP_QUOTE_NOT_FOUND",
    "DROP_SPAN_NOT_FOUND",
    "DROP_UNKNOWN_MARKER",
    "CitationDrop",
    "CitationReport",
    "CitationVerdict",
    "MarkerRef",
    "SourceBlock",
    "SpanMatch",
    "append_uncertainty_notice",
    "build_source_block",
    "extract_citations",
    "format_marker",
    "parse_markers",
    "strip_unresolved_markers",
    "verify_span",
]

_log = structlog.get_logger(__name__)

#: The answer cited a number that was never printed on a source.
DROP_UNKNOWN_MARKER = "unknown_marker"
#: The sentence carrying the marker has no verifiable content.
DROP_EMPTY_SPAN = "empty_span"
#: The sentence could not be matched against the cited chunk.
DROP_SPAN_NOT_FOUND = "span_not_found"
#: An explicit quotation did not occur in the cited chunk — a fabricated quote.
DROP_QUOTE_NOT_FOUND = "quote_not_found"

#: ``[3]`` and the forms a model reaches for anyway: ``[3, 5]`` and ``[3; 5]``.
#: Adjacent ``[3][5]`` is two matches, which is the form the answer prompt asks for.
_MARKER_RE = re.compile(r"\[\s*(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*\]")

#: Straight and typographic quotation marks, paired.
_QUOTE_RE = re.compile(r"\"([^\"]{2,})\"|“([^”]{2,})”|'([^']{4,})'|‘([^’]{4,})’")  # noqa: RUF001

#: Sentence boundary: terminator plus whitespace, or a hard line break. Bullet lists
#: and numbered steps are separate "sentences" because each carries its own claim.
_SENTENCE_RE = re.compile(r"(?<=[.!?:;])\s+|\n+")

#: Paragraph boundary, used to stop a short sentence borrowing context across a gap.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

#: Prefix length used as a poor-man's stemmer when comparing tokens. Four characters
#: collapses plurals and most English inflections without merging distinct terms.
_STEM_CHARS = 4

#: Function words that say nothing about whether a claim is supported. Deliberately
#: short: an aggressive stop list would discard domain vocabulary.
_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "can",
        "for",
        "from",
        "has",
        "have",
        "into",
        "its",
        "may",
        "must",
        "not",
        "per",
        "shall",
        "should",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "was",
        "were",
        "when",
        "which",
        "will",
        "with",
        "you",
        "your",
    }
)


class CitationVerdict(StrEnum):
    """Groundedness verdict for one answer."""

    #: Every attempted citation verified and the claims are covered.
    GROUNDED = "grounded"
    #: Citations verify but too few claim sentences carry one.
    PARTIAL = "partial"
    #: Citation validity fell below ``guardrail_min_groundedness``.
    UNGROUNDED = "ungrounded"
    #: The answer makes no factual claims (a refusal, a clarifying question).
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class MarkerRef:
    """One ``[n]`` occurrence in the answer.

    Attributes:
        number: The cited source number.
        marker: The rendered marker, e.g. ``"[3]"``.
        start: Start offset of the whole bracket group in the answer.
        end: End offset of the whole bracket group in the answer.
    """

    number: int
    marker: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SpanMatch:
    """Where an answer span was found inside a chunk.

    Attributes:
        text: The verbatim chunk substring that supported the span.
        start: Start offset within ``ChunkPayload.text``.
        end: End offset within ``ChunkPayload.text``.
        ratio: Similarity in ``[0, 1]``; 1.0 for an exact normalised match.
        exact: Whether the span occurred verbatim after normalisation.
    """

    text: str
    start: int
    end: int
    ratio: float
    exact: bool


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """The numbered sources handed to the model, plus the mapping back.

    Attributes:
        text: The rendered block, ready to drop into the answer user turn.
        snippets: The rendered snippets, in marker order.
        chunks: The chunks behind them; index ``i`` carries marker ``i + 1``.
    """

    text: str
    snippets: tuple[SourceSnippet, ...]
    chunks: tuple[RetrievedChunk, ...]

    def chunk_for(self, number: int) -> RetrievedChunk | None:
        """Resolve a marker number to its chunk.

        Args:
            number: The 1-based source number printed in the prompt.

        Returns:
            The chunk, or None when the model cited a number that was never printed.
        """
        if 1 <= number <= len(self.chunks):
            return self.chunks[number - 1]
        return None

    @property
    def size(self) -> int:
        """Number of sources in the block.

        Returns:
            How many markers the model may legitimately use.
        """
        return len(self.chunks)


class CitationDrop(BaseModel):
    """One citation the verifier refused to keep.

    Content-free by construction: stage 11 runs before the PII egress scan, so the
    span that failed is deliberately **not** recorded. The numbers are enough to
    debug a threshold and safe to trace, log and persist.
    """

    model_config = ConfigDict(extra="forbid")

    marker: str = Field(description="The marker as it appeared, e.g. '[3]'.")
    number: int = Field(description="The cited source number.")
    chunk_id: str | None = Field(
        default=None, description="Cited chunk, when the marker resolved to one."
    )
    document_id: str | None = Field(
        default=None, description="Owning document, when the marker resolved."
    )
    reason: str = Field(description="One of the module's DROP_* constants.")
    span_chars: int = Field(
        default=0, ge=0, description="Length of the span that failed verification."
    )
    best_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Best similarity achieved against the cited chunk.",
    )


class CitationReport(BaseModel):
    """Everything stage 11 concluded about one answer."""

    model_config = ConfigDict(extra="forbid")

    citations: list[Citation] = Field(
        default_factory=list,
        description="Verified citations, one per surviving marker, in marker order.",
    )
    dropped: list[CitationDrop] = Field(
        default_factory=list,
        description="Citations that failed verification, with the reason.",
    )
    cleaned_answer: str = Field(
        default="",
        description=(
            "The answer with markers that resolved to nothing removed. The "
            "orchestrator decides whether to use it."
        ),
    )
    markers_attempted: int = Field(
        default=0, ge=0, description="Marker occurrences found in the answer."
    )
    markers_verified: int = Field(
        default=0, ge=0, description="Marker occurrences whose span was verified."
    )
    unknown_markers: list[int] = Field(
        default_factory=list,
        description="Numbers cited that were never printed on a source.",
    )
    claim_sentences: int = Field(
        default=0, ge=0, description="Sentences that state a fact and need support."
    )
    cited_sentences: int = Field(
        default=0, ge=0, description="Claim sentences carrying a verified citation."
    )
    citation_validity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Verified marker occurrences over attempted ones — the metric "
            "MetricScores.citation_validity carries and the stage 11 groundedness "
            "gate compares against guardrail_min_groundedness."
        ),
    )
    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of claim sentences carrying a verified citation.",
    )
    groundedness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="min(citation_validity, coverage) — validity alone can be gamed.",
    )
    verdict: CitationVerdict = Field(
        default=CitationVerdict.NOT_APPLICABLE, description="Overall verdict."
    )
    needs_uncertainty_notice: bool = Field(
        default=False,
        description=(
            "True when citation_validity is below guardrail_min_groundedness and "
            "the answer should carry the uncertainty notice."
        ),
    )

    @property
    def cited_chunk_ids(self) -> list[str]:
        """Chunk ids the surviving citations point at, in marker order.

        Returns:
            The cited chunk ids, de-duplicated.
        """
        seen: dict[str, None] = {}
        for citation in self.citations:
            seen.setdefault(citation.chunk_id, None)
        return list(seen)


def format_marker(number: int) -> str:
    """Render a source number as the marker the prompt and the UI use.

    Args:
        number: The 1-based source number.

    Returns:
        ``"[n]"``.
    """
    return f"[{number}]"


def build_source_block(
    chunks: Sequence[RetrievedChunk], *, settings: Settings | None = None
) -> SourceBlock:
    """Render retrieved chunks as the numbered sources the answer prompt requires.

    Markers are assigned by position, so source ``n`` is ``chunks[n - 1]``. The
    snippet is the document's **own words** (``ChunkPayload.text``), never the
    contextual header that was prepended at embed time — the header is retrieval
    scaffolding and quoting it back would be quoting the pipeline.

    Args:
        chunks: The final ordered chunks from stage 5.
        settings: Settings supplying ``retrieval_snippet_chars``. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        The :class:`SourceBlock`. With no chunks the rendered text is the explicit
        "no sources were retrieved" marker, so the model cannot read an empty block
        as permission to answer from memory.
    """
    cfg = settings or get_settings()
    budget = cfg.retrieval_snippet_chars

    snippets: list[SourceSnippet] = []
    for index, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        effective = (
            payload.effective_from.date().isoformat()
            if payload.effective_from is not None
            else None
        )
        snippets.append(
            SourceSnippet(
                marker=format_marker(index),
                title=payload.title,
                text=_clip(payload.text, budget),
                source_uri=payload.source_uri,
                section_path=tuple(payload.section_path),
                page=payload.page,
                doc_type=payload.doc_type,
                effective_from=effective,
            )
        )
    return SourceBlock(
        text=render_numbered_sources(snippets),
        snippets=tuple(snippets),
        chunks=tuple(chunks),
    )


def _clip(text: str, budget: int) -> str:
    """Clip snippet text to the prompt budget on a word boundary.

    Args:
        text: The chunk text.
        budget: Character budget.

    Returns:
        The text, clipped with an ellipsis when it was too long. Verification always
        runs against the *full* chunk text, so a clipped snippet can only ever make
        the verifier more permissive, never less.
    """
    if budget <= 0 or len(text) <= budget:
        return text
    cut = text[:budget]
    space = cut.rfind(" ")
    if space > budget // 2:
        cut = cut[:space]
    return cut.rstrip() + " …"


def parse_markers(answer: str) -> list[MarkerRef]:
    """Find every ``[n]`` citation marker in an answer.

    Args:
        answer: The generated answer.

    Returns:
        One :class:`MarkerRef` per cited number, in the order they appear. A grouped
        marker such as ``[2, 5]`` yields two refs sharing the same offsets — the
        answer prompt forbids that form, but parsing it means a model that emits it
        still gets its citations verified instead of silently losing them.
    """
    refs: list[MarkerRef] = []
    for match in _MARKER_RE.finditer(answer):
        for part in re.split(r"[,;]", match.group(1)):
            digits = part.strip()
            if not digits:
                continue
            refs.append(
                MarkerRef(
                    number=int(digits),
                    marker=format_marker(int(digits)),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return refs


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise text for matching while keeping a map back to the original offsets.

    Normalisation is NFKC (so a full-width digit or a ligature matches its plain
    form), case folding, and collapsing every run of punctuation and whitespace into
    a single space. That makes the comparison tolerant of line wrapping, hyphenation
    at a line break, smart quotes and trailing commas — all differences that say
    nothing about whether the claim is supported.

    Args:
        text: Text to normalise.

    Returns:
        A ``(normalised, offsets)`` pair where ``offsets[i]`` is the index in
        ``text`` of the character that produced ``normalised[i]``. The map is exact
        even when a character expands under NFKC or case folding, because every
        expanded piece points back at the single source character.
    """
    pieces: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for index, char in enumerate(text):
        folded = unicodedata.normalize("NFKC", char).casefold()
        for piece in folded:
            if not piece.isalnum():
                pending_space = True
                continue
            if pending_space:
                if pieces:
                    pieces.append(" ")
                    offsets.append(index)
                pending_space = False
            pieces.append(piece)
            offsets.append(index)
    return "".join(pieces), offsets


def _to_original(
    start: int, end: int, offsets: Sequence[int], text: str
) -> tuple[int, int]:
    """Translate a normalised range back into original-text offsets.

    Args:
        start: Inclusive start in the normalised string.
        end: Exclusive end in the normalised string.
        offsets: The map from :func:`_normalise_with_map`.
        text: The original text.

    Returns:
        An ``(start, end)`` pair of offsets into ``text``. The end is advanced to the
        end of the character that produced the last matched piece, so a match on a
        ligature or a folded character does not cut it in half.
    """
    if not offsets or start >= len(offsets):
        return 0, 0
    first = offsets[start]
    last = offsets[min(end, len(offsets)) - 1]
    return first, min(len(text), last + 1)


def _tokens(normalised: str) -> list[str]:
    """Split a normalised string into tokens.

    Args:
        normalised: Output of :func:`_normalise_with_map`.

    Returns:
        The whitespace-separated tokens.
    """
    return [token for token in normalised.split(" ") if token]


def _content_tokens(normalised: str) -> list[str]:
    """Keep the tokens that carry meaning for matching.

    Args:
        normalised: Output of :func:`_normalise_with_map`.

    Returns:
        Tokens that are neither stop words nor too short to discriminate. Anything
        containing a digit is always kept: an amount, a date or a threshold is
        exactly the part of a claim that must be supported.
    """
    return [
        token
        for token in _tokens(normalised)
        if any(char.isdigit() for char in token)
        or (len(token) >= 3 and token not in _STOPWORDS)
    ]


def _numbers(normalised: str) -> set[str]:
    """Extract the multi-digit numbers a span asserts.

    Single digits are excluded: they are usually list numbering or a step count,
    not a claim. Two digits and up are amounts, thresholds, years and counts.

    Args:
        normalised: Output of :func:`_normalise_with_map`.

    Returns:
        The distinct numeric runs of at least two digits.
    """
    return set(re.findall(r"\d{2,}", normalised))


def _token_recall(needle: str, window: str) -> float:
    """Fraction of a span's content tokens that appear in a window of the chunk.

    Character-level similarity punishes a faithful paraphrase that reorders a
    sentence, which is the normal shape of a grounded answer. Token recall does not:
    it asks whether the chunk contains the things the sentence is *about*. Tokens are
    compared on their first four characters so that a plural or an inflection
    ("meals" against "meal", "travel" against "travelling") still counts.

    Args:
        needle: Normalised span.
        window: Normalised window of the chunk.

    Returns:
        Recall in ``[0, 1]``, or 0.0 when the span has no content tokens.
    """
    content = _content_tokens(needle)
    if not content:
        return 0.0
    stems = {token[:_STEM_CHARS] for token in _tokens(window)}
    hits = sum(1 for token in content if token[:_STEM_CHARS] in stems)
    return hits / len(content)


def _candidate_windows(
    needle: str, haystack: str, *, anchors: int, max_windows: int
) -> list[tuple[int, int]]:
    """Pick the windows of a chunk worth comparing a span against.

    Scoring every window of a chunk against every span is quadratic and wasteful.
    The longest content tokens of the span are its rarest, so where they occur in the
    chunk is where a real paraphrase must be — and a span whose rarest tokens appear
    nowhere is not a paraphrase of this chunk at all.

    Three windows are emitted per occurrence, placing the anchor at the start, the
    middle and the end. A paraphrase reorders words, so the anchor's position in the
    span says nothing about its position in the source: anchoring only one way finds
    the right chunk and then scores the wrong sentence of it.

    Args:
        needle: Normalised span.
        haystack: Normalised chunk text.
        anchors: How many distinct anchor tokens to try.
        max_windows: Hard cap on the windows returned.

    Returns:
        ``(low, high)`` offset pairs into ``haystack``, de-duplicated, in the order
        they were generated.
    """
    # Ordered by length then alphabetically: set iteration order depends on the
    # per-process hash seed, and a verifier whose result varies between runs is
    # worse than one that is merely imperfect.
    tokens = sorted(
        set(_content_tokens(needle)), key=lambda token: (-len(token), token)
    )
    width = len(needle)
    pad = max(8, width // 4)
    span = width + 2 * pad
    seen: set[tuple[int, int]] = set()
    windows: list[tuple[int, int]] = []
    for token in tokens[: max(1, anchors)]:
        if len(token) < 3:
            continue
        cursor = haystack.find(token)
        while cursor != -1 and len(windows) < max_windows:
            for start in (
                cursor - pad,
                cursor - width // 2,
                cursor + len(token) - width - pad,
            ):
                low = max(0, min(start, max(0, len(haystack) - span)))
                high = min(len(haystack), low + span)
                if (low, high) not in seen:
                    seen.add((low, high))
                    windows.append((low, high))
            cursor = haystack.find(token, cursor + 1)
        if len(windows) >= max_windows:
            break
    return windows[:max_windows]


def verify_span(
    span: str,
    chunk_text: str,
    *,
    threshold: float | None = None,
    verbatim: bool = False,
    check_numbers: bool | None = None,
    settings: Settings | None = None,
) -> SpanMatch | None:
    """Check that a span from the answer really occurs in a chunk.

    Three layers, in increasing tolerance.

    1. **Exact containment on the normalised forms.** Catches every citation where
       the model quoted or closely tracked the source and only whitespace, case,
       Unicode form or punctuation differs. This is the case the answer prompt asks
       for, so it is the case that must never be rejected.
    2. **A bounded fuzzy window search** anchored on the span's rarest tokens. A
       window is scored on character similarity *and* on content-token recall, and
       the better of the two decides — character similarity alone punishes a
       faithful paraphrase for reordering a sentence, and token recall alone would
       accept a bag of the right words in the wrong arrangement.
    3. **A numeric guard.** When ``check_numbers`` is on, every multi-digit number
       the span asserts must occur in the chunk. A wrong figure is the highest-cost
       hallucination in an enterprise answer and the cheapest one to catch, and the
       answer prompt already tells the model to quote numbers verbatim.

    Args:
        span: Text from the answer, with citation markers already removed.
        chunk_text: ``ChunkPayload.text`` of the cited chunk.
        threshold: Minimum score to accept. Defaults to ``citation_quote_min_ratio``
            when ``verbatim`` is set and ``citation_fuzzy_threshold`` otherwise.
        verbatim: Treat the span as an explicit quotation. Token recall is not
            consulted and the default threshold is much stricter, because a quotation
            that is merely *about* the right thing is a fabricated quotation.
        check_numbers: Apply the numeric guard. Defaults to
            ``citation_number_check``. Pass False for a sentence that cites several
            sources, where each source only has to support its own share.
        settings: Settings supplying the thresholds and bounds.

    Returns:
        The :class:`SpanMatch` describing where the support was found, or None when
        the span could not be located well enough. On success ``text`` is the
        verbatim chunk substring and the offsets index into ``chunk_text``.
    """
    cfg = settings or get_settings()
    if threshold is not None:
        floor = threshold
    elif verbatim:
        floor = float(rag_setting(cfg, "citation_quote_min_ratio"))
    else:
        floor = float(rag_setting(cfg, "citation_fuzzy_threshold"))
    budget = int(rag_setting(cfg, "citation_max_span_chars"))

    needle, _ = _normalise_with_map(span[:budget])
    haystack, offsets = _normalise_with_map(chunk_text)
    if not needle or not haystack:
        return None

    guard = (
        bool(rag_setting(cfg, "citation_number_check"))
        if check_numbers is None
        else check_numbers
    )
    if guard and not _numbers(needle) <= _numbers(haystack):
        return None

    exact = haystack.find(needle)
    if exact != -1:
        start, end = _to_original(exact, exact + len(needle), offsets, chunk_text)
        return SpanMatch(
            text=chunk_text[start:end], start=start, end=end, ratio=1.0, exact=True
        )

    candidates = _candidate_windows(
        needle,
        haystack,
        anchors=int(rag_setting(cfg, "citation_anchor_tokens")),
        max_windows=int(rag_setting(cfg, "citation_max_windows")),
    )
    if not candidates:
        return None

    recall_floor = float(rag_setting(cfg, "citation_token_recall_threshold"))
    best: SpanMatch | None = None
    for low, high in candidates:
        window = haystack[low:high]
        matcher = difflib.SequenceMatcher(None, needle, window, autojunk=False)
        ratio = matcher.ratio() if matcher.quick_ratio() >= floor else 0.0
        score = ratio
        if not verbatim:
            recall = _token_recall(needle, window)
            if recall >= recall_floor:
                score = max(score, recall)
        if score < floor or (best is not None and score <= best.ratio):
            continue
        start, end = _match_bounds(matcher, low, offsets, chunk_text, window_end=high)
        best = SpanMatch(
            text=chunk_text[start:end], start=start, end=end, ratio=score, exact=False
        )
    return best


def _match_bounds(
    matcher: difflib.SequenceMatcher[str],
    low: int,
    offsets: Sequence[int],
    chunk_text: str,
    *,
    window_end: int,
) -> tuple[int, int]:
    """Locate the supporting text inside a matched window.

    Args:
        matcher: The matcher already run over ``(needle, window)``.
        low: Offset of the window's start in the normalised chunk.
        offsets: The map from :func:`_normalise_with_map`.
        chunk_text: The original chunk text.
        window_end: Offset of the window's end in the normalised chunk.

    Returns:
        ``(start, end)`` offsets into ``chunk_text``. The matching blocks give the
        tightest useful span; when they cover almost nothing — which is what a
        recall-driven match looks like — the whole window is returned instead, so
        the citation still points somewhere a reader can check.
    """
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    span_start, span_end = low, window_end
    if blocks:
        covered = sum(block.size for block in blocks)
        candidate_start = low + blocks[0].b
        candidate_end = low + blocks[-1].b + blocks[-1].size
        if covered * 4 >= (candidate_end - candidate_start):
            span_start, span_end = candidate_start, candidate_end
    return _to_original(span_start, span_end, offsets, chunk_text)


def _split_sentences(answer: str) -> list[tuple[int, int]]:
    """Split an answer into sentence ranges.

    Args:
        answer: The generated answer.

    Returns:
        ``(start, end)`` offset pairs covering the answer, whitespace-trimmed and
        excluding empty ranges.
    """
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(answer):
        ranges.append((cursor, match.start()))
        cursor = match.end()
    ranges.append((cursor, len(answer)))

    trimmed: list[tuple[int, int]] = []
    for start, end in ranges:
        text = answer[start:end]
        lead = len(text) - len(text.lstrip())
        tail = len(text) - len(text.rstrip())
        if end - tail > start + lead:
            trimmed.append((start + lead, end - tail))
    return trimmed


def _strip_markers(text: str) -> str:
    """Remove citation markers from a span before it is verified.

    Args:
        text: A sentence from the answer.

    Returns:
        The sentence without ``[n]`` groups, with the resulting double spaces
        collapsed.
    """
    return re.sub(r"\s+", " ", _MARKER_RE.sub(" ", text)).strip()


def _quoted_literal(text: str) -> str | None:
    """Extract an explicit quotation from a sentence, if it has one.

    Args:
        text: A sentence with markers already removed.

    Returns:
        The longest quoted run, or None. Single quotes need four characters before
        they count, because an apostrophe in "the company's policy" would otherwise
        look like an opening quote.
    """
    candidates = [
        group
        for match in _QUOTE_RE.finditer(text)
        for group in match.groups()
        if group is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=len).strip() or None


def _is_claim(text: str, *, min_words: int) -> bool:
    """Decide whether a sentence states a fact that needs a citation.

    Args:
        text: A sentence with markers already removed.
        min_words: ``citation_min_claim_words``.

    Returns:
        True when the sentence is long enough to assert something and is not a
        question. Questions, headings and short connectives are excluded so that a
        well-formed answer's structure does not depress its coverage score.
    """
    stripped = text.strip()
    if not stripped or stripped.endswith("?"):
        return False
    return len(stripped.split()) >= min_words


def strip_unresolved_markers(answer: str, keep: Iterable[int]) -> str:
    """Remove markers the verifier could not turn into a citation.

    A marker with no citation behind it is worse than no marker: the UI renders it as
    a dead link and the reader reads it as evidence.

    Args:
        answer: The generated answer.
        keep: Source numbers that survived verification.

    Returns:
        The answer with every other ``[n]`` removed and the spacing tidied.
    """
    survivors = set(keep)

    def _replace(match: re.Match[str]) -> str:
        numbers = [
            int(part.strip())
            for part in re.split(r"[,;]", match.group(1))
            if part.strip()
        ]
        kept = [number for number in numbers if number in survivors]
        return "".join(format_marker(number) for number in kept)

    cleaned = _MARKER_RE.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)


def extract_citations(
    answer: str,
    sources: SourceBlock | Sequence[RetrievedChunk],
    *,
    settings: Settings | None = None,
) -> CitationReport:
    """Map ``[n]`` markers onto chunks, verify their spans and score groundedness.

    Args:
        answer: The generated answer, before the stage 12 output guard.
        sources: The :class:`SourceBlock` handed to the model, or the chunk list it
            was built from — the numbering is positional either way.
        settings: Settings supplying the thresholds. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        A :class:`CitationReport`. Every marker occurrence is accounted for: it is
        either behind a surviving :class:`~ragcore.models.retrieval.Citation` or in
        :attr:`CitationReport.dropped` with a reason.
    """
    cfg = settings or get_settings()
    block = (
        sources
        if isinstance(sources, SourceBlock)
        else build_source_block(sources, settings=cfg)
    )
    min_span = int(rag_setting(cfg, "citation_min_span_chars"))
    quote_ratio = float(rag_setting(cfg, "citation_quote_min_ratio"))
    min_claim_words = int(rag_setting(cfg, "citation_min_claim_words"))
    min_coverage = float(rag_setting(cfg, "citation_min_coverage"))

    sentences = _split_sentences(answer)
    refs = parse_markers(answer)

    best_per_marker: dict[int, Citation] = {}
    dropped: list[CitationDrop] = []
    unknown: list[int] = []
    verified_refs = 0
    cited_sentence_indexes: set[int] = set()

    for position, (start, end) in enumerate(sentences):
        sentence = answer[start:end]
        sentence_refs = [ref for ref in refs if start <= ref.start < end]
        if not sentence_refs:
            continue
        span = _span_for(answer, sentences, position, min_span_chars=min_span)
        quote = _quoted_literal(_strip_markers(sentence))
        for ref in sentence_refs:
            chunk = block.chunk_for(ref.number)
            if chunk is None:
                unknown.append(ref.number)
                dropped.append(
                    CitationDrop(
                        marker=ref.marker,
                        number=ref.number,
                        reason=DROP_UNKNOWN_MARKER,
                        span_chars=len(span),
                    )
                )
                continue
            if not span:
                dropped.append(
                    CitationDrop(
                        marker=ref.marker,
                        number=ref.number,
                        chunk_id=chunk.payload.chunk_id,
                        document_id=chunk.payload.document_id,
                        reason=DROP_EMPTY_SPAN,
                    )
                )
                continue

            use_quote = quote is not None and len(quote) >= min_span
            probe = quote if use_quote else span
            match = verify_span(
                probe or span,
                chunk.payload.text,
                threshold=quote_ratio if use_quote else None,
                verbatim=use_quote,
                # A sentence citing several sources spreads its numbers across them,
                # so the numeric guard would reject each source for the other's
                # figures. It only applies when one source carries the whole claim.
                check_numbers=len(sentence_refs) == 1,
                settings=cfg,
            )
            if match is None:
                dropped.append(
                    CitationDrop(
                        marker=ref.marker,
                        number=ref.number,
                        chunk_id=chunk.payload.chunk_id,
                        document_id=chunk.payload.document_id,
                        reason=(
                            DROP_QUOTE_NOT_FOUND if use_quote else DROP_SPAN_NOT_FOUND
                        ),
                        span_chars=len(probe or span),
                    )
                )
                continue

            verified_refs += 1
            cited_sentence_indexes.add(position)
            citation = chunk.to_citation(ref.marker, confidence=match.ratio).model_copy(
                update={
                    "quoted_span": match.text,
                    "char_start": match.start,
                    "char_end": match.end,
                    "confidence": round(match.ratio, 4),
                }
            )
            incumbent = best_per_marker.get(ref.number)
            if incumbent is None or citation.confidence > incumbent.confidence:
                best_per_marker[ref.number] = citation

    claim_positions = [
        position
        for position, (start, end) in enumerate(sentences)
        if _is_claim(_strip_markers(answer[start:end]), min_words=min_claim_words)
    ]
    claims = len(claim_positions)
    cited_claims = len(set(claim_positions) & cited_sentence_indexes)

    attempted = len(refs)
    if attempted:
        validity = verified_refs / attempted
    else:
        # No markers at all: fine for a refusal or a clarifying question, and a
        # complete failure of grounding for an answer that asserts things.
        validity = 0.0 if claims else 1.0
    coverage = (cited_claims / claims) if claims else 1.0
    groundedness = min(validity, coverage)

    if not claims and not attempted:
        verdict = CitationVerdict.NOT_APPLICABLE
    elif validity < cfg.guardrail_min_groundedness:
        verdict = CitationVerdict.UNGROUNDED
    elif coverage < min_coverage:
        verdict = CitationVerdict.PARTIAL
    else:
        verdict = CitationVerdict.GROUNDED

    citations = [best_per_marker[number] for number in sorted(best_per_marker)]
    cleaned = (
        strip_unresolved_markers(answer, best_per_marker)
        if rag_setting(cfg, "citation_strip_unresolved_markers")
        else answer
    )

    report = CitationReport(
        citations=citations,
        dropped=dropped,
        cleaned_answer=cleaned,
        markers_attempted=attempted,
        markers_verified=verified_refs,
        unknown_markers=sorted(set(unknown)),
        claim_sentences=claims,
        cited_sentences=cited_claims,
        citation_validity=round(validity, 4),
        coverage=round(coverage, 4),
        groundedness=round(groundedness, 4),
        verdict=verdict,
        needs_uncertainty_notice=validity < cfg.guardrail_min_groundedness,
    )
    _log.info(
        "citations.verified",
        sources=block.size,
        attempted=attempted,
        verified=verified_refs,
        dropped=len(dropped),
        unknown=len(report.unknown_markers),
        claim_sentences=claims,
        citation_validity=report.citation_validity,
        coverage=report.coverage,
        verdict=verdict.value,
    )
    return report


def _span_for(
    answer: str,
    sentences: Sequence[tuple[int, int]],
    position: int,
    *,
    min_span_chars: int,
) -> str:
    """Build the text a marker's citation is verified against.

    A marker attaches to its sentence. When that sentence is too short to carry
    meaning on its own — a list item like "EUR 60 per day [2]" or a table cell — the
    preceding sentence is prepended, provided it is in the same paragraph, so the
    verifier has something to match.

    Args:
        answer: The generated answer.
        sentences: Sentence ranges from :func:`_split_sentences`.
        position: Index of the sentence carrying the marker.
        min_span_chars: ``citation_min_span_chars``.

    Returns:
        The span with markers removed, possibly empty.
    """
    start, end = sentences[position]
    span = _strip_markers(answer[start:end])
    if len(span) >= min_span_chars or position == 0:
        return span
    previous_start, previous_end = sentences[position - 1]
    if _PARAGRAPH_RE.search(answer[previous_end:start]):
        return span
    previous = _strip_markers(answer[previous_start:previous_end])
    return f"{previous} {span}".strip() if previous else span


def append_uncertainty_notice(
    answer: str, report: CitationReport, *, settings: Settings | None = None
) -> str:
    """Append the groundedness notice when stage 11's gate says the answer needs it.

    Args:
        answer: The answer to annotate — normally
            :attr:`CitationReport.cleaned_answer`.
        report: The report produced by :func:`extract_citations`.
        settings: Settings supplying ``guardrail_min_groundedness``. Accepted so the
            caller can re-evaluate the gate against an override; the report already
            carries the decision.

    Returns:
        The answer, with :data:`~ragcore.llm.prompts.UNCERTAINTY_NOTICE` appended
        when the gate fires and the notice is not already present.
    """
    del settings
    if not report.needs_uncertainty_notice or UNCERTAINTY_NOTICE in answer:
        return answer
    return f"{answer.rstrip()}\n\n{UNCERTAINTY_NOTICE}"
