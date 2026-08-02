"""Contradiction detection and resolution — pipeline stage 7.

Two documents in an enterprise corpus routinely disagree: a policy is superseded
but the old PDF is still indexed, a handbook repeats a threshold that has since
changed, a regional standard contradicts the global one. A retriever ranks both
highly because both are about the same thing. If the answer silently picks one,
the reader has no way to tell they were handed the 2023 number.

So this stage does three things, in order:

1. **Cluster** retrieved chunks into claims — inverse-document-frequency-weighted
   term overlap, so two passages about the same subject group together while
   passages that merely share stop-words do not. Clustering is what turns "N chunks"
   into "a small number of candidate pairs", which is what keeps the model calls
   bounded.
2. **Detect** a genuine conflict in each cross-document pair. The primary detector
   is one structured ``MODEL_MAIN`` call per pair against
   :data:`~ragcore.llm.prompts.CONTRADICTION_SYSTEM`; the fallback — used when the
   model is unavailable, refuses, or is switched off with
   ``guardrail_contradiction_llm_enabled=False`` — is a deterministic quantity and
   polarity comparison. Both passages go into the prompt inside untrusted blocks:
   a poisoned document must not be able to win an adjudication by instruction.
3. **Resolve** by recency first (``effective_from``, then ``source_modified_at``)
   and authority second (``guardrail_doc_type_authority``), exactly as the contract
   orders it — and then **surface** the resolution. :func:`render_contradiction_notes`
   produces the note the answer prompt carries, which tells the model to present the
   current position *and* cite the conflicting source. A silently dropped
   contradiction would be indistinguishable from a corpus that agrees with itself.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.rag.guardrails.injection import wrap_untrusted
from ragcore.models.chat import GuardrailAction, GuardrailEvent, GuardrailKind
from ragcore.models.retrieval import Citation, RetrievedChunk
from ragcore.observability import observe_guardrail
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragcore.llm.client import LLMClient

__all__ = [
    "ClaimCluster",
    "ConflictResolution",
    "ConflictVerdict",
    "Contradiction",
    "ContradictionReport",
    "check_contradictions",
    "cluster_claims",
    "render_contradiction_notes",
    "resolve_conflict",
]

_log = structlog.get_logger(__name__)

#: Words carrying no topical signal. Kept deliberately small: an aggressive stop
#: list removes the modal verbs ("must", "may", "shall") that carry the polarity a
#: contradiction turns on.
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
        "being",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "had",
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
        "those",
        "to",
        "was",
        "were",
        "will",
        "with",
        "within",
        "you",
        "your",
        "our",
        "we",
        "they",
        "he",
        "she",
        "them",
        "his",
        "her",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "while",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "any",
        "because",
        "before",
        "below",
        "between",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "then",
        "through",
        "under",
        "until",
        "up",
        "very",
        "via",
    ]
)

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{1,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: A number with optional grouping and decimals, plus the unit that follows it.
_QUANTITY_RE = re.compile(
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|eur|usd|gbp|inr|chf|days?|hours?|weeks?|months?|years?)?",
    re.IGNORECASE,
)

#: Currency written before the amount, which is the common enterprise form.
_CURRENCY_PREFIX_RE = re.compile(r"(?:eur|usd|gbp|inr|chf|[€$£₹])\s*$", re.IGNORECASE)

#: Phrases that flip a statement's polarity.
_NEGATIVE_RE = re.compile(
    r"\b(?:must not|may not|shall not|cannot|can't|is not|are not|no longer|"
    r"prohibited|forbidden|not permitted|not allowed|never)\b",
    re.IGNORECASE,
)

#: Words that mark an obligation, so a polarity flip is worth reporting.
_OBLIGATION_RE = re.compile(
    r"\b(?:must|shall|required|require|requires|mandatory|permitted|allowed|"
    r"may|should)\b",
    re.IGNORECASE,
)

#: Shared terms kept per cluster. Sorted by weight with the term itself as the tie
#: break, because a set's iteration order varies per process and a guardrail that
#: decides differently on two runs of the same input is not a guardrail.
_KEY_TERMS_KEPT = 8

#: How many shared terms the polarity detector will try before giving up.
_POLARITY_TERMS_TRIED = 5

#: How many preceding content words identify the thing a number measures.
_QUANTITY_CONTEXT_WORDS = 3

#: How far back the context words are read from. Bounded so quantity extraction
#: stays linear in the passage length rather than quadratic.
_QUANTITY_PREFIX_CHARS = 160

#: Fraction of ``guardrail_contradiction_min_similarity`` at which a numeric
#: disagreement alone is enough to make a pair worth adjudicating. Two passages that
#: both say "the daily allowance is X" are about the same claim even when their
#: surrounding prose differs.
_NUMERIC_PAIR_SIMILARITY_FACTOR = 0.5


class ClaimCluster(BaseModel):
    """A group of retrieved chunks that appear to make the same claim."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(description="Stable id within this report, e.g. 'c1'.")
    chunk_ids: list[str] = Field(
        default_factory=list, description="Member chunk ids, best-scoring first."
    )
    document_ids: list[str] = Field(
        default_factory=list, description="Distinct documents represented."
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Highest-weight terms shared by the members, for tracing.",
    )

    @property
    def size(self) -> int:
        """Number of member chunks.

        Returns:
            The member count.
        """
        return len(self.chunk_ids)

    @property
    def is_cross_document(self) -> bool:
        """Whether the cluster spans more than one document.

        Two chunks of the same document restating a value are not a contradiction;
        they are a document repeating itself.

        Returns:
            True when at least two distinct documents are represented.
        """
        return len(self.document_ids) > 1


class ConflictVerdict(BaseModel):
    """Structured result of the ``MODEL_MAIN`` adjudication call.

    Flat and constraint-free on purpose: structured outputs strip most JSON Schema
    keywords, so the schema stays primitive and the values are clamped here.
    """

    model_config = ConfigDict(extra="ignore")

    conflicts: bool = Field(
        default=False, description="True only for an incompatible claim."
    )
    subject: str = Field(
        default="", description="What the two passages disagree about."
    )
    statement_a: str = Field(default="", description="The first passage's position.")
    statement_b: str = Field(default="", description="The second passage's position.")
    distinguishing_scope: str = Field(
        default="",
        description=(
            "Set when the passages differ in scope rather than conflicting, e.g. "
            "'different region'. Non-empty means conflicts should be False."
        ),
    )
    confidence: float = Field(default=0.5, description="Model confidence, 0..1.")


class ConflictResolution(BaseModel):
    """Which side of a conflict is current, and why."""

    model_config = ConfigDict(extra="forbid")

    winner: RetrievedChunk = Field(description="The currently authoritative chunk.")
    loser: RetrievedChunk = Field(description="The chunk it overrides.")
    basis: str = Field(
        description=(
            "effective_from | source_modified_at | authority | recency | "
            "indeterminate — the first discriminator that separated them."
        )
    )
    gap_days: float | None = Field(
        default=None, description="Age gap in days when a date decided it."
    )
    superseded: bool = Field(
        default=False,
        description=(
            "True when the gap exceeds guardrail_contradiction_recency_days, so the "
            "older statement is stale rather than a live disagreement."
        ),
    )
    winner_authority: int = Field(
        default=0, description="doc_type authority weight of the winner."
    )
    loser_authority: int = Field(
        default=0, description="doc_type authority weight of the loser."
    )

    @property
    def winner_chunk_id(self) -> str:
        """Chunk id of the current position.

        Returns:
            The winning chunk's id.
        """
        return self.winner.payload.chunk_id

    @property
    def loser_chunk_id(self) -> str:
        """Chunk id of the overridden position.

        Returns:
            The losing chunk's id.
        """
        return self.loser.payload.chunk_id


class Contradiction(BaseModel):
    """One resolved conflict, with both sides cited."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="What the sources disagree about.")
    current_chunk_id: str = Field(description="Chunk carrying the current position.")
    superseded_chunk_id: str = Field(description="Chunk carrying the older position.")
    current_statement: str = Field(
        default="", description="Verbatim fragment stating the current position."
    )
    superseded_statement: str = Field(
        default="", description="Verbatim fragment stating the older position."
    )
    basis: str = Field(description="Discriminator used: see ConflictResolution.basis.")
    gap_days: float | None = Field(
        default=None, description="Age gap in days, when dates decided it."
    )
    superseded: bool = Field(
        default=False, description="True when the older statement is simply stale."
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence that this is a conflict."
    )
    detection: str = Field(default="heuristic", description="'llm' or 'heuristic'.")
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Both sides, current first. The answer must cite both — resolving a "
            "conflict silently is the failure mode this stage exists to prevent."
        ),
    )

    @property
    def markers(self) -> list[str]:
        """Citation markers for both sides.

        Returns:
            The markers, current position first.
        """
        return [citation.marker for citation in self.citations]


class ContradictionReport(BaseModel):
    """Stage 7's output."""

    model_config = ConfigDict(extra="forbid")

    checked: bool = Field(
        default=False, description="False when the stage was skipped entirely."
    )
    clusters: list[ClaimCluster] = Field(
        default_factory=list, description="Claim clusters considered."
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="Resolved conflicts, most confident first."
    )
    pairs_examined: int = Field(
        default=0, ge=0, description="Cross-document pairs actually adjudicated."
    )
    degraded: bool = Field(
        default=False,
        description="True when a model call failed and the heuristic decided instead.",
    )
    notes: str = Field(
        default="",
        description=(
            "Rendered conflict note for the answer prompt's `notes` slot. Empty when "
            "there is nothing to surface."
        ),
    )
    events: list[GuardrailEvent] = Field(
        default_factory=list, description="Events to stream and persist."
    )

    @property
    def has_conflicts(self) -> bool:
        """Whether anything needs surfacing in the answer.

        Returns:
            True when at least one contradiction was resolved.
        """
        return bool(self.contradictions)


# --------------------------------------------------------------------- clustering
def _terms(text: str) -> Counter[str]:
    """Tokenise a passage into weighted content terms.

    Args:
        text: Passage text.

    Returns:
        A counter of content terms, stop-words removed.
    """
    counts: Counter[str] = Counter()
    for token in _WORD_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 3:
            continue
        counts[token] += 1
    return counts


def _idf(term_sets: Sequence[Counter[str]]) -> dict[str, float]:
    """Compute inverse document frequency across the candidate set.

    Weighting by IDF is what stops "policy", "employee" and "expenses" — words every
    chunk in a policy corpus contains — from making everything look like the same
    claim.

    Args:
        term_sets: Per-chunk term counters.

    Returns:
        Term to IDF weight.
    """
    total = max(1, len(term_sets))
    document_frequency: Counter[str] = Counter()
    for counts in term_sets:
        document_frequency.update(counts.keys())
    return {
        term: math.log(1.0 + total / (1.0 + frequency))
        for term, frequency in document_frequency.items()
    }


def _similarity(
    left: Counter[str], right: Counter[str], weights: Mapping[str, float]
) -> float:
    """Weighted overlap coefficient between two term sets.

    The overlap coefficient (shared mass over the *smaller* side) rather than
    Jaccard: a short chunk quoting a long one's rule is making the same claim, and
    Jaccard would score that pair down for a difference in length.

    Args:
        left: First term counter.
        right: Second term counter.
        weights: IDF weights.

    Returns:
        A similarity in ``[0, 1]``.
    """
    if not left or not right:
        return 0.0
    left_mass = sum(weights.get(term, 0.0) for term in left)
    right_mass = sum(weights.get(term, 0.0) for term in right)
    if left_mass <= 0.0 or right_mass <= 0.0:
        return 0.0
    shared = sum(weights.get(term, 0.0) for term in left.keys() & right.keys())
    return round(shared / min(left_mass, right_mass), 4)


def cluster_claims(
    chunks: Sequence[RetrievedChunk], *, settings: Settings | None = None
) -> list[ClaimCluster]:
    """Group chunks that appear to make the same claim.

    Single-linkage agglomeration at ``guardrail_contradiction_min_similarity`` over
    the IDF-weighted overlap. Single linkage is the right choice here because a
    claim can be carried across a chain of near-paraphrases; the cluster only has to
    be good enough to propose candidate pairs, which are then adjudicated properly.

    Args:
        chunks: Retrieved candidates, best-scoring first.
        settings: Process settings.

    Returns:
        Clusters of size two or more, largest first. Singletons are dropped — one
        chunk cannot contradict anything.
    """
    resolved = settings or get_settings()
    if len(chunks) < 2:
        return []

    threshold = resolved.guardrail_contradiction_min_similarity
    term_sets = [_terms(chunk.payload.text) for chunk in chunks]
    weights = _idf(term_sets)

    parent = list(range(len(chunks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if _similarity(term_sets[i], term_sets[j], weights) >= threshold:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for index in range(len(chunks)):
        grouped.setdefault(find(index), []).append(index)

    clusters: list[ClaimCluster] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        shared: Counter[str] = Counter()
        for index in members:
            shared.update(term_sets[index].keys())
        key_terms = sorted(
            (term for term, count in shared.items() if count == len(members)),
            key=lambda term: (-weights.get(term, 0.0), term),
        )[:_KEY_TERMS_KEPT]
        document_ids: dict[str, None] = {}
        for index in members:
            document_ids.setdefault(chunks[index].payload.document_id, None)
        clusters.append(
            ClaimCluster(
                cluster_id=f"c{len(clusters) + 1}",
                chunk_ids=[chunks[index].payload.chunk_id for index in members],
                document_ids=list(document_ids),
                key_terms=key_terms,
            )
        )

    clusters.sort(key=lambda cluster: cluster.size, reverse=True)
    max_clusters = int(resolved.guardrail_contradiction_max_clusters)
    return clusters[:max_clusters]


# ------------------------------------------------------------- heuristic detector
def _quantities(text: str) -> dict[str, dict[str, str]]:
    """Extract measured quantities keyed by what they measure.

    The key is the last few content words before the number, which is a crude but
    effective stand-in for "the thing being measured": ``"the daily meal allowance
    is EUR 45"`` and ``"the daily meal allowance is EUR 60"`` produce the same key
    with different values, which is precisely the conflict worth reporting.

    Args:
        text: Passage text.

    Returns:
        Mapping of context key to ``{normalised_value: raw_number}``. The raw number
        is kept so the statement sentence can be located in the original text.
    """
    found: dict[str, dict[str, str]] = {}
    for match in _QUANTITY_RE.finditer(text):
        prefix = text[max(0, match.start() - _QUANTITY_PREFIX_CHARS) : match.start()]
        words = [
            word
            for word in _WORD_RE.findall(prefix.lower())
            if word not in _STOPWORDS and len(word) >= 3
        ]
        if not words:
            continue
        key = " ".join(words[-_QUANTITY_CONTEXT_WORDS:])
        raw = match.group("number")
        number = raw.replace(",", "")
        if number.endswith(".0"):
            number = number[:-2]
        unit = (match.group("unit") or "").lower()
        if not unit:
            currency = _CURRENCY_PREFIX_RE.search(prefix.rstrip())
            unit = currency.group(0).strip().lower() if currency else ""
        found.setdefault(key, {})[f"{number}{unit}"] = raw
    return found


def _sentence_for(text: str, needle: str) -> str:
    """Return the sentence containing a fragment.

    Args:
        text: Passage text.
        needle: Fragment to locate, case-insensitively.

    Returns:
        The containing sentence, or the passage's first sentence when not found.
    """
    sentences = _SENTENCE_RE.split(text.strip())
    lowered = needle.lower()
    for sentence in sentences:
        if lowered and lowered in sentence.lower():
            return sentence.strip()
    return sentences[0].strip() if sentences else text.strip()


def _numeric_conflict(
    left: RetrievedChunk, right: RetrievedChunk
) -> tuple[str, str, str] | None:
    """Find a quantity the two passages disagree about.

    Args:
        left: First chunk.
        right: Second chunk.

    Returns:
        ``(subject, left_statement, right_statement)`` or None when no shared
        measure has differing values.
    """
    left_quantities = _quantities(left.payload.text)
    right_quantities = _quantities(right.payload.text)
    for key, left_values in left_quantities.items():
        right_values = right_quantities.get(key)
        if not right_values or left_values.keys() & right_values.keys():
            continue
        left_raw = left_values[sorted(left_values)[0]]
        right_raw = right_values[sorted(right_values)[0]]
        return (
            key,
            _sentence_for(left.payload.text, left_raw),
            _sentence_for(right.payload.text, right_raw),
        )
    return None


def _polarity_conflict(
    left: RetrievedChunk, right: RetrievedChunk, *, key_terms: Sequence[str]
) -> tuple[str, str, str] | None:
    """Find an obligation stated positively in one passage and negatively in another.

    Every shared key term is tried, not just the highest-weighted one: the term that
    best identifies the *claim* is not always the term that appears in the sentence
    carrying the rule.

    Args:
        left: First chunk.
        right: Second chunk.
        key_terms: Terms the two passages share, most distinctive first, used to
            locate the relevant sentences rather than comparing whole passages.

    Returns:
        ``(subject, left_statement, right_statement)`` or None.
    """
    for term in list(key_terms)[:_POLARITY_TERMS_TRIED]:
        left_sentence = _sentence_for(left.payload.text, term)
        right_sentence = _sentence_for(right.payload.text, term)
        if not (
            _OBLIGATION_RE.search(left_sentence)
            and _OBLIGATION_RE.search(right_sentence)
        ):
            continue
        if bool(_NEGATIVE_RE.search(left_sentence)) == bool(
            _NEGATIVE_RE.search(right_sentence)
        ):
            continue
        return term, left_sentence, right_sentence
    return None


def _heuristic_verdict(
    left: RetrievedChunk, right: RetrievedChunk, *, key_terms: Sequence[str]
) -> ConflictVerdict:
    """Decide a pair without a model call.

    Args:
        left: First chunk.
        right: Second chunk.
        key_terms: Shared cluster terms.

    Returns:
        A :class:`ConflictVerdict`. Confidence is deliberately modest: this detector
        is precise about numbers and crude about everything else.
    """
    numeric = _numeric_conflict(left, right)
    if numeric:
        subject, statement_a, statement_b = numeric
        return ConflictVerdict(
            conflicts=True,
            subject=subject,
            statement_a=statement_a,
            statement_b=statement_b,
            confidence=0.75,
        )
    polarity = _polarity_conflict(left, right, key_terms=key_terms)
    if polarity:
        subject, statement_a, statement_b = polarity
        return ConflictVerdict(
            conflicts=True,
            subject=subject,
            statement_a=statement_a,
            statement_b=statement_b,
            confidence=0.55,
        )
    return ConflictVerdict(conflicts=False)


# ------------------------------------------------------------------- adjudication
def _describe(chunk: RetrievedChunk) -> str:
    """Render a chunk's provenance line for the adjudication prompt.

    Args:
        chunk: The chunk to describe.

    Returns:
        A single line of metadata: type, effective date, modification date, title.
    """
    payload = chunk.payload
    parts = [f"document={payload.document_id}", f"type={payload.doc_type}"]
    if payload.effective_from:
        parts.append(f"effective_from={payload.effective_from.date().isoformat()}")
    if payload.source_modified_at:
        parts.append(f"modified={payload.source_modified_at.date().isoformat()}")
    if payload.title:
        parts.append(f"title={payload.title}")
    return "; ".join(parts)


async def _llm_verdict(
    left: RetrievedChunk,
    right: RetrievedChunk,
    *,
    question: str,
    settings: Settings,
    llm: LLMClient | None,
) -> ConflictVerdict | None:
    """Adjudicate one pair with a structured ``MODEL_MAIN`` call.

    Args:
        left: First chunk.
        right: Second chunk.
        question: The user's question, for scope judgements.
        settings: Process settings.
        llm: Client override.

    Returns:
        The verdict, or None when the call could not be made — the caller then falls
        back to the deterministic detector rather than losing the check.
    """
    if not settings.guardrail_contradiction_llm_enabled:
        return None
    if llm is None and not settings.anthropic_api_key:
        # No credentials: fall straight through to the deterministic detector rather
        # than spending a turn's latency on a call that cannot succeed.
        return None

    from ragcore.llm.prompts import CONTRADICTION_SYSTEM, prompt_metadata

    client = llm
    if client is None:
        from ragcore.llm.client import get_llm_client

        client = get_llm_client(settings)

    limit = int(settings.guardrail_contradiction_snippet_chars)
    user_turn = "\n\n".join(
        [
            f"<question>{question}</question>",
            f"<passage_a>{_describe(left)}</passage_a>",
            wrap_untrusted(left.payload.text[:limit], label="passage A", marker="A"),
            f"<passage_b>{_describe(right)}</passage_b>",
            wrap_untrusted(
                right.payload.text[:limit],
                label="passage B",
                marker="B",
                include_preamble=False,
            ),
        ]
    )
    try:
        verdict = await client.structured(
            system=CONTRADICTION_SYSTEM,
            messages=[{"role": "user", "content": user_turn}],
            schema=ConflictVerdict,
            model=settings.anthropic_model_main,
            effort=settings.anthropic_effort,
            name="guardrail.contradiction",
            metadata=prompt_metadata("contradiction"),
        )
    except Exception:
        _log.warning(
            "contradiction_adjudication_failed",
            left_chunk_id=left.payload.chunk_id,
            right_chunk_id=right.payload.chunk_id,
            exc_info=True,
        )
        return None

    verdict.confidence = max(0.0, min(1.0, verdict.confidence))
    if verdict.distinguishing_scope.strip():
        verdict.conflicts = False
    return verdict


# ---------------------------------------------------------------------- resolution
def _authority(chunk: RetrievedChunk, settings: Settings) -> int:
    """Authority weight of a chunk's document type.

    Args:
        chunk: The chunk.
        settings: Process settings supplying ``guardrail_doc_type_authority``.

    Returns:
        The configured weight, or 0 for an unrecognised type.
    """
    return int(settings.guardrail_doc_type_authority.get(chunk.payload.doc_type, 0))


def _gap_days(newer: datetime, older: datetime) -> float:
    """Whole-day gap between two instants.

    Args:
        newer: The later instant.
        older: The earlier instant.

    Returns:
        The gap in days, rounded to one decimal.
    """
    return round((newer - older).total_seconds() / 86_400.0, 1)


def resolve_conflict(
    left: RetrievedChunk,
    right: RetrievedChunk,
    *,
    settings: Settings | None = None,
) -> ConflictResolution:
    """Decide which side of a conflict is currently authoritative.

    The contract fixes the order: recency first (``effective_from``, then
    ``source_modified_at``), authority second. Recency wins because a superseded
    policy is still a policy — ranking by document type first would let last year's
    policy override this year's standard.

    Args:
        left: First chunk.
        right: Second chunk.
        settings: Process settings.

    Returns:
        A :class:`ConflictResolution` naming the winner, the discriminator that
        chose it, the age gap and whether the loser is superseded outright.
    """
    resolved = settings or get_settings()
    left_authority = _authority(left, resolved)
    right_authority = _authority(right, resolved)

    def build(
        winner: RetrievedChunk,
        loser: RetrievedChunk,
        basis: str,
        gap: float | None,
    ) -> ConflictResolution:
        horizon = resolved.guardrail_contradiction_recency_days
        return ConflictResolution(
            winner=winner,
            loser=loser,
            basis=basis,
            gap_days=gap,
            superseded=gap is not None and gap >= horizon,
            winner_authority=(left_authority if winner is left else right_authority),
            loser_authority=(left_authority if loser is left else right_authority),
        )

    for field, basis in (
        ("effective_from", "effective_from"),
        ("source_modified_at", "source_modified_at"),
    ):
        left_value: datetime | None = getattr(left.payload, field)
        right_value: datetime | None = getattr(right.payload, field)
        if left_value is None or right_value is None or left_value == right_value:
            continue
        if left_value > right_value:
            return build(left, right, basis, _gap_days(left_value, right_value))
        return build(right, left, basis, _gap_days(right_value, left_value))

    if left_authority != right_authority:
        if left_authority > right_authority:
            return build(left, right, "authority", None)
        return build(right, left, "authority", None)

    left_recency = left.payload.recency_at
    right_recency = right.payload.recency_at
    if left_recency != right_recency:
        if left_recency > right_recency:
            return build(left, right, "recency", _gap_days(left_recency, right_recency))
        return build(right, left, "recency", _gap_days(right_recency, left_recency))

    return build(left, right, "indeterminate", None)


# -------------------------------------------------------------------- entry point
def _default_markers(chunks: Sequence[RetrievedChunk]) -> dict[str, str]:
    """Assign ``[n]`` markers in retrieval order.

    Args:
        chunks: Candidates in the order the answer prompt numbers them.

    Returns:
        Mapping of chunk id to marker.
    """
    return {
        chunk.payload.chunk_id: f"[{index}]"
        for index, chunk in enumerate(chunks, start=1)
    }


def _citation_for(
    chunk: RetrievedChunk, marker: str, statement: str, *, confidence: float
) -> Citation:
    """Build a citation for one side of a contradiction.

    The statement is located inside the chunk so the citation carries verifiable
    offsets, exactly like a citation produced by stage 11. A statement the model
    paraphrased rather than quoted yields a citation without a span rather than a
    wrong one.

    Args:
        chunk: The cited chunk.
        marker: Citation marker.
        statement: The fragment stating this side's position.
        confidence: Confidence to record.

    Returns:
        A :class:`~ragcore.models.retrieval.Citation`.
    """
    citation = chunk.to_citation(marker, confidence=confidence)
    fragment = statement.strip()
    if fragment:
        start = chunk.payload.text.find(fragment)
        if start >= 0:
            citation = citation.model_copy(
                update={
                    "quoted_span": fragment,
                    "char_start": start,
                    "char_end": start + len(fragment),
                }
            )
    return citation


async def check_contradictions(
    chunks: Sequence[RetrievedChunk],
    *,
    question: str = "",
    markers: Mapping[str, str] | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> ContradictionReport:
    """Run stage 7 over the retrieved candidate set.

    Args:
        chunks: The chunks about to be rendered into the answer prompt, in the order
            they will be numbered.
        question: The rewritten question, used for scope judgements.
        markers: Chunk id to citation marker. Defaults to ``[1]``-based positional
            markers matching ``chunks``; pass the orchestrator's real mapping so the
            surfaced note cites the same numbers the answer does.
        settings: Process settings.
        llm: LLM client override.

    Returns:
        A :class:`ContradictionReport`. ``notes`` is non-empty exactly when there is
        something the answer must surface.
    """
    resolved = settings or get_settings()
    report = ContradictionReport()

    if not resolved.guardrail_contradiction_enabled:
        return report
    minimum = int(resolved.guardrail_contradiction_min_chunks)
    if len(chunks) < max(2, minimum):
        return report

    report.checked = True
    marker_map = dict(markers) if markers else _default_markers(chunks)
    by_id = {chunk.payload.chunk_id: chunk for chunk in chunks}
    report.clusters = cluster_claims(chunks, settings=resolved)

    pairs = _candidate_pairs(chunks, report.clusters, by_id, settings=resolved)
    max_pairs = int(resolved.guardrail_contradiction_max_pairs)
    # "Degraded" means the model was expected and did not deliver. A deployment that
    # switched the model off, or has no key, is running the documented deterministic
    # path — that is a configuration, not a degradation.
    llm_expected = bool(
        resolved.guardrail_contradiction_llm_enabled
        and (llm is not None or resolved.anthropic_api_key)
    )

    for left, right, key_terms in pairs[:max_pairs]:
        report.pairs_examined += 1
        verdict = await _llm_verdict(
            left, right, question=question, settings=resolved, llm=llm
        )
        detection = "llm"
        if verdict is None:
            report.degraded = report.degraded or llm_expected
            detection = "heuristic"
            verdict = _heuristic_verdict(left, right, key_terms=key_terms)
        if not verdict.conflicts:
            continue

        resolution = resolve_conflict(left, right, settings=resolved)
        left_won = resolution.winner_chunk_id == left.payload.chunk_id
        winner_statement = verdict.statement_a if left_won else verdict.statement_b
        loser_statement = verdict.statement_b if left_won else verdict.statement_a
        winner_marker = marker_map.get(resolution.winner_chunk_id, "[?]")
        loser_marker = marker_map.get(resolution.loser_chunk_id, "[?]")

        contradiction = Contradiction(
            subject=verdict.subject or (key_terms[0] if key_terms else "this topic"),
            current_chunk_id=resolution.winner_chunk_id,
            superseded_chunk_id=resolution.loser_chunk_id,
            current_statement=winner_statement.strip(),
            superseded_statement=loser_statement.strip(),
            basis=resolution.basis,
            gap_days=resolution.gap_days,
            superseded=resolution.superseded,
            confidence=max(0.0, min(1.0, verdict.confidence)),
            detection=detection,
            citations=[
                _citation_for(
                    resolution.winner,
                    winner_marker,
                    winner_statement,
                    confidence=verdict.confidence,
                ),
                _citation_for(
                    resolution.loser,
                    loser_marker,
                    loser_statement,
                    confidence=verdict.confidence,
                ),
            ],
        )
        report.contradictions.append(contradiction)

    report.contradictions.sort(key=lambda item: item.confidence, reverse=True)
    report.notes = render_contradiction_notes(report)

    if report.has_conflicts:
        event = GuardrailEvent(
            stage="retrieval",
            kind=GuardrailKind.CONTRADICTION.value,
            action=GuardrailAction.WARN.value,
            detail=(
                f"{len(report.contradictions)} conflicting source pair(s) resolved "
                f"and surfaced with both citations "
                f"({', '.join(item.basis for item in report.contradictions)})"
            ),
            entities=[item.subject for item in report.contradictions][:6],
            score=max(item.confidence for item in report.contradictions),
        )
        report.events.append(event)
        observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)
        _log.info(
            "contradictions_resolved",
            count=len(report.contradictions),
            pairs_examined=report.pairs_examined,
            degraded=report.degraded,
            bases=[item.basis for item in report.contradictions],
        )
    return report


def _candidate_pairs(
    chunks: Sequence[RetrievedChunk],
    clusters: Sequence[ClaimCluster],
    by_id: Mapping[str, RetrievedChunk],
    *,
    settings: Settings,
) -> list[tuple[RetrievedChunk, RetrievedChunk, list[str]]]:
    """Choose the cross-document pairs worth adjudicating.

    Two sources of candidates, deduplicated: pairs drawn from a cross-document claim
    cluster, and sub-threshold pairs that nonetheless disagree about a shared
    quantity or flip a shared obligation's polarity. The second source matters
    because the most consequential conflicts — a threshold that changed, a
    permission that became a prohibition — are usually surrounded by rewritten
    prose, which is exactly what drags the similarity below the clustering bar.

    Args:
        chunks: All candidates, best-scoring first.
        clusters: Claim clusters.
        by_id: Chunk lookup by id.
        settings: Process settings.

    Returns:
        Pairs as ``(left, right, key_terms)``, best-scoring first.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[RetrievedChunk, RetrievedChunk, list[str]]] = []

    def add(
        left: RetrievedChunk, right: RetrievedChunk, key_terms: Sequence[str]
    ) -> None:
        if left.payload.document_id == right.payload.document_id:
            return
        key = tuple(sorted((left.payload.chunk_id, right.payload.chunk_id)))
        if key in seen:
            return
        seen.add(key)
        pairs.append((left, right, list(key_terms)))

    for cluster in clusters:
        members = [
            by_id[chunk_id] for chunk_id in cluster.chunk_ids if chunk_id in by_id
        ]
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                add(left, right, cluster.key_terms)

    threshold = settings.guardrail_contradiction_min_similarity
    numeric_floor = threshold * _NUMERIC_PAIR_SIMILARITY_FACTOR
    term_sets = [_terms(chunk.payload.text) for chunk in chunks]
    weights = _idf(term_sets)
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = _similarity(term_sets[i], term_sets[j], weights)
            if similarity < numeric_floor or similarity >= threshold:
                continue
            shared = sorted(
                term_sets[i].keys() & term_sets[j].keys(),
                key=lambda term: (-weights.get(term, 0.0), term),
            )[:_KEY_TERMS_KEPT]
            disagrees = _numeric_conflict(chunks[i], chunks[j]) is not None or (
                _polarity_conflict(chunks[i], chunks[j], key_terms=shared) is not None
            )
            if not disagrees:
                continue
            add(chunks[i], chunks[j], shared)

    pairs.sort(key=lambda pair: pair[0].final_score + pair[1].final_score, reverse=True)
    return pairs


def render_contradiction_notes(report: ContradictionReport) -> str:
    """Render the note the answer prompt carries in its ``notes`` slot.

    This is the surfacing step. The note tells the model which position is current
    *and* requires it to cite the conflicting source, so the reader learns that the
    corpus disagrees with itself instead of receiving one number with no warning.

    Args:
        report: The report to render.

    Returns:
        A ``<conflicts>`` block, or an empty string when nothing conflicts.
    """
    if not report.contradictions:
        return ""
    lines = [
        "<conflicts>",
        "Retrieved sources disagree. For each item below, give the current "
        "position as the answer with its citation, then add one sentence noting "
        "that the other source states otherwise, citing it too. Do not merge the "
        "values and do not drop the older source silently.",
    ]
    for item in report.contradictions:
        padded = [*item.citations, None, None]
        current, superseded = padded[0], padded[1]
        current_marker = current.marker if current else "[?]"
        superseded_marker = superseded.marker if superseded else "[?]"
        status = "superseded" if item.superseded else "still in the corpus"
        gap = f", {item.gap_days:g} days apart" if item.gap_days is not None else ""
        lines.append(
            f"- {item.subject}: {current_marker} is current "
            f"(resolved by {item.basis}{gap}); {superseded_marker} is {status} "
            f"and must still be cited when you note the disagreement."
        )
    lines.append("</conflicts>")
    return "\n".join(lines)
