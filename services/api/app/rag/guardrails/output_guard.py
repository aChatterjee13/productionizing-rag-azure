"""Output guard — pipeline stage 12.

The last thing between a generated answer and the user. Four checks, in descending
order of severity:

1. **Classification, as defence in depth.** Every chunk handed to generation was
   already filtered in Qdrant by
   :func:`~ragcore.vectorstore.filters.build_acl_filter`, so nothing above the
   principal's clearance should be here at all. This check therefore does not exist
   to catch attackers — it exists to catch *us*. If it ever fires, a filter is
   broken, and the log line says so at ``error`` level with the chunk, document and
   ranks attached. The answer is replaced rather than edited: an answer derived from
   material the reader may not see is not salvageable by deleting a sentence.
2. **PII egress.** The answer is scanned again on the way out. Ingest-time redaction
   covers indexed text, but an answer can compose an identifier out of fragments, and
   a tool result never passed through ingestion at all. Redaction is the default;
   ``pii_block_on_egress`` makes it fatal.
3. **Groundedness.** ``citation_validity`` from stage 11 gates the answer: below
   ``guardrail_min_groundedness`` an uncertainty notice is appended, below
   ``guardrail_output_block_below_groundedness`` the answer is withheld. An answer
   whose markers point nowhere is a confident fabrication with footnotes.
4. **Refusal quality.** A refusal that says only "I don't know" is a bug, not a
   safety property: the contract requires a refusal to say what *is* covered and
   where the answer might live instead.

Everything is returned in one :class:`OutputDecision`;
:attr:`OutputDecision.redacted_text` is the only form that may be persisted.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.acl import Classification, Principal
from ragcore.models.chat import (
    GuardrailAction,
    GuardrailEvent,
    GuardrailKind,
    GuardrailStage,
)
from ragcore.models.retrieval import Citation, RetrievedChunk
from ragcore.observability import observe_guardrail
from ragcore.pii import PIIDetector, PIIReport, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "CLEARANCE_BLOCK_MESSAGE",
    "GROUNDEDNESS_BLOCK_MESSAGE",
    "PII_BLOCK_MESSAGE",
    "ClearanceReport",
    "ClearanceViolation",
    "OutputDecision",
    "RefusalQuality",
    "assess_refusal",
    "check_clearance",
    "citation_validity_score",
    "run_output_guard",
]

_log = structlog.get_logger(__name__)

#: Replaces the answer when a classification violation is found. Deliberately vague
#: about *what* was withheld: naming it would be the disclosure the check prevents.
CLEARANCE_BLOCK_MESSAGE = (
    "I can't return that answer. It drew on material above your access level, so "
    "it has been withheld and the incident logged for review. Ask the document "
    "owner if you need access."
)

#: Replaces the answer when PII egress is configured to be fatal.
PII_BLOCK_MESSAGE = (
    "I can't return that answer: it contained personal data that this assistant is "
    "configured not to emit. Rephrase the question so it does not require personal "
    "details, or ask the data owner directly."
)

#: Replaces the answer when groundedness collapses.
GROUNDEDNESS_BLOCK_MESSAGE = (
    "I couldn't ground an answer in the retrieved documents — the statements I "
    "assembled could not be traced back to a cited source, so I have withheld them "
    "rather than present them as fact. Try a narrower question, or name the "
    "document you expect the answer to be in."
)

#: Citation markers as the answer prompt requires them: ``[1]``, never ``[1, 2]``.
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

#: Whitespace runs, collapsed before any span comparison.
_WHITESPACE_RE = re.compile(r"\s+")

#: Phrases that mark an answer as a refusal rather than an attempt.
_REFUSAL_RE = re.compile(
    r"\b(?:i don't have|i do not have|i can't|i cannot|i'm not able|"
    r"outside the indexed|not in the indexed|no documents|nothing in the corpus|"
    r"couldn't ground|could not find|i don't know|i do not know)\b",
    re.IGNORECASE,
)

#: Phrases that show a refusal oriented the reader instead of stopping at "no".
_ORIENTATION_RE = re.compile(
    r"\b(?:indexed|covers?|covered|coverage|available|instead|try|narrow|"
    r"ask|owner|system|upload|index)\b",
    re.IGNORECASE,
)

#: Weight for a marker whose citation exists and points at a retrieved chunk but
#: carries no verified span — attributed, not quoted.
_UNQUOTED_MARKER_WEIGHT = 0.5


class ClearanceViolation(BaseModel):
    """One place where the answer exceeded the principal's clearance."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description=(
            "'chunk' — over-clearance material reached generation; 'citation' — the "
            "answer cited it; 'leaked_span' — its text appears verbatim in the "
            "answer."
        )
    )
    chunk_id: str = Field(description="Offending chunk id.")
    document_id: str = Field(description="Offending document id.")
    classification: str = Field(description="Chunk classification value.")
    classification_rank: int = Field(ge=0, description="Chunk classification rank.")
    principal_rank: int = Field(ge=0, description="Principal's clearance rank.")
    detail: str = Field(default="", description="Redacted explanation.")


class ClearanceReport(BaseModel):
    """Result of the defence-in-depth classification check."""

    model_config = ConfigDict(extra="forbid")

    violations: list[ClearanceViolation] = Field(
        default_factory=list, description="Every violation found, most severe first."
    )
    checked_chunks: int = Field(default=0, ge=0, description="Chunks inspected.")
    checked_citations: int = Field(default=0, ge=0, description="Citations inspected.")

    @property
    def ok(self) -> bool:
        """Whether the answer is safe to emit on classification grounds.

        Returns:
            True when no violation was found.
        """
        return not self.violations

    @property
    def leaked(self) -> bool:
        """Whether over-clearance text actually reached the answer.

        Returns:
            True when a verbatim span or a citation crossed the boundary.
        """
        return any(
            violation.kind in {"leaked_span", "citation"}
            for violation in self.violations
        )


class RefusalQuality(BaseModel):
    """Whether a refusal is the kind of refusal the contract asks for."""

    model_config = ConfigDict(extra="forbid")

    is_refusal: bool = Field(
        default=False, description="Whether the answer declines to answer."
    )
    acceptable: bool = Field(
        default=True, description="False for a bare or unhelpful refusal."
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Machine-readable faults: 'too_short', 'no_orientation'.",
    )


class OutputDecision(BaseModel):
    """Stage 12's verdict on one generated answer."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        default="",
        description="The answer to emit. Redacted, annotated or replaced as needed.",
    )
    redacted_text: str = Field(
        default="",
        description=(
            "Fully PII-redacted form of the emitted answer. The only form that may "
            "be persisted, traced or logged."
        ),
    )
    blocked: bool = Field(
        default=False, description="True when the original answer was withheld."
    )
    action: str = Field(
        default=GuardrailAction.ALLOW.value,
        description="Strongest action taken across the four checks.",
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Citations that survived the guard."
    )
    dropped_citations: list[Citation] = Field(
        default_factory=list, description="Citations removed, with the reason logged."
    )
    pii_types: list[str] = Field(
        default_factory=list, description="Entity types found on egress."
    )
    pii_redacted: bool = Field(
        default=False,
        description="Assertion that the egress redaction pass ran on this answer.",
    )
    groundedness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="citation_validity for this answer."
    )
    groundedness_applicable: bool = Field(
        default=False,
        description=(
            "False when the gate does not apply — a refusal, or a turn with no "
            "retrieved sources to be grounded in."
        ),
    )
    uncertainty_appended: bool = Field(
        default=False, description="True when the uncertainty notice was appended."
    )
    clearance: ClearanceReport = Field(
        default_factory=ClearanceReport, description="Classification check result."
    )
    refusal: RefusalQuality = Field(
        default_factory=RefusalQuality, description="Refusal-quality assessment."
    )
    events: list[GuardrailEvent] = Field(
        default_factory=list, description="Events to stream and persist."
    )


def _normalise(text: str) -> str:
    """Casefold and collapse whitespace for span comparison.

    Args:
        text: Text to normalise.

    Returns:
        The normalised text.
    """
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _markers(answer: str) -> list[str]:
    """Extract distinct citation markers from an answer, in first-seen order.

    Args:
        answer: The generated answer.

    Returns:
        Markers such as ``["[1]", "[3]"]``.
    """
    seen: dict[str, None] = {}
    for match in _MARKER_RE.finditer(answer):
        seen.setdefault(f"[{int(match.group(1))}]", None)
    return list(seen)


def citation_validity_score(
    answer: str,
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
) -> float:
    """Score how well the answer's markers are backed by verified citations.

    This is the ``citation_validity`` the contract names in ``MetricScores`` and the
    value the groundedness gate compares against ``guardrail_min_groundedness``. Per
    distinct marker:

    * ``1.0`` — a citation exists, points at a chunk that was actually retrieved, and
      carries a span verified against that chunk's text;
    * ``0.5`` — the citation exists and points at a retrieved chunk but quotes
      nothing, so the attribution is plausible but unproven;
    * ``0.0`` — no citation for the marker, or it points outside the retrieved set.
      That is a fabricated footnote and is the case worth catching.

    Args:
        answer: The generated answer.
        citations: Citations surviving stage 11's span verification.
        chunks: Chunks that were actually handed to generation.

    Returns:
        The mean weight over distinct markers, or ``1.0`` when the answer makes no
        cited claims — vacuous, so the caller decides whether the gate applies.
    """
    markers = _markers(answer)
    if not markers:
        return 1.0
    by_marker = {citation.marker: citation for citation in citations}
    retrieved = {chunk.payload.chunk_id for chunk in chunks}
    total = 0.0
    for marker in markers:
        citation = by_marker.get(marker)
        if citation is None or citation.chunk_id not in retrieved:
            continue
        total += 1.0 if citation.is_verified else _UNQUOTED_MARKER_WEIGHT
    return round(total / len(markers), 4)


def _leaked_span(answer: str, chunk_text: str, *, span_chars: int) -> bool:
    """Detect whether a chunk's text appears verbatim in the answer.

    Slides a window of ``span_chars`` over the normalised chunk text at half-window
    steps. A quoted sentence is caught; a shared stock phrase shorter than the window
    is not, which is the point — the question is whether *content* crossed, not
    whether two documents use the same words for "the following applies".

    Args:
        answer: The generated answer.
        chunk_text: Text of the chunk under suspicion.
        span_chars: Minimum overlap length treated as a leak.

    Returns:
        True when a window of the chunk occurs in the answer.
    """
    haystack = _normalise(answer)
    needle_source = _normalise(chunk_text)
    if not haystack or not needle_source:
        return False
    if len(needle_source) <= span_chars:
        return needle_source in haystack
    step = max(1, span_chars // 2)
    for start in range(0, len(needle_source) - span_chars + 1, step):
        if needle_source[start : start + span_chars] in haystack:
            return True
    return False


def check_clearance(
    *,
    answer: str,
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
    principal: Principal,
    settings: Settings | None = None,
) -> ClearanceReport:
    """Verify that nothing above the principal's clearance reached the answer.

    Uses :meth:`~ragcore.models.acl.AccessControl.permits`, the in-process mirror of
    the Qdrant filter — deliberately a *second* implementation of the same rule, so a
    bug in one is visible against the other.

    Args:
        answer: The generated answer.
        citations: Citations attached to the answer.
        chunks: Chunks handed to generation.
        principal: The caller.
        settings: Process settings.

    Returns:
        A :class:`ClearanceReport`. An empty report is the expected result on every
        turn; anything else means a filter is broken upstream.
    """
    resolved = settings or get_settings()
    report = ClearanceReport(
        checked_chunks=len(chunks), checked_citations=len(citations)
    )
    span_chars = int(resolved.guardrail_output_leak_span_chars)
    principal_rank = principal.clearance_rank()
    cited_chunk_ids = {citation.chunk_id for citation in citations}

    for chunk in chunks:
        payload = chunk.payload
        access = payload.access_control()
        over_clearance = payload.classification_rank > principal_rank
        if access.permits(principal) and not over_clearance:
            continue

        base = {
            "chunk_id": payload.chunk_id,
            "document_id": payload.document_id,
            "classification": payload.classification,
            "classification_rank": payload.classification_rank,
            "principal_rank": principal_rank,
        }
        report.violations.append(
            ClearanceViolation(
                kind="chunk",
                detail=(
                    "chunk above the principal's clearance reached generation; "
                    "the ACL filter should have excluded it"
                ),
                **base,
            )
        )
        if payload.chunk_id in cited_chunk_ids:
            report.violations.append(
                ClearanceViolation(
                    kind="citation",
                    detail="the answer cites an over-clearance chunk",
                    **base,
                )
            )
        if _leaked_span(answer, payload.text, span_chars=span_chars):
            report.violations.append(
                ClearanceViolation(
                    kind="leaked_span",
                    detail=(
                        f"a span of at least {span_chars} characters from an "
                        "over-clearance chunk appears in the answer"
                    ),
                    **base,
                )
            )

    order = {"leaked_span": 0, "citation": 1, "chunk": 2}
    report.violations.sort(key=lambda violation: order.get(violation.kind, 9))
    return report


def assess_refusal(text: str, *, settings: Settings | None = None) -> RefusalQuality:
    """Judge whether a refusal is useful.

    A refusal is acceptable when it is long enough to have said something and it
    orients the reader — names what is covered, or where the answer might live. "I
    don't know." satisfies neither and is reported as a fault, because the contract
    treats a bare refusal as a failure mode rather than a safe default.

    Args:
        text: The answer text.
        settings: Process settings.

    Returns:
        A :class:`RefusalQuality`.
    """
    resolved = settings or get_settings()
    quality = RefusalQuality()
    if not resolved.guardrail_refusal_check_enabled:
        return quality

    stripped = text.strip()
    quality.is_refusal = bool(_REFUSAL_RE.search(stripped))
    if not quality.is_refusal:
        return quality

    minimum = int(resolved.guardrail_refusal_min_chars)
    if len(stripped) < minimum:
        quality.reasons.append("too_short")
    if not _ORIENTATION_RE.search(stripped):
        quality.reasons.append("no_orientation")
    quality.acceptable = not quality.reasons
    return quality


def _egress_report(report: PIIReport, ignored: Sequence[str]) -> PIIReport:
    """Drop entity types that are noise in an answer.

    Dates and place names are the two entity types a policy answer legitimately
    contains on nearly every turn ("effective from 1 April 2025", "the Munich
    office"); redacting them would gut the answer to protect nothing.

    Args:
        report: The detector's report.
        ignored: Entity types to ignore on egress.

    Returns:
        A report without the ignored types.
    """
    skip = {name.upper() for name in ignored}
    if not skip:
        return report
    return PIIReport.from_findings(
        [finding for finding in report.findings if finding.entity_type not in skip]
    )


def _record(events: list[GuardrailEvent], event: GuardrailEvent) -> None:
    """Append an event and mirror it into the Prometheus counter.

    Args:
        events: The decision's event list.
        event: The event to record.
    """
    events.append(event)
    observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)


async def run_output_guard(
    *,
    answer: str,
    citations: Sequence[Citation],
    chunks: Sequence[RetrievedChunk],
    principal: Principal,
    settings: Settings | None = None,
    detector: PIIDetector | None = None,
    groundedness: float | None = None,
) -> OutputDecision:
    """Run stage 12 over one generated answer.

    Args:
        answer: The generated answer, after citation extraction.
        citations: Citations that survived stage 11's span verification.
        chunks: Chunks handed to generation, used for the clearance and groundedness
            checks.
        principal: The caller.
        settings: Process settings.
        detector: PII detector override.
        groundedness: Pre-computed ``citation_validity``. Recomputed from the answer
            when omitted, so the guard is correct even if stage 11 did not report it.

    Returns:
        An :class:`OutputDecision`. Emit :attr:`OutputDecision.text`; persist
        :attr:`OutputDecision.redacted_text`.
    """
    resolved = settings or get_settings()
    decision = OutputDecision(
        text=answer,
        redacted_text=answer,
        citations=list(citations),
    )
    # Bind to the model's own list: pydantic copies a list passed to the
    # constructor, so recording into a local would silently drop every event.
    events = decision.events
    # True once the answer has been replaced by one of this module's own canned
    # messages. Those are operator-authored text with no user or corpus content in
    # them, so scanning them for PII would only ever produce false positives.
    canned = False

    # ------------------------------------------------------- refusal quality
    decision.refusal = assess_refusal(answer, settings=resolved)
    if decision.refusal.is_refusal and not decision.refusal.acceptable:
        _record(
            events,
            GuardrailEvent(
                stage=GuardrailStage.OUTPUT.value,
                kind=GuardrailKind.GROUNDEDNESS.value,
                action=GuardrailAction.WARN.value,
                detail=(
                    "refusal does not tell the user what is covered or where to "
                    f"look next ({', '.join(decision.refusal.reasons)})"
                ),
                entities=list(decision.refusal.reasons),
            ),
        )

    # ------------------------------------------------------- classification
    decision.clearance = check_clearance(
        answer=answer,
        citations=citations,
        chunks=chunks,
        principal=principal,
        settings=resolved,
    )
    if not decision.clearance.ok:
        worst = decision.clearance.violations[0]
        enforce = resolved.guardrail_enforce_classification_on_output
        _log.error(
            "output_guard_clearance_violation",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            principal_clearance=Classification.from_rank(
                principal.clearance_rank()
            ).value,
            violations=[item.kind for item in decision.clearance.violations],
            chunk_ids=[item.chunk_id for item in decision.clearance.violations],
            document_ids=[item.document_id for item in decision.clearance.violations],
            enforced=enforce,
            hint="ACL filter bug: build_acl_filter should have excluded these chunks",
        )
        action = GuardrailAction.BLOCK.value if enforce else GuardrailAction.WARN.value
        _record(
            events,
            GuardrailEvent(
                stage=GuardrailStage.OUTPUT.value,
                kind=GuardrailKind.CLASSIFICATION.value,
                action=action,
                detail=(
                    f"{len(decision.clearance.violations)} classification violation(s)"
                    f"; worst: {worst.kind}. This indicates an upstream filter bug."
                ),
                entities=sorted(
                    {
                        violation.classification
                        for violation in decision.clearance.violations
                    }
                ),
            ),
        )
        if enforce:
            over_clearance_ids = {
                violation.chunk_id for violation in decision.clearance.violations
            }
            decision.dropped_citations = [
                citation
                for citation in decision.citations
                if citation.chunk_id in over_clearance_ids
            ]
            decision.citations = [
                citation
                for citation in decision.citations
                if citation.chunk_id not in over_clearance_ids
            ]
            decision.text = CLEARANCE_BLOCK_MESSAGE
            decision.blocked = True
            decision.action = GuardrailAction.BLOCK.value
            canned = True

    # -------------------------------------------------------------- pii egress
    pii_detector = detector or get_pii_detector(resolved)
    if resolved.pii_enabled and resolved.guardrail_output_pii_scan and not canned:
        raw_report = pii_detector.analyze(decision.text, language=resolved.pii_language)
        raw_report = await pii_detector.verify(decision.text, raw_report)
        egress = _egress_report(
            raw_report,
            list(resolved.guardrail_output_pii_ignore_entities),
        )
        decision.pii_types = list(egress.entity_types)
        decision.pii_redacted = True
        if egress.findings:
            if resolved.pii_block_on_egress and not decision.blocked:
                decision.text = PII_BLOCK_MESSAGE
                decision.blocked = True
                decision.action = GuardrailAction.BLOCK.value
                canned = True
                pii_action = GuardrailAction.BLOCK.value
            else:
                decision.text = pii_detector.redact(
                    decision.text, egress, mode=resolved.pii_redaction_mode
                )
                pii_action = GuardrailAction.REDACT.value
                if decision.action == GuardrailAction.ALLOW.value:
                    decision.action = GuardrailAction.REDACT.value
            _record(
                events,
                GuardrailEvent(
                    stage=GuardrailStage.OUTPUT.value,
                    kind=GuardrailKind.PII.value,
                    action=pii_action,
                    detail=(
                        f"{len(egress.findings)} personal-data span(s) found in the "
                        "generated answer on egress"
                    ),
                    entities=decision.pii_types,
                    score=egress.max_score,
                ),
            )
        else:
            events.append(
                GuardrailEvent.allow(
                    GuardrailStage.OUTPUT,
                    GuardrailKind.PII,
                    detail="no personal data in the emitted answer",
                )
            )

    # ------------------------------------------------------------ groundedness
    if not decision.blocked:
        score = (
            citation_validity_score(decision.text, decision.citations, chunks)
            if groundedness is None
            else max(0.0, min(1.0, groundedness))
        )
        applicable = bool(chunks) and not decision.refusal.is_refusal
        if applicable and groundedness is None and not _markers(decision.text):
            # Sources were retrieved and the answer asserts something, yet nothing is
            # attributed. That is the ungrounded case the vacuous 1.0 would hide.
            # Only applied to a self-computed score: when stage 11 supplied one it
            # already distinguished "no claims" from "uncited claims", and second-
            # guessing it here would block a legitimate one-line acknowledgement.
            score = 0.0
        decision.groundedness = score
        decision.groundedness_applicable = applicable

        if applicable:
            floor = float(resolved.guardrail_output_block_below_groundedness)
            if score < floor:
                decision.text = GROUNDEDNESS_BLOCK_MESSAGE
                decision.blocked = True
                decision.action = GuardrailAction.BLOCK.value
                canned = True
                _record(
                    events,
                    GuardrailEvent(
                        stage=GuardrailStage.OUTPUT.value,
                        kind=GuardrailKind.GROUNDEDNESS.value,
                        action=GuardrailAction.BLOCK.value,
                        detail=(
                            f"citation_validity {score:.2f} below the hard floor "
                            f"{floor:.2f}; answer withheld"
                        ),
                        score=score,
                    ),
                )
            elif score < resolved.guardrail_min_groundedness:
                from ragcore.llm.prompts import UNCERTAINTY_NOTICE

                # Idempotent: stage 11 exposes `append_uncertainty_notice` too, and
                # an answer carrying the notice twice reads as a malfunction.
                already = UNCERTAINTY_NOTICE in decision.text
                notice_state = "already present" if already else "appended"
                if not already:
                    decision.text = f"{decision.text.rstrip()}\n\n{UNCERTAINTY_NOTICE}"
                decision.uncertainty_appended = True
                if decision.action == GuardrailAction.ALLOW.value:
                    decision.action = GuardrailAction.WARN.value
                _record(
                    events,
                    GuardrailEvent(
                        stage=GuardrailStage.OUTPUT.value,
                        kind=GuardrailKind.GROUNDEDNESS.value,
                        action=GuardrailAction.WARN.value,
                        detail=(
                            f"citation_validity {score:.2f} below "
                            f"{resolved.guardrail_min_groundedness:.2f}; "
                            f"uncertainty notice {notice_state}"
                        ),
                        score=score,
                    ),
                )
            else:
                events.append(
                    GuardrailEvent.allow(
                        GuardrailStage.OUTPUT,
                        GuardrailKind.GROUNDEDNESS,
                        detail=f"citation_validity {score:.2f}",
                    )
                )

    # ------------------------------------------------- persistable redaction
    if resolved.pii_enabled and not canned:
        persist_report = pii_detector.analyze(
            decision.text, language=resolved.pii_language
        )
        decision.redacted_text = pii_detector.redact(
            decision.text, persist_report, mode=resolved.pii_redaction_mode
        )
        decision.pii_redacted = True
    else:
        decision.redacted_text = decision.text
        decision.pii_redacted = decision.pii_redacted or canned

    _log.info(
        "output_guard",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action=decision.action,
        blocked=decision.blocked,
        groundedness=decision.groundedness,
        groundedness_applicable=decision.groundedness_applicable,
        clearance_violations=len(decision.clearance.violations),
        pii_types=decision.pii_types,
        citations=len(decision.citations),
        dropped_citations=len(decision.dropped_citations),
    )
    return decision
