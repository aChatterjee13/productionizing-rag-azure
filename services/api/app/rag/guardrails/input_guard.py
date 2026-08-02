"""Input guard — pipeline stage 1.

Everything that has to happen to a user turn *before* it is logged, traced,
persisted, embedded or shown to a model:

* **Redact before you log.** The turn is PII-scanned and two texts come out.
  :attr:`InputDecision.redacted_text` is the only one that may be persisted or
  logged — it is what ``repositories.append_message(pii_redacted=True)`` wants.
  :attr:`InputDecision.text` is the prompt-safe form: the user's own words, so the
  question still means what they asked, with **credential-shaped** entities
  (``guardrail_input_credential_entities``) masked even there, because an API key
  or JWT pasted into a chat has no business reaching a model provider, a trace or a
  memory record.
* **Size cap.** ``guardrail_input_max_chars`` is enforced before anything expensive
  runs. The default is to refuse, not to silently truncate a question and answer
  the half that fit; ``guardrail_input_truncate`` flips that for operators who
  prefer a degraded answer to an error.
* **Unicode normalisation.** NFKC plus invisible-character stripping, before the
  injection scan, so a homoglyph or a zero-width-spaced ``i g n o r e`` does not
  walk past the detectors.
* **Language detection.** Dependency-free and deliberately cheap: language is a
  routing and filtering facet, not a correctness-critical value.
* **Injection heuristics** via :mod:`app.rag.guardrails.injection`.

The result is one :class:`InputDecision` carrying every
:class:`~ragcore.models.chat.GuardrailEvent` the turn produced, so the orchestrator
streams them and persists them without re-deriving anything.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.rag.guardrails.injection import (
    InjectionVerdict,
    sanitise_untrusted,
    scan_user_turn,
)
from ragcore.errors import GuardrailBlocked
from ragcore.models.acl import Principal
from ragcore.models.chat import (
    GuardrailAction,
    GuardrailEvent,
    GuardrailKind,
    GuardrailStage,
)
from ragcore.observability import observe_guardrail
from ragcore.pii import PIIDetector, PIIReport, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "EMPTY_INPUT_MESSAGE",
    "INJECTION_BLOCK_MESSAGE",
    "LANGUAGE_GUARDRAIL_KIND",
    "OVERSIZE_MESSAGE",
    "InputDecision",
    "detect_language",
    "run_input_guard",
]

_log = structlog.get_logger(__name__)

#: ``GuardrailEvent.kind`` for a language-policy notice. ``GuardrailEvent.kind`` is a
#: free-form string; this value extends the vocabulary in the same way
#: :class:`~ragcore.models.chat.GuardrailKind` already extends the contract with
#: ``size``. Registered in Addendum G of ``docs/CONTRACTS.md``.
LANGUAGE_GUARDRAIL_KIND = "language"

#: Shown when the turn is empty or shorter than ``guardrail_input_min_chars``.
EMPTY_INPUT_MESSAGE = (
    "I did not receive a question. Send the question you want answered from the "
    "indexed documents."
)

#: Shown when the turn exceeds ``guardrail_input_max_chars`` and truncation is off.
OVERSIZE_MESSAGE = (
    "That message is longer than this assistant accepts in one turn ({limit:,} "
    "characters; you sent {actual:,}). Send the specific question, or upload the "
    "long text as a document so it can be indexed and cited."
)

#: Shown when the injection score clears ``guardrail_injection_block_threshold``.
INJECTION_BLOCK_MESSAGE = (
    "I can't act on that message: it contains instructions aimed at changing how "
    "this assistant works rather than a question about the indexed documents. Ask "
    "the question directly and I'll answer it from the corpus."
)

#: Script ranges that identify a language without a word list. Ordered most
#: specific first; a script hit is far more reliable than a marker-word vote.
_SCRIPT_RANGES: tuple[tuple[str, str, str], ...] = (
    ("ja", "぀", "ヿ"),
    ("ko", "가", "힯"),
    ("zh", "一", "鿿"),
    ("ru", "Ѐ", "ӿ"),
    ("el", "Ͱ", "Ͽ"),
    ("he", "֐", "׿"),
    ("ar", "؀", "ۿ"),
    ("hi", "ऀ", "ॿ"),
    ("th", "฀", "๿"),
)

#: High-frequency function words per Latin-script language. Function words are the
#: right signal here: they are short, extremely common, and largely disjoint across
#: languages, so a handful of them beats any amount of content vocabulary.
_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "the",
            "and",
            "is",
            "of",
            "to",
            "in",
            "for",
            "what",
            "how",
            "our",
            "with",
            "policy",
        }
    ),
    "de": frozenset(
        {
            "der",
            "die",
            "das",
            "und",
            "ist",
            "nicht",
            "mit",
            "für",
            "wie",
            "wir",
            "eine",
            "auf",
        }
    ),
    "fr": frozenset(
        {
            "le",
            "la",
            "les",
            "et",
            "est",
            "pour",
            "dans",
            "une",
            "que",
            "nous",
            "avec",
            "sur",
        }
    ),
    "es": frozenset(
        {
            "el",
            "los",
            "las",
            "y",
            "es",
            "para",
            "con",
            "una",
            "que",
            "por",
            "como",
            "del",
        }
    ),
    "it": frozenset(
        {
            "il",
            "lo",
            "gli",
            "e",
            "è",
            "per",
            "con",
            "una",
            "che",
            "non",
            "come",
            "del",
        }
    ),
    "pt": frozenset(
        {
            "o",
            "os",
            "as",
            "e",
            "é",
            "para",
            "com",
            "uma",
            "que",
            "não",
            "como",
            "dos",
        }
    ),
    "nl": frozenset(
        {
            "de",
            "het",
            "een",
            "en",
            "is",
            "van",
            "voor",
            "met",
            "niet",
            "wij",
            "hoe",
            "wat",
        }
    ),
}

_WORD_RE = re.compile(r"[a-zà-öø-ÿ]+")

#: Characters inspected by the language detector. A long turn does not make the
#: guess better, and scanning all of it is wasted work.
_LANGUAGE_SAMPLE_CHARS = 2_000


class InputDecision(BaseModel):
    """Outcome of stage 1 for one user turn.

    Attributes are split deliberately: ``text`` is what may go into a prompt and
    ``redacted_text`` is what may be written down. Confusing the two is exactly the
    mistake the "never log or persist raw user content" rule exists to prevent.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool = Field(
        default=True, description="False when the pipeline must not continue."
    )
    action: str = Field(
        default=GuardrailAction.ALLOW.value,
        description="Strongest action taken: allow | redact | block | warn | clarify.",
    )
    text: str = Field(
        default="",
        description=(
            "Prompt-safe turn: normalised, size-capped, with credential-shaped "
            "entities masked. Never persist this — persist redacted_text."
        ),
    )
    redacted_text: str = Field(
        default="",
        description=(
            "Fully PII-redacted turn. The only form that may be logged, traced or "
            "written to the database."
        ),
    )
    refusal: str = Field(
        default="",
        description=(
            "User-facing message when the turn was refused. Empty when allowed."
        ),
    )
    original_chars: int = Field(
        default=0, ge=0, description="Length of the turn as received."
    )
    truncated: bool = Field(
        default=False, description="True when the size cap truncated the turn."
    )
    language: str = Field(default="en", description="Detected ISO 639-1 language code.")
    language_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the language guess."
    )
    pii_types: list[str] = Field(
        default_factory=list,
        description="Entity types found in the turn. Types only, never values.",
    )
    pii_redacted: bool = Field(
        default=False,
        description=(
            "Assertion that redaction ran, in the sense repositories.append_message "
            "requires. True whenever the PII pass completed, with or without hits."
        ),
    )
    credential_types: list[str] = Field(
        default_factory=list,
        description="Credential-shaped entities masked out of the prompt text too.",
    )
    injection: InjectionVerdict | None = Field(
        default=None, description="Injection verdict, when the scan ran."
    )
    events: list[GuardrailEvent] = Field(
        default_factory=list, description="Everything stage 1 decided, in order."
    )

    @property
    def blocked(self) -> bool:
        """Whether the turn was refused.

        Returns:
            True when the pipeline must stop and emit :attr:`refusal`.
        """
        return not self.allowed

    def raise_if_blocked(self) -> None:
        """Raise the contract's guardrail error when the turn was refused.

        Callers that would rather short-circuit with an exception than branch on
        :attr:`allowed` use this; the orchestrator branches, because a refusal is a
        normal answer with a ``guardrail`` SSE event, not an error.

        Raises:
            GuardrailBlocked: When :attr:`allowed` is False.
        """
        if self.allowed:
            return
        blocking = next(
            (event for event in self.events if event.blocked),
            None,
        )
        kind = blocking.kind if blocking else GuardrailKind.SIZE.value
        raise GuardrailBlocked(
            self.refusal or "input guard refused the turn",
            stage="input",
            kind=kind,
            entities=list(blocking.entities) if blocking else [],
        )


def detect_language(text: str, *, default: str = "en") -> tuple[str, float]:
    """Guess the language of a passage without an external dependency.

    Non-Latin scripts are identified by codepoint range, which is near-certain;
    Latin-script languages by a function-word vote, which is not, hence the returned
    confidence. Language drives a retrieval facet and the answer's language, so a
    wrong guess costs recall, never correctness — that is why this is fifty lines
    rather than a model call.

    Args:
        text: The passage to inspect. Only the first two thousand characters are
            considered.
        default: Code returned when detection is inconclusive.

    Returns:
        An ``(iso_639_1_code, confidence)`` pair. Confidence is the winning
        language's share of all marker-word hits, or ``1.0`` for a script match, and
        ``0.0`` when the guess is just ``default``.
    """
    sample = text[:_LANGUAGE_SAMPLE_CHARS]
    if not sample.strip():
        return default, 0.0

    for code, low, high in _SCRIPT_RANGES:
        hits = sum(1 for char in sample if low <= char <= high)
        if hits >= max(6, len(sample) // 50):
            return code, 1.0

    words = Counter(_WORD_RE.findall(sample.lower()))
    if not words:
        return default, 0.0

    scores = {
        code: sum(words[marker] for marker in markers)
        for code, markers in _LANGUAGE_MARKERS.items()
    }
    total = sum(scores.values())
    if not total:
        return default, 0.0
    best = max(scores, key=lambda code: scores[code])
    return best, round(scores[best] / total, 3)


def _credential_report(report: PIIReport, entity_types: list[str]) -> PIIReport:
    """Narrow a PII report to the entity types that must never reach a model.

    Args:
        report: The full report from the detector.
        entity_types: Entity type names treated as credentials.

    Returns:
        A report containing only the matching findings.
    """
    wanted = {name.upper() for name in entity_types}
    return PIIReport.from_findings(
        [finding for finding in report.findings if finding.entity_type in wanted]
    )


def _record(events: list[GuardrailEvent], event: GuardrailEvent) -> GuardrailEvent:
    """Append an event and mirror it into the Prometheus counter.

    Args:
        events: The decision's event list.
        event: The event to record.

    Returns:
        The same event, for chaining.
    """
    events.append(event)
    observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)
    return event


async def run_input_guard(
    message: str,
    *,
    principal: Principal,
    settings: Settings | None = None,
    detector: PIIDetector | None = None,
) -> InputDecision:
    """Run stage 1 over one user turn.

    Order matters and is not negotiable: normalise, then cap, then redact, then
    scan. Normalising first is what stops a homoglyph evading the detectors; capping
    before redaction bounds the work an oversized turn can cause; redacting before
    the injection scan means every log line and trace this function emits is already
    safe.

    Args:
        message: The raw user turn as it arrived over HTTP.
        principal: Resolved caller. Used for tenant-scoped audit logging only — this
            function makes no access-control decision.
        settings: Process settings. Defaults to
            :func:`~ragcore.settings.get_settings`.
        detector: PII detector override, mainly for tests.

    Returns:
        An :class:`InputDecision`. Callers must use :attr:`InputDecision.text` for
        prompting and :attr:`InputDecision.redacted_text` for anything persisted.
    """
    resolved = settings or get_settings()
    original_chars = len(message)

    text = message
    if resolved.guardrail_input_normalise_unicode:
        text = sanitise_untrusted(unicodedata.normalize("NFKC", message))
    text = text.strip()

    decision = InputDecision(
        text=text,
        redacted_text=text,
        original_chars=original_chars,
    )
    # Bind to the model's own list: pydantic copies a list passed to the
    # constructor, so recording into a local would silently drop every event.
    events = decision.events

    min_chars = int(resolved.guardrail_input_min_chars)
    if len(text) < min_chars:
        decision.allowed = False
        decision.action = GuardrailAction.CLARIFY.value
        decision.refusal = EMPTY_INPUT_MESSAGE
        _record(
            events,
            GuardrailEvent(
                stage="input",
                kind=GuardrailKind.SIZE.value,
                action=GuardrailAction.CLARIFY.value,
                detail=f"turn shorter than the {min_chars}-character minimum",
            ),
        )
        return decision

    limit = resolved.guardrail_input_max_chars
    if len(text) > limit:
        if resolved.guardrail_input_truncate:
            text = text[:limit]
            decision.truncated = True
            decision.action = GuardrailAction.WARN.value
            _record(
                events,
                GuardrailEvent(
                    stage="input",
                    kind=GuardrailKind.SIZE.value,
                    action=GuardrailAction.WARN.value,
                    detail=(
                        f"turn truncated from {original_chars:,} to {limit:,} "
                        "characters"
                    ),
                ),
            )
        else:
            decision.allowed = False
            decision.action = GuardrailAction.BLOCK.value
            decision.refusal = OVERSIZE_MESSAGE.format(
                limit=limit, actual=original_chars
            )
            decision.text = ""
            decision.redacted_text = ""
            _record(
                events,
                GuardrailEvent(
                    stage="input",
                    kind=GuardrailKind.SIZE.value,
                    action=GuardrailAction.BLOCK.value,
                    detail=(
                        f"turn of {original_chars:,} characters exceeds the "
                        f"{limit:,}-character cap"
                    ),
                ),
            )
            _log.info(
                "input_guard_oversize",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                chars=original_chars,
                limit=limit,
            )
            return decision

    # ---------------------------------------------------------------- pii
    pii_detector = detector or get_pii_detector(resolved)
    prompt_text = text
    redacted_text = text
    pii_types: list[str] = []
    credential_types: list[str] = []

    if resolved.pii_enabled:
        report = pii_detector.analyze(text, language=resolved.pii_language)
        report = await pii_detector.verify(text, report)
        redacted_text = pii_detector.redact(
            text, report, mode=resolved.pii_redaction_mode
        )
        pii_types = list(report.entity_types)
        decision.pii_redacted = True

        credential_entities = list(resolved.guardrail_input_credential_entities)
        credentials = _credential_report(report, credential_entities)
        if credentials.findings:
            prompt_text = pii_detector.redact(text, credentials, mode="mask")
            credential_types = list(credentials.entity_types)

        if pii_types:
            decision.action = GuardrailAction.REDACT.value
            _record(
                events,
                GuardrailEvent(
                    stage="input",
                    kind=GuardrailKind.PII.value,
                    action=GuardrailAction.REDACT.value,
                    detail=(
                        f"{len(report.findings)} personal-data span(s) redacted "
                        "before logging or persistence"
                    ),
                    entities=pii_types,
                    score=report.max_score,
                ),
            )
        else:
            events.append(
                GuardrailEvent.allow(
                    GuardrailStage.INPUT,
                    GuardrailKind.PII,
                    detail="no personal data detected in the turn",
                )
            )

    decision.text = prompt_text
    decision.redacted_text = redacted_text
    decision.pii_types = pii_types
    decision.credential_types = credential_types

    # ----------------------------------------------------------- language
    if resolved.guardrail_language_detect_enabled:
        language, confidence = detect_language(text, default=resolved.pii_language)
        decision.language = language
        decision.language_confidence = confidence
        allowed_languages = list(resolved.guardrail_allowed_languages)
        floor = float(resolved.guardrail_language_min_confidence)
        if (
            allowed_languages
            and confidence >= floor
            and language not in allowed_languages
        ):
            decision.action = GuardrailAction.WARN.value
            _record(
                events,
                GuardrailEvent(
                    stage="input",
                    kind=LANGUAGE_GUARDRAIL_KIND,
                    action=GuardrailAction.WARN.value,
                    detail=(
                        f"detected language '{language}' is outside the configured "
                        "set; answering anyway"
                    ),
                    entities=[language],
                    score=confidence,
                ),
            )

    # ---------------------------------------------------------- injection
    verdict = await scan_user_turn(prompt_text, settings=resolved)
    decision.injection = verdict
    if verdict.flagged:
        _record(events, verdict.to_event(stage="input"))
        if verdict.blocked:
            decision.allowed = False
            decision.action = GuardrailAction.BLOCK.value
            decision.refusal = INJECTION_BLOCK_MESSAGE
            _log.warning(
                "input_guard_injection_blocked",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                score=verdict.score,
                signals=[signal.name for signal in verdict.signals],
            )
        elif decision.action == GuardrailAction.ALLOW.value:
            decision.action = GuardrailAction.WARN.value

    _log.info(
        "input_guard",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        chars=original_chars,
        truncated=decision.truncated,
        language=decision.language,
        pii_types=pii_types,
        injection_score=verdict.score,
        action=decision.action,
        allowed=decision.allowed,
    )
    return decision
