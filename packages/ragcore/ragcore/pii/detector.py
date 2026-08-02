"""PII detection and redaction, with Presidio as an optional accelerator.

:class:`PIIDetector` is the single entry point used by the ingest pipeline
(stage: document scanning) and by the chat pipeline (stage 1 input guard, stage
12 output guard). It works in two modes:

* **Presidio mode** -- ``presidio-analyzer`` is importable and
  ``pii_use_presidio`` is on. The analyzer supplies NLP-backed entities (PERSON,
  LOCATION, DATE_TIME) and the custom recognisers from
  :mod:`ragcore.pii.recognizers` are registered into it.
* **Regex mode** -- the optional extra is absent, or the analyzer failed to
  build (a missing spaCy model, for example). The full regex set runs instead and
  a warning is logged once. The package always imports either way.

Redaction is implemented in-process for all three modes rather than through
Presidio's ``AnonymizerEngine``, so a redacted string is byte-identical whether
or not the optional extra is installed -- which matters because ``hash`` mode
output is a join key, and ``pii_redacted=True`` is an assertion the database
enforces.

Reports are safe to log and persist: :attr:`PIIFinding.snippet` is a
partially-masked preview, never the matched value. Redaction itself works from
offsets against the original text, so nothing is lost by masking the preview.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragcore.logging import get_logger
from ragcore.pii.recognizers import (
    REGEX_RECOGNIZERS,
    RegexMatch,
    build_presidio_recognizers,
    scan,
)
from ragcore.settings import Settings, get_settings

__all__ = [
    "PIIDetector",
    "PIIFinding",
    "PIIReport",
    "PIIVerdict",
    "PIIVerificationResult",
    "get_pii_detector",
    "reset_pii_detector_cache",
]

_log = get_logger(__name__)

#: Maximum characters kept in a masked snippet preview.
_MAX_SNIPPET_CHARS = 48
#: Characters of surrounding text sent to the LLM verifier per candidate.
_VERIFY_CONTEXT_CHARS = 60
#: Length of the HMAC prefix used as a stable pseudonym.
_HASH_TOKEN_CHARS = 16

_REDACTION_MODES = ("mask", "hash", "partial")


class PIIFinding(BaseModel):
    """One detected entity.

    Attributes:
        entity_type: Presidio-style entity name, e.g. ``"EMAIL_ADDRESS"``.
        start: Inclusive start offset in the analysed text.
        end: Exclusive end offset in the analysed text.
        score: Detector confidence in 0..1.
        snippet: Partially-masked preview of the matched value. Never the raw
            value, so a report can be logged or persisted as-is.
    """

    model_config = ConfigDict(extra="forbid")

    entity_type: str
    start: int
    end: int
    score: float
    snippet: str = ""

    @property
    def length(self) -> int:
        """Character length of the matched span.

        Returns:
            ``end - start``.
        """
        return self.end - self.start


class PIIReport(BaseModel):
    """The outcome of scanning one text.

    Attributes:
        findings: Non-overlapping findings, ordered by start offset.
        entity_types: Distinct entity types found, sorted.
        has_pii: Whether anything was found.
        max_score: Highest finding score, or 0.0.
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[PIIFinding] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    has_pii: bool = False
    max_score: float = 0.0

    @classmethod
    def empty(cls) -> PIIReport:
        """Build an empty report.

        Returns:
            A report with no findings.
        """
        return cls()

    @classmethod
    def from_findings(cls, findings: list[PIIFinding]) -> PIIReport:
        """Build a report from findings, deriving the summary fields.

        Args:
            findings: Findings, in any order.

        Returns:
            A report with ``entity_types``, ``has_pii`` and ``max_score`` set.
        """
        ordered = sorted(findings, key=lambda f: (f.start, f.end))
        return cls(
            findings=ordered,
            entity_types=sorted({f.entity_type for f in ordered}),
            has_pii=bool(ordered),
            max_score=max((f.score for f in ordered), default=0.0),
        )

    def spans(self) -> list[tuple[int, int]]:
        """Return the matched spans.

        Returns:
            ``(start, end)`` pairs in document order.
        """
        return [(f.start, f.end) for f in self.findings]


class PIIVerdict(BaseModel):
    """One LLM verdict on a candidate finding.

    Attributes:
        index: Index of the candidate as presented to the model.
        is_pii: Whether the candidate really is that entity type here.
        confidence: The model's confidence in 0..1.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    is_pii: bool
    confidence: float = 1.0


class PIIVerificationResult(BaseModel):
    """The structured response of the LLM verification pass.

    Attributes:
        verdicts: One verdict per presented candidate.
    """

    model_config = ConfigDict(extra="forbid")

    verdicts: list[PIIVerdict] = Field(default_factory=list)


def _mask_preview(value: str, keep: int) -> str:
    """Build a loggable preview of a matched value.

    Args:
        value: The raw matched text.
        keep: Trailing characters to preserve.

    Returns:
        A masked preview, truncated so a long match cannot bloat a log line.
    """
    trimmed = value[:_MAX_SNIPPET_CHARS]
    if keep <= 0 or len(trimmed) <= keep:
        return "*" * len(trimmed)
    return "*" * (len(trimmed) - keep) + trimmed[-keep:]


def _merge_overlaps(matches: list[RegexMatch]) -> list[RegexMatch]:
    """Drop overlapping matches, keeping the most confident.

    Ties break towards the longer span, so ``sk-ant-...`` detected by both the
    prefix recogniser and the keyword recogniser is reported once.

    Args:
        matches: Candidate matches.

    Returns:
        Non-overlapping matches in document order.
    """
    ranked = sorted(matches, key=lambda m: (-m.score, -(m.end - m.start), m.start))
    kept: list[RegexMatch] = []
    for candidate in ranked:
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda m: (m.start, m.end))


class PIIDetector:
    """Detect and redact personal data in text."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the detector and, when possible, the Presidio analyzer.

        Args:
            settings: Platform settings. Defaults to the process settings.
        """
        self._settings = settings or get_settings()
        self._analyzer = (
            _build_analyzer(self._settings)
            if self._settings.pii_enabled and self._settings.pii_use_presidio
            else None
        )
        self._allowed = frozenset(self._settings.pii_entities)

    @property
    def presidio_enabled(self) -> bool:
        """Whether the Presidio analyzer is in use.

        Returns:
            True in Presidio mode, False in regex-only mode.
        """
        return self._analyzer is not None

    def analyze(self, text: str, *, language: str = "en") -> PIIReport:
        """Scan text for personal data.

        Args:
            text: Text to scan.
            language: Language code passed to the analyzer.

        Returns:
            A :class:`PIIReport`. Findings below ``pii_score_threshold`` and
            entity types outside ``pii_entities`` are dropped, and overlapping
            findings are merged.
        """
        if not text or not self._settings.pii_enabled:
            return PIIReport.empty()

        matches = self._collect(text, language=language)
        threshold = self._settings.pii_score_threshold
        filtered = [
            match
            for match in matches
            if match.score >= threshold and match.entity_type in self._allowed
        ]
        keep = self._settings.pii_partial_keep_chars
        findings = [
            PIIFinding(
                entity_type=match.entity_type,
                start=match.start,
                end=match.end,
                score=round(match.score, 4),
                snippet=_mask_preview(match.value, keep),
            )
            for match in _merge_overlaps(filtered)
        ]
        return PIIReport.from_findings(findings)

    def _collect(self, text: str, *, language: str) -> list[RegexMatch]:
        """Run the configured engine over text.

        Args:
            text: Text to scan.
            language: Language code for the analyzer.

        Returns:
            Raw matches, unfiltered.
        """
        if self._analyzer is not None:
            try:
                results = self._analyzer.analyze(
                    text=text,
                    language=language or self._settings.pii_language,
                    score_threshold=0.0,
                )
            except Exception:
                _log.warning("presidio_analyze_failed", exc_info=True)
            else:
                return [
                    RegexMatch(
                        entity_type=str(result.entity_type),
                        start=int(result.start),
                        end=int(result.end),
                        score=float(result.score),
                        value=text[int(result.start) : int(result.end)],
                    )
                    for result in results
                ]
        return scan(text, recognizers=REGEX_RECOGNIZERS)

    def redact(self, text: str, report: PIIReport, *, mode: str = "mask") -> str:
        """Replace every finding in ``text``.

        Args:
            text: The **same** text the report was produced from. Offsets are
                applied literally, so passing different text corrupts the output.
            report: Findings to redact.
            mode: ``mask`` -> ``<EMAIL_ADDRESS>``; ``hash`` -> a stable HMAC
                pseudonym so redacted values still join across documents;
                ``partial`` -> keep the last ``pii_partial_keep_chars``
                characters.

        Returns:
            The redacted text.

        Raises:
            ValueError: If ``mode`` is not one of the three supported modes.
        """
        if mode not in _REDACTION_MODES:
            msg = f"unknown redaction mode {mode!r}; expected one of {_REDACTION_MODES}"
            raise ValueError(msg)
        if not report.findings or not text:
            return text

        result = text
        for finding in sorted(report.findings, key=lambda f: f.start, reverse=True):
            start = max(0, min(finding.start, len(result)))
            end = max(start, min(finding.end, len(result)))
            original = result[start:end]
            replacement = self._replacement(original, finding, mode)
            result = result[:start] + replacement + result[end:]
        return result

    def scan_and_redact(
        self, text: str, *, mode: str | None = None, language: str = "en"
    ) -> tuple[str, PIIReport]:
        """Scan and redact in one call, using the configured default mode.

        This is the shape the pipeline's redact-before-log rule wants: the
        redacted string is what gets logged or persisted, and the report is what
        gets attached to a :class:`ragcore.models.chat.GuardrailEvent`.

        Args:
            text: Text to scan.
            mode: Redaction mode. Defaults to ``pii_redaction_mode``.
            language: Language code for the analyzer.

        Returns:
            A ``(redacted_text, report)`` pair.
        """
        report = self.analyze(text, language=language)
        chosen = mode or self._settings.pii_redaction_mode
        return self.redact(text, report, mode=chosen), report

    def _replacement(self, original: str, finding: PIIFinding, mode: str) -> str:
        """Build the replacement string for one finding.

        Args:
            original: The matched text.
            finding: The finding being redacted.
            mode: Redaction mode.

        Returns:
            The replacement text.
        """
        if mode == "mask":
            return f"<{finding.entity_type}>"
        if mode == "hash":
            digest = hmac.new(
                self._settings.pii_hash_secret.encode("utf-8"),
                original.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:_HASH_TOKEN_CHARS]
            return f"<{finding.entity_type}:{digest}>"
        keep = self._settings.pii_partial_keep_chars
        if keep <= 0 or len(original) <= keep:
            return "*" * len(original)
        return "*" * (len(original) - keep) + original[-keep:]

    def pseudonym(self, value: str, *, entity_type: str = "VALUE") -> str:
        """Return the stable ``hash``-mode token for a value.

        Useful for joining an already-redacted corpus against a new document
        without redacting the whole document first.

        Args:
            value: The raw value.
            entity_type: Entity label to embed in the token.

        Returns:
            The same token :meth:`redact` would produce in ``hash`` mode.
        """
        digest = hmac.new(
            self._settings.pii_hash_secret.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:_HASH_TOKEN_CHARS]
        return f"<{entity_type}:{digest}>"

    async def verify(self, text: str, report: PIIReport) -> PIIReport:
        """Second-pass LLM verification of borderline findings.

        Disabled by default. When ``pii_llm_verify_enabled`` is on, findings at
        or above ``pii_llm_verify_min_score`` are sent to ``MODEL_CHEAP``, which
        rejects placeholders, example values and wrong-kind identifiers. Findings
        below the threshold are kept untouched, and any failure keeps the whole
        report as-is: verification may only ever *reduce* false positives, never
        drop protection because a model call failed.

        Args:
            text: The text the report was produced from.
            report: The report to verify.

        Returns:
            A report with rejected findings removed.
        """
        if not self._settings.pii_llm_verify_enabled or not report.findings:
            return report

        floor = self._settings.pii_llm_verify_min_score
        candidates = [f for f in report.findings if f.score >= floor]
        if not candidates:
            return report

        from ragcore.llm.client import get_llm_client
        from ragcore.llm.prompts import PII_VERIFICATION_SYSTEM, prompt_metadata

        lines: list[str] = []
        for index, finding in enumerate(candidates):
            window_start = max(0, finding.start - _VERIFY_CONTEXT_CHARS)
            window_end = min(len(text), finding.end + _VERIFY_CONTEXT_CHARS)
            lines.append(
                f"[{index}] type={finding.entity_type} "
                f"value={text[finding.start : finding.end]!r} "
                f"context={text[window_start:window_end]!r}"
            )
        user_turn = "<candidates>\n" + "\n".join(lines) + "\n</candidates>"

        try:
            client = get_llm_client(self._settings)
            verdicts = await client.structured(
                system=PII_VERIFICATION_SYSTEM,
                messages=[{"role": "user", "content": user_turn}],
                schema=PIIVerificationResult,
                model=self._settings.anthropic_model_cheap,
                effort=self._settings.anthropic_effort_cheap,
                thinking=False,
                name="pii.verify",
                metadata={
                    **prompt_metadata("pii_verification"),
                    "candidates": len(candidates),
                },
            )
        except Exception:
            _log.warning("pii_verify_failed", candidates=len(candidates), exc_info=True)
            return report

        rejected = {
            verdict.index
            for verdict in verdicts.verdicts
            if not verdict.is_pii and 0 <= verdict.index < len(candidates)
        }
        if not rejected:
            return report

        dropped = {
            (
                candidates[index].start,
                candidates[index].end,
                candidates[index].entity_type,
            )
            for index in rejected
        }
        kept = [
            f for f in report.findings if (f.start, f.end, f.entity_type) not in dropped
        ]
        _log.info(
            "pii_verify_rejected",
            rejected=len(rejected),
            kept=len(kept),
            entity_types=sorted({candidates[i].entity_type for i in rejected}),
        )
        return PIIReport.from_findings(kept)


def _build_analyzer(settings: Settings) -> Any:
    """Construct a Presidio analyzer with the custom recognisers registered.

    Args:
        settings: Settings supplying the default language.

    Returns:
        An ``AnalyzerEngine``, or None when Presidio is unavailable or failed to
        initialise (a missing spaCy model, for instance).
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        _log.warning(
            "presidio_unavailable",
            detail="falling back to the regex-only recogniser set",
        )
        return None
    try:
        engine = AnalyzerEngine()
        for recognizer in build_presidio_recognizers(language=settings.pii_language):
            engine.registry.add_recognizer(recognizer)
    except Exception:
        _log.warning("presidio_init_failed", exc_info=True)
        return None
    return engine


_DETECTORS: dict[tuple[Any, ...], PIIDetector] = {}


def get_pii_detector(settings: Settings | None = None) -> PIIDetector:
    """Return the process-wide detector for these settings.

    Building the Presidio analyzer loads an NLP model, so the instance is cached.

    Args:
        settings: Settings to build from. Defaults to the process settings.

    Returns:
        A cached :class:`PIIDetector`.
    """
    cfg = settings or get_settings()
    key: tuple[Any, ...] = (
        cfg.pii_enabled,
        cfg.pii_use_presidio,
        cfg.pii_language,
        cfg.pii_score_threshold,
        cfg.pii_redaction_mode,
        cfg.pii_partial_keep_chars,
        cfg.pii_hash_secret,
        tuple(cfg.pii_entities),
    )
    detector = _DETECTORS.get(key)
    if detector is None:
        detector = PIIDetector(cfg)
        _DETECTORS[key] = detector
    return detector


def reset_pii_detector_cache() -> None:
    """Drop cached detectors. Tests call this after mutating settings."""
    _DETECTORS.clear()
