"""Regex recognisers and checksum validators for PII detection.

Two roles:

* When Presidio is installed, :func:`build_presidio_recognizers` registers the
  entity types Presidio does not ship: Aadhaar, Indian PAN, SWIFT/BIC, generic
  API keys and JWTs.
* When Presidio is **not** installed, :func:`scan` runs the whole set --
  including the baseline recognisers for email, phone, IP, SSN, credit card and
  IBAN that Presidio would otherwise cover -- so :class:`ragcore.pii.PIIDetector`
  still works and the package still imports.

Precision comes from three mechanisms rather than from ever-longer patterns:

* **checksums** -- Luhn for card numbers, ISO 7064 mod-97 for IBAN, Verhoeff for
  Aadhaar, and a base64url header parse for JWTs. A pattern match that fails its
  checksum is dropped, not down-weighted.
* **context** -- nearby keywords raise the score; recognisers whose pattern is
  intrinsically ambiguous (an eight-letter uppercase word looks exactly like a
  BIC) require context and are dropped without it.
* **scores** -- every match carries a score so the detector can apply
  ``pii_score_threshold`` and route borderline findings to LLM verification.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BASELINE_RECOGNIZERS",
    "CONTEXT_BOOST",
    "CONTEXT_WINDOW",
    "CUSTOM_RECOGNIZERS",
    "REGEX_RECOGNIZERS",
    "RegexMatch",
    "RegexRecognizer",
    "build_presidio_recognizers",
    "iban_check",
    "jwt_check",
    "luhn_check",
    "scan",
    "verhoeff_check",
    "verhoeff_checksum",
]

#: Characters either side of a match that are searched for context keywords.
CONTEXT_WINDOW = 64
#: Score added when a context keyword is found near a match.
CONTEXT_BOOST = 0.15

_DIGITS = re.compile(r"\d")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")

_MIN_CARD_DIGITS = 13
_MAX_CARD_DIGITS = 19
_MIN_IBAN_CHARS = 15
_MAX_IBAN_CHARS = 34
_AADHAAR_DIGITS = 12
_JWT_PARTS = 3
_MAX_TRIM_RETRIES = 4

# Verhoeff tables (ISO 7064-style dihedral group D5), used for Aadhaar.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _digits_of(value: str) -> str:
    """Extract the digits from a formatted number.

    Args:
        value: Raw matched text.

    Returns:
        Only the digit characters, in order.
    """
    return "".join(_DIGITS.findall(value))


def luhn_check(value: str) -> bool:
    """Validate a payment card number with the Luhn algorithm.

    Args:
        value: Card number, with or without spaces and hyphens.

    Returns:
        True when the length is 13-19 digits and the Luhn checksum passes.
    """
    digits = _digits_of(value)
    if not _MIN_CARD_DIGITS <= len(digits) <= _MAX_CARD_DIGITS:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_check(value: str) -> bool:
    """Validate an IBAN with the ISO 7064 mod-97 checksum.

    Args:
        value: IBAN, with or without grouping spaces.

    Returns:
        True when the length and mod-97 remainder are both valid.
    """
    normalized = _NON_ALNUM.sub("", value).upper()
    if not _MIN_IBAN_CHARS <= len(normalized) <= _MAX_IBAN_CHARS:
        return False
    if not normalized[:2].isalpha() or not normalized[2:4].isdigit():
        return False
    rearranged = normalized[4:] + normalized[:4]
    numeric = "".join(
        str(ord(char) - ord("A") + 10) if char.isalpha() else char
        for char in rearranged
    )
    if not numeric.isdigit():
        return False
    return int(numeric) % 97 == 1


def verhoeff_check(value: str) -> bool:
    """Validate a number whose last digit is a Verhoeff check digit.

    Args:
        value: Digit string, with or without separators.

    Returns:
        True when the Verhoeff checksum passes.
    """
    digits = _digits_of(value)
    if not digits:
        return False
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(char)]]
    return checksum == 0


def verhoeff_checksum(value: str) -> int:
    """Compute the Verhoeff check digit for a number that lacks one.

    Args:
        value: Digit string without its check digit.

    Returns:
        The check digit to append.
    """
    digits = _digits_of(value)
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[(index + 1) % 8][int(char)]]
    return _VERHOEFF_INV[checksum]


def _aadhaar_check(value: str) -> bool:
    """Validate an Aadhaar number: 12 digits, first not 0 or 1, Verhoeff valid.

    Args:
        value: Candidate Aadhaar number.

    Returns:
        True when the candidate is a plausible Aadhaar number.
    """
    digits = _digits_of(value)
    if len(digits) != _AADHAAR_DIGITS or digits[0] in {"0", "1"}:
        return False
    return verhoeff_check(digits)


def jwt_check(value: str) -> bool:
    """Validate that a token really is a JWT by decoding its header.

    Args:
        value: Candidate token.

    Returns:
        True when the value has three dot-separated parts and the first decodes
        to a JSON object carrying an ``alg`` claim.
    """
    parts = value.split(".")
    if len(parts) != _JWT_PARTS or not parts[0]:
        return False
    padded = parts[0] + "=" * (-len(parts[0]) % 4)
    try:
        header = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, TypeError):
        return False
    return isinstance(header, dict) and "alg" in header


@dataclass(frozen=True, slots=True)
class RegexMatch:
    """One recogniser hit.

    Attributes:
        entity_type: Presidio-style entity name, e.g. ``"CREDIT_CARD"``.
        start: Inclusive start offset in the scanned text.
        end: Exclusive end offset in the scanned text.
        score: Confidence in 0..1, context boost included.
        value: The matched substring. Callers must not log or persist it.
    """

    entity_type: str
    start: int
    end: int
    score: float
    value: str


@dataclass(frozen=True, slots=True)
class RegexRecognizer:
    """A pattern, an optional checksum, and optional context keywords.

    Attributes:
        entity_type: Entity name reported for a match.
        pattern: Compiled pattern.
        score: Base confidence before any context boost.
        validator: Checksum or structural validator. A match that fails is
            discarded outright.
        context: Keywords that, when found nearby, raise confidence.
        require_context: Drop matches with no nearby context keyword. Used for
            patterns that are intrinsically ambiguous.
        group: Capture group carrying the entity, when the pattern also matches a
            surrounding keyword.
        trim_retry: Re-run the validator on progressively shorter prefixes when
            the full match fails it. Regex matching is greedy and knows nothing
            about checksums, so ``4111 1111 1111 1111 2024`` is matched whole and
            fails Luhn; trimming the trailing group recovers the real card.
        name: Recogniser name, for Presidio registration and debugging.
    """

    entity_type: str
    pattern: re.Pattern[str]
    score: float = 0.6
    validator: Callable[[str], bool] | None = None
    context: tuple[str, ...] = ()
    require_context: bool = False
    group: int = 0
    trim_retry: bool = False
    name: str = ""

    @property
    def recognizer_name(self) -> str:
        """Stable name for this recogniser.

        Returns:
            ``name`` when set, else a name derived from the entity type.
        """
        return self.name or f"ragcore_{self.entity_type.lower()}"

    def _validated(self, value: str) -> str | None:
        """Return the longest prefix of ``value`` that passes the validator.

        Args:
            value: The raw matched text.

        Returns:
            The accepted text, or None when nothing passed.
        """
        if self.validator is None:
            return value
        if self.validator(value):
            return value
        if not self.trim_retry:
            return None
        current = value
        for _ in range(_MAX_TRIM_RETRIES):
            cut = max(current.rfind(" "), current.rfind("-"))
            if cut <= 0:
                return None
            current = current[:cut]
            if self.validator(current):
                return current
        return None

    def find(self, text: str) -> list[RegexMatch]:
        """Scan text for this entity.

        Args:
            text: Text to scan.

        Returns:
            Matches that passed the validator and the context requirement.
        """
        if not text:
            return []
        lowered = text.lower() if self.context else ""
        found: list[RegexMatch] = []
        for match in self.pattern.finditer(text):
            start, end = match.span(self.group)
            if start < 0 or end <= start:
                continue
            value = self._validated(text[start:end])
            if value is None:
                continue
            end = start + len(value)
            score = self.score
            if self.context:
                window = lowered[max(0, start - CONTEXT_WINDOW) : end + CONTEXT_WINDOW]
                if any(keyword in window for keyword in self.context):
                    score = min(1.0, score + CONTEXT_BOOST)
                elif self.require_context:
                    continue
            found.append(
                RegexMatch(
                    entity_type=self.entity_type,
                    start=start,
                    end=end,
                    score=round(score, 4),
                    value=value,
                )
            )
        return found


_CREDENTIAL_CONTEXT = (
    "api key",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "key=",
    "password",
    "secret",
    "token",
)

#: Entity types Presidio does not ship; registered into the analyzer when
#: Presidio is available, and scanned directly when it is not.
CUSTOM_RECOGNIZERS: tuple[RegexRecognizer, ...] = (
    RegexRecognizer(
        entity_type="AADHAAR",
        pattern=re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
        score=0.75,
        validator=_aadhaar_check,
        context=("aadhaar", "aadhar", "uidai", "uid"),
        name="ragcore_aadhaar",
    ),
    RegexRecognizer(
        entity_type="PAN",
        pattern=re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        score=0.7,
        context=("pan", "permanent account", "income tax", "itr"),
        name="ragcore_pan",
    ),
    RegexRecognizer(
        entity_type="SWIFT_CODE",
        pattern=re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
        score=0.55,
        context=("swift", "bic", "bank", "iban", "wire", "beneficiary"),
        require_context=True,
        name="ragcore_swift",
    ),
    RegexRecognizer(
        entity_type="JWT",
        pattern=re.compile(
            r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*"
        ),
        score=0.9,
        validator=jwt_check,
        name="ragcore_jwt",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}\b"),
        score=0.9,
        name="ragcore_api_key_sk",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        score=0.9,
        name="ragcore_api_key_aws",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        score=0.9,
        name="ragcore_api_key_github",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        score=0.9,
        name="ragcore_api_key_slack",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        score=0.9,
        name="ragcore_api_key_google",
    ),
    RegexRecognizer(
        entity_type="API_KEY",
        pattern=re.compile(
            r"(?i)(?:api[_-]?key|secret|token|password|passwd)"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-.]{16,})"
        ),
        score=0.6,
        group=1,
        context=_CREDENTIAL_CONTEXT,
        name="ragcore_api_key_assignment",
    ),
)

#: Entity types Presidio covers. Used only in regex-only fallback mode, so the
#: detector still finds the obvious identifiers without the optional extra.
BASELINE_RECOGNIZERS: tuple[RegexRecognizer, ...] = (
    RegexRecognizer(
        entity_type="EMAIL_ADDRESS",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        score=0.9,
        name="ragcore_email",
    ),
    RegexRecognizer(
        entity_type="CREDIT_CARD",
        pattern=re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        score=0.8,
        validator=luhn_check,
        trim_retry=True,
        context=("card", "credit", "debit", "visa", "mastercard", "amex", "cvv"),
        name="ragcore_credit_card",
    ),
    RegexRecognizer(
        entity_type="IBAN_CODE",
        pattern=re.compile(
            r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Za-z0-9]{4})+(?:[ ]?[A-Za-z0-9]{1,3})?\b"
        ),
        score=0.8,
        validator=iban_check,
        trim_retry=True,
        context=("iban", "account", "bank", "payment", "transfer"),
        name="ragcore_iban",
    ),
    RegexRecognizer(
        entity_type="US_SSN",
        pattern=re.compile(r"\b(?!000|666)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        score=0.7,
        context=("ssn", "social security", "taxpayer"),
        name="ragcore_us_ssn",
    ),
    RegexRecognizer(
        entity_type="PHONE_NUMBER",
        pattern=re.compile(r"\+\d{1,3}[ .-]?(?:\(?\d{1,4}\)?[ .-]?){1,4}\d{2,4}\b"),
        score=0.6,
        context=("phone", "mobile", "tel", "call", "contact", "fax"),
        name="ragcore_phone_international",
    ),
    RegexRecognizer(
        entity_type="PHONE_NUMBER",
        pattern=re.compile(r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"),
        score=0.55,
        context=("phone", "mobile", "tel", "call", "contact", "fax"),
        name="ragcore_phone_national",
    ),
    RegexRecognizer(
        entity_type="IP_ADDRESS",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
        score=0.6,
        name="ragcore_ipv4",
    ),
)

#: The complete regex-only recogniser set.
REGEX_RECOGNIZERS: tuple[RegexRecognizer, ...] = (
    *CUSTOM_RECOGNIZERS,
    *BASELINE_RECOGNIZERS,
)


def scan(
    text: str,
    *,
    recognizers: Sequence[RegexRecognizer] = REGEX_RECOGNIZERS,
    entities: Iterable[str] | None = None,
    min_score: float = 0.0,
) -> list[RegexMatch]:
    """Run recognisers over text.

    Args:
        text: Text to scan.
        recognizers: Recognisers to apply. Defaults to the full set.
        entities: Restrict to these entity types. None applies all.
        min_score: Drop matches scoring below this.

    Returns:
        Matches sorted by start offset, then by descending score.
    """
    allowed = set(entities) if entities is not None else None
    found: list[RegexMatch] = []
    for recognizer in recognizers:
        if allowed is not None and recognizer.entity_type not in allowed:
            continue
        for match in recognizer.find(text):
            if match.score >= min_score:
                found.append(match)
    found.sort(key=lambda m: (m.start, -m.score, m.entity_type))
    return found


def build_presidio_recognizers(
    *, language: str = "en", recognizers: Sequence[RegexRecognizer] | None = None
) -> list[Any]:
    """Build Presidio recognisers for the custom entity types.

    Args:
        language: Language the recognisers apply to.
        recognizers: Recognisers to wrap. Defaults to
            :data:`CUSTOM_RECOGNIZERS`.

    Returns:
        A list of ``presidio_analyzer.PatternRecognizer`` instances, or an empty
        list when Presidio is not installed. Note that Presidio reports the whole
        pattern match, so a recogniser with a capture group over-selects slightly
        compared with :func:`scan`.
    """
    try:
        from presidio_analyzer import Pattern, PatternRecognizer
    except ImportError:
        return []

    class _ValidatingPatternRecognizer(PatternRecognizer):  # type: ignore[misc]
        """Pattern recogniser that also enforces a checksum validator."""

        def __init__(
            self, validator: Callable[[str], bool] | None, **kwargs: Any
        ) -> None:
            """Initialise the recogniser.

            Args:
                validator: Checksum or structural validator, or None.
                **kwargs: Forwarded to ``PatternRecognizer``.
            """
            super().__init__(**kwargs)
            self._validator = validator

        def validate_result(self, pattern_text: str) -> bool | None:
            """Accept or reject a raw pattern match.

            Args:
                pattern_text: The matched text.

            Returns:
                True or False when a validator is configured, else None to leave
                the pattern score untouched.
            """
            if self._validator is None:
                return None
            return bool(self._validator(pattern_text))

    built: list[Any] = []
    for recognizer in recognizers if recognizers is not None else CUSTOM_RECOGNIZERS:
        built.append(
            _ValidatingPatternRecognizer(
                recognizer.validator,
                supported_entity=recognizer.entity_type,
                name=recognizer.recognizer_name,
                supported_language=language,
                context=list(recognizer.context) or None,
                patterns=[
                    Pattern(
                        name=recognizer.recognizer_name,
                        regex=recognizer.pattern.pattern,
                        score=recognizer.score,
                    )
                ],
            )
        )
    return built
