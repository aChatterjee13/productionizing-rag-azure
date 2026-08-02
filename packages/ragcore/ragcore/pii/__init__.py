"""PII detection and redaction (requirement #9).

:func:`get_pii_detector` returns the cached detector. Presidio is an optional
extra (``uv sync --extra pii``): when it is missing the detector falls back to
the regex-only recogniser set in :mod:`ragcore.pii.recognizers`, logs a warning,
and keeps working, so this package always imports.

Redaction has three modes -- ``mask`` (``<EMAIL_ADDRESS>``), ``hash`` (a stable
HMAC pseudonym so redacted values still join) and ``partial`` (keep the last few
characters). ``mode`` defaults to ``pii_redaction_mode``.
"""

from __future__ import annotations

from ragcore.pii.detector import (
    PIIDetector,
    PIIFinding,
    PIIReport,
    PIIVerdict,
    PIIVerificationResult,
    get_pii_detector,
    reset_pii_detector_cache,
)
from ragcore.pii.recognizers import (
    BASELINE_RECOGNIZERS,
    CUSTOM_RECOGNIZERS,
    REGEX_RECOGNIZERS,
    RegexMatch,
    RegexRecognizer,
    build_presidio_recognizers,
    iban_check,
    jwt_check,
    luhn_check,
    scan,
    verhoeff_check,
    verhoeff_checksum,
)

__all__ = [
    "BASELINE_RECOGNIZERS",
    "CUSTOM_RECOGNIZERS",
    "REGEX_RECOGNIZERS",
    "PIIDetector",
    "PIIFinding",
    "PIIReport",
    "PIIVerdict",
    "PIIVerificationResult",
    "RegexMatch",
    "RegexRecognizer",
    "build_presidio_recognizers",
    "get_pii_detector",
    "iban_check",
    "jwt_check",
    "luhn_check",
    "reset_pii_detector_cache",
    "scan",
    "verhoeff_check",
    "verhoeff_checksum",
]
