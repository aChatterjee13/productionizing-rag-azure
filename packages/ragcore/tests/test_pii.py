"""Tests for :mod:`ragcore.pii`.

Presidio is forced off (``pii_use_presidio=False``) so the suite exercises the
regex-only fallback path, which is the one that must work when the optional extra
is absent. Every custom recogniser is covered, along with all three redaction
modes and the guarantee that a report never carries a raw value.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragcore.pii.detector import PIIDetector, PIIFinding, PIIReport
from ragcore.pii.recognizers import (
    BASELINE_RECOGNIZERS,
    CUSTOM_RECOGNIZERS,
    iban_check,
    jwt_check,
    luhn_check,
    scan,
    verhoeff_check,
    verhoeff_checksum,
)
from ragcore.settings import Settings

VALID_CARD = "4111 1111 1111 1111"
INVALID_CARD = "4111 1111 1111 1112"
VALID_IBAN = "GB82 WEST 1234 5698 7654 32"
INVALID_IBAN = "GB82 WEST 1234 5698 7654 33"
VALID_PAN = "ABCDE1234F"
VALID_SWIFT = "DEUTDEFF500"
VALID_JWT = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
)
AADHAAR_PREFIX = "23456789012"
VALID_AADHAAR = AADHAAR_PREFIX + str(verhoeff_checksum(AADHAAR_PREFIX))
INVALID_AADHAAR = AADHAAR_PREFIX + str((verhoeff_checksum(AADHAAR_PREFIX) + 1) % 10)


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "pii_use_presidio": False,
        "pii_llm_verify_enabled": False,
        "langfuse_enabled": False,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)


def detector(**overrides: Any) -> PIIDetector:
    return PIIDetector(make_settings(**overrides))


# ------------------------------------------------------------------- validators


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (VALID_CARD, True),
        ("4111-1111-1111-1111", True),
        (INVALID_CARD, False),
        ("411111111111", False),
        ("41111111111111111111", False),
    ],
)
def test_luhn_check(value: str, expected: bool) -> None:
    assert luhn_check(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (VALID_IBAN, True),
        ("GB82WEST12345698765432", True),
        ("DE89 3704 0044 0532 0130 00", True),
        (INVALID_IBAN, False),
        ("GB82", False),
    ],
)
def test_iban_check(value: str, expected: bool) -> None:
    assert iban_check(value) is expected


def test_verhoeff_round_trip() -> None:
    assert verhoeff_check(VALID_AADHAAR) is True
    assert verhoeff_check(INVALID_AADHAAR) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (VALID_JWT, True),
        ("eyJhbGciOiJIUzI1NiJ9.payload", False),
        ("eyJub3RhbGciOiJ4In0.a.b", False),
    ],
)
def test_jwt_check(value: str, expected: bool) -> None:
    assert jwt_check(value) is expected


# ----------------------------------------------------------------- recognisers


@pytest.mark.parametrize(
    ("text", "entity_type", "expected_value"),
    [
        (f"Aadhaar number {VALID_AADHAAR} on file.", "AADHAAR", VALID_AADHAAR),
        (f"PAN {VALID_PAN} is registered.", "PAN", VALID_PAN),
        (f"SWIFT code {VALID_SWIFT} for the wire.", "SWIFT_CODE", VALID_SWIFT),
        (f"Authorization: Bearer {VALID_JWT}", "JWT", VALID_JWT),
        (
            "key: sk-ant-abcdef0123456789xyz",
            "API_KEY",
            "sk-ant-abcdef0123456789xyz",
        ),
        ("AKIAIOSFODNN7EXAMPLE is the access key id", "API_KEY", None),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "API_KEY", None),
        (f"Charge the credit card {VALID_CARD} today.", "CREDIT_CARD", VALID_CARD),
        (f"Send to IBAN {VALID_IBAN} please.", "IBAN_CODE", VALID_IBAN),
        ("mail jane.doe@example.com now", "EMAIL_ADDRESS", "jane.doe@example.com"),
        ("SSN 123-45-6789 on the form", "US_SSN", "123-45-6789"),
        ("call +49 30 1234 5678 for support", "PHONE_NUMBER", None),
        ("server at 10.20.30.40 responded", "IP_ADDRESS", "10.20.30.40"),
    ],
)
def test_each_recognizer_detects_its_entity(
    text: str, entity_type: str, expected_value: str | None
) -> None:
    matches = [m for m in scan(text) if m.entity_type == entity_type]
    assert matches, f"{entity_type} was not detected in {text!r}"
    if expected_value is not None:
        assert matches[0].value == expected_value


def test_swift_requires_context_to_avoid_flagging_uppercase_words() -> None:
    assert scan("DEUTSCHE BAHN operates trains") == []
    assert scan("BIC DEUTDEFF for the transfer")


@pytest.mark.parametrize(
    ("text", "entity_type"),
    [
        (f"card {INVALID_CARD} declined", "CREDIT_CARD"),
        (f"iban {INVALID_IBAN} rejected", "IBAN_CODE"),
        (f"aadhaar {INVALID_AADHAAR} rejected", "AADHAAR"),
    ],
)
def test_checksum_failures_are_dropped(text: str, entity_type: str) -> None:
    assert not [m for m in scan(text) if m.entity_type == entity_type]


def test_card_followed_by_another_number_is_still_found() -> None:
    matches = [
        m
        for m in scan(f"credit card {VALID_CARD} 2029 expiry")
        if m.entity_type == "CREDIT_CARD"
    ]
    assert matches[0].value == VALID_CARD


def test_clean_text_produces_no_findings() -> None:
    text = "Quarterly revenue grew 12 percent across EMEA during 2026."
    assert scan(text) == []


def test_context_raises_the_score() -> None:
    without = scan(f"value {VALID_CARD} here")[0]
    with_context = scan(f"credit card {VALID_CARD} here")[0]
    assert with_context.score > without.score


def test_custom_and_baseline_sets_are_disjoint() -> None:
    custom = {r.entity_type for r in CUSTOM_RECOGNIZERS}
    baseline = {r.entity_type for r in BASELINE_RECOGNIZERS}
    assert custom.isdisjoint(baseline)


# -------------------------------------------------------------------- detector


def test_analyze_reports_summary_fields() -> None:
    report = detector().analyze(
        f"mail jane.doe@example.com and card {VALID_CARD} (credit)"
    )
    assert report.has_pii is True
    assert report.entity_types == ["CREDIT_CARD", "EMAIL_ADDRESS"]
    assert report.max_score == max(f.score for f in report.findings)
    assert [f.start for f in report.findings] == sorted(
        f.start for f in report.findings
    )


def test_analyze_respects_the_entity_allowlist() -> None:
    report = detector(pii_entities=["EMAIL_ADDRESS"]).analyze(
        f"mail jane.doe@example.com and card {VALID_CARD} (credit)"
    )
    assert report.entity_types == ["EMAIL_ADDRESS"]


def test_analyze_respects_the_score_threshold() -> None:
    text = "server at 10.20.30.40 responded"
    assert detector(pii_score_threshold=0.5).analyze(text).has_pii is True
    assert detector(pii_score_threshold=0.95).analyze(text).has_pii is False


def test_analyze_returns_empty_when_pii_is_disabled() -> None:
    report = detector(pii_enabled=False).analyze("mail jane.doe@example.com")
    assert report.has_pii is False
    assert report.findings == []


def test_overlapping_detections_are_merged() -> None:
    report = detector().analyze('api_key = "sk-ant-abcdef0123456789xyz"')
    api_keys = [f for f in report.findings if f.entity_type == "API_KEY"]
    assert len(api_keys) == 1


def test_snippet_never_carries_the_raw_value() -> None:
    email = "jane.doe@example.com"
    report = detector().analyze(f"mail {email} now")
    snippet = report.findings[0].snippet
    assert email not in snippet
    assert snippet.startswith("*")
    dumped = report.model_dump_json()
    assert "jane.doe@example.com" not in dumped


# ------------------------------------------------------------------- redaction


def test_mask_mode_replaces_with_the_entity_type() -> None:
    engine = detector()
    text = f"mail jane.doe@example.com and card {VALID_CARD} (credit)"
    report = engine.analyze(text)
    redacted = engine.redact(text, report, mode="mask")
    assert redacted == "mail <EMAIL_ADDRESS> and card <CREDIT_CARD> (credit)"


def test_hash_mode_is_stable_and_secret_dependent() -> None:
    text = "mail jane.doe@example.com now"
    engine = detector(pii_hash_secret="secret-one")
    report = engine.analyze(text)
    first = engine.redact(text, report, mode="hash")
    second = engine.redact(text, report, mode="hash")
    assert first == second
    assert "jane.doe@example.com" not in first

    other = detector(pii_hash_secret="secret-two")
    assert other.redact(text, other.analyze(text), mode="hash") != first


def test_hash_mode_joins_the_same_value_across_documents() -> None:
    engine = detector()
    left = "contact jane.doe@example.com for access"
    right = "escalate to jane.doe@example.com immediately"
    token_left = engine.redact(left, engine.analyze(left), mode="hash").split()[1]
    token_right = engine.redact(right, engine.analyze(right), mode="hash").split()[2]
    assert token_left == token_right
    assert token_left == engine.pseudonym(
        "jane.doe@example.com", entity_type="EMAIL_ADDRESS"
    )


def test_partial_mode_keeps_the_configured_tail() -> None:
    engine = detector(pii_partial_keep_chars=4)
    text = f"card {VALID_CARD} on file"
    report = engine.analyze(text)
    redacted = engine.redact(text, report, mode="partial")
    assert redacted == "card ***************1111 on file"


def test_partial_mode_with_zero_keep_masks_everything() -> None:
    engine = detector(pii_partial_keep_chars=0)
    text = "mail jane.doe@example.com now"
    redacted = engine.redact(text, engine.analyze(text), mode="partial")
    assert redacted == "mail ******************** now"


def test_scan_and_redact_uses_the_configured_mode() -> None:
    engine = detector(pii_redaction_mode="partial", pii_partial_keep_chars=2)
    redacted, report = engine.scan_and_redact("mail jane.doe@example.com now")
    assert report.has_pii is True
    assert redacted.endswith("om now")
    assert "jane.doe" not in redacted


def test_unknown_redaction_mode_raises() -> None:
    engine = detector()
    text = "mail jane.doe@example.com now"
    with pytest.raises(ValueError, match="unknown redaction mode"):
        engine.redact(text, engine.analyze(text), mode="obfuscate")


def test_redact_without_findings_returns_the_input() -> None:
    engine = detector()
    assert engine.redact("nothing here", PIIReport.empty()) == "nothing here"


def test_redaction_is_order_independent_for_multiple_spans() -> None:
    engine = detector()
    text = (
        f"one jane.doe@example.com two {VALID_CARD} (credit) three ops@example.com four"
    )
    report = engine.analyze(text)
    assert len(report.findings) == 3
    redacted = engine.redact(text, report, mode="mask")
    assert redacted == (
        "one <EMAIL_ADDRESS> two <CREDIT_CARD> (credit) three <EMAIL_ADDRESS> four"
    )


# ---------------------------------------------------------------------- report


def test_report_from_findings_derives_summary() -> None:
    report = PIIReport.from_findings(
        [
            PIIFinding(entity_type="B", start=10, end=12, score=0.4),
            PIIFinding(entity_type="A", start=0, end=2, score=0.9),
        ]
    )
    assert [f.entity_type for f in report.findings] == ["A", "B"]
    assert report.entity_types == ["A", "B"]
    assert report.max_score == 0.9
    assert report.spans() == [(0, 2), (10, 12)]


def test_finding_length() -> None:
    assert PIIFinding(entity_type="A", start=3, end=9, score=1.0).length == 6


# ---------------------------------------------------------------- verification


async def test_verify_is_a_no_op_when_disabled() -> None:
    engine = detector(pii_llm_verify_enabled=False)
    text = "mail jane.doe@example.com now"
    report = engine.analyze(text)
    assert await engine.verify(text, report) is report


async def test_verify_is_a_no_op_without_findings() -> None:
    engine = detector(pii_llm_verify_enabled=True)
    empty = PIIReport.empty()
    assert await engine.verify("nothing here", empty) is empty


async def test_verify_keeps_the_report_when_the_model_call_fails() -> None:
    engine = detector(
        pii_llm_verify_enabled=True,
        pii_llm_verify_min_score=0.1,
        anthropic_base_url="http://127.0.0.1:1/never",
        anthropic_max_retries=0,
        anthropic_timeout_seconds=0.05,
    )
    text = "mail jane.doe@example.com now"
    report = engine.analyze(text)
    verified = await engine.verify(text, report)
    assert verified.entity_types == report.entity_types
