"""Stage 11: a citation is kept only when the cited chunk really says it.

The two behaviours requirement #9 turns on are pinned first — a fabricated quote is
caught, and a citation that differs from its source only by whitespace, case or
punctuation is not — followed by the accounting that feeds ``citation_validity`` and
the groundedness gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.rag.citations import (
    DROP_QUOTE_NOT_FOUND,
    DROP_SPAN_NOT_FOUND,
    DROP_UNKNOWN_MARKER,
    CitationVerdict,
    append_uncertainty_notice,
    build_source_block,
    extract_citations,
    parse_markers,
    strip_unresolved_markers,
    verify_span,
)
from ragcore.dedupe import content_sha256, simhash_hex
from ragcore.llm.prompts import UNCERTAINTY_NOTICE
from ragcore.models.acl import AccessControl, Classification
from ragcore.models.chunk import ChunkPayload
from ragcore.models.retrieval import RetrievedChunk
from ragcore.settings import Settings

MEALS = (
    "The daily meal allowance is EUR 60 per day for employees travelling within "
    "the EU. Expenses above 500 EUR require prior approval from a director."
)
VPN = (
    "To connect to the corporate network, install the ACME VPN client and "
    "authenticate with your Entra credentials. Split tunnelling is not permitted."
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-travel",
    title: str = "Travel Policy",
) -> RetrievedChunk:
    access = AccessControl(
        tenant_id="tenant-acme", classification=Classification.PUBLIC
    )
    now = datetime.now(UTC)
    payload = ChunkPayload.from_access_control(
        access,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        source_type="blob",
        source_id="src-1",
        source_uri=f"https://example.invalid/{document_id}",
        title=title,
        section_path=["Expenses", "Meals"],
        page=4,
        text=text,
        contextual_header=f"{title} > Expenses > Meals",
        summary=None,
        keywords=[],
        doc_type="policy",
        tags=[],
        author=None,
        language="en",
        content_sha256=content_sha256(text),
        simhash=simhash_hex(text),
        token_count=max(1, len(text) // 4),
        effective_from=datetime(2025, 4, 1, tzinfo=UTC),
        created_at=now,
        updated_at=now,
        version=1,
        is_deleted=False,
        pii_types=[],
        pii_redacted=True,
        ingest_run_id="run-1",
    )
    return RetrievedChunk(payload=payload, fusion_score=0.03, final_score=0.9)


# ----------------------------------------------------------------- source block
def test_the_source_block_is_numbered_positionally() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS), chunk("c2", VPN)], settings=cfg)

    assert "[1] Travel Policy" in block.text
    assert "[2] Travel Policy" in block.text
    assert "section: Expenses > Meals" in block.text
    assert "effective from: 2025-04-01" in block.text
    assert block.chunk_for(1) is not None
    assert block.chunk_for(1).payload.chunk_id == "c1"
    assert block.chunk_for(3) is None
    assert block.size == 2


def test_an_empty_source_block_says_so_explicitly() -> None:
    block = build_source_block([], settings=make_settings())
    assert "no sources were retrieved" in block.text
    assert block.size == 0


def test_snippets_are_clipped_to_the_prompt_budget() -> None:
    cfg = make_settings(retrieval_snippet_chars=40)
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    assert block.snippets[0].text.endswith("…")
    assert len(block.snippets[0].text) <= 45


# --------------------------------------------------------------------- markers
@pytest.mark.parametrize(
    ("answer", "numbers"),
    [
        ("A fact [1].", [1]),
        ("A fact [1][2].", [1, 2]),
        ("A fact [1, 2].", [1, 2]),
        ("A fact [ 3 ].", [3]),
        ("An array literal [abc] is not a marker.", []),
        ("No markers here.", []),
    ],
)
def test_parse_markers(answer: str, numbers: list[int]) -> None:
    assert [ref.number for ref in parse_markers(answer)] == numbers


def test_strip_unresolved_markers_keeps_only_survivors() -> None:
    cleaned = strip_unresolved_markers("First [1]. Second [2][3].", keep={1, 3})
    assert cleaned == "First [1]. Second [3]."


# ------------------------------------------------------------------ span checks
def test_whitespace_case_and_punctuation_differences_still_verify() -> None:
    cfg = make_settings()
    reflowed = "the   DAILY meal\n\tallowance is EUR 60 per day,"

    match = verify_span(reflowed, MEALS, settings=cfg)

    assert match is not None
    assert match.exact is True
    assert match.ratio == 1.0
    assert MEALS[match.start : match.end] == match.text
    assert match.text.startswith("The daily meal allowance")


def test_a_fabricated_quote_is_not_verifiable() -> None:
    cfg = make_settings()
    invented = "the daily meal allowance is EUR 250 for senior managers abroad"

    assert verify_span(invented, MEALS, verbatim=True, settings=cfg) is None


def test_a_faithful_paraphrase_verifies_below_full_confidence() -> None:
    cfg = make_settings()
    paraphrase = "Employees can claim up to EUR 60 each day for meals in the EU"

    match = verify_span(paraphrase, MEALS, settings=cfg)

    assert match is not None
    assert match.exact is False
    assert 0.0 < match.ratio < 1.0
    assert MEALS[match.start : match.end] == match.text


def test_a_number_the_source_does_not_contain_fails_the_guard() -> None:
    cfg = make_settings()
    wrong = "The daily meal allowance is EUR 95 per day for employees in the EU"

    assert verify_span(wrong, MEALS, settings=cfg) is None
    # The guard is what rejects it, not the lexical similarity.
    assert verify_span(wrong, MEALS, check_numbers=False, settings=cfg) is not None


def test_an_unrelated_sentence_never_verifies() -> None:
    cfg = make_settings()
    unrelated = "Employees receive a company car after three years of service"

    assert verify_span(unrelated, MEALS, settings=cfg) is None


# ------------------------------------------------------------ full extraction
def test_a_grounded_answer_scores_one_and_carries_real_offsets() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS), chunk("c2", VPN)], settings=cfg)
    answer = (
        "The daily meal\nallowance is EUR 60 per day for employees travelling "
        "within the EU [1]. Split tunnelling is not permitted [2]."
    )

    report = extract_citations(answer, block, settings=cfg)

    assert report.citation_validity == 1.0
    assert report.coverage == 1.0
    assert report.verdict is CitationVerdict.GROUNDED
    assert report.needs_uncertainty_notice is False
    assert [citation.marker for citation in report.citations] == ["[1]", "[2]"]
    assert report.cited_chunk_ids == ["c1", "c2"]
    for citation in report.citations:
        source = block.chunk_for(int(citation.marker.strip("[]")))
        assert source is not None
        assert citation.quoted_span
        assert (
            source.payload.text[citation.char_start : citation.char_end]
            == citation.quoted_span
        )
        assert citation.is_verified is True


def test_a_fabricated_quote_is_dropped_and_reported() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = (
        'The policy states "the daily meal allowance is EUR 250 for senior '
        'managers travelling abroad" [1].'
    )

    report = extract_citations(answer, block, settings=cfg)

    assert report.citations == []
    assert report.citation_validity == 0.0
    assert report.verdict is CitationVerdict.UNGROUNDED
    assert report.needs_uncertainty_notice is True
    assert [drop.reason for drop in report.dropped] == [DROP_QUOTE_NOT_FOUND]
    drop = report.dropped[0]
    assert drop.chunk_id == "c1"
    assert drop.marker == "[1]"
    # A drop never carries the offending text: stage 11 runs before PII egress.
    assert "250" not in drop.model_dump_json()
    # The dead marker is removed from the cleaned copy.
    assert "[1]" not in report.cleaned_answer


def test_a_hallucinated_sentence_is_dropped_even_without_a_quote() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = "Employees also receive a company car after three years of service [1]."

    report = extract_citations(answer, block, settings=cfg)

    assert [drop.reason for drop in report.dropped] == [DROP_SPAN_NOT_FOUND]
    assert report.citation_validity == 0.0


def test_a_marker_with_no_source_is_reported_as_unknown() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = "Expenses above 500 EUR require prior approval from a director [7]."

    report = extract_citations(answer, block, settings=cfg)

    assert report.unknown_markers == [7]
    assert [drop.reason for drop in report.dropped] == [DROP_UNKNOWN_MARKER]
    assert report.dropped[0].chunk_id is None
    assert "[7]" not in report.cleaned_answer


def test_partial_grounding_scores_between_zero_and_one() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = (
        "Expenses above 500 EUR require prior approval from a director [1]. "
        "Employees also receive a company car after three years of service [1]."
    )

    report = extract_citations(answer, block, settings=cfg)

    assert report.markers_attempted == 2
    assert report.markers_verified == 1
    assert report.citation_validity == 0.5
    # One surviving citation for the marker, carrying the verified instance.
    assert len(report.citations) == 1
    assert report.coverage == 0.5
    assert report.groundedness == 0.5


def test_an_answer_that_cites_nothing_is_ungrounded() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)

    report = extract_citations(
        "Employees may claim a meal allowance whenever they travel.",
        block,
        settings=cfg,
    )

    assert report.markers_attempted == 0
    assert report.claim_sentences == 1
    assert report.citation_validity == 0.0
    assert report.verdict is CitationVerdict.UNGROUNDED
    assert report.needs_uncertainty_notice is True


def test_a_refusal_makes_no_claims_and_needs_no_citations() -> None:
    cfg = make_settings()
    block = build_source_block([], settings=cfg)

    report = extract_citations("I don't know.", block, settings=cfg)

    assert report.verdict is CitationVerdict.NOT_APPLICABLE
    assert report.citation_validity == 1.0
    assert report.needs_uncertainty_notice is False


def test_a_short_list_item_borrows_the_preceding_sentence() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = "The daily meal allowance is set out in the policy:\nEUR 60 per day [1]."

    report = extract_citations(answer, block, settings=cfg)

    assert report.markers_verified == 1


def test_a_sentence_citing_two_sources_relaxes_the_numeric_guard() -> None:
    cfg = make_settings()
    old = chunk(
        "c-2023", "The daily meal allowance is EUR 45 per day.", document_id="d23"
    )
    new = chunk(
        "c-2025", "The daily meal allowance is EUR 60 per day.", document_id="d25"
    )
    block = build_source_block([old, new], settings=cfg)
    answer = "The daily meal allowance rose from EUR 45 [1] to EUR 60 [2]."

    report = extract_citations(answer, block, settings=cfg)

    assert report.markers_verified == 2
    assert report.citation_validity == 1.0


def test_the_best_instance_wins_when_a_marker_is_cited_twice() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    answer = (
        "Employees can claim up to EUR 60 each day for meals in the EU [1]. "
        "The daily meal allowance is EUR 60 per day for employees travelling "
        "within the EU [1]."
    )

    report = extract_citations(answer, block, settings=cfg)

    assert len(report.citations) == 1
    assert report.citations[0].confidence == 1.0
    assert report.markers_verified == 2


def test_extract_citations_accepts_a_bare_chunk_list() -> None:
    cfg = make_settings()
    report = extract_citations(
        "Split tunnelling is not permitted [1].", [chunk("c2", VPN)], settings=cfg
    )
    assert report.citation_validity == 1.0


def test_the_uncertainty_notice_is_appended_once() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    report = extract_citations(
        "Employees also receive a company car after three years of service [1].",
        block,
        settings=cfg,
    )

    once = append_uncertainty_notice(report.cleaned_answer, report)
    twice = append_uncertainty_notice(once, report)

    assert once.endswith(UNCERTAINTY_NOTICE)
    assert twice == once


def test_the_notice_is_not_appended_to_a_grounded_answer() -> None:
    cfg = make_settings()
    block = build_source_block([chunk("c1", MEALS)], settings=cfg)
    report = extract_citations(
        "Expenses above 500 EUR require prior approval from a director [1].",
        block,
        settings=cfg,
    )

    assert report.needs_uncertainty_notice is False
    assert append_uncertainty_notice("text", report) == "text"
