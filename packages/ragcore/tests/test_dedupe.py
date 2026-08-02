"""Tests for content fingerprinting and near-duplicate detection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragcore.dedupe import (
    SIMHASH_BITS,
    content_sha256,
    dedupe_chunks,
    hamming64,
    is_near_duplicate,
    normalise_text,
    shingles,
    simhash64,
    simhash_hex,
)


@dataclass
class Candidate:
    """Stand-in for a RetrievedChunk: just the two fingerprints and a label."""

    label: str
    sha: str
    sim: str


def candidate(label: str, text: str, *, fingerprint: bool = True) -> Candidate:
    return Candidate(
        label=label,
        sha=content_sha256(text),
        sim=simhash_hex(text) if fingerprint else "",
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_collapses_case_whitespace_and_punctuation():
    assert normalise_text("  Hello,   WORLD!\n\tagain  ") == "hello world again"


def test_normalise_applies_nfkc_so_compatibility_forms_match():
    assert normalise_text("ﬁle") == normalise_text("file")
    # Decomposed "e" + combining acute must equal the precomposed form.
    assert normalise_text("Caf\u00e9") == normalise_text("Cafe\u0301")


def test_normalise_keeps_non_latin_words():
    assert normalise_text("東京 タワー") == "東京 タワー"
    assert normalise_text("Köln, Zürich") == "köln zürich"


def test_normalise_casefolds_rather_than_lowercasing():
    """casefold() folds ß to ss, so 'GRUSSE' and 'Grüße' still differ only by umlaut."""
    assert normalise_text("Grüße") == "grüsse"
    assert normalise_text("GRÜSSE") == normalise_text("Grüße")


# ---------------------------------------------------------------------------
# content_sha256
# ---------------------------------------------------------------------------


def test_content_sha256_is_stable_and_64_hex_chars():
    digest = content_sha256("the quick brown fox")
    assert digest == content_sha256("the quick brown fox")
    assert len(digest) == 64
    int(digest, 16)


def test_content_sha256_ignores_cosmetic_differences():
    """A re-export with different wrapping must not look like new content."""
    assert content_sha256("Annual Leave Policy\n2024") == content_sha256(
        "annual   leave policy 2024"
    )


def test_content_sha256_distinguishes_real_differences():
    assert content_sha256("holiday is 25 days") != content_sha256("holiday is 28 days")


# ---------------------------------------------------------------------------
# simhash
# ---------------------------------------------------------------------------


def test_simhash_is_deterministic_across_calls():
    """Never hash() — its per-process salt would invalidate stored fingerprints."""
    text = "employees accrue twenty five days of annual leave each calendar year"
    assert simhash64(text) == simhash64(text)
    assert simhash_hex(text) == simhash_hex(text)


def test_simhash_fits_in_64_unsigned_bits():
    value = simhash64("a moderately long sentence about expense reimbursement rules")
    assert 0 <= value < 1 << SIMHASH_BITS


def test_simhash_hex_is_16_lowercase_hex_chars():
    hex_value = simhash_hex("expense reimbursement rules for contractors")
    assert len(hex_value) == 16
    assert hex_value == hex_value.lower()
    int(hex_value, 16)


def test_simhash_hex_zero_pads():
    assert simhash_hex("") == "0" * 16


def test_empty_text_has_no_fingerprint():
    assert simhash64("") == 0
    assert simhash64("   \n\t  ") == 0
    assert simhash64("!!! ???") == 0


def test_near_duplicate_text_lands_close():
    original = (
        "Employees accrue twenty five days of annual leave each calendar year. "
        "Unused days may be carried over until the end of March."
    )
    edited = (
        "Employees accrue twenty five days of annual leave each calendar year. "
        "Unused days may be carried over until the end of April."
    )
    distance = hamming64(simhash64(original), simhash64(edited))
    assert distance <= 8, f"one-word edit moved the simhash {distance} bits"


def test_unrelated_text_lands_far_apart():
    a = simhash64("the expense policy requires receipts for claims over fifty euro")
    b = simhash64("kubernetes ingress controllers terminate tls at the edge proxy")
    assert hamming64(a, b) > 12


def test_shingles_capture_word_order():
    """Reordering words changes the shingles, which is why simhash notices."""
    forward = shingles("the cat sat on the mat", width=3)
    backward = shingles("the mat sat on the cat", width=3)
    assert forward != backward
    assert (
        hamming64(
            simhash64("the cat sat on the mat"), simhash64("the mat sat on the cat")
        )
        > 0
    )


def test_shingles_of_short_text_fall_back_to_words():
    assert shingles("two words", width=4) == ["two", "words"]
    assert shingles("", width=4) == []


def test_shingles_rejects_a_non_positive_width():
    with pytest.raises(ValueError, match="positive"):
        shingles("anything", width=0)


# ---------------------------------------------------------------------------
# hamming64 / is_near_duplicate
# ---------------------------------------------------------------------------


def test_hamming_basics():
    assert hamming64(0, 0) == 0
    assert hamming64(0, 1) == 1
    assert hamming64(0, (1 << SIMHASH_BITS) - 1) == SIMHASH_BITS
    assert hamming64(0b1010, 0b0101) == 4


def test_is_near_duplicate_respects_the_threshold():
    a = "0000000000000000"
    b = "0000000000000007"  # three bits set
    assert is_near_duplicate(a, b, max_distance=3) is True
    assert is_near_duplicate(a, b, max_distance=2) is False
    assert is_near_duplicate(a, a, max_distance=0) is True


def test_missing_fingerprint_is_never_a_duplicate():
    """A blank simhash must not be read as the value zero."""
    assert is_near_duplicate("", "0000000000000000", max_distance=3) is False
    assert is_near_duplicate("0000000000000000", "", max_distance=64) is False
    assert is_near_duplicate("not-hex-at-all!!", "0000000000000000") is False


# ---------------------------------------------------------------------------
# dedupe_chunks
# ---------------------------------------------------------------------------


def test_dedupe_keeps_distinct_chunks_and_reports_no_drops():
    items = [
        candidate("a", "annual leave accrues at two point one days per month"),
        candidate("b", "expense claims require an itemised receipt"),
    ]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim
    )
    assert [c.label for c in kept] == ["a", "b"]
    assert dropped == []


def test_dedupe_drops_exact_duplicates_with_a_reason():
    text = "annual leave accrues at two point one days per month"
    items = [candidate("first", text), candidate("second", text)]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim
    )
    assert [c.label for c in kept] == ["first"]
    assert len(dropped) == 1
    chunk, reason = dropped[0]
    assert chunk.label == "second"
    # An identical text has distance 0, so the simhash layer claims it first; either
    # reason is a duplicate report, but it must name the layer.
    assert reason.startswith("duplicate:")


def test_dedupe_drops_near_duplicates_and_names_the_distance():
    original = (
        "Employees accrue twenty five days of annual leave each calendar year, "
        "and unused days may be carried over until the end of March."
    )
    edited = (
        "Employees accrue twenty five days of annual leave each calendar year, "
        "and unused days may be carried over until the end of April."
    )
    items = [candidate("original", original), candidate("edited", edited)]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim, max_distance=8
    )
    assert [c.label for c in kept] == ["original"]
    assert len(dropped) == 1
    chunk, reason = dropped[0]
    assert chunk.label == "edited"
    assert reason.startswith("duplicate:simhash:")
    assert int(reason.rsplit(":", 1)[1]) <= 8


def test_dedupe_keeps_near_duplicates_below_the_threshold():
    original = "the reimbursement window closes thirty days after the expense date"
    edited = "the reimbursement window closes sixty days after the expense date"
    items = [candidate("original", original), candidate("edited", edited)]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim, max_distance=0
    )
    assert [c.label for c in kept] == ["original", "edited"]
    assert dropped == []


def test_first_occurrence_wins_so_callers_should_pass_best_first():
    text = "identical content ingested from two different sources"
    items = [candidate("higher-ranked", text), candidate("lower-ranked", text)]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim
    )
    assert [c.label for c in kept] == ["higher-ranked"]
    assert [c.label for c, _ in dropped] == ["lower-ranked"]


def test_chunks_without_a_simhash_skip_the_near_duplicate_layer():
    """Short chunks opt out with an empty simhash and must not collide."""
    items = [
        candidate("short-a", "yes", fingerprint=False),
        candidate("short-b", "no", fingerprint=False),
    ]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim
    )
    assert [c.label for c in kept] == ["short-a", "short-b"]
    assert dropped == []


def test_exact_layer_still_applies_without_a_simhash():
    items = [
        candidate("a", "same text", fingerprint=False),
        candidate("b", "same text", fingerprint=False),
    ]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim
    )
    assert [c.label for c in kept] == ["a"]
    assert dropped == [(items[1], "duplicate:sha256")]


def test_every_input_is_accounted_for():
    """Nothing is discarded silently: kept + dropped always reconstructs the input."""
    texts = [
        "annual leave accrues at two point one days per month",
        "annual leave accrues at two point one days per month",
        "expense claims require an itemised receipt",
        "expense claims require an itemized receipt",
        "kubernetes ingress controllers terminate tls at the edge",
    ]
    items = [candidate(str(index), text) for index, text in enumerate(texts)]
    kept, dropped = dedupe_chunks(
        items, key=lambda c: c.sha, simhash_key=lambda c: c.sim, max_distance=3
    )
    assert len(kept) + len(dropped) == len(items)
    assert {c.label for c in kept} | {c.label for c, _ in dropped} == {
        c.label for c in items
    }
    assert all(reason.startswith("duplicate:") for _, reason in dropped)


def test_dedupe_of_an_empty_list():
    kept, dropped = dedupe_chunks([], key=lambda c: c, simhash_key=lambda c: c)
    assert kept == []
    assert dropped == []
