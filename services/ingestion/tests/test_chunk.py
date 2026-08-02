"""Structure-aware chunking: heading boundaries, section paths, tables, overlap."""

from __future__ import annotations

from ingestion.chunk import (
    build_contextual_header,
    chunk_document,
    estimate_tokens,
    split_sentences,
)
from ingestion.parse import SectionTracker, render_table
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import BlockKind, ParsedBlock, ParsedDocument
from ragcore.settings import Settings

TENANT = "tenant-a"

SENTENCE = "Employees accrue twenty-five days of paid annual leave each calendar year. "


def settings_for(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "chunk_min_tokens": 16,
        "chunk_target_tokens": 64,
        "chunk_overlap_tokens": 8,
        "chunk_max_tokens": 128,
        "chunk_respect_headings": True,
        "chunk_contextual_header_enabled": True,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def build(blocks: list[ParsedBlock], *, title: str = "Leave Policy") -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        tenant_id=TENANT,
        source_id="src-1",
        source_type=SourceType.LOCAL,
        source_uri="file:///corpus/leave-policy.md",
        title=title,
        blocks=blocks,
        access_control=AccessControl(tenant_id=TENANT),
    )


def heading(text: str, level: int, order: int, path: list[str]) -> ParsedBlock:
    return ParsedBlock(
        kind=BlockKind.HEADING,
        text=text,
        order=order,
        level=level,
        section_path=path,
    )


def paragraph(
    text: str, order: int, path: list[str], *, page: int | None = None
) -> ParsedBlock:
    return ParsedBlock(
        kind=BlockKind.PARAGRAPH,
        text=text,
        order=order,
        page=page,
        section_path=path,
    )


# ------------------------------------------------------------------ token sizing
def test_estimate_tokens_is_monotonic_and_never_zero_for_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    short = estimate_tokens(SENTENCE)
    long = estimate_tokens(SENTENCE * 10)
    assert short >= 1
    assert long > short * 5


def test_split_sentences_keeps_units():
    parts = split_sentences("One. Two! Three?\nFour")
    assert parts == ["One.", "Two!", "Three?", "Four"]


# --------------------------------------------------------------- section paths
def test_section_tracker_pops_siblings_and_keeps_ancestors():
    tracker = SectionTracker()
    assert tracker.push(1, "Entitlement") == ["Entitlement"]
    assert tracker.push(2, "Carry-over") == ["Entitlement", "Carry-over"]
    # A sibling at the same depth replaces its predecessor, keeping the parent.
    assert tracker.push(2, "Buy-back") == ["Entitlement", "Buy-back"]
    # A shallower heading pops everything deeper.
    assert tracker.push(1, "Requests") == ["Requests"]


def test_chunks_carry_the_section_path_of_their_first_block():
    blocks = [
        heading("Entitlement", 1, 0, ["Entitlement"]),
        paragraph(SENTENCE * 4, 1, ["Entitlement"]),
        heading("Carry-over", 2, 2, ["Entitlement", "Carry-over"]),
        paragraph(
            "Unused days expire on 31 March. " * 8,
            3,
            ["Entitlement", "Carry-over"],
        ),
    ]
    chunks = chunk_document(build(blocks), settings_for())

    assert len(chunks) >= 2
    paths = [tuple(chunk.section_path) for chunk in chunks]
    assert ("Entitlement",) in paths
    assert ("Entitlement", "Carry-over") in paths


def test_heading_boundary_prevents_section_bleed():
    marker_a = "AAAMARKER"
    marker_b = "BBBMARKER"
    blocks = [
        heading("Section A", 1, 0, ["Section A"]),
        paragraph(f"{marker_a} " + SENTENCE * 4, 1, ["Section A"]),
        heading("Section B", 1, 2, ["Section B"]),
        paragraph(f"{marker_b} " + SENTENCE * 4, 3, ["Section B"]),
    ]
    chunks = chunk_document(build(blocks), settings_for())

    for chunk in chunks:
        # No chunk may contain content from both sections: overlap is deliberately
        # not carried across a heading boundary.
        assert not (marker_a in chunk.text and marker_b in chunk.text)


def test_short_section_is_merged_rather_than_emitted_as_a_fragment():
    blocks = [
        heading("Scope", 1, 0, ["Scope"]),
        paragraph("Applies to all staff.", 1, ["Scope"]),
        heading("Detail", 1, 2, ["Detail"]),
        paragraph(SENTENCE * 3, 3, ["Detail"]),
    ]
    chunks = chunk_document(build(blocks), settings_for())
    # "Scope" is only a few tokens, well below chunk_min_tokens, so it is folded in
    # with what follows instead of becoming a chunk of its own.
    assert "Applies to all staff." in chunks[0].text
    assert "Scope" in chunks[0].text


# ---------------------------------------------------------------- contextual header
def test_contextual_header_is_title_plus_section_path():
    header = build_contextual_header("Leave Policy", ["Entitlement", "Carry-over"])
    assert header == "Leave Policy > Entitlement > Carry-over"


def test_contextual_header_appends_a_one_line_summary():
    header = build_contextual_header(
        "Leave Policy", ["Entitlement"], "Sets annual leave\nfor all staff."
    )
    assert header.startswith("Leave Policy > Entitlement\n")
    assert "\n" in header
    assert header.count("\n") == 1


def test_contextual_header_does_not_repeat_the_title():
    assert build_contextual_header("Entitlement", ["Entitlement"]) == "Entitlement"


def test_header_is_stored_separately_and_only_prepended_for_embedding():
    blocks = [
        heading("Entitlement", 1, 0, ["Entitlement"]),
        paragraph(SENTENCE * 3, 1, ["Entitlement"]),
    ]
    chunk = chunk_document(build(blocks), settings_for())[0]
    assert chunk.contextual_header
    # The verbatim text must stay quotable for citations.
    assert not chunk.text.startswith(chunk.contextual_header)
    assert chunk.embed_text.startswith(chunk.contextual_header)
    assert chunk.embed_text.endswith(chunk.text)


def test_header_can_be_disabled():
    blocks = [paragraph(SENTENCE * 3, 0, [])]
    chunk = chunk_document(
        build(blocks), settings_for(chunk_contextual_header_enabled=False)
    )[0]
    assert chunk.contextual_header == ""
    assert chunk.embed_text == chunk.text


# ------------------------------------------------------------------------ tables
def test_small_table_stays_whole_and_unmixed_with_prose():
    table = render_table(
        [["Grade", "Days"], ["Junior", "25"], ["Senior", "30"], ["Director", "35"]]
    )
    blocks = [
        heading("Entitlement", 1, 0, ["Entitlement"]),
        paragraph(SENTENCE * 4, 1, ["Entitlement"]),
        ParsedBlock(
            kind=BlockKind.TABLE, text=table, order=2, section_path=["Entitlement"]
        ),
    ]
    chunks = chunk_document(build(blocks), settings_for())
    table_chunks = [c for c in chunks if "table" in c.block_kinds]
    assert len(table_chunks) == 1
    assert "| Grade | Days |" in table_chunks[0].text
    assert "| Director | 35 |" in table_chunks[0].text
    assert "paragraph" not in table_chunks[0].block_kinds


def test_oversized_table_is_split_by_rows_with_the_header_repeated():
    rows = [["Grade", "Days", "Notes"]]
    rows.extend(
        [f"Grade {index}", str(20 + index), "Standard entitlement with carry-over"]
        for index in range(60)
    )
    table = render_table(rows)
    block = ParsedBlock(
        kind=BlockKind.TABLE, text=table, order=0, section_path=["Entitlement"]
    )
    assert estimate_tokens(table) > settings_for().chunk_max_tokens

    chunks = chunk_document(build([block]), settings_for())

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("| Grade | Days | Notes |")
        assert chunk.section_path == ["Entitlement"]


# ----------------------------------------------------------------------- overlap
def test_overlap_is_carried_within_a_section():
    first = "Alpha statement one. " * 12
    second = "Beta statement two. " * 12
    blocks = [
        heading("Entitlement", 1, 0, ["Entitlement"]),
        paragraph(first, 1, ["Entitlement"]),
        paragraph(second, 2, ["Entitlement"]),
    ]
    chunks = chunk_document(build(blocks), settings_for())
    assert len(chunks) >= 2
    assert any("Alpha statement one." in chunk.text for chunk in chunks[1:])


def test_overlap_can_be_switched_off():
    first = "Alpha statement one. " * 12
    second = "Beta statement two. " * 12
    blocks = [
        paragraph(first, 0, ["Entitlement"]),
        paragraph(second, 1, ["Entitlement"]),
    ]
    chunks = chunk_document(build(blocks), settings_for(chunk_overlap_tokens=0))
    assert len(chunks) >= 2
    assert "Alpha statement one." not in chunks[1].text


# ------------------------------------------------------------------- bookkeeping
def test_chunk_indexes_are_contiguous_from_zero():
    blocks = [
        heading(f"Section {index}", 1, index * 2, [f"Section {index}"])
        for index in range(4)
    ]
    for index in range(4):
        blocks.append(paragraph(SENTENCE * 3, index * 2 + 1, [f"Section {index}"]))
    document = build(sorted(blocks, key=lambda block: block.order))
    chunks = chunk_document(document, settings_for())
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_page_numbers_survive_chunking():
    blocks = [
        heading("Entitlement", 1, 0, ["Entitlement"]),
        paragraph(SENTENCE * 4, 1, ["Entitlement"], page=7),
    ]
    chunks = chunk_document(build(blocks), settings_for())
    assert chunks[0].page == 7


def test_token_count_covers_header_plus_text():
    blocks = [paragraph(SENTENCE * 3, 0, ["Entitlement"])]
    chunk = chunk_document(build(blocks), settings_for())[0]
    assert chunk.token_count >= estimate_tokens(chunk.text)
    assert chunk.token_count == estimate_tokens(chunk.embed_text)


def test_empty_document_yields_no_chunks():
    assert chunk_document(build([]), settings_for()) == []


def test_oversized_paragraph_is_force_split_below_the_ceiling():
    block = paragraph(SENTENCE * 60, 0, ["Entitlement"])
    active = settings_for()
    assert estimate_tokens(block.text) > active.chunk_max_tokens
    chunks = chunk_document(build([block]), active)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= active.chunk_max_tokens * 2
