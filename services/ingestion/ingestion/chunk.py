"""Structure-aware chunking with contextual headers.

The chunker packs :class:`~ragcore.models.document.ParsedBlock`s into chunks that are
useful to a retriever rather than merely uniform in size:

* **Headings are boundaries.** A new heading starts a new chunk once the current one
  has reached ``chunk_min_tokens``; below that the section is merged into its
  neighbour rather than emitted as a fragment.
* **Tables stay whole.** A ``TABLE`` block is never mixed with prose, and a table too
  large for ``chunk_max_tokens`` is split by *rows* with its header row repeated, so
  no row ever loses its column names.
* **Sizing is token-based** with ``chunk_target_tokens`` and ``chunk_overlap_tokens``
  of sentence-aligned overlap carried into the next chunk.
* **Every chunk carries a contextual header** — ``"<title> > <section path>"`` plus an
  optional one-line document summary. It is prepended *at embed time* only
  (:attr:`ragcore.models.chunk.ChunkPayload.embed_text`) and stored in its own payload
  field, so a citation span stays verbatim and the header never pollutes a quote.

Token sizing uses :func:`estimate_tokens`, a deterministic offline estimator.
``LLMClient.count_tokens`` is the authority at chat time, but calling it per chunk
would mean one API round trip per chunk of every document in the corpus; ingest-scale
sizing must be local. ``tiktoken`` is deliberately not used anywhere — it is the wrong
tokenizer for Claude.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.document import BlockKind, ParsedBlock, ParsedDocument
from ragcore.settings import Settings, get_settings

__all__ = [
    "ChunkDraft",
    "build_contextual_header",
    "chunk_document",
    "estimate_tokens",
    "split_sentences",
]

#: Word-ish tokens: alphanumeric runs plus individual punctuation marks.
_WORD_RE = re.compile(r"\w+|[^\w\s]")

#: Sentence boundary: terminal punctuation followed by whitespace, or a newline.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

#: Average characters per Claude token for English prose. Used as one of two
#: estimates; the larger of the two wins so chunks never overflow the real budget.
_CHARS_PER_TOKEN = 3.6

#: Multiplier applied to the word-ish token count.
_TOKENS_PER_WORD = 1.15

#: Kinds that must never be merged with surrounding prose.
_ATOMIC_KINDS: frozenset[BlockKind] = frozenset({BlockKind.TABLE, BlockKind.CODE})


def estimate_tokens(text: str) -> int:
    """Estimate a text's Claude token count without an API call.

    Two independent estimates are computed — characters per token and words per
    token — and the larger is returned, so the chunker errs towards slightly smaller
    chunks rather than overflowing a downstream budget.

    Args:
        text: The text to measure.

    Returns:
        An estimated token count, at least 1 for non-empty text and 0 for empty.
    """
    if not text:
        return 0
    by_chars = len(text) / _CHARS_PER_TOKEN
    by_words = len(_WORD_RE.findall(text)) * _TOKENS_PER_WORD
    return max(1, int(max(by_chars, by_words)))


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish units for overlap and force-splitting.

    Args:
        text: Text to split.

    Returns:
        Non-empty sentence fragments in order. Text with no terminal punctuation
        comes back as a single fragment.
    """
    parts = [part.strip() for part in _SENTENCE_RE.split(text)]
    return [part for part in parts if part]


def build_contextual_header(
    title: str,
    section_path: Sequence[str],
    summary: str | None = None,
) -> str:
    """Build the contextual header prepended to a chunk at embed time.

    Args:
        title: Document title.
        section_path: Heading breadcrumb for the chunk.
        summary: Optional one-line document summary appended on its own line.

    Returns:
        A header such as ``"Leave Policy > Entitlement > Carry-over"``, optionally
        followed by a newline and the summary. Empty when there is nothing to say.
    """
    crumbs = [part.strip() for part in [title, *section_path] if part and part.strip()]
    header = " > ".join(dict.fromkeys(crumbs))
    if summary:
        one_line = " ".join(summary.split())
        header = f"{header}\n{one_line}" if header else one_line
    return header


class ChunkDraft(BaseModel):
    """A chunk before embedding, dedupe and payload construction.

    Attributes:
        chunk_index: Zero-based position within the document.
        text: Verbatim chunk text. Citation spans must match this exactly.
        contextual_header: Header prepended at embed time, stored separately.
        section_path: Heading breadcrumb of the chunk's first block.
        page: Page number of the chunk's first block, when paginated.
        token_count: Estimated tokens of header plus text.
        block_kinds: Structural kinds the chunk was assembled from, for diagnostics.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_index: int = Field(ge=0, description="Zero-based position in the document.")
    text: str = Field(description="Verbatim chunk text.")
    contextual_header: str = Field(
        default="", description="Header prepended at embed time only."
    )
    section_path: list[str] = Field(
        default_factory=list, description="Heading breadcrumb for this chunk."
    )
    page: int | None = Field(default=None, description="1-based page number.")
    token_count: int = Field(
        default=0, ge=0, description="Estimated tokens of header plus text."
    )
    block_kinds: list[str] = Field(
        default_factory=list, description="Block kinds this chunk was built from."
    )

    @property
    def embed_text(self) -> str:
        """Text that will be embedded.

        Returns:
            The header followed by the chunk text, mirroring
            :attr:`ragcore.models.chunk.ChunkPayload.embed_text` so the value indexed
            at ingest time and the value described by the payload never diverge.
        """
        if not self.contextual_header:
            return self.text
        return f"{self.contextual_header}\n\n{self.text}"


class _Accumulator:
    """Mutable state for one in-progress chunk."""

    def __init__(self) -> None:
        """Start empty."""
        self.parts: list[str] = []
        self.kinds: list[str] = []
        self.section_path: list[str] = []
        self.page: int | None = None
        self.tokens = 0
        self.started = False

    def add(self, block: ParsedBlock, tokens: int) -> None:
        """Append a block's text.

        Args:
            block: The block to add.
            tokens: Its estimated token count.
        """
        if not self.started:
            self.section_path = list(block.section_path)
            self.started = True
        # A heading has no page of its own in most formats; the chunk's page is the
        # first page that actually carries content, which is what a citation needs.
        if self.page is None and block.page is not None:
            self.page = block.page
        self.parts.append(block.text)
        self.kinds.append(block.kind.value)
        self.tokens += tokens

    def add_text(self, text: str, tokens: int, *, kind: str = "overlap") -> None:
        """Append raw text such as an overlap prefix or a table fragment.

        Args:
            text: Text to append.
            tokens: Its estimated token count.
            kind: Label recorded in ``block_kinds``.
        """
        if not text:
            return
        self.parts.append(text)
        self.kinds.append(kind)
        self.tokens += tokens

    def seed_context(self, section_path: Sequence[str], page: int | None) -> None:
        """Pre-set the breadcrumb for a chunk that starts with overlap text.

        Args:
            section_path: Breadcrumb to record.
            page: Page number to record.
        """
        if not self.started:
            self.section_path = list(section_path)
            self.started = True
        if self.page is None and page is not None:
            self.page = page

    @property
    def text(self) -> str:
        """Assembled chunk text.

        Returns:
            The parts joined by blank lines.
        """
        return "\n\n".join(part for part in self.parts if part).strip()

    @property
    def is_empty(self) -> bool:
        """Whether anything has been accumulated.

        Returns:
            True when no text is buffered.
        """
        return not self.text


def chunk_document(
    parsed: ParsedDocument,
    settings: Settings | None = None,
    *,
    doc_summary: str | None = None,
) -> list[ChunkDraft]:
    """Chunk a parsed document, respecting headings and tables.

    Args:
        parsed: The parsed document.
        settings: Process settings; ``get_settings()`` is used when omitted.
        doc_summary: One-line document summary folded into each contextual header.
            Defaults to ``parsed.summary``.

    Returns:
        Chunks in reading order, each with a contextual header and an estimated token
        count. An empty document yields an empty list rather than a blank chunk.
    """
    active = settings or get_settings()
    summary = doc_summary if doc_summary is not None else parsed.summary
    if not active.chunk_contextual_header_enabled:
        summary = None

    drafts: list[ChunkDraft] = []
    accumulator = _Accumulator()
    pending_overlap = ""

    def flush(*, carry_overlap: bool = True) -> None:
        """Emit the accumulated chunk and prepare the next overlap prefix.

        Args:
            carry_overlap: Whether the emitted chunk's tail should be carried into
                the next chunk. False at a heading boundary and between the fragments
                of a force-split table, where continuation overlap would either cross
                a section boundary or duplicate a repeated table header.
        """
        nonlocal accumulator, pending_overlap
        if accumulator.is_empty:
            accumulator = _Accumulator()
            return
        text = accumulator.text
        header = (
            build_contextual_header(parsed.title, accumulator.section_path, summary)
            if active.chunk_contextual_header_enabled
            else ""
        )
        drafts.append(
            ChunkDraft(
                chunk_index=len(drafts),
                text=text,
                contextual_header=header,
                section_path=list(accumulator.section_path),
                page=accumulator.page,
                token_count=estimate_tokens(f"{header}\n\n{text}" if header else text),
                block_kinds=list(dict.fromkeys(accumulator.kinds)),
            )
        )
        pending_overlap = (
            _tail_for_overlap(text, active.chunk_overlap_tokens)
            if carry_overlap
            else ""
        )
        accumulator = _Accumulator()

    def start_new(section_path: Sequence[str], page: int | None) -> None:
        """Seed a fresh chunk with the carried overlap text.

        Args:
            section_path: Breadcrumb of the block about to be added.
            page: Page of the block about to be added.
        """
        nonlocal pending_overlap
        if pending_overlap:
            accumulator.seed_context(section_path, page)
            accumulator.add_text(pending_overlap, estimate_tokens(pending_overlap))
            pending_overlap = ""

    for block in parsed.blocks:
        tokens = estimate_tokens(block.text)

        if (
            active.chunk_respect_headings
            and block.kind is BlockKind.HEADING
            and accumulator.tokens >= active.chunk_min_tokens
        ):
            flush(carry_overlap=False)

        if block.kind in _ATOMIC_KINDS:
            if tokens > active.chunk_max_tokens:
                flush(carry_overlap=False)
                for fragment in _split_atomic(block, active.chunk_target_tokens):
                    accumulator.seed_context(block.section_path, block.page)
                    accumulator.add_text(
                        fragment, estimate_tokens(fragment), kind=block.kind.value
                    )
                    flush(carry_overlap=False)
                continue
            if accumulator.tokens + tokens > active.chunk_target_tokens:
                flush(carry_overlap=False)
            # A table or code block is never diluted with continuation prose, so any
            # pending overlap is discarded rather than prepended here.
            pending_overlap = ""
            accumulator.add(block, tokens)
            continue

        if tokens > active.chunk_max_tokens:
            flush()
            for fragment in _split_prose(block.text, active.chunk_target_tokens):
                start_new(block.section_path, block.page)
                accumulator.seed_context(block.section_path, block.page)
                accumulator.add_text(
                    fragment, estimate_tokens(fragment), kind=block.kind.value
                )
                flush()
            continue

        # Only split on size once the current chunk is worth emitting. Without the
        # minimum, a heading immediately followed by a long paragraph would be flushed
        # as a chunk containing nothing but the heading.
        if (
            accumulator.tokens >= active.chunk_min_tokens
            and accumulator.tokens + tokens > active.chunk_target_tokens
        ):
            flush()
        start_new(block.section_path, block.page)
        accumulator.add(block, tokens)

    flush()
    return _merge_runts(drafts, active)


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    """Take the trailing sentences of a chunk to overlap into the next one.

    Args:
        text: The emitted chunk text.
        overlap_tokens: Token budget for the overlap.

    Returns:
        The trailing text, sentence-aligned, within the budget. Empty when overlap is
        disabled.
    """
    if overlap_tokens <= 0 or not text:
        return ""
    sentences = split_sentences(text)
    tail: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        cost = estimate_tokens(sentence)
        if used + cost > overlap_tokens and tail:
            break
        tail.insert(0, sentence)
        used += cost
        if used >= overlap_tokens:
            break
    return " ".join(tail)


def _split_prose(text: str, target_tokens: int) -> list[str]:
    """Force-split an oversized prose block on sentence boundaries.

    Args:
        text: Block text.
        target_tokens: Token target per fragment.

    Returns:
        Fragments, each at or below the target except when a single sentence exceeds
        it (which is then emitted alone rather than cut mid-word).
    """
    fragments: list[str] = []
    current: list[str] = []
    used = 0
    for sentence in split_sentences(text) or [text]:
        cost = estimate_tokens(sentence)
        if current and used + cost > target_tokens:
            fragments.append(" ".join(current))
            current = []
            used = 0
        current.append(sentence)
        used += cost
    if current:
        fragments.append(" ".join(current))
    return fragments


def _split_atomic(block: ParsedBlock, target_tokens: int) -> list[str]:
    """Split an oversized table or code block without losing its structure.

    A pipe table is split by rows with the header and separator rows repeated in
    every fragment. Anything else is split on line boundaries.

    Args:
        block: The oversized block.
        target_tokens: Token target per fragment.

    Returns:
        Fragments in order.
    """
    lines = block.text.split("\n")
    header: list[str] = []
    body = lines
    if block.kind is BlockKind.TABLE and len(lines) >= 2 and lines[0].startswith("|"):
        header = lines[:2]
        body = lines[2:]

    header_cost = estimate_tokens("\n".join(header))
    fragments: list[str] = []
    current: list[str] = []
    used = header_cost
    for line in body:
        cost = estimate_tokens(line)
        if current and used + cost > target_tokens:
            fragments.append("\n".join([*header, *current]))
            current = []
            used = header_cost
        current.append(line)
        used += cost
    if current:
        fragments.append("\n".join([*header, *current]))
    return fragments or [block.text]


def _merge_runts(drafts: list[ChunkDraft], settings: Settings) -> list[ChunkDraft]:
    """Fold a trailing under-sized chunk into its predecessor.

    A final fragment of a few tokens retrieves badly and wastes a point in the index.
    Merging only happens when the combined size still fits ``chunk_max_tokens`` and
    both chunks share a section path, so a genuine short section survives on its own.

    Args:
        drafts: Chunks in reading order.
        settings: Process settings.

    Returns:
        The possibly-shortened chunk list, re-indexed.
    """
    if len(drafts) < 2:
        return drafts
    out = list(drafts)
    last = out[-1]
    previous = out[-2]
    combined = f"{previous.text}\n\n{last.text}"
    if (
        last.token_count < settings.chunk_min_tokens
        and previous.section_path == last.section_path
        and estimate_tokens(combined) <= settings.chunk_max_tokens
    ):
        merged = previous.model_copy(
            update={
                "text": combined,
                "token_count": estimate_tokens(
                    f"{previous.contextual_header}\n\n{combined}"
                    if previous.contextual_header
                    else combined
                ),
                "block_kinds": list(
                    dict.fromkeys([*previous.block_kinds, *last.block_kinds])
                ),
            }
        )
        out = [*out[:-2], merged]
    return [
        draft.model_copy(update={"chunk_index": index})
        for index, draft in enumerate(out)
    ]
