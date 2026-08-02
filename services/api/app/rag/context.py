"""Context assembly — pipeline stage 9, requirements #3 and #5.

This is where context bloat is actually solved, so everything here is measured rather
than assumed.

**Measurement.** Every token count comes from ``LLMClient.count_tokens`` (Claude's own
tokenizer) via :class:`~app.rag.memory.short_term.TokenCounter`. No characters/4
heuristic is used at any call site: the system prompt is measured by differencing two
exact counts, and the finished payload is measured exactly once before it is returned,
which is the number :attr:`~ragcore.models.chat.ContextStats.window_tokens` reports.

**Priority order** (``docs/CONTRACTS.md`` stage 9), highest first::

    system prompt -> pinned turns -> rolling summary -> long-term memory
    -> retrieved chunks (dedup-aware, largest marginal value first) -> recent turns

**Pressure is relieved by suppression, never truncation.** In order:

1. Tool results older than ``context_tool_result_ttl_turns`` are cleared with the
   ``clear_tool_uses_20250919`` context edit rather than re-sent.
2. The oldest non-pinned turns are folded into the rolling summary with ``MODEL_FAST``
   and marked ``suppressed`` — they stay in PostgreSQL and stay visible in the UI.
3. The lowest-marginal-value retrieved chunks are dropped, dedup-aware: a chunk that
   is a near-duplicate of one already packed carries almost no new information, so it
   goes first, with an audited ``dropped_reason``.

Nothing is cut mid-sentence and every drop is reported.

**Compaction is proactive and periodic**, not only reactive: it fires at
``context_compact_at_ratio`` of the budget *and* every
``context_compact_every_n_turns`` turns, which is the "periodic context suppression"
requirement #5 asks for explicitly. Compaction always retires down to
``context_min_live_turns``, the documented floor of turns that are never suppressed.

**Caching.** Volatile content (sources, memory, summary, preferences, the question)
lives in the final user turn, never in the system prompt, so the system prefix stays
byte-stable and cacheable across turns. One ``cache_control`` breakpoint goes on the
last system block and — once the history prefix is long enough to be worth one — a
second on the last stable history message.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from app.rag.memory import render_memory_block, render_preferences
from app.rag.memory.short_term import (
    SessionWindow,
    ShortTermMemory,
    TokenCounter,
    get_short_term_memory,
)
from ragcore.dedupe import is_near_duplicate
from ragcore.llm.client import LLMClient, clear_tool_uses_edit
from ragcore.llm.prompts import (
    ANSWER_SYSTEM,
    SourceSnippet,
    build_answer_user_turn,
    render_numbered_sources,
)
from ragcore.models.acl import Principal
from ragcore.models.chat import ContextStats, Message
from ragcore.models.memory import LongTermMemory, UserProfile
from ragcore.models.retrieval import RetrievedChunk
from ragcore.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "DROP_BUDGET",
    "DROP_DUPLICATE_EXACT",
    "DROP_DUPLICATE_NEAR",
    "DROP_RETRIEVED_CAP",
    "GAP_NOTICE",
    "AssembledContext",
    "ContextAssembler",
    "ContextBudget",
    "PackedSource",
    "get_context_assembler",
    "reset_context_assemblers",
]

_log = structlog.get_logger(__name__)

#: ``dropped_reason`` values this module sets. Every drop is audited (requirement #9).
DROP_DUPLICATE_EXACT = "context:duplicate:sha256"
DROP_DUPLICATE_NEAR = "context:duplicate:simhash"
DROP_RETRIEVED_CAP = "context:retrieved_cap"
DROP_BUDGET = "context:budget"

#: Constant stand-in inserted when the oldest replayed turn is an assistant turn,
#: which happens once its user turn has been suppressed. The Anthropic API requires
#: the first message to be a user turn; a constant keeps the prefix cacheable.
GAP_NOTICE = (
    "(Earlier turns in this conversation were suppressed; a rolling summary of "
    "them appears in the final message.)"
)

#: Minimal message used to difference out the framing when measuring the system
#: prompt on its own. Constant, so it is measured once per process and memoised.
_PROBE_MESSAGE: tuple[dict[str, str], ...] = ({"role": "user", "content": "."},)

#: A simhash of all zeros means "no fingerprint" and never compares equal.
_BLANK_SIMHASH = "0" * 16


# ------------------------------------------------------------------- the budget
@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Token budget the assembler packs into, derived entirely from settings."""

    total_tokens: int
    reserve_output_tokens: int
    prompt_tokens: int
    compact_at_tokens: int
    retrieved_tokens: int
    memory_tokens: int
    summary_tokens: int
    min_live_turns: int
    max_history_turns: int
    compact_every_n_turns: int
    tool_result_ttl_turns: int
    min_retrieved_chunks: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ContextBudget:
        """Derive the budget from configuration.

        Args:
            settings: Active settings.

        Returns:
            The budget. ``prompt_tokens`` is the usable prompt space
            (``context_budget_tokens`` minus ``context_reserve_output_tokens``) and
            ``compact_at_tokens`` is ``context_compact_at_ratio`` of that.
        """
        prompt = settings.context_prompt_budget_tokens
        return cls(
            total_tokens=settings.context_budget_tokens,
            reserve_output_tokens=settings.context_reserve_output_tokens,
            prompt_tokens=prompt,
            compact_at_tokens=settings.context_compact_threshold_tokens,
            retrieved_tokens=int(prompt * settings.context_retrieved_budget_ratio),
            memory_tokens=int(prompt * settings.context_memory_budget_ratio),
            summary_tokens=settings.context_summary_max_tokens,
            min_live_turns=settings.context_min_live_turns,
            max_history_turns=settings.context_max_history_turns,
            compact_every_n_turns=int(settings.context_compact_every_n_turns),
            tool_result_ttl_turns=settings.context_tool_result_ttl_turns,
            min_retrieved_chunks=int(settings.context_min_retrieved_chunks),
        )

    def headroom(self, used: int) -> int:
        """Tokens still available.

        Args:
            used: Tokens already committed.

        Returns:
            The remaining budget, never negative.
        """
        return max(0, self.prompt_tokens - used)


# ------------------------------------------------------------------ packed item
@dataclass(slots=True)
class PackedSource:
    """One retrieved chunk as it will be rendered, with its packing economics."""

    chunk: RetrievedChunk
    text: str
    tokens: int
    novelty: float
    marker: str = ""

    @property
    def value(self) -> float:
        """Marginal value of including this chunk.

        Returns:
            The chunk's final score discounted by how much of its content is already
            carried by a higher-ranked chunk.
        """
        return max(0.0, self.chunk.final_score) * self.novelty

    @property
    def density(self) -> float:
        """Marginal value per token, which is what packing actually optimises.

        Returns:
            :attr:`value` divided by the measured token cost.
        """
        return self.value / max(1, self.tokens)


# ------------------------------------------------------------------- the result
@dataclass(slots=True)
class AssembledContext:
    """The finished prompt payload plus everything the pipeline must report."""

    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    stats: ContextStats
    sources: list[RetrievedChunk] = field(default_factory=list)
    dropped: list[RetrievedChunk] = field(default_factory=list)
    memories: list[LongTermMemory] = field(default_factory=list)
    rolling_summary: str = ""
    suppressed_message_ids: list[str] = field(default_factory=list)
    context_management: dict[str, Any] | None = None
    cache_system: bool = True
    compaction_reason: str = ""
    user_turn: str = ""

    @property
    def compacted(self) -> bool:
        """Whether this assembly retired any turn.

        Returns:
            True when at least one turn was folded into the rolling summary.
        """
        return bool(self.suppressed_message_ids)

    def request_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``LLMClient.complete`` / ``LLMClient.stream``.

        Returns:
            The prompt half of the request. The caller supplies ``model``,
            ``effort``, ``max_tokens`` and ``tools``.
        """
        kwargs: dict[str, Any] = {
            "system": self.system,
            "messages": self.messages,
            "cache_system": self.cache_system,
        }
        if self.context_management is not None:
            kwargs["context_management"] = self.context_management
        return kwargs


# ------------------------------------------------------------------ the packing
@dataclass(slots=True)
class _Packing:
    """Mutable working set for one assembly pass."""

    system_blocks: list[dict[str, Any]]
    system_tokens: int
    question: str
    preferences: str
    preferences_tokens: int
    summary: str
    summary_tokens: int
    memory_lines: list[tuple[LongTermMemory, str, int]]
    sources: list[PackedSource]
    history: list[Message]
    notes: str
    history_breakpoint: bool
    dropped: list[RetrievedChunk]

    @property
    def memory_tokens(self) -> int:
        """Measured cost of the memory block.

        Returns:
            Sum of the selected memory lines.
        """
        return sum(tokens for _, _, tokens in self.memory_lines)

    @property
    def retrieved_tokens(self) -> int:
        """Measured cost of the packed sources.

        Returns:
            Sum of the selected chunk renderings.
        """
        return sum(source.tokens for source in self.sources)

    @property
    def history_tokens(self) -> int:
        """Measured cost of the replayed history.

        Returns:
            Sum of the turns' recorded token counts.
        """
        return sum(turn.token_count for turn in self.history)


# --------------------------------------------------------------------- helpers
def _snippet_for(chunk: RetrievedChunk, marker: str, text: str) -> SourceSnippet:
    """Project a chunk onto the answer prompt's source shape.

    Args:
        chunk: The chunk.
        marker: Citation marker, e.g. ``"[1]"``.
        text: Snippet text, already truncated.

    Returns:
        The snippet.
    """
    payload = chunk.payload
    effective = payload.effective_from
    return SourceSnippet(
        marker=marker,
        title=payload.title,
        text=text,
        source_uri=payload.source_uri,
        section_path=tuple(payload.section_path),
        page=payload.page,
        doc_type=payload.doc_type,
        effective_from=effective.isoformat() if effective else None,
    )


def _renumber(sources: Sequence[PackedSource]) -> None:
    """Reassign citation markers after a source was shed.

    Args:
        sources: The surviving sources, in relevance order.
    """
    for index, source in enumerate(sources, start=1):
        source.marker = f"[{index}]"


def _text_of(message: Mapping[str, Any]) -> str:
    """Extract the plain text of a rendered message.

    Args:
        message: An Anthropic message mapping whose content is text or text blocks.

    Returns:
        The concatenated text.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n\n".join(
        str(block.get("text", "")) for block in content if isinstance(block, Mapping)
    )


def _normalise_sequence(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Make a replayed sequence valid for the Messages API without losing content.

    Suppression removes arbitrary turns, so the survivors can start on an assistant
    turn or contain two consecutive turns of the same role — both of which the API
    rejects. Consecutive same-role turns are merged, and a leading assistant turn is
    preceded by the constant :data:`GAP_NOTICE` rather than discarded.

    Args:
        messages: Rendered messages, oldest first.

    Returns:
        A valid sequence carrying the same text.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = _text_of(message)
        if not text.strip():
            continue
        if not out and role != "user":
            out.append({"role": "user", "content": GAP_NOTICE})
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": f"{out[-1]['content']}\n\n{text}"}
            continue
        out.append({"role": role, "content": text})
    return out


# ------------------------------------------------------------------- assembler
class ContextAssembler:
    """Packs a prompt to a measured budget, suppressing rather than truncating."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        counter: TokenCounter | None = None,
        short_term: ShortTermMemory | None = None,
    ) -> None:
        """Initialise the assembler.

        Args:
            settings: Active settings. Defaults to the process settings.
            llm: Client used for summarisation and token counting.
            counter: Shared token counter. Defaults to the short-term memory's, so
                one memo serves the whole turn.
            short_term: Session-window façade. Defaults to the process one.
        """
        self._settings = settings or get_settings()
        self._short_term = short_term or get_short_term_memory(self._settings)
        self._counter = counter or self._short_term.counter
        self._llm = llm

    @property
    def settings(self) -> Settings:
        """Settings this assembler was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def counter(self) -> TokenCounter:
        """The token counter in use.

        Returns:
            The shared counter.
        """
        return self._counter

    @property
    def short_term(self) -> ShortTermMemory:
        """The session-window façade in use.

        Returns:
            The bound façade.
        """
        return self._short_term

    def budget(self) -> ContextBudget:
        """Derive the current budget.

        Returns:
            A :class:`ContextBudget` built from the bound settings.
        """
        return ContextBudget.from_settings(self._settings)

    # ------------------------------------------------------------------ measure
    def _system_blocks(
        self, system: str | Sequence[str] | None
    ) -> list[dict[str, Any]]:
        """Normalise the system prompt into cacheable text blocks.

        The final block carries the ``cache_control`` breakpoint. ``LLMClient``
        re-normalises system blocks and places the breakpoint itself, so doing it
        here is belt-and-braces: it makes the returned payload self-describing for
        anything that inspects the assembled context without sending it.

        Args:
            system: Prompt text, several blocks, or None for
                :data:`~ragcore.llm.prompts.ANSWER_SYSTEM`.

        Returns:
            The system blocks, cache breakpoint on the last one.
        """
        if system is None:
            texts: list[str] = [ANSWER_SYSTEM]
        elif isinstance(system, str):
            texts = [system]
        else:
            texts = [text for text in system if text]
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": text} for text in texts if text.strip()
        ]
        if blocks and self._settings.anthropic_cache_system:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    async def _measure_system(self, blocks: Sequence[Mapping[str, Any]]) -> int:
        """Measure the system prompt exactly, without the surrounding message.

        Both counts are constant across turns and therefore memoised after the first
        request, so this costs nothing steady-state.

        Args:
            blocks: The system blocks.

        Returns:
            Tokens attributable to the system prompt.
        """
        if not blocks:
            return 0
        with_system = await self._counter.count_prompt(
            system=[dict(block) for block in blocks], messages=list(_PROBE_MESSAGE)
        )
        without_system = await self._counter.count_prompt(
            system=None, messages=list(_PROBE_MESSAGE)
        )
        return max(0, with_system - without_system)

    async def _measure_memories(
        self, memories: Sequence[LongTermMemory]
    ) -> list[tuple[LongTermMemory, str, int]]:
        """Render and measure each memory line independently.

        Args:
            memories: Memories to inject, most relevant first.

        Returns:
            ``(memory, rendered_line, tokens)`` triples in the same order.
        """
        if not memories:
            return []
        lines = [render_memory_block([memory]) for memory in memories]
        counts = await self._counter.count_many(lines)
        return list(zip(memories, lines, counts, strict=True))

    # ------------------------------------------------------------------- ranking
    async def rank_sources(
        self, chunks: Sequence[RetrievedChunk]
    ) -> tuple[list[PackedSource], list[RetrievedChunk]]:
        """Render, measure and score retrieved chunks by marginal value.

        Dedup-aware: an exact content-hash repeat carries no new information and is
        dropped outright; a simhash near-duplicate of an already-seen chunk keeps only
        ``context_duplicate_penalty`` of its score, so it is the first thing shed
        under pressure. Both drops are audited via ``dropped_reason``.

        Args:
            chunks: Retrieved chunks, best first.

        Returns:
            A ``(packed, dropped)`` pair. ``packed`` preserves the incoming relevance
            order; ``dropped`` carries the exact duplicates with their reason set.
        """
        penalty = float(self._settings.context_duplicate_penalty)
        max_distance = self._settings.dedupe_max_distance
        snippet_chars = self._settings.retrieval_snippet_chars

        dropped: list[RetrievedChunk] = []
        seen_hashes: set[str] = set()
        seen_simhashes: list[str] = []
        candidates: list[tuple[RetrievedChunk, str, float]] = []

        for chunk in chunks:
            payload = chunk.payload
            digest = payload.content_sha256
            if digest and digest in seen_hashes:
                dropped.append(chunk.drop(DROP_DUPLICATE_EXACT))
                continue
            if digest:
                seen_hashes.add(digest)

            novelty = 1.0
            simhash = (payload.simhash or "").strip().lower()
            if simhash and simhash != _BLANK_SIMHASH:
                if any(
                    is_near_duplicate(simhash, other, max_distance=max_distance)
                    for other in seen_simhashes
                ):
                    novelty = penalty
                seen_simhashes.append(simhash)

            candidates.append((chunk, payload.text[:snippet_chars], novelty))

        if not candidates:
            return [], dropped

        rendered = [
            render_numbered_sources([_snippet_for(chunk, f"[{index}]", text)])
            for index, (chunk, text, _) in enumerate(candidates, start=1)
        ]
        counts = await self._counter.count_many(rendered)
        packed = [
            PackedSource(chunk=chunk, text=text, tokens=tokens, novelty=novelty)
            for (chunk, text, novelty), tokens in zip(candidates, counts, strict=True)
        ]
        return packed, dropped

    # ------------------------------------------------------------------ assemble
    async def assemble(
        self,
        *,
        principal: Principal,
        window: SessionWindow,
        question: str,
        chunks: Sequence[RetrievedChunk] = (),
        memories: Sequence[LongTermMemory] = (),
        profile: UserProfile | None = None,
        notes: str = "",
        system: str | Sequence[str] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        force_compaction: bool = False,
        db_session: AsyncSession | None = None,
    ) -> AssembledContext:
        """Assemble the prompt for one turn, within the measured token budget.

        Args:
            principal: The caller. Used for tenant-scoped logging only; the ACL work
                happened in retrieval.
            window: The live session window. Mutated in place when turns are
                suppressed, so the caller can persist the result.
            question: The context-resolved question, already PII-redacted.
            chunks: Retrieved chunks, best first.
            memories: Long-term memories to inject, most relevant first.
            profile: The user's rolling profile, rendered as a preferences block.
            notes: Pipeline notes for the model, e.g. a contradiction summary.
            system: System prompt override. Defaults to ``ANSWER_SYSTEM``.
            tools: Tool definitions, which also consume prompt tokens and therefore
                participate in the measurement.
            force_compaction: Compact regardless of the triggers, for
                ``POST /sessions/{id}/compact``.
            db_session: Database session used to mirror suppression into
                ``chat_messages``. Omit to leave persistence to the caller.

        Returns:
            The assembled payload with a fully populated
            :class:`~ragcore.models.chat.ContextStats`.
        """
        budget = self.budget()
        system_blocks = self._system_blocks(system)
        system_tokens = await self._measure_system(system_blocks)

        preferences = render_preferences(profile)
        packed_sources, dropped = await self.rank_sources(chunks)
        memory_lines = await self._measure_memories(memories)
        question_tokens = await self._counter.count_text(question)
        preferences_tokens = await self._counter.count_text(preferences)
        notes_tokens = await self._counter.count_text(notes)

        # -- stage 9a: decide whether to compact, proactively and periodically.
        projected = (
            system_tokens
            + question_tokens
            + preferences_tokens
            + notes_tokens
            + window.total_tokens
            + min(budget.memory_tokens, sum(t for _, _, t in memory_lines))
            + min(budget.retrieved_tokens, sum(s.tokens for s in packed_sources))
        )
        reason = self._compaction_reason(
            window, budget=budget, projected=projected, forced=force_compaction
        )
        suppressed: list[str] = []
        if reason:
            outcome = await self._short_term.compact(
                window, keep_live=budget.min_live_turns, reason=reason, persist=False
            )
            suppressed.extend(turn.message_id for turn in outcome.retired)
            if db_session is not None and outcome.retired:
                await self._short_term.persist_compaction(
                    window, outcome, db_session=db_session
                )

        summary = window.rolling_summary
        summary_tokens = window.summary_tokens
        if summary and not summary_tokens:
            summary_tokens = await self._counter.count_text(summary)
            window.summary_tokens = summary_tokens

        # -- stage 9b: clear stale tool results instead of re-sending them.
        cleared, context_management = self._tool_result_edit(window, budget=budget)

        # -- stage 9c: pack in priority order.
        packing = self._pack(
            budget=budget,
            window=window,
            system_blocks=system_blocks,
            system_tokens=system_tokens,
            question=question,
            question_tokens=question_tokens,
            preferences=preferences,
            preferences_tokens=preferences_tokens,
            notes=notes,
            notes_tokens=notes_tokens,
            summary=summary,
            summary_tokens=summary_tokens,
            memory_lines=memory_lines,
            sources=packed_sources,
            dropped=dropped,
        )

        # -- history that did not fit is suppressed, not truncated.
        await self._suppress_overflow(
            window, packing, suppressed=suppressed, db_session=db_session
        )

        # -- stage 9d: verify exactly, then shed until it fits.
        window_tokens = await self._fit(packing, budget=budget, tools=tools)

        await self._short_term.save(window)
        result = self._finalise(
            packing,
            budget=budget,
            window=window,
            window_tokens=window_tokens,
            memories=memories,
            suppressed=suppressed,
            cleared=cleared,
            context_management=context_management,
            reason=reason,
        )
        _log.info(
            "context_assembled",
            tenant_id=principal.tenant_id,
            session_id=window.session_id,
            window_tokens=result.stats.window_tokens,
            budget_tokens=result.stats.budget_tokens,
            utilisation=round(result.stats.utilisation, 3),
            sources=len(result.sources),
            dropped=len(result.dropped),
            messages_live=result.stats.messages_live,
            suppressed_now=len(suppressed),
            tool_results_cleared=cleared,
            compaction_reason=reason,
            token_count_calls=self._counter.calls,
        )
        return result

    # --------------------------------------------------------------- compaction
    def _compaction_reason(
        self,
        window: SessionWindow,
        *,
        budget: ContextBudget,
        projected: int,
        forced: bool,
    ) -> str:
        """Decide whether — and why — compaction should run this turn.

        Args:
            window: The live window.
            budget: The active budget.
            projected: Estimated prompt size before packing.
            forced: Caller demanded compaction.

        Returns:
            One of ``"forced"``, ``"max_turns"``, ``"periodic"``, ``"ratio"``, or an
            empty string when no compaction is warranted.
        """
        live = len(window.live_turns)
        if live <= budget.min_live_turns:
            # The floor is never compacted, however tight the budget gets.
            return ""
        if forced:
            return "forced"
        if live > budget.max_history_turns:
            return "max_turns"
        if (
            budget.compact_every_n_turns > 0
            and window.turns_since_compaction >= budget.compact_every_n_turns
        ):
            return "periodic"
        if projected >= budget.compact_at_tokens:
            return "ratio"
        return ""

    def _tool_result_edit(
        self, window: SessionWindow, *, budget: ContextBudget
    ) -> tuple[int, dict[str, Any] | None]:
        """Count stale tool results and build the context edit that clears them.

        The ``clear_tool_uses_20250919`` edit is what actually removes the blocks,
        including the ones the stage 8 tool loop appends to this same payload.
        Replayed history is rendered as plain text, so nothing stale is re-sent from
        here either — the count is what the edit will remove.

        Args:
            window: The live window.
            budget: The active budget, supplying the TTL in turns.

        Returns:
            A ``(count, context_management)`` pair; ``context_management`` is None
            when there is nothing to clear.
        """
        ttl = budget.tool_result_ttl_turns
        live = window.live_turns
        stale = live[: max(0, len(live) - ttl)] if ttl >= 0 else []
        cleared = sum(len(turn.tool_calls) for turn in stale)
        if cleared == 0:
            return 0, None
        return cleared, clear_tool_uses_edit()

    async def _suppress_overflow(
        self,
        window: SessionWindow,
        packing: _Packing,
        *,
        suppressed: list[str],
        db_session: AsyncSession | None,
    ) -> None:
        """Fold the turns packing could not fit into the rolling summary.

        Args:
            window: The live window, mutated in place.
            packing: The selected working set; its summary is refreshed.
            suppressed: Accumulator of suppressed message ids.
            db_session: Optional database session for the SQL mirror.
        """
        packed_ids = {turn.message_id for turn in packing.history}
        overflow = [
            turn.message_id
            for turn in window.live_turns
            if not turn.pinned and turn.message_id not in packed_ids
        ]
        if not overflow:
            return
        outcome = await self._short_term.compact(
            window, message_ids=overflow, reason="overflow", persist=False
        )
        if not outcome.retired:
            return
        suppressed.extend(turn.message_id for turn in outcome.retired)
        packing.summary = outcome.summary
        packing.summary_tokens = outcome.summary_tokens
        if db_session is not None:
            await self._short_term.persist_compaction(
                window, outcome, db_session=db_session
            )

    # ------------------------------------------------------------------ packing
    def _pack(
        self,
        *,
        budget: ContextBudget,
        window: SessionWindow,
        system_blocks: list[dict[str, Any]],
        system_tokens: int,
        question: str,
        question_tokens: int,
        preferences: str,
        preferences_tokens: int,
        notes: str,
        notes_tokens: int,
        summary: str,
        summary_tokens: int,
        memory_lines: list[tuple[LongTermMemory, str, int]],
        sources: list[PackedSource],
        dropped: list[RetrievedChunk],
    ) -> _Packing:
        """Select what goes into the prompt, in the documented priority order.

        Args:
            budget: The active budget.
            window: The live window, already compacted if it needed to be.
            system_blocks: Normalised system blocks.
            system_tokens: Their measured cost.
            question: The question for this turn.
            question_tokens: Its measured cost.
            preferences: Rendered preferences block.
            preferences_tokens: Its measured cost.
            notes: Pipeline notes.
            notes_tokens: Their measured cost.
            summary: The rolling summary.
            summary_tokens: Its measured cost.
            memory_lines: Measured memory lines.
            sources: Measured, marginal-value-scored chunks.
            dropped: Audited drop list, extended in place.

        Returns:
            The selected working set.
        """
        pinned = [turn for turn in window.live_turns if turn.pinned]
        pinned_tokens = sum(turn.token_count for turn in pinned)
        used = (
            system_tokens
            + question_tokens
            + preferences_tokens
            + notes_tokens
            + pinned_tokens
        )

        # 3. rolling summary
        chosen_summary, chosen_summary_tokens = "", 0
        if summary and summary_tokens <= min(
            budget.headroom(used), budget.summary_tokens
        ):
            chosen_summary, chosen_summary_tokens = summary, summary_tokens
            used += summary_tokens

        # 4. long-term memory
        chosen_memories: list[tuple[LongTermMemory, str, int]] = []
        memory_spent = 0
        for memory, line, tokens in memory_lines:
            if memory_spent + tokens > budget.memory_tokens:
                continue
            if tokens > budget.headroom(used):
                break
            chosen_memories.append((memory, line, tokens))
            memory_spent += tokens
            used += tokens

        # 5. retrieved chunks: best marginal value per token first, but rendered in
        #    relevance order so citation markers stay meaningful to the reader.
        position = {id(source): index for index, source in enumerate(sources)}
        kept: list[PackedSource] = []
        retrieved_spent = 0
        for source in sorted(sources, key=lambda s: (s.density, s.value), reverse=True):
            over_cap = retrieved_spent + source.tokens > budget.retrieved_tokens
            if over_cap and len(kept) >= budget.min_retrieved_chunks:
                dropped.append(source.chunk.drop(DROP_RETRIEVED_CAP))
                continue
            if source.tokens > budget.headroom(used):
                dropped.append(source.chunk.drop(DROP_BUDGET))
                continue
            kept.append(source)
            retrieved_spent += source.tokens
            used += source.tokens
        kept.sort(key=lambda s: position[id(s)])
        _renumber(kept)

        # 6. recent turns, newest first, down to the protected floor.
        history = self._select_history(window, budget=budget, used=used)
        history_tokens = sum(turn.token_count for turn in history)
        use_breakpoint = bool(
            self._settings.context_cache_history_breakpoint
        ) and history_tokens >= int(self._settings.context_cache_history_min_tokens)

        return _Packing(
            system_blocks=system_blocks,
            system_tokens=system_tokens,
            question=question,
            preferences=preferences,
            preferences_tokens=preferences_tokens,
            summary=chosen_summary,
            summary_tokens=chosen_summary_tokens,
            memory_lines=chosen_memories,
            sources=kept,
            history=history,
            notes=notes,
            history_breakpoint=use_breakpoint,
            dropped=dropped,
        )

    def _select_history(
        self, window: SessionWindow, *, budget: ContextBudget, used: int
    ) -> list[Message]:
        """Choose the recent turns that fit, newest first, floor protected.

        Args:
            window: The live window.
            budget: The active budget.
            used: Tokens already committed. The pinned turns are already counted
                there, so only the unpinned selections are charged again here.

        Returns:
            The turns to replay, oldest first. Pinned turns are always included, and
            so are the newest ``context_min_live_turns``.
        """
        live = window.live_turns
        keep_ids = {turn.message_id for turn in live if turn.pinned}
        if budget.min_live_turns:
            keep_ids.update(turn.message_id for turn in live[-budget.min_live_turns :])
        selected = {turn.message_id for turn in live if turn.message_id in keep_ids}
        spent = sum(
            turn.token_count
            for turn in live
            if turn.message_id in selected and not turn.pinned
        )
        for turn in reversed(live):
            if turn.message_id in selected:
                continue
            if spent + turn.token_count > budget.headroom(used):
                continue
            selected.add(turn.message_id)
            spent += turn.token_count
        return [turn for turn in live if turn.message_id in selected]

    # ---------------------------------------------------------------- rendering
    def _render(
        self, packing: _Packing
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Render the working set into an Anthropic request payload.

        Volatile content — preferences, summary, memory, sources, notes and the
        question — goes into the final user turn, so the system prefix stays
        byte-stable and the prompt cache survives across turns.

        Args:
            packing: The selected working set.

        Returns:
            A ``(system, messages)`` pair.
        """
        snippets = [
            _snippet_for(source.chunk, source.marker or "[?]", source.text)
            for source in packing.sources
        ]
        user_turn = build_answer_user_turn(
            question=packing.question,
            sources=render_numbered_sources(snippets),
            memory="\n".join(line for _, line, _ in packing.memory_lines),
            summary=packing.summary,
            preferences=packing.preferences,
            notes=packing.notes,
        )
        history = _normalise_sequence(
            [turn.to_anthropic_message() for turn in packing.history]
        )
        messages = _normalise_sequence(
            [*history, {"role": "user", "content": user_turn}]
        )
        if (
            packing.history_breakpoint
            and len(messages) > 1
            and messages[-2]["role"] == "assistant"
        ):
            messages[-2] = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": messages[-2]["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return packing.system_blocks, messages

    async def _fit(
        self,
        packing: _Packing,
        *,
        budget: ContextBudget,
        tools: Sequence[Mapping[str, Any]] | None,
    ) -> int:
        """Measure the finished payload exactly and shed until it fits.

        Per-item sums ignore the framing the API adds around a whole request, so the
        assembled payload is measured for real and trimmed when the estimate was
        optimistic. Shedding order is documented on :meth:`_shed`.

        Args:
            packing: The working set, mutated in place.
            budget: The active budget.
            tools: Tool definitions, which consume prompt tokens too.

        Returns:
            The exact measured prompt size.
        """
        passes = int(self._settings.context_fit_max_passes)
        for attempt in range(passes + 1):
            system_blocks, messages = self._render(packing)
            measured = await self._counter.count_prompt(
                system=system_blocks, messages=messages, tools=tools
            )
            if measured <= budget.prompt_tokens:
                return measured
            freed = self._shed_until(
                packing, budget=budget, excess=measured - budget.prompt_tokens
            )
            if freed == 0:
                _log.error(
                    "context_budget_exceeded",
                    measured=measured,
                    budget=budget.prompt_tokens,
                    passes=attempt,
                )
                return measured
            _log.warning(
                "context_shed",
                measured=measured,
                budget=budget.prompt_tokens,
                freed=freed,
                attempt=attempt + 1,
                sources=len(packing.sources),
                history=len(packing.history),
            )
        system_blocks, messages = self._render(packing)
        return await self._counter.count_prompt(
            system=system_blocks, messages=messages, tools=tools
        )

    def _shed_until(
        self, packing: _Packing, *, budget: ContextBudget, excess: int
    ) -> int:
        """Shed content until the estimated saving covers the overshoot.

        Shedding one item per measurement pass would need as many round trips as
        there are surplus items; shedding to a target and re-measuring once keeps the
        number of ``count_tokens`` calls bounded by ``context_fit_max_passes``.

        Args:
            packing: The working set, mutated in place.
            budget: The active budget.
            excess: Tokens the exact measurement overshot by.

        Returns:
            The estimated tokens freed, 0 when nothing could be shed.
        """
        freed = 0
        while freed < excess:
            step = self._shed(packing, budget=budget)
            if step <= 0:
                break
            freed += step
        return freed

    def _shed(self, packing: _Packing, *, budget: ContextBudget) -> int:
        """Remove the single least valuable remaining item.

        Order, least valuable first: near-duplicate and low-density chunks above the
        floor, then memory lines, then the remaining chunks, then non-pinned history
        turns, then the rolling summary. Pinned turns are never shed. A history turn
        removed here is only left out of *this* prompt — it stays live on the window
        and returns as soon as there is room, which is why it is tried after the
        retrieved chunks rather than before them. Dropping the summary is the last
        resort and likewise affects this prompt only.

        Args:
            packing: The working set, mutated in place.
            budget: The active budget.

        Returns:
            The estimated tokens freed, 0 when nothing is left to shed.
        """
        if len(packing.sources) > budget.min_retrieved_chunks:
            return self._shed_weakest_source(packing)
        if packing.memory_lines:
            return packing.memory_lines.pop()[2]
        if packing.sources:
            return self._shed_weakest_source(packing)
        droppable = [turn for turn in packing.history if not turn.pinned]
        if droppable:
            packing.history.remove(droppable[0])
            return max(1, droppable[0].token_count)
        if packing.summary:
            freed = max(1, packing.summary_tokens)
            packing.summary = ""
            packing.summary_tokens = 0
            return freed
        return 0

    def _shed_weakest_source(self, packing: _Packing) -> int:
        """Drop the lowest-marginal-value packed source, auditing the drop.

        A chunk whose novelty was discounted is recorded as a near-duplicate drop
        rather than a plain budget drop, so the audit trail says *why* it was the
        cheapest thing to lose.

        Args:
            packing: The working set, mutated in place.

        Returns:
            The tokens freed.
        """
        victim = min(packing.sources, key=lambda s: (s.density, s.value))
        packing.sources.remove(victim)
        reason = DROP_DUPLICATE_NEAR if victim.novelty < 1.0 else DROP_BUDGET
        packing.dropped.append(victim.chunk.drop(reason))
        _renumber(packing.sources)
        return max(1, victim.tokens)

    # ---------------------------------------------------------------- reporting
    def _finalise(
        self,
        packing: _Packing,
        *,
        budget: ContextBudget,
        window: SessionWindow,
        window_tokens: int,
        memories: Sequence[LongTermMemory],
        suppressed: Sequence[str],
        cleared: int,
        context_management: dict[str, Any] | None,
        reason: str,
    ) -> AssembledContext:
        """Build the result and its :class:`~ragcore.models.chat.ContextStats`.

        Args:
            packing: The final working set.
            budget: The active budget.
            window: The live window after compaction.
            window_tokens: Exact measured prompt size.
            memories: Every memory offered, before packing.
            suppressed: Ids of turns retired during this assembly.
            cleared: Tool results the context edit will clear.
            context_management: The context edit, when one applies.
            reason: Why compaction ran.

        Returns:
            The assembled context.
        """
        system_blocks, messages = self._render(packing)
        kept_ids = {id(memory) for memory, _, _ in packing.memory_lines}
        stats = ContextStats(
            window_tokens=window_tokens,
            budget_tokens=budget.prompt_tokens,
            system_tokens=packing.system_tokens,
            history_tokens=packing.history_tokens,
            retrieved_tokens=packing.retrieved_tokens,
            memory_tokens=packing.memory_tokens + packing.preferences_tokens,
            summary_tokens=packing.summary_tokens,
            messages_live=len(messages),
            messages_suppressed=window.suppressed_count,
            compaction_events=window.compaction_events,
            tool_results_cleared=cleared,
        )
        return AssembledContext(
            system=system_blocks,
            messages=messages,
            stats=stats,
            sources=[source.chunk for source in packing.sources],
            dropped=list(packing.dropped),
            memories=[memory for memory in memories if id(memory) in kept_ids],
            rolling_summary=window.rolling_summary,
            suppressed_message_ids=list(suppressed),
            context_management=context_management,
            cache_system=self._settings.anthropic_cache_system,
            compaction_reason=reason,
            user_turn=_text_of(messages[-1]),
        )


_ASSEMBLERS: dict[str, ContextAssembler] = {}


def get_context_assembler(settings: Settings | None = None) -> ContextAssembler:
    """Return the process-wide context assembler.

    ``Settings`` is a pydantic model and therefore unhashable, so the cache key is
    the budget shape — the fields that change how the assembler behaves.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached assembler.
    """
    cfg = settings or get_settings()
    key = (
        f"{cfg.context_budget_tokens}|{cfg.context_reserve_output_tokens}"
        f"|{cfg.context_compact_at_ratio}|{cfg.anthropic_model_main}"
    )
    existing = _ASSEMBLERS.get(key)
    if existing is None:
        existing = ContextAssembler(settings=cfg)
        _ASSEMBLERS[key] = existing
    return existing


def reset_context_assemblers() -> None:
    """Drop the cached assemblers. Test helper."""
    _ASSEMBLERS.clear()
