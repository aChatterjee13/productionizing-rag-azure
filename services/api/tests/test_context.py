"""Context assembly: the budget is never exceeded and suppression is lossless.

These tests exercise the real :class:`app.rag.context.ContextAssembler` against a
deterministic fake tokenizer. The fake stands in for ``LLMClient.count_tokens`` only —
every packing, compaction and shedding decision under test is the production one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

from app.rag.context import (
    DROP_DUPLICATE_EXACT,
    DROP_RETRIEVED_CAP,
    ContextAssembler,
    ContextBudget,
)
from app.rag.memory.short_term import (
    InMemorySessionStore,
    SessionWindow,
    ShortTermMemory,
    TokenCounter,
)
from ragcore.dedupe import content_sha256, simhash_hex
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chat import Message, Role, ToolCall
from ragcore.models.chunk import ChunkPayload
from ragcore.models.memory import LongTermMemory, MemoryKind, UserProfile
from ragcore.models.retrieval import RetrievedChunk
from ragcore.settings import Settings

TENANT = "tenant-acme"
USER = "user-1"
SYSTEM = "Test answering engine. Answer only from the numbered sources, with [n]."
SUMMARY_TEXT = "Rolling summary: the user asked about travel policy and approvals."


# ------------------------------------------------------------------- fake model
@dataclass
class _FakeResponse:
    text: str
    refused: bool = False


class FakeLLM:
    """Deterministic stand-in for ``LLMClient``.

    ``count_tokens`` is a stable word count plus per-message framing, which makes the
    packing arithmetic reproducible while keeping the *shape* of the real tokenizer
    (longer text costs more, framing is not free).
    """

    def __init__(self, summary: str = SUMMARY_TEXT) -> None:
        """Initialise the fake with the summary it will always return."""
        self.summary = summary
        self.count_calls = 0
        self.complete_calls = 0

    @staticmethod
    def _words(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value.split())
        if isinstance(value, dict):
            return FakeLLM._words(value.get("text") or value.get("content"))
        if isinstance(value, list):
            return sum(FakeLLM._words(item) for item in value)
        return 0

    async def count_tokens(
        self,
        *,
        system: Any = None,
        messages: Any,
        model: str | None = None,
        tools: Any = None,
    ) -> int:
        self.count_calls += 1
        total = 5 + self._words(system)
        for message in messages:
            total += 4 + self._words(message)
        total += 25 * len(tools or [])
        return total

    async def complete(self, **kwargs: Any) -> _FakeResponse:
        self.complete_calls += 1
        return _FakeResponse(text=self.summary)


# ---------------------------------------------------------------------- helpers
def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "local",
        "redis_enabled": False,
        "langfuse_enabled": False,
        "context_budget_tokens": 1_400,
        "context_reserve_output_tokens": 400,
        "context_compact_at_ratio": 0.75,
        "context_max_history_turns": 20,
        "context_min_live_turns": 2,
        "context_summary_max_tokens": 120,
        "context_retrieved_budget_ratio": 0.55,
        "context_memory_budget_ratio": 0.10,
        "context_tool_result_ttl_turns": 2,
        "retrieval_snippet_chars": 600,
    }
    base.update(overrides)
    return Settings(**base)


def build_assembler(
    settings: Settings, llm: FakeLLM | None = None
) -> tuple[ContextAssembler, ShortTermMemory, FakeLLM]:
    model = llm or FakeLLM()
    counter = TokenCounter(model, settings=settings)
    short_term = ShortTermMemory(
        settings=settings,
        store=InMemorySessionStore(),
        counter=counter,
        llm=model,
    )
    assembler = ContextAssembler(
        settings=settings, llm=model, counter=counter, short_term=short_term
    )
    return assembler, short_term, model


def principal(**overrides: Any) -> Principal:
    fields: dict[str, Any] = {
        "user_id": USER,
        "tenant_id": TENANT,
        "roles": ["rag.user"],
        "groups": ["g-acme-engineering"],
        "max_classification": Classification.CONFIDENTIAL,
    }
    fields.update(overrides)
    return Principal(**fields)


def make_chunk(
    index: int,
    *,
    text: str,
    score: float,
    document_id: str | None = None,
    simhash: str | None = None,
) -> RetrievedChunk:
    payload = ChunkPayload.from_access_control(
        AccessControl(tenant_id=TENANT, classification=Classification.INTERNAL),
        chunk_id=f"doc-{index}::0000",
        document_id=document_id or f"doc-{index}",
        chunk_index=0,
        source_type="blob",
        source_id="src-1",
        source_uri=f"https://example.invalid/doc-{index}",
        title=f"Document {index}",
        text=text,
        content_sha256=content_sha256(text),
        simhash=simhash or simhash_hex(text),
        token_count=len(text.split()),
    )
    return RetrievedChunk(payload=payload, fusion_score=score, final_score=score)


async def fill_window(
    short_term: ShortTermMemory, *, turns: int, words: int = 40
) -> SessionWindow:
    window = SessionWindow.empty(tenant_id=TENANT, user_id=USER, session_id="session-1")
    for index in range(turns):
        role = Role.USER if index % 2 == 0 else Role.ASSISTANT
        await short_term.record_turn(
            window,
            role=role,
            content=f"turn {index} "
            + " ".join(f"word{index}x{n}" for n in range(words)),
            pii_redacted=True,
            message_id=f"m{index:02d}",
        )
    return window


def all_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(str(block.get("text", "")) for block in content)
    return "\n".join(parts)


# ------------------------------------------------------------------------ tests
async def test_budget_is_never_exceeded_under_heavy_pressure() -> None:
    settings = make_settings(
        context_budget_tokens=1_000, context_reserve_output_tokens=300
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=12, words=60)

    chunks = [
        make_chunk(
            i,
            text=" ".join(f"chunk{i}word{n}" for n in range(200)),
            score=1.0 - i / 20,
        )
        for i in range(10)
    ]
    memories = [
        LongTermMemory(
            memory_id=f"mem-{i}",
            user_id=USER,
            tenant_id=TENANT,
            kind=MemoryKind.PREFERENCE,
            text=f"Prefers detail level {i} " + " ".join(["filler"] * 30),
            pii_redacted=True,
        )
        for i in range(6)
    ]

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="What is the current meal allowance for European travel?",
        chunks=chunks,
        memories=memories,
        profile=UserProfile(user_id=USER, tenant_id=TENANT, preferred_style="bullets"),
        system=SYSTEM,
    )

    assert result.stats.budget_tokens == settings.context_prompt_budget_tokens
    assert result.stats.window_tokens <= result.stats.budget_tokens
    assert result.stats.utilisation <= 1.0
    # Something had to give, and every drop is auditable.
    assert result.dropped
    assert all(chunk.dropped_reason for chunk in result.dropped)


async def test_budget_respected_with_the_real_answer_prompt() -> None:
    settings = make_settings(
        context_budget_tokens=6_000, context_reserve_output_tokens=1_000
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=6)

    chunks = [
        make_chunk(
            i, text=" ".join(f"c{i}w{n}" for n in range(150)), score=0.9 - i / 20
        )
        for i in range(8)
    ]
    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Which approval is needed above 500 EUR?",
        chunks=chunks,
    )

    assert result.stats.window_tokens <= result.stats.budget_tokens
    assert result.stats.system_tokens > 0
    assert result.sources


async def test_suppression_preserves_pinned_turns_and_produces_a_summary() -> None:
    settings = make_settings()
    assembler, short_term, model = build_assembler(settings)
    window = await fill_window(short_term, turns=10)
    assert window.pin("m00")
    pinned_content = next(t for t in window.turns if t.message_id == "m00").content

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Remind me what we decided.",
        system=SYSTEM,
        force_compaction=True,
    )

    assert result.compacted
    assert "m00" not in result.suppressed_message_ids
    assert any(turn.message_id == "m00" for turn in window.live_turns)
    assert pinned_content in all_text(result.messages)

    # A summary was produced by MODEL_FAST and now stands in for the retired turns.
    assert window.rolling_summary == SUMMARY_TEXT
    assert model.complete_calls >= 1
    assert result.stats.summary_tokens > 0
    assert SUMMARY_TEXT in all_text(result.messages)
    assert result.stats.messages_suppressed == len(result.suppressed_message_ids)


async def test_periodic_compaction_fires_without_budget_pressure() -> None:
    settings = make_settings(
        context_budget_tokens=200_000,
        context_reserve_output_tokens=16_000,
        context_compact_every_n_turns=4,
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=8, words=5)
    assert window.turns_since_compaction >= 4

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Anything else?",
        system=SYSTEM,
    )

    assert result.compaction_reason == "periodic"
    assert result.suppressed_message_ids
    assert window.turns_since_compaction == 0
    assert len(window.live_turns) == settings.context_min_live_turns


async def test_no_compaction_below_the_floor() -> None:
    settings = make_settings(
        context_budget_tokens=200_000, context_reserve_output_tokens=16_000
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=2, words=5)

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Hello again",
        system=SYSTEM,
        force_compaction=True,
    )

    assert result.compaction_reason == ""
    assert result.suppressed_message_ids == []
    assert len(window.live_turns) == 2


async def test_exact_duplicate_chunks_are_dropped_and_audited() -> None:
    settings = make_settings(
        context_budget_tokens=200_000, context_reserve_output_tokens=16_000
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=2, words=5)

    shared = "The meal allowance for European travel is EUR 60 per day."
    chunks = [
        make_chunk(1, text=shared, score=0.9),
        make_chunk(2, text=shared, score=0.8, document_id="doc-other"),
        make_chunk(
            3, text="Approvals above EUR 500 need director sign-off.", score=0.7
        ),
    ]
    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="What is the allowance?",
        chunks=chunks,
        system=SYSTEM,
    )

    assert len(result.sources) == 2
    assert [chunk.dropped_reason for chunk in result.dropped] == [DROP_DUPLICATE_EXACT]


async def test_near_duplicates_are_shed_before_novel_chunks() -> None:
    settings = make_settings(
        context_budget_tokens=700, context_reserve_output_tokens=400
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=2, words=5)

    body = " ".join(f"policy{n}" for n in range(120))
    novel = " ".join(f"unrelated{n}" for n in range(120))
    twin_hash = simhash_hex(body)
    chunks = [
        make_chunk(1, text=body, score=0.95, simhash=twin_hash),
        # Same simhash, different bytes: a re-publication of the same passage.
        make_chunk(2, text=body + " restated", score=0.90, simhash=twin_hash),
        make_chunk(3, text=novel, score=0.85),
    ]

    packed, _ = await assembler.rank_sources(chunks)
    assert packed[0].novelty == 1.0
    assert packed[1].novelty < 1.0  # the re-publication carries little new signal
    assert packed[2].novelty == 1.0
    # Lower marginal value than the lower-scored but novel chunk.
    assert packed[1].density < packed[2].density

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="What does the policy say?",
        chunks=chunks,
        system=SYSTEM,
    )

    kept = {chunk.payload.document_id for chunk in result.sources}
    assert "doc-1" in kept
    assert "doc-3" in kept
    assert "doc-2" not in kept
    assert [chunk.dropped_reason for chunk in result.dropped] == [DROP_RETRIEVED_CAP]
    assert result.stats.window_tokens <= result.stats.budget_tokens


async def test_stale_tool_results_are_cleared_by_the_context_edit() -> None:
    settings = make_settings(
        context_budget_tokens=200_000,
        context_reserve_output_tokens=16_000,
        context_tool_result_ttl_turns=1,
    )
    assembler, _short_term, _ = build_assembler(settings)
    window = SessionWindow.empty(
        tenant_id=TENANT, user_id=USER, session_id="session-tools"
    )
    now = datetime.now(UTC)
    for index in range(4):
        window.append(
            Message(
                message_id=f"t{index}",
                session_id="session-tools",
                role=Role.USER if index % 2 == 0 else Role.ASSISTANT,
                content=f"turn {index}",
                token_count=6,
                created_at=now + timedelta(minutes=index),
                tool_calls=[
                    ToolCall(
                        tool_call_id=f"tc{index}",
                        tool_name="order_lookup",
                        kind="rest",
                        latency_ms=12.0,
                    )
                ],
            )
        )

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="And now?",
        system=SYSTEM,
    )

    assert result.stats.tool_results_cleared >= 1
    assert result.context_management is not None
    edits = result.context_management["edits"]
    assert edits[0]["type"] == "clear_tool_uses_20250919"


async def test_cache_control_sits_on_the_stable_prefix_only() -> None:
    settings = make_settings(
        context_budget_tokens=200_000,
        context_reserve_output_tokens=16_000,
        context_cache_history_min_tokens=10,
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=4, words=20)

    first = await assembler.assemble(
        principal=principal(),
        window=window,
        question="First question?",
        system=SYSTEM,
    )
    second = await assembler.assemble(
        principal=principal(),
        window=window,
        question="A completely different second question?",
        system=SYSTEM,
    )

    # The system prefix is byte-stable across turns, which is what makes it cacheable.
    assert first.system == second.system
    assert first.system[-1]["cache_control"] == {"type": "ephemeral"}
    assert first.cache_system is True

    # Volatile content never enters the system prefix.
    assert "First question?" not in first.system[-1]["text"]

    # The second breakpoint lands on the last stable history turn, not on the
    # volatile final user turn.
    breakpointed = [
        index
        for index, message in enumerate(first.messages)
        if not isinstance(message["content"], str)
    ]
    assert breakpointed == [len(first.messages) - 2]
    assert first.messages[-1]["role"] == "user"
    assert isinstance(first.messages[-1]["content"], str)


async def test_stats_are_fully_populated() -> None:
    settings = make_settings(
        context_budget_tokens=200_000, context_reserve_output_tokens=16_000
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=4, words=20)
    window.rolling_summary = SUMMARY_TEXT
    window.summary_tokens = 0

    memories = [
        LongTermMemory(
            memory_id="mem-a",
            user_id=USER,
            tenant_id=TENANT,
            kind=MemoryKind.FACT,
            text="Works in the Munich office.",
            pii_redacted=True,
        )
    ]
    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Where do I file expenses?",
        chunks=[make_chunk(1, text="File expenses in Concur.", score=0.9)],
        memories=memories,
        profile=UserProfile(
            user_id=USER, tenant_id=TENANT, preferred_language="en", summary="Engineer"
        ),
        system=SYSTEM,
    )

    stats = result.stats
    assert stats.system_tokens > 0
    assert stats.history_tokens > 0
    assert stats.retrieved_tokens > 0
    assert stats.memory_tokens > 0
    assert stats.summary_tokens > 0
    assert stats.messages_live == len(result.messages)
    assert stats.budget_tokens == settings.context_prompt_budget_tokens
    assert 0.0 < stats.utilisation <= 1.0
    assert result.memories == memories


async def test_message_sequence_is_valid_after_arbitrary_suppression() -> None:
    settings = make_settings(
        context_budget_tokens=200_000,
        context_reserve_output_tokens=16_000,
        context_min_live_turns=1,
    )
    assembler, short_term, _ = build_assembler(settings)
    window = await fill_window(short_term, turns=6, words=5)
    # Retire the very first user turn so the survivors start on an assistant turn.
    window.suppress(["m00"], summary=SUMMARY_TEXT, summary_tokens=12)

    result = await assembler.assemble(
        principal=principal(),
        window=window,
        question="Continue please",
        system=SYSTEM,
    )

    roles = [message["role"] for message in result.messages]
    assert roles[0] == "user"
    assert roles[-1] == "user"
    assert all(first != second for first, second in pairwise(roles))


def test_budget_is_derived_entirely_from_settings() -> None:
    settings = make_settings(
        context_budget_tokens=100_000,
        context_reserve_output_tokens=20_000,
        context_compact_at_ratio=0.5,
    )
    budget = ContextBudget.from_settings(settings)

    assert budget.prompt_tokens == 80_000
    assert budget.compact_at_tokens == 40_000
    assert budget.retrieved_tokens == int(
        80_000 * settings.context_retrieved_budget_ratio
    )
    assert budget.memory_tokens == int(80_000 * settings.context_memory_budget_ratio)
    assert budget.headroom(80_001) == 0


async def test_token_counter_memoises_and_never_estimates() -> None:
    settings = make_settings()
    model = FakeLLM()
    counter = TokenCounter(model, settings=settings)

    first = await counter.count_text("the same sentence measured twice")
    second = await counter.count_text("the same sentence measured twice")

    assert first == second
    assert counter.calls == 1
    assert await counter.count_text("   ") == 0
    assert counter.calls == 1
