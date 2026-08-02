"""Chat surface: messages, tool calls, requests, context stats and guardrail events.

:class:`ContextStats` and :class:`GuardrailEvent` exist so requirements #3, #5 and #9
are *observable* rather than implicit: the UI can show exactly how many turns were
suppressed, how the token budget was spent, and which guardrail fired.

The SSE event names emitted by ``POST /api/v1/chat`` are enumerated in
:class:`SSEEvent`; a client must ignore unknown event names.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.retrieval import Citation, MetadataFilter

__all__ = [
    "ChatRequest",
    "ContextStats",
    "GuardrailAction",
    "GuardrailEvent",
    "GuardrailKind",
    "GuardrailStage",
    "Message",
    "Role",
    "SSEEvent",
    "ToolCall",
]


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class Role(StrEnum):
    """Who authored a stored message.

    Only user and assistant turns are persisted; the system prompt is rebuilt from
    :mod:`ragcore.llm.prompts` on every request so it stays a stable cache prefix.
    """

    USER = "user"
    ASSISTANT = "assistant"


class SSEEvent(StrEnum):
    """Event names emitted by ``POST /api/v1/chat`` as SSE ``event:`` fields.

    Each carries a JSON ``data:`` payload. Clients must ignore unknown names so the
    server can add events without breaking older web builds.
    """

    SESSION = "session"
    RETRIEVAL = "retrieval"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOKEN = "token"
    CITATIONS = "citations"
    CONTEXT_STATS = "context_stats"
    GUARDRAIL = "guardrail"
    USAGE = "usage"
    DONE = "done"
    ERROR = "error"


class GuardrailStage(StrEnum):
    """Pipeline stage a guardrail fired in."""

    INPUT = "input"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"


class GuardrailKind(StrEnum):
    """Which guardrail fired."""

    PII = "pii"
    INJECTION = "injection"
    OOD = "ood"
    CONTRADICTION = "contradiction"
    CLASSIFICATION = "classification"
    GROUNDEDNESS = "groundedness"
    SIZE = "size"


class GuardrailAction(StrEnum):
    """What the guardrail did about it."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    WARN = "warn"
    CLARIFY = "clarify"


class ToolCall(BaseModel):
    """One tool invocation recorded against a message.

    ``arguments`` is persisted after PII redaction: model-authored arguments can echo
    user content, so they go through the same redaction path as message text.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="Matches the model's tool_use block id.")
    tool_name: str = Field(description="Name of the invoked tool.")
    kind: str = Field(description="rest | mcp | retrieval.")
    arguments: dict = Field(
        default_factory=dict, description="Redacted arguments the model supplied."
    )
    result_summary: str | None = Field(
        default=None, description="Short redacted preview of the result."
    )
    is_error: bool = Field(default=False, description="Whether the invocation failed.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Invocation latency.")
    created_at: datetime = Field(
        default_factory=_utcnow, description="When the invocation completed."
    )


class Message(BaseModel):
    """A persisted chat turn.

    Suppression is how context management stays lossless: an over-budget turn is
    marked ``suppressed`` and folded into the rolling summary rather than truncated,
    so it still exists in history and in the UI. ``pinned`` turns are never suppressed.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(description="Unique message id.")
    session_id: str = Field(description="Owning session id.")
    role: Role = Field(description="Author of this turn.")
    content: str = Field(
        description="Turn text. Always PII-redacted before it is logged or persisted."
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Verified citations for an assistant turn."
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations made while producing this turn.",
    )
    token_count: int = Field(default=0, ge=0, description="Tokens this turn occupies.")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation time.")
    suppressed: bool = Field(
        default=False,
        description="Dropped from the live window and folded into the rolling summary.",
    )
    pinned: bool = Field(
        default=False, description="Never suppressed, however tight the budget."
    )

    @property
    def is_live(self) -> bool:
        """Whether this turn is still in the live context window.

        Returns:
            True when the turn is pinned or not suppressed.
        """
        return self.pinned or not self.suppressed

    def to_anthropic_message(self) -> dict[str, Any]:
        """Render as an Anthropic ``messages`` entry.

        Returns:
            A mapping with ``role`` and ``content``. Citations and tool metadata are
            deliberately excluded — history is replayed as plain text and the tool
            loop rebuilds its own blocks.
        """
        return {"role": self.role.value, "content": self.content}


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(description="The user's turn.")
    session_id: str | None = Field(
        default=None,
        description="Existing session to continue. Omit to start a new one.",
    )
    filters: MetadataFilter | None = Field(
        default=None, description="Facet filter to apply on top of the ACL filter."
    )
    allow_tools: bool = Field(
        default=True, description="Permit the tool loop for this turn."
    )
    stream: bool = Field(
        default=True,
        description="Stream SSE events. When false the endpoint returns one JSON body.",
    )


class ContextStats(BaseModel):
    """Surfaced to the UI — requirements #3 and #5 must be observable, not implicit."""

    model_config = ConfigDict(extra="forbid")

    window_tokens: int = Field(
        default=0, ge=0, description="Tokens actually sent in this request's prompt."
    )
    budget_tokens: int = Field(
        default=0, ge=0, description="Token budget the assembler packed into."
    )
    system_tokens: int = Field(
        default=0, ge=0, description="Tokens in the system prompt."
    )
    history_tokens: int = Field(default=0, ge=0, description="Tokens in recent turns.")
    retrieved_tokens: int = Field(
        default=0, ge=0, description="Tokens in retrieved chunks."
    )
    memory_tokens: int = Field(
        default=0, ge=0, description="Tokens in long-term memory injections."
    )
    summary_tokens: int = Field(
        default=0, ge=0, description="Tokens in the rolling conversation summary."
    )
    messages_live: int = Field(
        default=0, ge=0, description="Turns present in the live window."
    )
    messages_suppressed: int = Field(
        default=0, ge=0, description="Turns folded into the summary instead of sent."
    )
    compaction_events: int = Field(
        default=0, ge=0, description="Compactions performed in this session so far."
    )
    cache_read_tokens: int = Field(
        default=0, ge=0, description="Prompt tokens served from the Anthropic cache."
    )
    cache_write_tokens: int = Field(
        default=0, ge=0, description="Prompt tokens written to the Anthropic cache."
    )
    tool_results_cleared: int = Field(
        default=0,
        ge=0,
        description=(
            "Tool results removed by the clear_tool_uses_20250919 context edit."
        ),
    )

    @property
    def utilisation(self) -> float:
        """Fraction of the budget the prompt consumed.

        Returns:
            ``window_tokens / budget_tokens``, or 0.0 when no budget is recorded.
        """
        if self.budget_tokens <= 0:
            return 0.0
        return self.window_tokens / self.budget_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens that came from cache.

        Returns:
            Cache reads divided by total prompt tokens, or 0.0 when nothing was sent.
        """
        total = self.window_tokens
        if total <= 0:
            return 0.0
        return min(1.0, self.cache_read_tokens / total)


class GuardrailEvent(BaseModel):
    """One guardrail decision, streamed to the UI and persisted with the message."""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(description="input | retrieval | output.")
    kind: str = Field(
        description=(
            "pii | injection | ood | contradiction | classification | groundedness."
        )
    )
    action: str = Field(description="allow | redact | block | warn | clarify.")
    detail: str = Field(
        description="Redacted explanation. Never contains the offending content."
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Entity type names involved, never entity values.",
    )
    score: float | None = Field(
        default=None, description="Guardrail score, when the check produced one."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="When the decision was made."
    )

    @classmethod
    def allow(
        cls, stage: GuardrailStage, kind: GuardrailKind, detail: str = ""
    ) -> Self:
        """Build a pass-through event, recorded so traces show the check ran.

        Args:
            stage: Pipeline stage.
            kind: Guardrail kind.
            detail: Optional redacted note.

        Returns:
            A :class:`GuardrailEvent` with action ``allow``.
        """
        return cls(
            stage=stage.value,
            kind=kind.value,
            action=GuardrailAction.ALLOW.value,
            detail=detail,
        )

    @property
    def blocked(self) -> bool:
        """Whether this event short-circuited the pipeline.

        Returns:
            True when the action is ``block``.
        """
        return self.action == GuardrailAction.BLOCK.value
