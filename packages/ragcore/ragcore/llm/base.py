"""The provider seam for chat completions.

:class:`LLMClient` was the only implementation for as long as Anthropic was the only
provider. :class:`LLMProvider` names the surface the rest of the platform actually
depends on — five methods, all keyword-only — so Azure OpenAI and Ollama can be
dropped in behind :func:`ragcore.llm.client.get_llm_client` without touching the
twenty modules that call it.

**Anthropic-shaped by design.** The message, system-block and tool shapes on this
interface are Anthropic's, because that is what every call site already builds and
what :mod:`ragcore.llm.prompts` emits. A non-Anthropic provider translates on the way
in and back on the way out; the alternative — a neutral intermediate format — would
mean rewriting every prompt in the repository to gain nothing the translation does not
already give.

**Features that do not port.** Extended thinking, context-management edits, the remote
MCP connector and prompt caching exist only on Anthropic. Rather than let a provider
quietly ignore them, :class:`ragcore.settings.Settings` refuses at construction when
one is enabled against a provider that cannot honour it, so a deployment either does
what its configuration says or fails before it serves a request. The parameters stay
on the signatures below so call sites need no branching: on a provider without them
they are inert, and settings validation is what guarantees they were never switched on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ragcore.llm.client import LLMResponse, StreamEvent
    from ragcore.settings import Settings

__all__ = ["LLMProvider"]

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """What the platform requires of a chat-completion backend.

    Implemented by :class:`ragcore.llm.client.LLMClient` (Anthropic) and
    :class:`ragcore.llm.openai_compatible.OpenAICompatibleClient` (Azure OpenAI and
    Ollama). ``runtime_checkable`` so tests can assert a new backend is complete
    before it is wired in; note that this checks method *names* only, which is why
    the conformance test compares signatures explicitly.
    """

    @property
    def settings(self) -> Settings:
        """Settings this provider was built from.

        Returns:
            The bound settings.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying transport."""
        ...

    async def complete(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        tools: Sequence[Any] | None = None,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
        thinking: bool = True,
        context_management: Mapping[str, Any] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        name: str = "llm.complete",
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Run one non-streaming completion.

        Args:
            system: System prompt, as a string or Anthropic text blocks.
            messages: Conversation turns, oldest first, never ending on assistant.
            tools: Tool definitions the model may call.
            mcp_servers: Remote MCP servers. Anthropic only.
            model: Model id. Defaults to the provider's configured main model.
            effort: Reasoning effort. Anthropic only.
            max_tokens: Output cap.
            cache_system: Place a prompt-cache breakpoint. Anthropic only.
            thinking: Enable extended thinking. Anthropic only.
            context_management: Context-edit payload. Anthropic only.
            tool_choice: Force or forbid tool use.
            name: Logical call-site name, used for tracing and metrics.
            metadata: Structural metadata for the trace; never raw content.

        Returns:
            The parsed response.
        """
        ...

    def stream(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        tools: Sequence[Any] | None = None,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
        thinking: bool = True,
        context_management: Mapping[str, Any] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        name: str = "llm.stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one completion as typed events.

        Args:
            system: See :meth:`complete`.
            messages: See :meth:`complete`.
            tools: See :meth:`complete`.
            mcp_servers: See :meth:`complete`.
            model: See :meth:`complete`.
            effort: See :meth:`complete`.
            max_tokens: See :meth:`complete`.
            cache_system: See :meth:`complete`.
            thinking: See :meth:`complete`.
            context_management: See :meth:`complete`.
            tool_choice: See :meth:`complete`.
            name: See :meth:`complete`.
            metadata: See :meth:`complete`.

        Returns:
            An async iterator of :class:`~ragcore.llm.client.StreamEvent`, ending
            with exactly one ``DONE`` carrying the assembled response.
        """
        ...

    async def structured(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        schema: type[T],
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
        thinking: bool = True,
        name: str = "llm.structured",
        metadata: Mapping[str, Any] | None = None,
    ) -> T:
        """Return a validated instance of ``schema``.

        Args:
            system: See :meth:`complete`.
            messages: See :meth:`complete`.
            schema: Pydantic model the answer must satisfy.
            model: See :meth:`complete`.
            effort: See :meth:`complete`.
            max_tokens: See :meth:`complete`.
            cache_system: See :meth:`complete`.
            thinking: See :meth:`complete`.
            name: See :meth:`complete`.
            metadata: See :meth:`complete`.

        Returns:
            The parsed and validated model instance.
        """
        ...

    async def classify(
        self,
        *,
        system: Any,
        text: str,
        labels: list[str],
        model: str | None = None,
        name: str = "llm.classify",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Pick exactly one label.

        Args:
            system: Instruction describing the labels.
            text: Content to classify.
            labels: Permitted answers; the return value is always one of these.
            model: See :meth:`complete`.
            name: See :meth:`complete`.
            metadata: See :meth:`complete`.

        Returns:
            One member of ``labels``, falling back to the first on an unusable
            answer so a classifier can never fail a request open.
        """
        ...

    async def count_tokens(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        model: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Count the prompt tokens this request would spend.

        The context packer budgets against this, so an under-count overflows the
        window and an over-count wastes it. Anthropic answers exactly, over the
        network; the OpenAI-compatible provider counts locally with ``tiktoken``,
        which is exact for OpenAI models and an approximation for anything served
        by Ollama.

        Args:
            system: See :meth:`complete`.
            messages: See :meth:`complete`.
            model: See :meth:`complete`.
            tools: Tool definitions, which occupy prompt tokens too.

        Returns:
            Prompt tokens, never negative.
        """
        ...
