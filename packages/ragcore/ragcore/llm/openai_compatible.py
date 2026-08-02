"""Chat completions for Azure OpenAI and Ollama.

One class serves both, because both speak the OpenAI chat-completions schema: Azure
through ``AsyncAzureOpenAI`` (deployment names in place of model ids, ``api-version``
on every request), Ollama through ``AsyncOpenAI`` pointed at its ``/v1`` endpoint. The
differences that remain are narrow enough to name in one enum rather than justify two
classes — see :attr:`OpenAICompatibleClient.flavour`.

**The translation is the substance of this module.** Every call site in the platform
builds Anthropic-shaped requests, so the work is turning those into OpenAI-shaped ones
and the answers back again:

``system``
    Anthropic takes a top-level list of text blocks; OpenAI takes a leading message
    with ``role="system"``. Cache breakpoints are dropped, having no meaning here.
``tool_result``
    Anthropic packs results into the *content list of a user message*; OpenAI wants one
    separate ``role="tool"`` message per result. One inbound message therefore fans out
    to several, which is why :func:`_to_openai_messages` returns a list rather than
    mapping one-to-one.
``tool_use``
    Anthropic carries a parsed ``input`` object; OpenAI carries ``arguments`` as a JSON
    *string*, so arguments are dumped on the way out and parsed on the way back. A
    model that emits malformed JSON yields an empty argument object rather than raising,
    because the tool dispatcher already refuses calls whose arguments fail validation
    and a hard error here would lose the rest of the turn.
``finish_reason``
    Mapped onto Anthropic's ``stop_reason`` vocabulary so
    :class:`~ragcore.llm.client.LLMResponse` means the same thing whoever produced it —
    the orchestrator branches on ``stop_reason`` and must not learn two dialects.

**Token counting is local.** Anthropic has a counting endpoint; OpenAI does not, so
:meth:`~OpenAICompatibleClient.count_tokens` uses ``tiktoken``. That is exact for
OpenAI models and an *approximation* for whatever Ollama is serving, since Llama,
Mistral and Qwen tokenise differently. The context packer budgets against this number,
so the approximation is deliberately biased to over-count — see
:data:`_OLLAMA_TOKEN_SAFETY`.

**Context edits are performed locally.** ``clear_tool_uses`` is a server-side edit on
Anthropic and has no OpenAI equivalent, so
:meth:`~OpenAICompatibleClient._apply_context_edits` applies it to the outgoing
messages instead: stale tool results have their bodies replaced before the request
leaves. They are replaced rather than removed because OpenAI rejects an assistant turn
whose ``tool_calls`` have no answering ``role="tool"`` message — dropping them would
turn a context saving into a 400.

**What this provider does not do.** Extended thinking, the remote MCP connector,
prompt caching and context *compaction* are Anthropic features with no counterpart
here. Compaction is a server-side summarisation pass, not a rewrite rule, so unlike
``clear_tool_uses`` it cannot be reproduced locally. The parameters stay on the
signatures and are inert; :class:`ragcore.settings.Settings` refuses to construct when
one is *enabled* against this provider, so being inert here is unreachable rather than
silent.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from enum import StrEnum
from typing import Any, TypeVar

import openai
import tiktoken
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ragcore.errors import ConfigError, RagError
from ragcore.llm.client import (
    LLMRefusedError,
    LLMResponse,
    LLMUsage,
    StreamEvent,
    StreamEventType,
    _first_json_object,
    _json_schema_for,
    _log_retry,
    _normalize_messages,
    _normalize_system,
)
from ragcore.logging import get_logger
from ragcore.observability.langfuse import get_tracer
from ragcore.observability.metrics import observe_llm_call
from ragcore.settings import Settings, get_settings

__all__ = ["OpenAICompatibleClient", "OpenAIFlavour"]

_log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Multiplier applied to a tiktoken count when Ollama is serving. Llama-family
#: tokenisers typically produce more tokens than ``o200k_base`` for the same text, and
#: the context packer treats this number as a budget: over-counting wastes a little
#: window, under-counting overflows it and the request fails outright.
_OLLAMA_TOKEN_SAFETY = 1.15

#: Per-message framing overhead in the OpenAI chat format (role, separators). The
#: published figure for chat models; close enough that the safety factor above covers
#: the rest.
_TOKENS_PER_MESSAGE = 4

#: Body written over a tool result the context edit cleared. Anthropic's server-side
#: edit leaves a comparable marker; a bare empty string reads to the model as a tool
#: that returned nothing, which is a different and misleading fact.
_CLEARED_PLACEHOLDER = "[tool result cleared to reclaim context]"

#: The edit type this provider implements locally. Anything else in a
#: ``context_management`` payload is Anthropic-only and refused by settings.
_CLEAR_TOOL_USES = "clear_tool_uses_20250919"

#: OpenAI ``finish_reason`` to Anthropic ``stop_reason``. The orchestrator, the tool
#: loop and the output guard all branch on the Anthropic vocabulary.
_FINISH_REASONS: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class OpenAIFlavour(StrEnum):
    """Which OpenAI-compatible service is on the other end.

    Attributes:
        AZURE_OPENAI: Azure-hosted OpenAI. Deployment names stand in for model ids
            and structured outputs support a strict JSON schema.
        OLLAMA: A local Ollama daemon's ``/v1`` endpoint. No API key, no strict
            schema support, and token counts are approximate.
    """

    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"


def _is_retryable(exc: BaseException) -> bool:
    """Classify an exception as worth retrying.

    Ordered most-specific-first, and never string-matches a message. Mirrors
    :func:`ragcore.llm.client._is_retryable` so a provider swap does not quietly
    change failure behaviour.

    Args:
        exc: The raised exception.

    Returns:
        True for rate limits, 5xx and connection failures; False for 404 and every
        other 4xx.
    """
    if isinstance(exc, openai.NotFoundError):
        return False
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, openai.APIConnectionError)


def _system_text(system: Any, *, cache: bool) -> str:
    """Flatten Anthropic system blocks into one string.

    Args:
        system: A string, a block mapping, or a sequence of either.
        cache: Ignored. Accepted so the caller need not know the provider; prompt
            caching is an Anthropic feature and there is no breakpoint to place.

    Returns:
        The concatenated system text, empty when there is none.
    """
    del cache
    blocks = _normalize_system(system, cache=False)
    return "\n\n".join(str(block.get("text", "")) for block in blocks).strip()


def _block_text(content: Any) -> str:
    """Render a tool result's content as plain text.

    Anthropic allows a string or a list of blocks; OpenAI wants a string.

    Args:
        content: The ``content`` of a ``tool_result`` block.

    Returns:
        Flattened text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _to_openai_messages(system: Any, messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Translate Anthropic messages into OpenAI chat messages.

    A single inbound message can produce several outbound ones: Anthropic packs tool
    results into a user turn's content list, while OpenAI requires one ``role="tool"``
    message per result, and those must precede any accompanying user text.

    Args:
        system: System prompt in any shape :func:`_normalize_system` accepts.
        messages: Anthropic message mappings.

    Returns:
        Messages ready for ``chat.completions.create``.
    """
    out: list[dict[str, Any]] = []
    text = _system_text(system, cache=False)
    if text:
        out.append({"role": "system", "content": text})

    for message in messages:
        role = str(message.get("role", "user")) if isinstance(message, Mapping) else ""
        content = message.get("content") if isinstance(message, Mapping) else None

        if isinstance(content, str):
            out.append({"role": role or "user", "content": content})
            continue

        blocks = list(content) if isinstance(content, Sequence) else []
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            if kind == "text":
                texts.append(str(block.get("text", "")))
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif kind == "tool_result":
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": _block_text(block.get("content")),
                    }
                )
            # `thinking` and `redacted_thinking` blocks are Anthropic-only and have
            # nowhere to go; dropping them is the whole of the degradation.

        # Results first: OpenAI rejects a tool message that does not directly answer
        # an assistant turn's tool_calls.
        out.extend(results)
        joined = "\n".join(part for part in texts if part)
        if tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": joined or None,
                    "tool_calls": tool_calls,
                }
            )
        elif joined or not results:
            out.append({"role": role or "user", "content": joined})
    return out


def _to_openai_tools(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    """Translate Anthropic tool definitions into OpenAI function tools.

    Args:
        tools: Anthropic tool mappings with ``name``/``description``/``input_schema``.

    Returns:
        OpenAI tool definitions, or None when there are none.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name", "")),
                    "description": str(tool.get("description", "")),
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _to_openai_tool_choice(tool_choice: Mapping[str, Any] | None) -> Any:
    """Translate an Anthropic ``tool_choice`` into OpenAI's form.

    Args:
        tool_choice: Anthropic choice mapping, or None.

    Returns:
        An OpenAI ``tool_choice`` value, or None to leave it unset.
    """
    if not tool_choice:
        return None
    kind = str(tool_choice.get("type", ""))
    if kind == "any":
        return "required"
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": str(tool_choice["name"])},
        }
    return None


def _tool_calls_from(message: Any) -> list[dict[str, Any]]:
    """Read tool calls off an OpenAI assistant message.

    Args:
        message: The ``choices[0].message`` object.

    Returns:
        Calls in the platform's shape. Malformed argument JSON becomes an empty
        object rather than raising: the dispatcher validates arguments anyway, and
        failing here would discard the rest of the turn.
    """
    calls: list[dict[str, Any]] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        raw = getattr(function, "arguments", "") or ""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            _log.warning(
                "llm_tool_arguments_unparseable",
                tool=getattr(function, "name", ""),
                chars=len(raw),
            )
            parsed = {}
        calls.append(
            {
                "id": str(getattr(call, "id", "")),
                "name": str(getattr(function, "name", "")),
                "input": parsed if isinstance(parsed, dict) else {},
                "kind": "tool_use",
            }
        )
    return calls


def _clear_stale_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep: int,
    clear_inputs: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Apply the ``clear_tool_uses`` context edit locally.

    Anthropic performs this edit server-side; there is no OpenAI equivalent, so the
    provider does it before sending. The result is that stale tool output stops
    occupying the window on every provider, and ``context.py``'s reported "cleared"
    count describes something that actually happened.

    Content is **replaced, not removed**. OpenAI rejects a request whose assistant
    turn has ``tool_calls`` with no answering ``role="tool"`` message, so dropping
    the message would turn a context saving into a 400. Replacing the body keeps the
    conversation structurally valid while freeing the tokens the body occupied — the
    same thing the server-side edit does.

    Args:
        messages: Translated OpenAI messages, modified in place on a copy.
        keep: How many of the most recent tool-calling turns keep their results.
        clear_inputs: Also blank the arguments on the cleared calls, not just the
            results.

    Returns:
        A ``(messages, cleared)`` pair, where ``cleared`` counts the results whose
        bodies were replaced.
    """
    turns = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    if len(turns) <= max(keep, 0):
        return messages, 0

    stale_turns = turns[: len(turns) - max(keep, 0)]
    stale_ids: set[str] = set()
    out = [dict(message) for message in messages]
    for index in stale_turns:
        calls = [dict(call) for call in out[index].get("tool_calls") or []]
        for call in calls:
            identifier = str(call.get("id", ""))
            if identifier:
                stale_ids.add(identifier)
            if clear_inputs:
                function = dict(call.get("function") or {})
                function["arguments"] = "{}"
                call["function"] = function
        out[index]["tool_calls"] = calls

    cleared = 0
    for message in out:
        if message.get("role") != "tool":
            continue
        if str(message.get("tool_call_id", "")) not in stale_ids:
            continue
        message["content"] = _CLEARED_PLACEHOLDER
        cleared += 1
    return out, cleared


def _encoding_for(model: str) -> tiktoken.Encoding:
    """Resolve a tokeniser for a model id.

    Args:
        model: Model or deployment name.

    Returns:
        The model's encoding when tiktoken knows it, else ``o200k_base`` — the
        current OpenAI default and the closest available stand-in for a local model.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


class OpenAICompatibleClient:
    """Chat completions against Azure OpenAI or Ollama.

    Satisfies :class:`ragcore.llm.base.LLMProvider`. Construct through
    :func:`ragcore.llm.client.get_llm_client`, which picks the implementation from
    ``settings.llm_provider`` and caches per transport.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Build a client for the configured flavour.

        Args:
            settings: Active settings. Defaults to the process settings.
            client: Pre-built AsyncOpenAI-compatible client. Injectable for tests;
                one is constructed from settings otherwise.

        Raises:
            ConfigError: When the flavour's required coordinates are missing.
        """
        self._settings = settings or get_settings()
        self._flavour = OpenAIFlavour(self._settings.llm_provider)
        self._tracer = get_tracer(self._settings)
        self._client = client if client is not None else self._build_client()

    @property
    def settings(self) -> Settings:
        """Settings this client was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def flavour(self) -> OpenAIFlavour:
        """Which service this client talks to.

        Returns:
            The resolved flavour.
        """
        return self._flavour

    @property
    def raw_client(self) -> Any:
        """The underlying SDK client.

        Returns:
            The AsyncOpenAI or AsyncAzureOpenAI instance.
        """
        return self._client

    def _build_client(self) -> Any:
        """Construct the SDK client for this flavour.

        Returns:
            An AsyncOpenAI-compatible client.

        Raises:
            ConfigError: When required coordinates are absent.
        """
        cfg = self._settings
        timeout = float(cfg.llm_timeout_seconds)
        if self._flavour is OpenAIFlavour.AZURE_OPENAI:
            if not cfg.azure_openai_endpoint:
                msg = (
                    "azure_openai_endpoint is required when llm_provider='azure_openai'"
                )
                raise ConfigError(msg, code="llm_not_configured")
            return openai.AsyncAzureOpenAI(
                azure_endpoint=cfg.azure_openai_endpoint,
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.azure_openai_api_version,
                timeout=timeout,
                max_retries=0,  # tenacity owns retries, so the SDK must not double them
            )
        return openai.AsyncOpenAI(
            base_url=cfg.ollama_base_url.rstrip("/") + "/v1",
            # Ollama ignores the key but the SDK refuses to construct without one.
            api_key="ollama",
            timeout=timeout,
            max_retries=0,
        )

    async def aclose(self) -> None:
        """Close the underlying transport."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    def _retrying(self) -> AsyncRetrying:
        """Build the retry policy for one call.

        Returns:
            A tenacity ``AsyncRetrying`` configured from settings.
        """
        return AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_random_exponential(
                multiplier=self._settings.anthropic_retry_base_delay_seconds,
                max=self._settings.anthropic_retry_max_delay_seconds,
            ),
            stop=stop_after_attempt(self._settings.llm_max_retries + 1),
            reraise=True,
            before_sleep=_log_retry,
        )

    def _model(self, model: str | None) -> str:
        """Resolve the model or deployment to call.

        Args:
            model: Caller-supplied model id, or None for the configured default.

        Returns:
            The model id, mapped through ``llm_model_aliases`` so a call site asking
            for a Claude model by name reaches the configured equivalent here.
        """
        requested = model or self._settings.llm_model_main
        return self._settings.llm_model_aliases.get(requested, requested)

    def _max_tokens(self, max_tokens: int | None) -> int:
        """Resolve the output cap.

        Args:
            max_tokens: Caller-supplied cap, or None.

        Returns:
            The cap to send.
        """
        return int(max_tokens or self._settings.anthropic_max_tokens)

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """Send one non-streaming request with retries.

        Args:
            kwargs: Request payload.

        Returns:
            The SDK completion object.
        """
        retrying = self._retrying()

        async def _attempt() -> Any:
            return await self._client.chat.completions.create(**kwargs)

        return await retrying(_attempt)

    def _parse(self, completion: Any, *, requested_model: str) -> LLMResponse:
        """Turn an OpenAI completion into the platform's response type.

        Args:
            completion: The SDK completion.
            requested_model: Model asked for, used when the reply names none.

        Returns:
            The parsed response.
        """
        choices = getattr(completion, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        finish = str(getattr(choices[0], "finish_reason", "") or "") if choices else ""
        stop_reason = _FINISH_REASONS.get(finish, finish or "end_turn")

        text = str(getattr(message, "content", "") or "") if message else ""
        refusal = str(getattr(message, "refusal", "") or "") if message else ""
        refused = bool(refusal) or stop_reason == "refusal"
        if refused:
            stop_reason = "refusal"
            text = ""

        usage_obj = getattr(completion, "usage", None)
        cached = 0
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        usage = LLMUsage(
            # Match Anthropic's convention: input_tokens excludes the cached bucket
            # so the three input buckets stay additive.
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            cache_read_tokens=cached,
            cache_write_tokens=0,
            model=str(getattr(completion, "model", requested_model) or requested_model),
            settings=self._settings,
        )
        return LLMResponse(
            text=text,
            tool_calls=_tool_calls_from(message) if message else [],
            stop_reason=stop_reason,
            usage=usage,
            refused=refused,
            raw=completion,
            thinking="",
            stop_details=refusal or None,
        )

    def _record(
        self,
        *,
        name: str,
        response: LLMResponse,
        requested_model: str,
        latency_ms: float,
        trace_input: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Emit a Langfuse generation and Prometheus samples for one call.

        Identical in shape to :meth:`ragcore.llm.client.LLMClient._record`, so a
        dashboard does not need to know which provider served a turn.

        Args:
            name: Logical call site.
            response: The parsed response.
            requested_model: Model asked for.
            latency_ms: Wall-clock duration.
            trace_input: Structural request description, never raw content.
            metadata: Extra structural metadata from the caller.
        """
        usage = response.usage
        observe_llm_call(
            model=usage.model,
            operation=name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=usage.cost_usd(),
            latency_ms=latency_ms,
            outcome="refused" if response.refused else "ok",
        )
        self._tracer.generation(
            name,
            model=usage.model,
            input=dict(trace_input),
            output={
                "text_chars": len(response.text),
                "thinking_chars": 0,
                "tool_calls": [call.get("name") for call in response.tool_calls],
                "refused": response.refused,
            },
            usage=usage.as_dict(),
            metadata={
                "provider": str(self._flavour),
                "requested_model": requested_model,
                "served_model": usage.model,
                "stop_reason": response.stop_reason,
                "refused": response.refused,
                "latency_ms": round(latency_ms, 2),
                **dict(metadata or {}),
            },
        )

    @staticmethod
    def _trace_input(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Describe a request without copying any of its content.

        Args:
            kwargs: The request payload.

        Returns:
            Structural counts only.
        """
        messages = kwargs.get("messages") or []
        return {
            "model": kwargs.get("model"),
            "messages": len(messages),
            "tools": len(kwargs.get("tools") or []),
            "max_tokens": kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens"),
        }

    def _apply_context_edits(
        self,
        translated: list[dict[str, Any]],
        context_management: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Honour the context edits this provider can perform locally.

        Only ``clear_tool_uses`` is implementable here; compaction is a server-side
        summarisation pass with no local equivalent, and settings refuse it against
        this provider rather than let it be dropped.

        Args:
            translated: Messages already in OpenAI form.
            context_management: The caller's edit payload, or None.

        Returns:
            The messages, with stale tool results cleared when asked.
        """
        if not context_management:
            return translated
        edits = context_management.get("edits") or []
        wanted = [
            edit
            for edit in edits
            if isinstance(edit, Mapping) and edit.get("type") == _CLEAR_TOOL_USES
        ]
        if not wanted:
            return translated

        cleared_inputs = any(edit.get("clear_tool_inputs") for edit in wanted)
        out, cleared = _clear_stale_tool_results(
            translated,
            keep=self._settings.context_tool_result_ttl_turns,
            clear_inputs=cleared_inputs,
        )
        if cleared:
            _log.info(
                "llm_tool_results_cleared",
                provider=str(self._flavour),
                cleared=cleared,
                keep=self._settings.context_tool_result_ttl_turns,
            )
        return out

    def _build_request(
        self,
        *,
        system: Any,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        model: str | None,
        max_tokens: int | None,
        tool_choice: Mapping[str, Any] | None,
        context_management: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the request payload.

        Args:
            system: System prompt.
            messages: Conversation turns.
            tools: Tool definitions.
            model: Model override.
            max_tokens: Output cap.
            tool_choice: Tool-choice override.
            context_management: Context edits to apply locally before sending.

        Returns:
            Keyword arguments for ``chat.completions.create``.
        """
        resolved = self._model(model)
        translated = _to_openai_messages(system, _normalize_messages(messages))
        kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": self._apply_context_edits(translated, context_management),
            "max_completion_tokens": self._max_tokens(max_tokens),
        }
        translated = _to_openai_tools(tools)
        if translated:
            kwargs["tools"] = translated
            choice = _to_openai_tool_choice(tool_choice)
            if choice is not None:
                kwargs["tool_choice"] = choice
        return kwargs

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
            system: System prompt.
            messages: Conversation turns.
            tools: Tool definitions.
            mcp_servers: Ignored; Anthropic only.
            model: Model override.
            effort: Ignored; Anthropic only.
            max_tokens: Output cap.
            cache_system: Ignored; Anthropic only.
            thinking: Ignored; Anthropic only.
            context_management: Ignored; Anthropic only.
            tool_choice: Tool-choice override.
            name: Logical call site.
            metadata: Structural trace metadata.

        Returns:
            The parsed response.
        """
        del mcp_servers, effort, cache_system, thinking
        kwargs = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            context_management=context_management,
        )
        started = time.perf_counter()
        completion = await self._create(kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = self._parse(completion, requested_model=str(kwargs["model"]))
        self._record(
            name=name,
            response=response,
            requested_model=str(kwargs["model"]),
            latency_ms=latency_ms,
            trace_input=self._trace_input(kwargs),
            metadata=metadata,
        )
        return response

    async def stream(
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

        Tool calls arrive as deltas keyed by index, with ``arguments`` accumulating
        one fragment at a time; a ``TOOL_USE`` event is emitted only once the stream
        ends and every fragment is in hand, so a caller never sees a partial call.

        Args:
            system: System prompt.
            messages: Conversation turns.
            tools: Tool definitions.
            mcp_servers: Ignored; Anthropic only.
            model: Model override.
            effort: Ignored; Anthropic only.
            max_tokens: Output cap.
            cache_system: Ignored; Anthropic only.
            thinking: Ignored; Anthropic only.
            context_management: Ignored; Anthropic only.
            tool_choice: Tool-choice override.
            name: Logical call site.
            metadata: Structural trace metadata.

        Yields:
            Text deltas, then completed tool calls, then usage, then exactly one
            ``DONE`` carrying the assembled response.
        """
        del mcp_servers, effort, cache_system, thinking
        kwargs = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            context_management=context_management,
        )
        kwargs["max_completion_tokens"] = self._max_tokens(
            max_tokens or self._settings.anthropic_max_tokens_streaming
        )
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        requested = str(kwargs["model"])

        started = time.perf_counter()
        text_parts: list[str] = []
        partial: dict[int, dict[str, Any]] = {}
        served = requested
        finish = ""
        usage = LLMUsage(model=requested, settings=self._settings)

        try:
            stream = await self._create(kwargs)
            async for chunk in stream:
                served = str(getattr(chunk, "model", served) or served)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    cached = 0
                    details = getattr(chunk_usage, "prompt_tokens_details", None)
                    if details is not None:
                        cached = int(getattr(details, "cached_tokens", 0) or 0)
                    prompt = int(getattr(chunk_usage, "prompt_tokens", 0) or 0)
                    usage = LLMUsage(
                        input_tokens=max(prompt - cached, 0),
                        output_tokens=int(
                            getattr(chunk_usage, "completion_tokens", 0) or 0
                        ),
                        cache_read_tokens=cached,
                        model=served,
                        settings=self._settings,
                    )

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish = str(getattr(choice, "finish_reason", "") or "") or finish
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(str(piece))
                    yield StreamEvent(
                        type=StreamEventType.TEXT, text=str(piece), index=0
                    )

                for call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(call, "index", 0) or 0)
                    slot = partial.setdefault(index, {"id": "", "name": "", "args": ""})
                    if getattr(call, "id", None):
                        slot["id"] = str(call.id)
                    function = getattr(call, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            slot["name"] = str(function.name)
                        if getattr(function, "arguments", None):
                            slot["args"] += str(function.arguments)
        except Exception as exc:
            _log.warning("llm_stream_failed", error=type(exc).__name__)
            yield StreamEvent(type=StreamEventType.ERROR, error=type(exc).__name__)
            return

        tool_calls: list[dict[str, Any]] = []
        for index in sorted(partial):
            slot = partial[index]
            raw = slot["args"]
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                _log.warning("llm_tool_arguments_unparseable", tool=slot["name"])
                parsed = {}
            call = {
                "id": slot["id"],
                "name": slot["name"],
                "input": parsed if isinstance(parsed, dict) else {},
                "kind": "tool_use",
            }
            tool_calls.append(call)
            yield StreamEvent(
                type=StreamEventType.TOOL_USE, tool_call=call, index=index
            )

        stop_reason = _FINISH_REASONS.get(finish, finish or "end_turn")
        refused = stop_reason == "refusal"
        response = LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            refused=refused,
            raw=None,
            thinking="",
        )
        if refused:
            yield StreamEvent(
                type=StreamEventType.REFUSAL, stop_reason=stop_reason, refused=True
            )
        yield StreamEvent(
            type=StreamEventType.USAGE, usage=usage, stop_reason=stop_reason
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(
            name=name,
            response=response,
            requested_model=requested,
            latency_ms=latency_ms,
            trace_input=self._trace_input(kwargs),
            metadata=metadata,
        )
        yield StreamEvent(
            type=StreamEventType.DONE,
            usage=usage,
            stop_reason=stop_reason,
            refused=refused,
            response=response,
        )

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

        Azure OpenAI is asked for a strict JSON schema, which the service enforces.
        Ollama's ``/v1`` surface only offers free-form JSON mode, so the schema is
        stated in the system prompt and the answer is extracted with the same
        tolerant reader the Anthropic path uses.

        Args:
            system: System prompt.
            messages: Conversation turns.
            schema: Pydantic model the answer must satisfy.
            model: Model override.
            effort: Ignored; Anthropic only.
            max_tokens: Output cap.
            cache_system: Ignored; Anthropic only.
            thinking: Ignored; Anthropic only.
            name: Logical call site.
            metadata: Structural trace metadata.

        Returns:
            The validated model instance.

        Raises:
            LLMRefusedError: When the model refused.
            RagError: When no valid instance could be parsed.
        """
        del effort, cache_system, thinking
        json_schema = _json_schema_for(schema)
        instruction = (
            "Reply with a single JSON object and nothing else. It must satisfy this "
            f"JSON Schema:\n{json.dumps(json_schema)}"
        )
        blocks = _normalize_system(system, cache=False)
        merged = "\n\n".join(
            [*(str(b.get("text", "")) for b in blocks), instruction]
        ).strip()

        kwargs = self._build_request(
            system=merged,
            messages=messages,
            tools=None,
            model=model,
            max_tokens=max_tokens or self._settings.anthropic_max_tokens_structured,
            tool_choice=None,
        )
        if self._flavour is OpenAIFlavour.AZURE_OPENAI:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": False,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        completion = await self._create(kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = self._parse(completion, requested_model=str(kwargs["model"]))
        self._record(
            name=name,
            response=response,
            requested_model=str(kwargs["model"]),
            latency_ms=latency_ms,
            trace_input=self._trace_input(kwargs),
            metadata=metadata,
        )
        if response.refused:
            msg = "model refused to produce structured output"
            raise LLMRefusedError(msg, code="llm_refused")

        payload = _first_json_object(response.text)
        if payload is None:
            msg = "structured output did not contain a JSON object"
            raise RagError(msg, code="llm_unparseable_structured_output")
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            msg = "structured output did not satisfy the schema"
            raise RagError(msg, code="llm_invalid_structured_output") from exc

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
            labels: Permitted answers.
            model: Model override.
            name: Logical call site.
            metadata: Structural trace metadata.

        Returns:
            One member of ``labels``. An unusable answer falls back to the first
            label rather than raising, matching the Anthropic path: a classifier is
            always on a path that must produce a decision.

        Raises:
            ValueError: If ``labels`` is empty, which is a programming error.
        """
        if not labels:
            msg = "labels must not be empty"
            raise ValueError(msg)

        instruction = (
            "Answer with exactly one of these labels and nothing else: "
            + ", ".join(labels)
        )
        blocks = _normalize_system(system, cache=False)
        merged = "\n\n".join(
            [*(str(b.get("text", "")) for b in blocks), instruction]
        ).strip()

        kwargs = self._build_request(
            system=merged,
            messages=[{"role": "user", "content": text}],
            tools=None,
            model=model or self._settings.llm_model_cheap,
            max_tokens=self._settings.anthropic_max_tokens_classify,
            tool_choice=None,
        )
        started = time.perf_counter()
        completion = await self._create(kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = self._parse(completion, requested_model=str(kwargs["model"]))
        self._record(
            name=name,
            response=response,
            requested_model=str(kwargs["model"]),
            latency_ms=latency_ms,
            trace_input=self._trace_input(kwargs),
            metadata=metadata,
        )

        answer = response.text.strip().strip(".\"'").lower()
        for label in labels:
            if answer == label.lower():
                return label
        for label in labels:
            if label.lower() in answer:
                return label
        _log.warning("llm_classify_unusable", labels=len(labels), chars=len(answer))
        return labels[0]

    async def count_tokens(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        model: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Count the prompt tokens this request would spend.

        Counted locally with ``tiktoken``: exact for OpenAI models, approximate for
        anything Ollama serves, where the result is scaled by
        :data:`_OLLAMA_TOKEN_SAFETY` so the context packer errs toward a smaller
        prompt rather than an overflowing one.

        Args:
            system: System prompt.
            messages: Conversation turns.
            model: Model override.
            tools: Tool definitions, which occupy prompt tokens too.

        Returns:
            Prompt tokens, never negative.
        """
        resolved = self._model(model)
        encoding = _encoding_for(resolved)
        translated = _to_openai_messages(system, list(messages))

        total = 0
        for message in translated:
            total += _TOKENS_PER_MESSAGE
            content = message.get("content")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                total += len(encoding.encode(str(function.get("name", ""))))
                total += len(encoding.encode(str(function.get("arguments", ""))))

        for tool in _to_openai_tools(tools) or []:
            total += len(encoding.encode(json.dumps(tool)))

        if self._flavour is OpenAIFlavour.OLLAMA:
            total = int(total * _OLLAMA_TOKEN_SAFETY)
        return max(total, 0)
