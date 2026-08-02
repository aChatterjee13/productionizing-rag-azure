"""Anthropic Messages API client, hardened against the 2026 API shape.

Every rule in `docs/CONTRACTS.md` LLM_FACTS is enforced here once, so no call
site has to remember it:

* adaptive thinking only (``{"type": "adaptive"}``); ``budget_tokens`` is a 400,
  and ``{"type": "disabled"}`` is only accepted at effort ``high`` or lower.
* effort nested inside ``output_config``, never top-level.
* no ``temperature`` / ``top_p`` / ``top_k`` -- they are 400s on
  ``claude-opus-5``. Steer with prompting instead.
* no assistant prefill: a trailing ``role: "assistant"`` message raises before a
  request is made.
* ``cache_control`` on the **final** system block only, and any caller-supplied
  breakpoint is stripped first.
* ``betas=["server-side-fallback-2026-07-01"]`` + ``fallbacks="default"`` on
  every call, which means every call goes through ``client.beta.messages``.
* ``stop_reason == "refusal"`` is checked **before** ``response.content`` is
  touched, and surfaces as ``refused=True`` with empty text.
* ``RateLimitError`` and 5xx retry with jittered exponential backoff;
  ``NotFoundError`` and other 4xx do not.
* every call emits a Langfuse generation plus Prometheus token/cost/latency
  samples.

Observability content policy: the Langfuse generation carries the **shape** of a
request (message count, block kinds, tool names, token counts, cost) and never
raw prompt or answer text, because that text has not necessarily passed PII
redaction yet. Callers that want traced content pass already-redacted values in
``metadata``.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ragcore.errors import RagError
from ragcore.llm.base import LLMProvider
from ragcore.llm.pricing import MODEL_CHEAP, MODEL_FAST, MODEL_MAIN, pricing_for
from ragcore.logging import get_logger
from ragcore.observability.langfuse import Tracer, get_tracer
from ragcore.observability.metrics import observe_llm_call
from ragcore.settings import Settings, get_settings

__all__ = [
    "BETA_COMPACTION",
    "BETA_CONTEXT_MANAGEMENT",
    "BETA_SERVER_FALLBACK",
    "CLEAR_TOOL_USES_EDIT",
    "COMPACT_EDIT",
    "FALLBACKS_DEFAULT",
    "MODEL_CHEAP",
    "MODEL_FAST",
    "MODEL_MAIN",
    "LLMClient",
    "LLMRefusedError",
    "LLMResponse",
    "LLMUsage",
    "StreamEvent",
    "StreamEventType",
    "clear_tool_uses_edit",
    "compaction_edit",
    "get_llm_client",
    "reset_llm_client_cache",
]

T = TypeVar("T", bound=BaseModel)

_log = get_logger(__name__)

#: Opt into server-side refusal fallbacks (scalar ``fallbacks="default"`` form).
BETA_SERVER_FALLBACK = "server-side-fallback-2026-07-01"
#: Required for ``context_management`` edits other than compaction.
BETA_CONTEXT_MANAGEMENT = "context-management-2025-06-27"
#: Required for the separate compaction edit.
BETA_COMPACTION = "compact-2026-01-12"
#: Context edit that clears stale tool results (requirement #3/#5).
CLEAR_TOOL_USES_EDIT = "clear_tool_uses_20250919"
#: Context edit that summarises history server-side. Not the same as clearing.
COMPACT_EDIT = "compact_20260112"
#: Scalar fallbacks mode: let the API pick the substitute model by category.
FALLBACKS_DEFAULT = "default"

#: Efforts at which ``thinking={"type": "disabled"}`` is accepted.
_DISABLE_THINKING_MAX_EFFORT = frozenset({"low", "medium", "high"})

#: JSON Schema keywords structured outputs reject; stripped before sending.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "default",
        "deprecated",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "patternProperties",
        "readOnly",
        "uniqueItems",
        "writeOnly",
    }
)
#: String formats structured outputs understand.
_SUPPORTED_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "time",
        "uri",
        "uuid",
    }
)

_CHARS_PER_TOKEN_ESTIMATE = 4


class LLMRefusedError(RagError):
    """The model declined the request and the caller needs a value, not a flag.

    :meth:`LLMClient.complete` and :meth:`LLMClient.stream` report refusals as
    ``refused=True``; :meth:`LLMClient.structured` cannot, because it must return
    a validated model, so it raises this instead.
    """

    status_code = 422
    code = "llm_refused"


class StreamEventType(StrEnum):
    """Kinds of event :meth:`LLMClient.stream` yields."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    USAGE = "usage"
    REFUSAL = "refusal"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class LLMUsage:
    """Token accounting for one model call.

    ``input_tokens`` excludes cached tokens, matching the Anthropic ``usage``
    object, so the three input buckets are additive.

    Attributes:
        input_tokens: Uncached input tokens.
        output_tokens: Generated tokens, thinking included.
        cache_read_tokens: ``usage.cache_read_input_tokens``.
        cache_write_tokens: ``usage.cache_creation_input_tokens``.
        model: Model that actually served the call. With server-side fallbacks
            this can differ from the requested model, and cost follows the
            serving model.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = MODEL_MAIN
    settings: Settings | None = field(default=None, repr=False, compare=False)

    def cost_usd(self) -> float:
        """Price this call.

        Returns:
            Cost in USD using the per-model rate table, with cache reads at
            0.1x and cache writes at 1.25x the input rate.
        """
        return pricing_for(self.model, self.settings).cost_usd(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """Every token billed for this call.

        Returns:
            Sum of the input, cache and output buckets.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def prompt_tokens(self) -> int:
        """Total prompt size regardless of how it was billed.

        Returns:
            Uncached plus cache-read plus cache-write input tokens.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def as_dict(self) -> dict[str, Any]:
        """Render usage for tracing and persistence.

        Returns:
            A JSON-serialisable mapping including derived cost.
        """
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_write_tokens,
            "total": self.total_tokens,
            "model": self.model,
            "cost_usd": round(self.cost_usd(), 8),
        }

    @classmethod
    def from_message(
        cls,
        message: Any,
        *,
        model: str,
        settings: Settings | None = None,
    ) -> LLMUsage:
        """Read usage off an Anthropic message.

        Args:
            message: The API response (or a test double exposing ``usage``).
            model: Requested model, used when the response does not name one.
            settings: Settings supplying the rate table.

        Returns:
            A populated :class:`LLMUsage`; missing fields count as zero.
        """
        usage = _attr(message, "usage")
        return cls(
            input_tokens=int(_attr(usage, "input_tokens", 0) or 0),
            output_tokens=int(_attr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(_attr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(_attr(usage, "cache_creation_input_tokens", 0) or 0),
            model=str(_attr(message, "model", model) or model),
            settings=settings,
        )


@dataclass(slots=True)
class LLMResponse:
    """One completed model call.

    Attributes:
        text: Concatenated text blocks. Empty when the model refused.
        tool_calls: ``{"id", "name", "input", "kind"}`` per tool-use block.
        stop_reason: Raw ``stop_reason`` from the API.
        usage: Token accounting.
        refused: True when ``stop_reason == "refusal"``.
        raw: The underlying SDK message, for callers that need to echo
            ``response.content`` back (thinking blocks, compaction blocks).
        thinking: Summarised reasoning, when ``anthropic_thinking_display`` asks
            for it.
        stop_details: Refusal category payload, which may be None even on a
            refusal.
    """

    text: str
    tool_calls: list[dict[str, Any]]
    stop_reason: str
    usage: LLMUsage
    refused: bool
    raw: Any
    thinking: str = ""
    stop_details: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model asked for at least one tool.

        Returns:
            True when ``tool_calls`` is non-empty.
        """
        return bool(self.tool_calls)


@dataclass(slots=True)
class StreamEvent:
    """One typed event from :meth:`LLMClient.stream`.

    Attributes:
        type: Which kind of event this is.
        text: Delta text for ``TEXT``/``THINKING``, or a redacted message for
            ``ERROR``.
        index: Content-block index the delta belongs to.
        tool_call: Completed tool call for ``TOOL_USE``.
        usage: Final token accounting for ``USAGE`` and ``DONE``.
        stop_reason: Final stop reason for ``USAGE``, ``REFUSAL`` and ``DONE``.
        refused: True on ``REFUSAL`` and on a ``DONE`` that followed one.
        response: The assembled :class:`LLMResponse`, set on ``DONE``.
        error: Exception class name for ``ERROR``.
    """

    type: StreamEventType
    text: str = ""
    index: int | None = None
    tool_call: dict[str, Any] | None = None
    usage: LLMUsage | None = None
    stop_reason: str | None = None
    refused: bool = False
    response: LLMResponse | None = None
    error: str | None = None


def clear_tool_uses_edit(*, clear_tool_inputs: bool = False) -> dict[str, Any]:
    """Build the ``context_management`` payload that clears stale tool results.

    Args:
        clear_tool_inputs: Also drop the ``tool_use`` parameters, not just the
            results.

    Returns:
        A payload for ``LLMClient.complete(context_management=...)``.
    """
    edit: dict[str, Any] = {"type": CLEAR_TOOL_USES_EDIT}
    if clear_tool_inputs:
        edit["clear_tool_inputs"] = True
    return {"edits": [edit]}


def compaction_edit() -> dict[str, Any]:
    """Build the ``context_management`` payload that enables server compaction.

    Compaction is separate from clearing: when it is on, the caller must append
    the whole ``response.content`` (compaction blocks included) back into
    ``messages``.

    Returns:
        A payload for ``LLMClient.complete(context_management=...)``.
    """
    return {"edits": [{"type": COMPACT_EDIT}]}


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain mapping.

    Args:
        obj: SDK model, mapping, or None.
        key: Field name.
        default: Value when absent.

    Returns:
        The field value or ``default``.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_retryable(exc: BaseException) -> bool:
    """Classify an exception as worth retrying.

    Ordered most-specific-first, and never string-matches a message.

    Args:
        exc: The raised exception.

    Returns:
        True for rate limits, 5xx and connection failures; False for 404 and
        every other 4xx.
    """
    if isinstance(exc, anthropic.NotFoundError):
        return False
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, anthropic.APIConnectionError)


def _normalize_system(
    system: str | Sequence[Any] | Mapping[str, Any] | None,
    *,
    cache: bool,
) -> list[dict[str, Any]]:
    """Render the system prompt as text blocks with at most one cache breakpoint.

    Any ``cache_control`` supplied by the caller is dropped: the breakpoint
    belongs on the final block and nowhere else, or the cached prefix splits.

    Args:
        system: A string, a block mapping, or a sequence of either.
        cache: Whether to place ``cache_control`` on the last block.

    Returns:
        A list of Anthropic system blocks; empty when there is no system text.
    """
    if system is None:
        raw: list[Any] = []
    elif isinstance(system, str | Mapping):
        raw = [system]
    else:
        raw = list(system)

    blocks: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            block = {k: v for k, v in item.items() if k != "cache_control"}
            block.setdefault("type", "text")
        else:
            block = {"type": "text", "text": str(item)}
        if not str(block.get("text", "")).strip():
            continue
        blocks.append(block)

    if blocks and cache:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def _normalize_messages(messages: Sequence[Any]) -> list[Any]:
    """Validate the message list and reject assistant prefill.

    Args:
        messages: Anthropic message dicts (or mappings).

    Returns:
        A shallow copy of the list.

    Raises:
        ValueError: If the list is empty or ends on an assistant turn, which is
            a 400 on ``claude-opus-5``.
    """
    normalized = [dict(m) if isinstance(m, Mapping) else m for m in messages]
    if not normalized:
        msg = "messages must not be empty"
        raise ValueError(msg)
    if _attr(normalized[-1], "role") == "assistant":
        msg = (
            "assistant prefill is not supported: a trailing role='assistant' "
            "message returns 400. Use structured outputs or a system instruction."
        )
        raise ValueError(msg)
    return normalized


def _prepare_mcp(
    mcp_servers: Sequence[Mapping[str, Any]],
    tools: Sequence[Any] | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Pair every MCP server with a matching ``mcp_toolset`` entry.

    Passing ``mcp_servers`` without a matching toolset is a validation error, so
    the missing halves are filled in here rather than at each call site.

    Args:
        mcp_servers: Connector entries, e.g. from
            :meth:`ragcore.models.tool.McpServerSpec.to_connector_entries`.
        tools: Existing tool definitions.

    Returns:
        A ``(servers, tools)`` pair ready to send.

    Raises:
        ValueError: If a server lacks a name or url, or two servers share a name.
    """
    servers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in mcp_servers:
        entry = dict(raw)
        entry.setdefault("type", "url")
        name = entry.get("name")
        url = entry.get("url")
        if not name or not url:
            msg = "each mcp_servers entry needs both 'name' and 'url'"
            raise ValueError(msg)
        if name in seen:
            msg = f"duplicate mcp server name {name!r}"
            raise ValueError(msg)
        seen.add(str(name))
        servers.append(entry)

    tool_list: list[Any] = [
        dict(t) if isinstance(t, Mapping) else t for t in tools or []
    ]
    declared = {
        _attr(t, "mcp_server_name")
        for t in tool_list
        if _attr(t, "type") == "mcp_toolset"
    }
    for name in seen:
        if name not in declared:
            tool_list.append({"type": "mcp_toolset", "mcp_server_name": name})
    return servers, tool_list


def _sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema keywords structured outputs reject.

    Also sets ``additionalProperties: false`` on every object node, which the
    API requires.

    Args:
        node: A schema fragment.

    Returns:
        A sanitised copy.
    """
    if isinstance(node, list):
        return [_sanitize_schema(item) for item in node]
    if not isinstance(node, Mapping):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        cleaned[key] = _sanitize_schema(value)
    if cleaned.get("type") == "object" or "properties" in cleaned:
        cleaned["additionalProperties"] = False
    return cleaned


def _json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Build a structured-output schema from a pydantic model.

    Args:
        model: The pydantic model the response must satisfy.

    Returns:
        A sanitised JSON Schema.
    """
    return _sanitize_schema(model.model_json_schema())


def _first_json_object(text: str) -> Any:
    """Parse the first JSON object in a text block.

    Args:
        text: Model output. With ``output_config.format`` this is already pure
            JSON; the brace scan is a defensive fallback.

    Returns:
        The parsed value.

    Raises:
        ValueError: If no JSON object could be parsed.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as exc:
                msg = "structured output was not valid JSON"
                raise ValueError(msg) from exc
        msg = "structured output contained no JSON object"
        raise ValueError(msg) from None


def _parse_message(message: Any, *, model: str, settings: Settings) -> LLMResponse:
    """Turn an SDK message into an :class:`LLMResponse`.

    ``stop_reason`` is inspected before ``content`` is touched, so a refusal
    (HTTP 200 with a possibly-empty content list) can never raise an IndexError.

    Args:
        message: The SDK message.
        model: Requested model id.
        settings: Settings supplying the rate table.

    Returns:
        The parsed response.
    """
    stop_reason = str(_attr(message, "stop_reason", "") or "")
    usage = LLMUsage.from_message(message, model=model, settings=settings)
    stop_details = _attr(message, "stop_details")

    if stop_reason == "refusal":
        return LLMResponse(
            text="",
            tool_calls=[],
            stop_reason=stop_reason,
            usage=usage,
            refused=True,
            raw=message,
            stop_details=stop_details,
        )

    texts: list[str] = []
    thinking: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in _attr(message, "content", []) or []:
        block_type = _attr(block, "type")
        if block_type == "text":
            texts.append(str(_attr(block, "text", "") or ""))
        elif block_type == "thinking":
            thinking.append(str(_attr(block, "thinking", "") or ""))
        elif block_type in {"tool_use", "mcp_tool_use"}:
            call = {
                "id": _attr(block, "id"),
                "name": _attr(block, "name"),
                "input": _attr(block, "input") or {},
                "kind": "mcp" if block_type == "mcp_tool_use" else "tool",
            }
            server = _attr(block, "server_name")
            if server:
                call["server_name"] = server
            tool_calls.append(call)

    return LLMResponse(
        text="".join(texts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
        refused=False,
        raw=message,
        thinking="".join(thinking),
        stop_details=stop_details,
    )


def _build_async_client(settings: Settings) -> anthropic.AsyncAnthropic:
    """Construct the SDK client with our retry policy, not its own.

    Args:
        settings: Settings supplying key, base URL and timeout.

    Returns:
        An ``AsyncAnthropic`` with ``max_retries=0`` so tenacity owns retries
        and backoff is not applied twice.
    """
    kwargs: dict[str, Any] = {
        "timeout": settings.anthropic_timeout_seconds,
        "max_retries": 0,
    }
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    if settings.anthropic_base_url:
        kwargs["base_url"] = settings.anthropic_base_url
    return anthropic.AsyncAnthropic(**kwargs)


class LLMClient:
    """Async Anthropic client with the platform's obligations built in."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            settings: Platform settings. Defaults to the process settings.
            client: Pre-built ``AsyncAnthropic`` (or a test double exposing
                ``messages`` and ``beta.messages``).
            tracer: Tracer to emit generations to. Defaults to
                :func:`ragcore.observability.get_tracer`.
        """
        self._settings = settings or get_settings()
        self._client = (
            client if client is not None else _build_async_client(self._settings)
        )
        self._tracer = tracer if tracer is not None else get_tracer(self._settings)

    # ------------------------------------------------------------------ helpers
    @property
    def settings(self) -> Settings:
        """Settings this client was built from.

        Returns:
            The bound :class:`ragcore.settings.Settings`.
        """
        return self._settings

    @property
    def raw_client(self) -> Any:
        """The underlying SDK client.

        Returns:
            The ``AsyncAnthropic`` instance or injected double.
        """
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP transport, if the client owns one."""
        closer = getattr(self._client, "close", None)
        if closer is None:
            return
        result = closer()
        if asyncio.iscoroutine(result):
            await result

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
            stop=stop_after_attempt(self._settings.anthropic_max_retries + 1),
            reraise=True,
            before_sleep=_log_retry,
        )

    def _namespace(self, *, use_beta: bool) -> Any:
        """Pick the stable or beta messages namespace.

        Args:
            use_beta: Whether any beta flag is in play. Fallbacks, the MCP
                connector and context management all require the beta endpoint.

        Returns:
            The namespace exposing ``create``/``stream``.
        """
        if use_beta:
            return self._client.beta.messages
        return self._client.messages

    def _thinking_param(self, *, enabled: bool, effort: str) -> dict[str, str] | None:
        """Build the ``thinking`` parameter.

        Args:
            enabled: Whether adaptive thinking is wanted.
            effort: Resolved effort level.

        Returns:
            ``{"type": "adaptive", ...}`` when enabled, ``{"type": "disabled"}``
            when disabling is legal at this effort, otherwise None (which runs
            adaptive by omission, because thinking is on by default).
        """
        if enabled:
            param = {"type": "adaptive"}
            display = self._settings.anthropic_thinking_display
            if display:
                param["display"] = display
            return param
        if effort in _DISABLE_THINKING_MAX_EFFORT:
            return {"type": "disabled"}
        _log.warning(
            "thinking_disable_ignored",
            effort=effort,
            reason="thinking cannot be disabled above effort 'high'",
        )
        return None

    def _betas(
        self, *, mcp: bool, context_management: Mapping[str, Any] | None
    ) -> list[str]:
        """Collect the beta flags this request needs.

        Args:
            mcp: Whether remote MCP servers are attached.
            context_management: The context-management payload, if any.

        Returns:
            An ordered, de-duplicated list of beta flags.
        """
        betas: list[str] = []
        if self._settings.anthropic_fallbacks_enabled:
            betas.append(BETA_SERVER_FALLBACK)
        if mcp:
            betas.append(self._settings.tool_mcp_beta_flag)
        if context_management:
            edits = context_management.get("edits") or []
            kinds = {_attr(edit, "type") for edit in edits}
            if kinds - {COMPACT_EDIT}:
                betas.append(BETA_CONTEXT_MANAGEMENT)
            if COMPACT_EDIT in kinds:
                betas.append(BETA_COMPACTION)
        return list(dict.fromkeys(betas))

    def _build_request(
        self,
        *,
        system: Any,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        mcp_servers: Sequence[Mapping[str, Any]] | None,
        model: str,
        effort: str,
        max_tokens: int,
        cache_system: bool,
        thinking: bool,
        context_management: Mapping[str, Any] | None,
        tool_choice: Mapping[str, Any] | None,
        output_format: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        """Assemble request kwargs.

        Args:
            system: System prompt as text or blocks.
            messages: Conversation turns.
            tools: Tool definitions.
            mcp_servers: Remote MCP connector entries.
            model: Resolved model id.
            effort: Resolved effort level.
            max_tokens: Resolved output cap.
            cache_system: Whether to place the cache breakpoint.
            thinking: Whether adaptive thinking is wanted.
            context_management: Context edits payload.
            tool_choice: Tool-choice payload.
            output_format: ``output_config.format`` payload.

        Returns:
            A ``(kwargs, use_beta)`` pair. Note that no sampling parameter is
            ever added: ``temperature``, ``top_p`` and ``top_k`` are 400s.
        """
        tool_list: list[Any] | None = list(tools) if tools else None
        servers: list[dict[str, Any]] | None = None
        if mcp_servers:
            servers, prepared = _prepare_mcp(mcp_servers, tool_list)
            tool_list = prepared

        output_config: dict[str, Any] = {"effort": effort}
        if output_format:
            output_config["format"] = dict(output_format)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _normalize_messages(messages),
            "output_config": output_config,
        }

        system_blocks = _normalize_system(
            system, cache=cache_system and self._settings.anthropic_cache_system
        )
        if system_blocks:
            kwargs["system"] = system_blocks

        thinking_param = self._thinking_param(enabled=thinking, effort=effort)
        if thinking_param is not None:
            kwargs["thinking"] = thinking_param

        if tool_list:
            kwargs["tools"] = tool_list
        if servers:
            kwargs["mcp_servers"] = servers
        if tool_choice:
            kwargs["tool_choice"] = dict(tool_choice)
        if context_management:
            kwargs["context_management"] = dict(context_management)

        betas = self._betas(mcp=bool(servers), context_management=context_management)
        if betas:
            kwargs["betas"] = betas
            if self._settings.anthropic_fallbacks_enabled:
                kwargs["fallbacks"] = FALLBACKS_DEFAULT
        return kwargs, bool(betas)

    def _resolve(
        self,
        *,
        model: str | None,
        effort: str | None,
        max_tokens: int | None,
        thinking: bool,
        default_model: str,
        default_effort: str,
        default_max_tokens: int,
    ) -> tuple[str, str, int, bool]:
        """Fill unset call parameters from settings.

        Args:
            model: Caller-supplied model or None.
            effort: Caller-supplied effort or None.
            max_tokens: Caller-supplied output cap or None.
            thinking: Caller-supplied thinking flag.
            default_model: Settings default for this call kind.
            default_effort: Settings default for this call kind.
            default_max_tokens: Settings default for this call kind.

        Returns:
            A ``(model, effort, max_tokens, thinking)`` tuple.
        """
        return (
            model or default_model,
            effort or default_effort,
            max_tokens or default_max_tokens,
            thinking and self._settings.anthropic_thinking,
        )

    async def _create(self, kwargs: dict[str, Any], *, use_beta: bool) -> Any:
        """Send one non-streaming request with retries.

        Args:
            kwargs: Request payload.
            use_beta: Whether to use the beta namespace.

        Returns:
            The SDK message.
        """
        namespace = self._namespace(use_beta=use_beta)
        retrying = self._retrying()

        # The SDK's ``create`` is wrapped by ``@required_args``, a *plain* ``def``
        # that returns a coroutine. ``inspect.iscoroutinefunction`` therefore
        # reports False, and tenacity's ``AsyncRetrying`` would take its
        # sync branch: it would build the coroutine, never await it, and hand the
        # raw coroutine object back as the "result" -- silently, with no request
        # ever sent. Wrapping the call in a real ``async def`` keeps tenacity on
        # its async branch so the request is awaited and failures stay retryable.
        async def _attempt() -> Any:
            return await namespace.create(**kwargs)

        return await retrying(_attempt)

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

        Args:
            name: Logical call site.
            response: The parsed response.
            requested_model: Model asked for, which a fallback may have replaced.
            latency_ms: Wall-clock duration.
            trace_input: Structural request description (no raw content).
            metadata: Extra structural metadata from the caller.
        """
        usage = response.usage
        outcome = "refused" if response.refused else "ok"
        observe_llm_call(
            model=usage.model,
            operation=name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=usage.cost_usd(),
            latency_ms=latency_ms,
            outcome=outcome,
        )
        merged: dict[str, Any] = {
            "requested_model": requested_model,
            "served_model": usage.model,
            "stop_reason": response.stop_reason,
            "refused": response.refused,
            "latency_ms": round(latency_ms, 2),
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            **dict(metadata or {}),
        }
        self._tracer.generation(
            name,
            model=usage.model,
            input=dict(trace_input),
            output={
                "text_chars": len(response.text),
                "thinking_chars": len(response.thinking),
                "tool_calls": [call.get("name") for call in response.tool_calls],
                "refused": response.refused,
            },
            usage=usage.as_dict(),
            metadata=merged,
        )

    @staticmethod
    def _trace_input(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Describe a request without copying any of its content.

        Args:
            kwargs: The request payload.

        Returns:
            A structural summary safe to persist in Langfuse.
        """
        messages = kwargs.get("messages") or []
        system_blocks = kwargs.get("system") or []
        tools = kwargs.get("tools") or []
        return {
            "message_count": len(messages),
            "roles": [_attr(m, "role") for m in messages],
            "system_blocks": len(system_blocks),
            "system_chars": sum(
                len(str(_attr(b, "text", "") or "")) for b in system_blocks
            ),
            "cached_prefix": any(_attr(b, "cache_control") for b in system_blocks),
            "tools": [_attr(t, "name") or _attr(t, "type") for t in tools],
            "mcp_servers": [_attr(s, "name") for s in kwargs.get("mcp_servers") or []],
            "effort": (kwargs.get("output_config") or {}).get("effort"),
            "max_tokens": kwargs.get("max_tokens"),
            "betas": kwargs.get("betas"),
            "context_management": bool(kwargs.get("context_management")),
        }

    # -------------------------------------------------------------------- calls
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
            system: System prompt: a string, a block, or a sequence of either.
                Only the final block gets a cache breakpoint.
            messages: Conversation turns. Must not end on an assistant turn.
            tools: Anthropic tool definitions.
            mcp_servers: Remote MCP connector entries; a matching
                ``mcp_toolset`` tool is added automatically and the MCP beta flag
                is set.
            model: Model id. Defaults to ``settings.anthropic_model_main``
                (``MODEL_MAIN``).
            effort: ``low``|``medium``|``high``|``xhigh``|``max``. Defaults to
                ``settings.anthropic_effort``.
            max_tokens: Output cap. Defaults to ``settings.anthropic_max_tokens``.
            cache_system: Place ``cache_control`` on the final system block.
            thinking: Use adaptive thinking. False disables it, which the API
                only allows at effort ``high`` or lower.
            context_management: e.g. :func:`clear_tool_uses_edit`.
            tool_choice: Anthropic ``tool_choice`` payload.
            name: Operation label for traces and metrics.
            metadata: Structural metadata for the trace. No raw user content.

        Returns:
            An :class:`LLMResponse`; ``refused=True`` with empty text when the
            model declined.
        """
        (
            resolved_model,
            resolved_effort,
            resolved_max,
            resolved_thinking,
        ) = self._resolve(
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            thinking=thinking,
            default_model=self._settings.anthropic_model_main,
            default_effort=self._settings.anthropic_effort,
            default_max_tokens=self._settings.anthropic_max_tokens,
        )
        kwargs, use_beta = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            mcp_servers=mcp_servers,
            model=resolved_model,
            effort=resolved_effort,
            max_tokens=resolved_max,
            cache_system=cache_system,
            thinking=resolved_thinking,
            context_management=context_management,
            tool_choice=tool_choice,
            output_format=None,
        )
        trace_input = self._trace_input(kwargs)
        started = time.perf_counter()
        try:
            message = await self._create(kwargs, use_beta=use_beta)
        except anthropic.APIError:
            observe_llm_call(
                model=resolved_model,
                operation=name,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                outcome="error",
            )
            _log.error("llm_call_failed", operation=name, model=resolved_model)
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = _parse_message(
            message, model=resolved_model, settings=self._settings
        )
        if response.refused:
            _log.warning(
                "llm_refused",
                operation=name,
                model=response.usage.model,
                category=_attr(response.stop_details, "category"),
            )
        self._record(
            name=name,
            response=response,
            requested_model=resolved_model,
            latency_ms=latency_ms,
            trace_input=trace_input,
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

        Event order is: interleaved ``THINKING``/``TEXT`` deltas and completed
        ``TOOL_USE`` calls, then ``REFUSAL`` if the model declined, then
        ``USAGE``, then exactly one ``DONE`` carrying the assembled
        :class:`LLMResponse`.

        The stream is retried only while nothing has been yielded yet; once a
        delta has reached the caller a failure emits ``ERROR`` and re-raises,
        because silently restarting would duplicate output.

        Args:
            system: See :meth:`complete`.
            messages: See :meth:`complete`.
            tools: See :meth:`complete`.
            mcp_servers: See :meth:`complete`.
            model: See :meth:`complete`.
            effort: See :meth:`complete`.
            max_tokens: Defaults to ``settings.anthropic_max_tokens_streaming``.
            cache_system: See :meth:`complete`.
            thinking: See :meth:`complete`.
            context_management: See :meth:`complete`.
            tool_choice: See :meth:`complete`.
            name: Operation label for traces and metrics.
            metadata: Structural metadata for the trace.

        Yields:
            :class:`StreamEvent` values.

        Raises:
            anthropic.APIError: Propagated after an ``ERROR`` event when the
                stream cannot be completed.
        """
        (
            resolved_model,
            resolved_effort,
            resolved_max,
            resolved_thinking,
        ) = self._resolve(
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            thinking=thinking,
            default_model=self._settings.anthropic_model_main,
            default_effort=self._settings.anthropic_effort,
            default_max_tokens=self._settings.anthropic_max_tokens_streaming,
        )
        kwargs, use_beta = self._build_request(
            system=system,
            messages=messages,
            tools=tools,
            mcp_servers=mcp_servers,
            model=resolved_model,
            effort=resolved_effort,
            max_tokens=resolved_max,
            cache_system=cache_system,
            thinking=resolved_thinking,
            context_management=context_management,
            tool_choice=tool_choice,
            output_format=None,
        )
        trace_input = self._trace_input(kwargs)
        namespace = self._namespace(use_beta=use_beta)

        attempt = 0
        produced = False
        started = time.perf_counter()
        final: Any = None
        while True:
            state: dict[int, dict[str, Any]] = {}
            try:
                async with namespace.stream(**kwargs) as stream_obj:
                    async for raw_event in stream_obj:
                        for event in _translate_stream_event(raw_event, state):
                            produced = True
                            yield event
                    final = await stream_obj.get_final_message()
            except Exception as exc:
                retry = (
                    not produced
                    and _is_retryable(exc)
                    and attempt < self._settings.anthropic_max_retries
                )
                if not retry:
                    observe_llm_call(
                        model=resolved_model,
                        operation=name,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        outcome="error",
                    )
                    _log.error(
                        "llm_stream_failed",
                        operation=name,
                        model=resolved_model,
                        error=type(exc).__name__,
                    )
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        text="the model stream failed",
                        error=type(exc).__name__,
                    )
                    raise
                attempt += 1
                delay = min(
                    self._settings.anthropic_retry_base_delay_seconds
                    * 2 ** (attempt - 1),
                    self._settings.anthropic_retry_max_delay_seconds,
                )
                jitter = delay * random.random()  # noqa: S311 - backoff jitter
                _log.warning(
                    "llm_stream_retry",
                    operation=name,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                await asyncio.sleep(jitter)
                continue
            break

        latency_ms = (time.perf_counter() - started) * 1000.0
        response = _parse_message(final, model=resolved_model, settings=self._settings)
        if response.refused:
            _log.warning("llm_refused", operation=name, model=response.usage.model)
            yield StreamEvent(
                type=StreamEventType.REFUSAL,
                stop_reason=response.stop_reason,
                refused=True,
                usage=response.usage,
            )
        yield StreamEvent(
            type=StreamEventType.USAGE,
            usage=response.usage,
            stop_reason=response.stop_reason,
            refused=response.refused,
        )
        self._record(
            name=name,
            response=response,
            requested_model=resolved_model,
            latency_ms=latency_ms,
            trace_input=trace_input,
            metadata=metadata,
        )
        yield StreamEvent(
            type=StreamEventType.DONE,
            usage=response.usage,
            stop_reason=response.stop_reason,
            refused=response.refused,
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
        """Run one completion constrained to a pydantic model.

        Uses ``output_config.format`` with a JSON Schema derived from the model,
        which is the supported replacement for assistant prefill.

        Args:
            system: See :meth:`complete`.
            messages: See :meth:`complete`.
            schema: Pydantic model the response must satisfy.
            model: Defaults to ``settings.anthropic_model_fast`` (``MODEL_FAST``).
            effort: Defaults to ``settings.anthropic_effort_fast``.
            max_tokens: Defaults to ``settings.anthropic_max_tokens_structured``.
            cache_system: See :meth:`complete`.
            thinking: See :meth:`complete`.
            name: Operation label for traces and metrics.
            metadata: Structural metadata for the trace.

        Returns:
            A validated instance of ``schema``.

        Raises:
            LLMRefusedError: If the model declined, since there is no value to
                return.
            ValueError: If the response was not parseable as the schema.
        """
        (
            resolved_model,
            resolved_effort,
            resolved_max,
            resolved_thinking,
        ) = self._resolve(
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            thinking=thinking,
            default_model=self._settings.anthropic_model_fast,
            default_effort=self._settings.anthropic_effort_fast,
            default_max_tokens=self._settings.anthropic_max_tokens_structured,
        )
        kwargs, use_beta = self._build_request(
            system=system,
            messages=messages,
            tools=None,
            mcp_servers=None,
            model=resolved_model,
            effort=resolved_effort,
            max_tokens=resolved_max,
            cache_system=cache_system,
            thinking=resolved_thinking,
            context_management=None,
            tool_choice=None,
            output_format={
                "type": "json_schema",
                "schema": _json_schema_for(schema),
            },
        )
        trace_input = self._trace_input(kwargs)
        started = time.perf_counter()
        message = await self._create(kwargs, use_beta=use_beta)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = _parse_message(
            message, model=resolved_model, settings=self._settings
        )
        self._record(
            name=name,
            response=response,
            requested_model=resolved_model,
            latency_ms=latency_ms,
            trace_input=trace_input,
            metadata={**dict(metadata or {}), "schema": schema.__name__},
        )
        if response.refused:
            msg = f"model declined the structured request for {schema.__name__}"
            raise LLMRefusedError(msg, detail={"schema": schema.__name__})
        parsed = _first_json_object(response.text)
        return schema.model_validate(parsed)

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
        """Assign one of ``labels`` to ``text`` using the cheap model.

        Order ``labels`` so the **safest** option is first: a refusal, an
        unparseable response, or an unknown label falls back to ``labels[0]``
        with a warning rather than raising into the pipeline.

        Args:
            system: Classification instructions (stable, so it caches).
            text: The content to classify. Passed in the user turn.
            labels: Allowed labels, safest first.
            model: Defaults to ``settings.anthropic_model_cheap``
                (``MODEL_CHEAP``).
            name: Operation label for traces and metrics.
            metadata: Structural metadata for the trace.

        Returns:
            One of ``labels``.

        Raises:
            ValueError: If ``labels`` is empty.
        """
        if not labels:
            msg = "classify requires at least one label"
            raise ValueError(msg)

        resolved_model = model or self._settings.anthropic_model_cheap
        kwargs, use_beta = self._build_request(
            system=system,
            messages=[{"role": "user", "content": text}],
            tools=None,
            mcp_servers=None,
            model=resolved_model,
            effort=self._settings.anthropic_effort_cheap,
            max_tokens=self._settings.anthropic_max_tokens_classify,
            cache_system=True,
            thinking=False,
            context_management=None,
            tool_choice=None,
            output_format={
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": list(labels)}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        )
        trace_input = self._trace_input(kwargs)
        started = time.perf_counter()
        message = await self._create(kwargs, use_beta=use_beta)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = _parse_message(
            message, model=resolved_model, settings=self._settings
        )
        self._record(
            name=name,
            response=response,
            requested_model=resolved_model,
            latency_ms=latency_ms,
            trace_input=trace_input,
            metadata={**dict(metadata or {}), "labels": list(labels)},
        )

        label: str | None = None
        if not response.refused:
            try:
                parsed = _first_json_object(response.text)
            except ValueError:
                label = None
            else:
                candidate = (
                    parsed.get("label") if isinstance(parsed, Mapping) else parsed
                )
                label = str(candidate) if candidate is not None else None

        if label in labels:
            return str(label)
        lowered = {value.lower(): value for value in labels}
        if label and label.lower() in lowered:
            return lowered[label.lower()]
        _log.warning(
            "classify_fallback",
            operation=name,
            model=resolved_model,
            refused=response.refused,
            chosen=labels[0],
        )
        return labels[0]

    async def count_tokens(
        self,
        *,
        system: Any = None,
        messages: Sequence[Any],
        model: str | None = None,
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Count prompt tokens with Claude's own tokenizer.

        Never uses ``tiktoken``, which is the wrong tokenizer for Claude. If the
        API is unreachable after retries the call degrades to a coarse
        characters/4 estimate and logs, so context assembly cannot hard-fail on
        a counting outage.

        Args:
            system: System prompt as text or blocks.
            messages: Conversation turns.
            model: Defaults to ``settings.anthropic_model_main``.
            tools: Tool definitions, which also consume prompt tokens.

        Returns:
            Prompt tokens.
        """
        resolved_model = model or self._settings.anthropic_model_main
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": [dict(m) if isinstance(m, Mapping) else m for m in messages],
        }
        system_blocks = _normalize_system(system, cache=False)
        if system_blocks:
            kwargs["system"] = system_blocks
        if tools:
            kwargs["tools"] = list(tools)

        # Wrapped in a real ``async def`` for the same reason as ``_create``: a
        # non-``async def`` callable sends tenacity down its sync branch, which
        # would return an un-awaited coroutine whose ``input_tokens`` reads as 0
        # and silently zeroes the context budget.
        async def _attempt() -> Any:
            return await self._client.messages.count_tokens(**kwargs)

        try:
            retrying = self._retrying()
            result = await retrying(_attempt)
        except anthropic.APIError:
            estimate = _estimate_tokens(system_blocks, kwargs["messages"])
            _log.warning(
                "count_tokens_estimated",
                model=resolved_model,
                estimate=estimate,
                exc_info=True,
            )
            return estimate
        return int(_attr(result, "input_tokens", 0) or 0)


def _estimate_tokens(system_blocks: Sequence[Any], messages: Sequence[Any]) -> int:
    """Estimate prompt tokens without calling the API.

    Args:
        system_blocks: Normalised system blocks.
        messages: Conversation turns.

    Returns:
        A coarse characters/4 estimate.
    """
    chars = sum(len(str(_attr(b, "text", "") or "")) for b in system_blocks)
    for message in messages:
        content = _attr(message, "content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, Iterable):
            for block in content:
                chars += len(str(_attr(block, "text", "") or ""))
    return max(1, chars // _CHARS_PER_TOKEN_ESTIMATE)


def _log_retry(state: RetryCallState) -> None:
    """Log a retry before tenacity sleeps.

    Args:
        state: Tenacity call state.
    """
    exc = state.outcome.exception() if state.outcome is not None else None
    _log.warning(
        "llm_call_retry",
        attempt=state.attempt_number,
        sleep=round(state.next_action.sleep, 3) if state.next_action else None,
        error=type(exc).__name__ if exc else None,
    )


def _translate_stream_event(
    raw_event: Any, state: dict[int, dict[str, Any]]
) -> list[StreamEvent]:
    """Convert one SDK stream event into zero or more typed events.

    Tool-use blocks arrive as a start event plus partial-JSON deltas, so their
    input is accumulated in ``state`` and emitted once the block stops.

    Args:
        raw_event: SDK stream event.
        state: Per-stream accumulator keyed by content-block index.

    Returns:
        The typed events to yield.
    """
    kind = _attr(raw_event, "type")
    index = _attr(raw_event, "index")

    if kind == "content_block_start":
        block = _attr(raw_event, "content_block")
        block_type = _attr(block, "type")
        if block_type in {"tool_use", "mcp_tool_use"} and index is not None:
            state[int(index)] = {
                "id": _attr(block, "id"),
                "name": _attr(block, "name"),
                "kind": "mcp" if block_type == "mcp_tool_use" else "tool",
                "server_name": _attr(block, "server_name"),
                "json": "",
                "input": _attr(block, "input") or {},
            }
        return []

    if kind == "content_block_delta":
        delta = _attr(raw_event, "delta")
        delta_type = _attr(delta, "type")
        if delta_type == "text_delta":
            return [
                StreamEvent(
                    type=StreamEventType.TEXT,
                    text=str(_attr(delta, "text", "") or ""),
                    index=index,
                )
            ]
        if delta_type == "thinking_delta":
            return [
                StreamEvent(
                    type=StreamEventType.THINKING,
                    text=str(_attr(delta, "thinking", "") or ""),
                    index=index,
                )
            ]
        if delta_type == "input_json_delta" and index is not None:
            entry = state.get(int(index))
            if entry is not None:
                entry["json"] += str(_attr(delta, "partial_json", "") or "")
        return []

    if kind == "content_block_stop" and index is not None:
        entry = state.pop(int(index), None)
        if entry is None:
            return []
        arguments = entry["input"]
        if entry["json"]:
            try:
                arguments = json.loads(entry["json"])
            except json.JSONDecodeError:
                _log.warning("tool_use_json_incomplete", tool=entry.get("name"))
                arguments = entry["input"]
        call = {
            "id": entry["id"],
            "name": entry["name"],
            "input": arguments,
            "kind": entry["kind"],
        }
        if entry.get("server_name"):
            call["server_name"] = entry["server_name"]
        return [StreamEvent(type=StreamEventType.TOOL_USE, tool_call=call, index=index)]

    return []


_CLIENTS: dict[tuple[Any, ...], LLMProvider] = {}


def get_llm_client(settings: Settings | None = None) -> LLMProvider:
    """Return the process-wide chat-completion client for these settings.

    Dispatches on ``llm_provider``: Anthropic gets :class:`LLMClient`, Azure OpenAI
    and Ollama share
    :class:`~ragcore.llm.openai_compatible.OpenAICompatibleClient`. The import is
    deferred because that module imports this one for the shared response types.

    Args:
        settings: Settings to build from. Defaults to the process settings.

    Returns:
        A cached client satisfying :class:`~ragcore.llm.base.LLMProvider`.
        ``Settings`` is unhashable, so the cache is keyed on the fields that
        actually change the transport — the provider included, so switching it in a
        test never hands back the previous backend.
    """
    cfg = settings or get_settings()
    key: tuple[Any, ...] = (
        cfg.llm_provider,
        cfg.anthropic_base_url,
        cfg.anthropic_api_key,
        cfg.anthropic_timeout_seconds,
        cfg.anthropic_max_retries,
        cfg.azure_openai_endpoint,
        cfg.azure_openai_api_key,
        cfg.azure_openai_api_version,
        cfg.ollama_base_url,
        cfg.llm_timeout_seconds,
        cfg.llm_max_retries,
    )
    client = _CLIENTS.get(key)
    if client is None:
        if cfg.llm_provider == "anthropic":
            client = LLMClient(cfg)
        else:
            from ragcore.llm.openai_compatible import OpenAICompatibleClient

            client = OpenAICompatibleClient(cfg)
        _CLIENTS[key] = client
    return client


def reset_llm_client_cache() -> None:
    """Drop cached clients. Tests call this after mutating settings."""
    _CLIENTS.clear()
