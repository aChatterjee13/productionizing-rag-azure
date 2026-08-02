"""Tests for :mod:`ragcore.llm.client`.

The Anthropic SDK is mocked and the tests assert on the **request kwargs**,
because the obligations in LLM_FACTS are properties of the payload we send:
no sampling parameters, effort nested inside ``output_config``, exactly one
``cache_control`` breakpoint on the final system block, and the server-side
fallback beta plus ``fallbacks="default"`` on every call. A refusal is asserted
to produce ``refused=True`` rather than raising or indexing ``content[0]``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from ragcore.llm.client import (
    BETA_COMPACTION,
    BETA_CONTEXT_MANAGEMENT,
    BETA_SERVER_FALLBACK,
    FALLBACKS_DEFAULT,
    LLMClient,
    LLMRefusedError,
    LLMUsage,
    StreamEventType,
    clear_tool_uses_edit,
    compaction_edit,
    get_llm_client,
    reset_llm_client_cache,
)
from ragcore.observability.langfuse import NoopTracer
from ragcore.settings import Settings

# --------------------------------------------------------------------- fixtures


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "anthropic_api_key": "test-key",
        "anthropic_retry_base_delay_seconds": 0.001,
        "anthropic_retry_max_delay_seconds": 0.002,
        "langfuse_enabled": False,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text)


def tool_block(block_id: str, name: str, payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


def make_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_write: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def make_message(
    content: list[Any] | None = None,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5",
    usage: SimpleNamespace | None = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content if content is not None else [text_block("hello")],
        stop_reason=stop_reason,
        model=model,
        usage=usage or make_usage(),
        stop_details=stop_details,
    )


class FakeStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        """Record the scripted events and the final message."""
        self.events = events
        self.final = final

    async def __aenter__(self) -> FakeStream:
        """Enter the stream context."""
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        """Leave the stream context without suppressing exceptions."""
        return False

    def __aiter__(self) -> Any:
        """Iterate the scripted events."""

        async def gen() -> Any:
            for event in self.events:
                yield event

        return gen()

    async def get_final_message(self) -> Any:
        return self.final


class FakeMessages:
    def __init__(self, label: str) -> None:
        """Initialise the recorder for one namespace."""
        self.label = label
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.responses: list[Any] = []
        self.count_responses: list[Any] = []
        self.stream_events: list[Any] = []
        self.stream_final: Any = None
        self.stream_errors: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else make_message()
        if isinstance(item, BaseException):
            raise item
        return item

    async def count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(kwargs)
        item = (
            self.count_responses.pop(0)
            if self.count_responses
            else SimpleNamespace(input_tokens=123)
        )
        if isinstance(item, BaseException):
            raise item
        return item

    def stream(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        if self.stream_errors:
            error = self.stream_errors.pop(0)

            class Failing:
                """Stream manager that fails on entry."""

                async def __aenter__(self) -> Any:
                    """Fail to open the stream."""
                    raise error

                async def __aexit__(self, *_exc: object) -> bool:
                    """Leave without suppressing exceptions."""
                    return False

            return Failing()
        return FakeStream(list(self.stream_events), self.stream_final)

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


class FakeClient:
    def __init__(self) -> None:
        """Build the stable and beta namespaces."""
        self.messages = FakeMessages("stable")
        self.beta_messages = FakeMessages("beta")
        self.beta = SimpleNamespace(messages=self.beta_messages)

    def count_calls_model(self) -> str:
        return self.messages.count_calls[-1]["model"]


def build_client(**overrides: Any) -> tuple[LLMClient, FakeClient]:
    fake = FakeClient()
    client = LLMClient(make_settings(**overrides), client=fake, tracer=NoopTracer())
    return client, fake


def api_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    if status == 429:
        return anthropic.RateLimitError("slow down", response=response, body=None)
    if status == 404:
        return anthropic.NotFoundError("no such model", response=response, body=None)
    return anthropic.InternalServerError("boom", response=response, body=None)


# ------------------------------------------------------------------ request shape


async def test_complete_sends_contract_compliant_kwargs() -> None:
    client, fake = build_client()
    await client.complete(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert fake.messages.calls == [], "fallbacks require the beta namespace"
    sent = fake.beta_messages.last

    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sent

    assert sent["output_config"] == {"effort": "high"}
    assert "effort" not in sent

    assert sent["betas"] == [BETA_SERVER_FALLBACK]
    assert sent["fallbacks"] == FALLBACKS_DEFAULT

    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in str(sent["thinking"])

    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 16000


async def test_cache_control_only_on_final_system_block() -> None:
    client, fake = build_client()
    await client.complete(
        system=[
            "Stable instructions.",
            {"type": "text", "text": "More stable instructions."},
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    blocks = fake.beta_messages.last["system"]
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


async def test_caller_supplied_cache_control_is_relocated() -> None:
    client, fake = build_client()
    await client.complete(
        system=[
            {"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "B"},
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    blocks = fake.beta_messages.last["system"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


async def test_cache_control_can_be_disabled() -> None:
    client, fake = build_client()
    await client.complete(
        system="Stable instructions.",
        messages=[{"role": "user", "content": "hi"}],
        cache_system=False,
    )
    assert "cache_control" not in fake.beta_messages.last["system"][0]


async def test_settings_can_disable_caching_globally() -> None:
    client, fake = build_client(anthropic_cache_system=False)
    await client.complete(
        system="Stable instructions.",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "cache_control" not in fake.beta_messages.last["system"][0]


async def test_effort_and_model_are_overridable() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-5",
        effort="xhigh",
        max_tokens=999,
    )
    sent = fake.beta_messages.last
    assert sent["model"] == "claude-sonnet-5"
    assert sent["output_config"]["effort"] == "xhigh"
    assert sent["max_tokens"] == 999


async def test_thinking_disabled_is_sent_at_high_effort() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        effort="high",
        thinking=False,
    )
    assert fake.beta_messages.last["thinking"] == {"type": "disabled"}


async def test_thinking_disabled_is_omitted_above_high_effort() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        effort="max",
        thinking=False,
    )
    assert "thinking" not in fake.beta_messages.last


async def test_assistant_prefill_is_rejected_before_any_request() -> None:
    client, fake = build_client()
    with pytest.raises(ValueError, match="prefill"):
        await client.complete(
            system="s",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": '{"answer": "'},
            ],
        )
    assert fake.beta_messages.calls == []


async def test_empty_messages_are_rejected() -> None:
    client, _ = build_client()
    with pytest.raises(ValueError, match="must not be empty"):
        await client.complete(system="s", messages=[])


# ------------------------------------------------------------------- MCP + edits


async def test_mcp_servers_add_toolset_and_beta_flag() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        mcp_servers=[{"type": "url", "name": "svc", "url": "https://example.test/mcp"}],
    )
    sent = fake.beta_messages.last
    assert sent["mcp_servers"][0]["name"] == "svc"
    assert {"type": "mcp_toolset", "mcp_server_name": "svc"} in sent["tools"]
    assert client.settings.tool_mcp_beta_flag in sent["betas"]
    assert BETA_SERVER_FALLBACK in sent["betas"]


async def test_existing_mcp_toolset_is_not_duplicated() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "svc"}],
        mcp_servers=[{"name": "svc", "url": "https://example.test/mcp"}],
    )
    toolsets = [
        tool
        for tool in fake.beta_messages.last["tools"]
        if tool.get("type") == "mcp_toolset"
    ]
    assert len(toolsets) == 1


async def test_mcp_server_without_url_is_rejected() -> None:
    client, _ = build_client()
    with pytest.raises(ValueError, match="mcp_servers"):
        await client.complete(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            mcp_servers=[{"name": "svc"}],
        )


async def test_clear_tool_uses_edit_adds_context_management_beta() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        context_management=clear_tool_uses_edit(),
    )
    sent = fake.beta_messages.last
    assert sent["context_management"]["edits"][0]["type"] == "clear_tool_uses_20250919"
    assert BETA_CONTEXT_MANAGEMENT in sent["betas"]
    assert BETA_COMPACTION not in sent["betas"]


async def test_compaction_edit_adds_compaction_beta_only() -> None:
    client, fake = build_client()
    await client.complete(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        context_management=compaction_edit(),
    )
    sent = fake.beta_messages.last
    assert BETA_COMPACTION in sent["betas"]
    assert BETA_CONTEXT_MANAGEMENT not in sent["betas"]


# --------------------------------------------------------------------- responses


async def test_refusal_returns_refused_instead_of_raising() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(
        make_message(
            content=[],
            stop_reason="refusal",
            stop_details=SimpleNamespace(type="refusal", category="cyber"),
        )
    )
    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.refused is True
    assert response.text == ""
    assert response.tool_calls == []
    assert response.stop_reason == "refusal"
    assert response.usage.input_tokens == 10


async def test_refusal_never_reads_content_blocks() -> None:
    """A refusal with a non-text block must not be indexed or parsed."""
    client, fake = build_client()

    class Exploding:
        type = "text"

        @property
        def text(self) -> str:
            raise AssertionError("content must not be read on a refusal")

    fake.beta_messages.responses.append(
        make_message(content=[Exploding()], stop_reason="refusal")
    )
    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.refused is True


async def test_text_thinking_and_tool_calls_are_parsed() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(
        make_message(
            content=[
                thinking_block("weighing options"),
                text_block("Here is the answer "),
                text_block("[1]."),
                tool_block("tu_1", "search", {"q": "policy"}),
            ],
            stop_reason="tool_use",
        )
    )
    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.text == "Here is the answer [1]."
    assert response.thinking == "weighing options"
    assert response.has_tool_calls
    assert response.tool_calls[0] == {
        "id": "tu_1",
        "name": "search",
        "input": {"q": "policy"},
        "kind": "tool",
    }


async def test_usage_and_cost_follow_the_serving_model() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(
        make_message(
            model="claude-haiku-4-5",
            usage=make_usage(input_tokens=1_000_000, output_tokens=1_000_000),
        )
    )
    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.usage.model == "claude-haiku-4-5"
    assert response.usage.cost_usd() == pytest.approx(6.0)


def test_usage_cost_prices_cache_buckets_separately() -> None:
    usage = LLMUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        model="claude-opus-5",
    )
    # 5 + 25 + 0.5 (0.1x read) + 6.25 (1.25x write)
    assert usage.cost_usd() == pytest.approx(36.75)
    assert usage.total_tokens == 4_000_000
    assert usage.prompt_tokens == 3_000_000


# ----------------------------------------------------------------------- retries


async def test_rate_limit_is_retried() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.extend([api_error(429), make_message()])
    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.text == "hello"
    assert len(fake.beta_messages.calls) == 2


async def test_server_error_is_retried() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.extend([api_error(503), make_message()])
    await client.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert len(fake.beta_messages.calls) == 2


async def test_not_found_is_not_retried() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(api_error(404))
    with pytest.raises(anthropic.NotFoundError):
        await client.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert len(fake.beta_messages.calls) == 1


async def test_retries_are_bounded_by_settings() -> None:
    client, fake = build_client(anthropic_max_retries=1)
    fake.beta_messages.responses.extend([api_error(429), api_error(429)])
    with pytest.raises(anthropic.RateLimitError):
        await client.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert len(fake.beta_messages.calls) == 2


# -------------------------------------------------------------------- structured


class Extracted(BaseModel):
    name: str
    count: int = 0


async def test_structured_sends_json_schema_and_validates() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(
        make_message(content=[text_block('{"name": "Ada", "count": 2}')])
    )
    result = await client.structured(
        system="extract",
        messages=[{"role": "user", "content": "Ada, twice"}],
        schema=Extracted,
    )
    assert result == Extracted(name="Ada", count=2)

    sent = fake.beta_messages.last
    assert sent["model"] == "claude-sonnet-5"
    assert sent["output_config"]["effort"] == "medium"
    schema = sent["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["schema"]["additionalProperties"] is False


async def test_structured_raises_on_refusal() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(make_message(content=[], stop_reason="refusal"))
    with pytest.raises(LLMRefusedError):
        await client.structured(
            system="extract",
            messages=[{"role": "user", "content": "x"}],
            schema=Extracted,
        )


# ---------------------------------------------------------------------- classify


async def test_classify_returns_the_chosen_label() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(
        make_message(content=[text_block('{"label": "out_of_domain"}')])
    )
    label = await client.classify(
        system="route",
        text="what is the capital of France",
        labels=["out_of_domain", "in_domain"],
    )
    assert label == "out_of_domain"

    sent = fake.beta_messages.last
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["max_tokens"] == 256
    assert sent["thinking"] == {"type": "disabled"}
    enum = sent["output_config"]["format"]["schema"]["properties"]["label"]["enum"]
    assert enum == ["out_of_domain", "in_domain"]


async def test_classify_falls_back_to_the_first_label() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(make_message(content=[text_block("nonsense")]))
    label = await client.classify(
        system="route", text="x", labels=["out_of_domain", "in_domain"]
    )
    assert label == "out_of_domain"


async def test_classify_falls_back_when_refused() -> None:
    client, fake = build_client()
    fake.beta_messages.responses.append(make_message(content=[], stop_reason="refusal"))
    label = await client.classify(system="route", text="x", labels=["block", "allow"])
    assert label == "block"


async def test_classify_requires_labels() -> None:
    client, _ = build_client()
    with pytest.raises(ValueError, match="at least one label"):
        await client.classify(system="route", text="x", labels=[])


# ------------------------------------------------------------------ count_tokens


async def test_count_tokens_uses_the_anthropic_tokenizer() -> None:
    client, fake = build_client()
    total = await client.count_tokens(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert total == 123
    assert fake.count_calls_model() == "claude-opus-5"


async def test_count_tokens_degrades_to_an_estimate() -> None:
    client, fake = build_client(anthropic_max_retries=0)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake.messages.count_responses.append(anthropic.APIConnectionError(request=request))
    total = await client.count_tokens(
        system="x" * 40, messages=[{"role": "user", "content": "y" * 40}]
    )
    assert total == 20


# ------------------------------------------------------------------------ stream


async def test_stream_yields_typed_events() -> None:
    client, fake = build_client()
    beta = fake.beta_messages
    beta.stream_events = [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking="considering"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="text_delta", text="Hello "),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="text_delta", text="world"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=2,
            content_block=SimpleNamespace(
                type="tool_use", id="tu_9", name="lookup", input={}
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=2,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"id":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=2,
            delta=SimpleNamespace(type="input_json_delta", partial_json=' "42"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=2),
    ]
    beta.stream_final = make_message(
        content=[text_block("Hello world"), tool_block("tu_9", "lookup", {"id": "42"})],
        stop_reason="tool_use",
    )

    events = [
        event
        async for event in client.stream(
            system="s", messages=[{"role": "user", "content": "hi"}]
        )
    ]
    kinds = [event.type for event in events]

    assert StreamEventType.THINKING in kinds
    assert "".join(e.text for e in events if e.type is StreamEventType.TEXT) == (
        "Hello world"
    )
    tool_events = [e for e in events if e.type is StreamEventType.TOOL_USE]
    assert tool_events[0].tool_call == {
        "id": "tu_9",
        "name": "lookup",
        "input": {"id": "42"},
        "kind": "tool",
    }
    assert kinds[-2] is StreamEventType.USAGE
    assert kinds[-1] is StreamEventType.DONE
    assert events[-1].response is not None
    assert events[-1].response.text == "Hello world"
    assert events[-1].usage is not None

    sent = beta.stream_calls[-1]
    assert sent["max_tokens"] == 64000
    assert sent["fallbacks"] == FALLBACKS_DEFAULT
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sent


async def test_stream_emits_refusal_event() -> None:
    client, fake = build_client()
    fake.beta_messages.stream_events = []
    fake.beta_messages.stream_final = make_message(content=[], stop_reason="refusal")
    events = [
        event
        async for event in client.stream(
            system="s", messages=[{"role": "user", "content": "hi"}]
        )
    ]
    kinds = [event.type for event in events]
    assert StreamEventType.REFUSAL in kinds
    assert events[-1].refused is True


async def test_stream_retries_before_the_first_event_then_errors() -> None:
    client, fake = build_client(anthropic_max_retries=1)
    fake.beta_messages.stream_errors = [api_error(429), api_error(429)]
    events: list[Any] = []
    with pytest.raises(anthropic.RateLimitError):
        async for event in client.stream(
            system="s", messages=[{"role": "user", "content": "hi"}]
        ):
            events.append(event)
    assert [event.type for event in events] == [StreamEventType.ERROR]
    assert len(fake.beta_messages.stream_calls) == 2


# ------------------------------------------------------------------------ caches


def test_get_llm_client_is_cached_per_transport() -> None:
    reset_llm_client_cache()
    settings = make_settings()
    first = get_llm_client(settings)
    assert get_llm_client(settings) is first
    other = get_llm_client(make_settings(anthropic_base_url="https://proxy.test"))
    assert other is not first
    reset_llm_client_cache()


# --------------------------------------------------- SDK-shaped transport (retries)


class SdkShapedMessages(FakeMessages):
    """A namespace shaped like the real SDK rather than like a test double.

    The Anthropic SDK decorates ``messages.create`` with ``@required_args``, whose
    wrapper is a **plain** ``def`` that returns a coroutine. That makes
    ``inspect.iscoroutinefunction`` report ``False`` even though the call is
    awaitable, which is exactly the case ``FakeMessages`` (an ``async def``) cannot
    reproduce. Retry wrappers that branch on that introspection will build the
    coroutine, never await it, and hand the raw coroutine object back as the result
    without ever sending a request.
    """

    def create(self, **kwargs: Any) -> Any:
        return super().create(**kwargs)

    def count_tokens(self, **kwargs: Any) -> Any:
        return super().count_tokens(**kwargs)


class SdkShapedClient(FakeClient):
    """A fake client whose namespaces mimic the SDK's decorator shape."""

    def __init__(self) -> None:
        """Swap both namespaces for SDK-shaped ones."""
        super().__init__()
        self.messages = SdkShapedMessages("stable")
        self.beta_messages = SdkShapedMessages("beta")
        self.beta = SimpleNamespace(messages=self.beta_messages)


def build_sdk_shaped_client(**overrides: Any) -> tuple[LLMClient, SdkShapedClient]:
    """Build an ``LLMClient`` over an SDK-shaped transport."""
    fake = SdkShapedClient()
    client = LLMClient(make_settings(**overrides), client=fake, tracer=NoopTracer())
    return client, fake


async def test_complete_awaits_a_non_async_def_create() -> None:
    """``complete`` must return a parsed response, not an un-awaited coroutine.

    Regression: the retry wrapper used to hand ``namespace.create`` straight to
    tenacity, which branches on ``inspect.iscoroutinefunction``. Against the real
    SDK that is ``False``, so the request was never awaited and the "response" was
    a coroutine object -- ``text`` came back empty and no HTTP call was made.
    """
    client, fake = build_sdk_shaped_client()
    fake.beta_messages.responses = [make_message([text_block("real answer")])]

    response = await client.complete(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(fake.beta_messages.calls) == 1, "the request must actually be sent"
    assert response.text == "real answer"
    assert response.refused is False
    assert response.usage.input_tokens == 10


async def test_complete_still_retries_a_non_async_def_create() -> None:
    """Retries must survive the wrapping that fixes the await."""
    client, fake = build_sdk_shaped_client(anthropic_max_retries=1)
    fake.beta_messages.responses = [
        api_error(429),
        make_message([text_block("second try")]),
    ]

    response = await client.complete(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )

    assert len(fake.beta_messages.calls) == 2
    assert response.text == "second try"


async def test_count_tokens_awaits_a_non_async_def_count_tokens() -> None:
    """``count_tokens`` must return the real count, never a silent zero."""
    client, fake = build_sdk_shaped_client()
    fake.messages.count_responses = [SimpleNamespace(input_tokens=4242)]

    total = await client.count_tokens(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )

    assert len(fake.messages.count_calls) == 1
    assert total == 4242
