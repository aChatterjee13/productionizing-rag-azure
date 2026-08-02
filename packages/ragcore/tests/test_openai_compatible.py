"""Azure OpenAI and Ollama behind the same interface as Anthropic.

The risk in a second provider is not that it fails loudly — it is that it succeeds
*differently*: a tool call whose arguments never made the trip, a ``stop_reason`` the
orchestrator does not recognise, a token count the context packer trusts. So these
tests concentrate on the translation boundary rather than on the SDK:

* **Shape in, shape out.** Anthropic messages, tools and tool choices become OpenAI
  ones and back again, including the fan-out where one Anthropic user turn carrying
  several ``tool_result`` blocks becomes several ``role="tool"`` messages.
* **One vocabulary.** ``finish_reason`` is mapped onto Anthropic's ``stop_reason``
  words, because the orchestrator branches on them and must not learn two dialects.
* **Conformance.** Both clients are compared method-by-method against
  :class:`~ragcore.llm.base.LLMProvider`, signature included — a ``runtime_checkable``
  Protocol only checks that names exist, which would let a typo'd keyword through.
* **Refusal to pretend.** Settings must reject a provider configured for a capability
  it does not have, rather than accept it and quietly do less.

The SDK is stubbed throughout. Nothing here reaches a network.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from ragcore.llm.base import LLMProvider
from ragcore.llm.client import LLMClient, StreamEventType, get_llm_client
from ragcore.llm.openai_compatible import (
    OpenAICompatibleClient,
    OpenAIFlavour,
    _to_openai_messages,
    _to_openai_tool_choice,
    _to_openai_tools,
)
from ragcore.observability.langfuse import NoopTracer
from ragcore.settings import Settings

ALIASES = {
    "claude-opus-5": "gpt-4o",
    "claude-sonnet-5": "gpt-4o-mini",
    "claude-haiku-4-5": "gpt-4o-mini",
}


def make_settings(**overrides: Any) -> Settings:
    """Build settings for a non-Anthropic provider.

    Args:
        **overrides: Fields to override.

    Returns:
        Settings with the Anthropic-only features switched off, which is what a
        non-Anthropic provider requires.
    """
    base: dict[str, Any] = {
        "llm_provider": "ollama",
        "llm_model_aliases": ALIASES,
        "anthropic_thinking": False,
        "anthropic_cache_system": False,
        "tool_mcp_enabled": False,
        "context_compaction_enabled": False,
        "anthropic_retry_base_delay_seconds": 0.001,
        "anthropic_retry_max_delay_seconds": 0.002,
        "langfuse_enabled": False,
        "log_json": False,
    }
    base.update(overrides)
    return Settings(**base)


def usage(prompt: int = 10, completion: int = 5, cached: int = 0) -> SimpleNamespace:
    """Build a stub OpenAI usage object.

    Args:
        prompt: Prompt tokens, cached included.
        completion: Completion tokens.
        cached: Cached prompt tokens.

    Returns:
        A usage stand-in.
    """
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def completion(
    *,
    content: str | None = "hello",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    refusal: str | None = None,
    model: str = "gpt-4o",
) -> SimpleNamespace:
    """Build a stub OpenAI completion.

    Args:
        content: Assistant text.
        tool_calls: Tool calls on the message.
        finish_reason: Raw OpenAI finish reason.
        refusal: Refusal string, when the model declined.
        model: Serving model.

    Returns:
        A completion stand-in.
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage(),
        model=model,
    )


def tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    """Build a stub OpenAI tool call.

    Args:
        name: Function name.
        arguments: Raw JSON argument string.
        call_id: Tool call id.

    Returns:
        A tool-call stand-in.
    """
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    """Records the request and replays a scripted reply."""

    def __init__(self, reply: Any) -> None:
        """Store the scripted reply.

        Args:
            reply: Returned from ``create``, or raised when it is an exception.
        """
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeOpenAI:
    """Minimal stand-in for AsyncOpenAI."""

    def __init__(self, reply: Any) -> None:
        """Wire a completions stub under the SDK's attribute path.

        Args:
            reply: Passed through to :class:`FakeCompletions`.
        """
        self.chat = SimpleNamespace(completions=FakeCompletions(reply))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def build(reply: Any, **overrides: Any) -> tuple[OpenAICompatibleClient, FakeOpenAI]:
    """Build a client wired to a stubbed SDK.

    Args:
        reply: What the stub returns from ``create``.
        **overrides: Settings overrides.

    Returns:
        The client and its stub.
    """
    fake = FakeOpenAI(reply)
    client = OpenAICompatibleClient(make_settings(**overrides), client=fake)
    client._tracer = NoopTracer()
    return client, fake


# ------------------------------------------------------------------ conformance


class TestProviderConformance:
    """Both backends present the same interface, keyword for keyword."""

    @pytest.mark.parametrize(
        "method",
        ["complete", "stream", "structured", "classify", "count_tokens"],
    )
    @pytest.mark.parametrize(
        "implementation", [LLMClient, OpenAICompatibleClient], ids=lambda c: c.__name__
    )
    def test_signature_matches_the_protocol(
        self, implementation: type, method: str
    ) -> None:
        """Every parameter on the Protocol exists, with the same default.

        ``runtime_checkable`` only checks that a name is present, so a provider that
        renamed a keyword or changed a default would still pass ``isinstance`` and
        then fail at the twentieth call site instead of here.

        Args:
            implementation: The client class under test.
            method: Method name to compare.
        """
        expected = inspect.signature(getattr(LLMProvider, method))
        actual = inspect.signature(getattr(implementation, method))
        for name, parameter in expected.parameters.items():
            if name == "self":
                continue
            assert name in actual.parameters, f"{implementation.__name__} lost {name!r}"
            assert actual.parameters[name].default == parameter.default, (
                f"{implementation.__name__}.{method} changed the default for {name!r}"
            )

    def test_instances_satisfy_the_protocol(self) -> None:
        """The runtime check still has to pass for both."""
        client, _ = build(completion())
        assert isinstance(client, LLMProvider)


# ------------------------------------------------------------------ translation


class TestMessageTranslation:
    """Anthropic request shapes become OpenAI ones without losing anything."""

    def test_system_becomes_a_leading_system_message(self) -> None:
        """Anthropic's top-level system prompt is a message here."""
        out = _to_openai_messages("be terse", [{"role": "user", "content": "hi"}])
        assert out[0] == {"role": "system", "content": "be terse"}
        assert out[1] == {"role": "user", "content": "hi"}

    def test_absent_system_adds_no_message(self) -> None:
        """No system prompt means no empty system turn."""
        out = _to_openai_messages(None, [{"role": "user", "content": "hi"}])
        assert [m["role"] for m in out] == ["user"]

    def test_tool_use_becomes_an_assistant_tool_call(self) -> None:
        """`input` is an object on Anthropic and a JSON string on OpenAI."""
        out = _to_openai_messages(
            None,
            [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "looking"},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "search_corpus",
                            "input": {"query": "acls"},
                        },
                    ],
                },
            ],
        )
        assistant = out[-1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "looking"
        call = assistant["tool_calls"][0]
        assert call["id"] == "tu_1"
        assert call["function"]["name"] == "search_corpus"
        assert json.loads(call["function"]["arguments"]) == {"query": "acls"}

    def test_tool_results_fan_out_into_separate_tool_messages(self) -> None:
        """Anthropic packs results into one user turn; OpenAI wants one each."""
        out = _to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "first",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_2",
                            "content": "second",
                        },
                    ],
                }
            ],
        )
        assert [m["role"] for m in out] == ["tool", "tool"]
        assert [m["tool_call_id"] for m in out] == ["tu_1", "tu_2"]
        assert [m["content"] for m in out] == ["first", "second"]

    def test_tool_results_precede_accompanying_user_text(self) -> None:
        """A tool message that does not directly answer a call is rejected."""
        out = _to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "result",
                        },
                        {"type": "text", "text": "and now summarise"},
                    ],
                }
            ],
        )
        assert [m["role"] for m in out] == ["tool", "user"]

    def test_block_list_tool_result_content_is_flattened(self) -> None:
        """A result whose content is itself blocks still becomes a string."""
        out = _to_openai_messages(
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": [
                                {"type": "text", "text": "a"},
                                {"type": "text", "text": "b"},
                            ],
                        }
                    ],
                }
            ],
        )
        assert out[0]["content"] == "a\nb"

    def test_thinking_blocks_are_dropped(self) -> None:
        """Thinking has nowhere to go and must not leak in as text."""
        out = _to_openai_messages(
            None,
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret reasoning"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            ],
        )
        assert all("secret reasoning" not in str(m) for m in out)

    def test_tools_become_function_definitions(self) -> None:
        """`input_schema` is `parameters` on OpenAI."""
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        out = _to_openai_tools(
            [{"name": "search", "description": "find", "input_schema": schema}]
        )
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "find",
                    "parameters": schema,
                },
            }
        ]

    @pytest.mark.parametrize(
        ("anthropic_choice", "expected"),
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            (
                {"type": "tool", "name": "search"},
                {"type": "function", "function": {"name": "search"}},
            ),
            (None, None),
        ],
    )
    def test_tool_choice_translation(
        self, anthropic_choice: dict[str, Any] | None, expected: Any
    ) -> None:
        """Forcing and forbidding tool use must survive the trip.

        Args:
            anthropic_choice: The Anthropic tool_choice.
            expected: The OpenAI equivalent.
        """
        assert _to_openai_tool_choice(anthropic_choice) == expected


# -------------------------------------------------------------------- responses


class TestComplete:
    """`complete` returns the platform's response type, not the SDK's."""

    async def test_text_and_usage_are_parsed(self) -> None:
        """Cached tokens come out of the input bucket, as on Anthropic."""
        client, _ = build(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="hi", tool_calls=None, refusal=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=usage(prompt=10, completion=5, cached=4),
                model="gpt-4o",
            )
        )
        response = await client.complete(messages=[{"role": "user", "content": "q"}])
        assert response.text == "hi"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 6
        assert response.usage.cache_read_tokens == 4
        assert response.usage.prompt_tokens == 10

    async def test_tool_calls_are_parsed_back_into_objects(self) -> None:
        """`arguments` is a JSON string on the wire and an object here."""
        client, _ = build(
            completion(
                content=None,
                tool_calls=[tool_call("search", '{"query": "acls"}')],
                finish_reason="tool_calls",
            )
        )
        response = await client.complete(messages=[{"role": "user", "content": "q"}])
        assert response.has_tool_calls
        assert response.stop_reason == "tool_use"
        assert response.tool_calls[0]["input"] == {"query": "acls"}

    async def test_malformed_tool_arguments_do_not_raise(self) -> None:
        """The dispatcher validates arguments; losing the turn helps nobody."""
        client, _ = build(
            completion(
                content=None,
                tool_calls=[tool_call("search", "{not json")],
                finish_reason="tool_calls",
            )
        )
        response = await client.complete(messages=[{"role": "user", "content": "q"}])
        assert response.tool_calls[0]["input"] == {}

    @pytest.mark.parametrize(
        ("finish", "stop_reason"),
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "refusal"),
        ],
    )
    async def test_finish_reason_maps_to_the_anthropic_vocabulary(
        self, finish: str, stop_reason: str
    ) -> None:
        """The orchestrator branches on stop_reason and knows one dialect.

        Args:
            finish: Raw OpenAI finish reason.
            stop_reason: Expected Anthropic-vocabulary stop reason.
        """
        client, _ = build(completion(finish_reason=finish))
        response = await client.complete(messages=[{"role": "user", "content": "q"}])
        assert response.stop_reason == stop_reason

    async def test_a_refusal_empties_the_text(self) -> None:
        """A refusal must not be mistaken for an answer."""
        client, _ = build(completion(content="partial", refusal="I can't help"))
        response = await client.complete(messages=[{"role": "user", "content": "q"}])
        assert response.refused is True
        assert response.text == ""

    async def test_model_aliases_are_applied(self) -> None:
        """Call sites name Claude models; the backend must receive its own."""
        client, fake = build(completion())
        await client.complete(
            messages=[{"role": "user", "content": "q"}], model="claude-opus-5"
        )
        assert fake.calls[0]["model"] == "gpt-4o"

    async def test_anthropic_only_parameters_are_not_sent(self) -> None:
        """Thinking, caching, MCP and context edits have no wire form here."""
        client, fake = build(completion())
        await client.complete(
            messages=[{"role": "user", "content": "q"}],
            thinking=True,
            cache_system=True,
            effort="high",
            mcp_servers=[{"type": "url", "url": "https://example.test"}],
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
        )
        sent = fake.calls[0]
        for forbidden in (
            "thinking",
            "mcp_servers",
            "context_management",
            "effort",
            "output_config",
        ):
            assert forbidden not in sent


class TestStream:
    """Streaming assembles deltas into the same response type."""

    async def test_text_deltas_then_done(self) -> None:
        """DONE carries the assembled response exactly once."""

        async def chunks() -> Any:
            for piece in ("he", "llo"):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=piece, tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                    model="gpt-4o",
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=usage(),
                model="gpt-4o",
            )

        client, _ = build(chunks())
        events = [
            event
            async for event in client.stream(
                messages=[{"role": "user", "content": "q"}]
            )
        ]
        text = "".join(e.text for e in events if e.type is StreamEventType.TEXT)
        done = [e for e in events if e.type is StreamEventType.DONE]
        assert text == "hello"
        assert len(done) == 1
        assert done[0].response is not None
        assert done[0].response.text == "hello"

    async def test_tool_call_fragments_are_accumulated(self) -> None:
        """Arguments arrive in pieces; a partial call must never be emitted."""

        async def chunks() -> Any:
            fragments = ['{"que', 'ry": "a', 'cls"}']
            for index, fragment in enumerate(fragments):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1" if index == 0 else None,
                                        function=SimpleNamespace(
                                            name="search" if index == 0 else None,
                                            arguments=fragment,
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                    model="gpt-4o",
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="tool_calls",
                    )
                ],
                usage=usage(),
                model="gpt-4o",
            )

        client, _ = build(chunks())
        events = [
            event
            async for event in client.stream(
                messages=[{"role": "user", "content": "q"}]
            )
        ]
        tool_events = [e for e in events if e.type is StreamEventType.TOOL_USE]
        assert len(tool_events) == 1
        assert tool_events[0].tool_call["name"] == "search"
        assert tool_events[0].tool_call["input"] == {"query": "acls"}

    async def test_a_transport_failure_becomes_an_error_event(self) -> None:
        """A stream that dies mid-flight must not raise past the caller."""
        client, _ = build(RuntimeError("connection reset"))
        events = [
            event
            async for event in client.stream(
                messages=[{"role": "user", "content": "q"}]
            )
        ]
        assert [e.type for e in events] == [StreamEventType.ERROR]
        assert events[0].error == "RuntimeError"


class Answer(BaseModel):
    """Schema for the structured-output tests."""

    verdict: str
    score: int


class TestStructuredAndClassify:
    """The two shapes callers rely on for machine-readable answers."""

    async def test_structured_parses_and_validates(self) -> None:
        """A valid object round-trips into the model."""
        client, _ = build(completion(content='{"verdict": "yes", "score": 3}'))
        answer = await client.structured(
            messages=[{"role": "user", "content": "q"}], schema=Answer
        )
        assert answer.verdict == "yes"
        assert answer.score == 3

    async def test_structured_tolerates_surrounding_prose(self) -> None:
        """Small models pad JSON with commentary; the object still wins."""
        client, _ = build(
            completion(content='Sure!\n{"verdict": "no", "score": 1}\nHope that helps.')
        )
        answer = await client.structured(
            messages=[{"role": "user", "content": "q"}], schema=Answer
        )
        assert answer.verdict == "no"

    async def test_ollama_asks_for_json_object_mode(self) -> None:
        """Ollama's /v1 surface has no strict schema support."""
        client, fake = build(completion(content='{"verdict": "y", "score": 1}'))
        await client.structured(
            messages=[{"role": "user", "content": "q"}], schema=Answer
        )
        assert fake.calls[0]["response_format"] == {"type": "json_object"}

    async def test_azure_asks_for_a_json_schema(self) -> None:
        """Azure enforces the schema server-side, so send it."""
        client, fake = build(
            completion(content='{"verdict": "y", "score": 1}'),
            llm_provider="azure_openai",
            azure_openai_endpoint="https://example.openai.azure.com",
        )
        await client.structured(
            messages=[{"role": "user", "content": "q"}], schema=Answer
        )
        assert fake.calls[0]["response_format"]["type"] == "json_schema"

    async def test_classify_returns_a_member_of_labels(self) -> None:
        """The answer is matched against the label set, not echoed."""
        client, _ = build(completion(content="  Yes.  "))
        assert (
            await client.classify(system="pick", text="t", labels=["yes", "no"])
            == "yes"
        )

    async def test_classify_falls_back_rather_than_failing(self) -> None:
        """A classifier sits on a path that must produce a decision."""
        client, _ = build(completion(content="completely unrelated"))
        assert (
            await client.classify(system="pick", text="t", labels=["yes", "no"])
            == "yes"
        )

    async def test_classify_rejects_an_empty_label_set(self) -> None:
        """No labels is a programming error, not a runtime fallback."""
        client, _ = build(completion())
        with pytest.raises(ValueError, match="labels must not be empty"):
            await client.classify(system="pick", text="t", labels=[])


class TestCountTokens:
    """The context packer budgets against this number."""

    async def test_counts_grow_with_the_prompt(self) -> None:
        """A longer prompt must never count as fewer tokens."""
        client, _ = build(completion())
        short = await client.count_tokens(messages=[{"role": "user", "content": "hi"}])
        long = await client.count_tokens(
            messages=[{"role": "user", "content": "hi " * 500}]
        )
        assert 0 < short < long

    async def test_tools_are_counted(self) -> None:
        """Tool definitions occupy prompt tokens too."""
        client, _ = build(completion())
        messages = [{"role": "user", "content": "hi"}]
        without = await client.count_tokens(messages=messages)
        with_tools = await client.count_tokens(
            messages=messages,
            tools=[
                {
                    "name": "search",
                    "description": "find things",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        assert with_tools > without

    async def test_ollama_counts_are_padded_above_azure(self) -> None:
        """Llama tokenisers differ from o200k_base, so over-count deliberately.

        Under-counting overflows the window and fails the request; over-counting
        only wastes a little of it.
        """
        messages = [{"role": "user", "content": "the quick brown fox " * 40}]
        ollama, _ = build(completion())
        azure, _ = build(
            completion(),
            llm_provider="azure_openai",
            azure_openai_endpoint="https://example.openai.azure.com",
        )
        assert await ollama.count_tokens(messages=messages) > await azure.count_tokens(
            messages=messages
        )


# ---------------------------------------------------------------------- wiring


class TestProviderSelection:
    """Settings choose the backend, and refuse impossible combinations."""

    def test_factory_returns_the_configured_backend(self) -> None:
        """`get_llm_client` dispatches on llm_provider."""
        assert isinstance(get_llm_client(make_settings()), OpenAICompatibleClient)

    def test_flavour_follows_the_provider(self) -> None:
        """Azure and Ollama share a class but not a flavour."""
        client, _ = build(completion())
        assert client.flavour is OpenAIFlavour.OLLAMA

    @pytest.mark.parametrize(
        "feature",
        [
            "anthropic_thinking",
            "anthropic_cache_system",
            "tool_mcp_enabled",
            "context_compaction_enabled",
        ],
    )
    def test_anthropic_only_features_are_refused_at_construction(
        self, feature: str
    ) -> None:
        """A config that cannot be honoured must not start.

        Args:
            feature: The Anthropic-only setting under test.
        """
        with pytest.raises(ValueError, match="Anthropic-only"):
            make_settings(**{feature: True})

    def test_azure_without_an_endpoint_is_refused(self) -> None:
        """There is no default resource to fall back to."""
        with pytest.raises(ValueError, match="azure_openai_endpoint is required"):
            make_settings(llm_provider="azure_openai")

    def test_unmapped_model_slots_are_refused(self) -> None:
        """Otherwise every turn fails separately at the backend."""
        with pytest.raises(ValueError, match="llm_model_aliases"):
            make_settings(llm_model_aliases={"claude-opus-5": "gpt-4o"})

    def test_anthropic_keeps_its_features_and_needs_no_aliases(self) -> None:
        """None of the above applies to the default provider."""
        settings = Settings(anthropic_thinking=True, tool_mcp_enabled=True)
        assert settings.llm_provider == "anthropic"

    def test_ollama_is_priced_at_zero(self) -> None:
        """Local inference costs nothing per token.

        Falling through to the Anthropic rate card would make the self-hosted
        deployment look like the most expensive one on the dashboard.
        """
        assert make_settings().price_for_model("llama3.1") == (0.0, 0.0)
