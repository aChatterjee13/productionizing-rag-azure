"""Tests for the tool registry, the REST executor and the router (requirement #4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.rag.query_transform import TransformedQuery
from app.rag.tools.builtin import (
    CONTEXT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    BuiltinExecutor,
    builtin_tools,
    filter_from_arguments,
    render_chunks,
)
from app.rag.tools.registry import (
    NEVER_FORWARD,
    RateLimiter,
    ToolPolicy,
    build_registry,
    load_tool_document,
    redact_arguments,
    register_tool,
    tool_config,
)
from app.rag.tools.rest_tool import (
    CircuitBreaker,
    CircuitState,
    InvalidToolArgumentsError,
    RestExecutor,
    project_response,
    render_template,
    validate_arguments,
)
from app.rag.tools.router import (
    LoopGuard,
    LoopVerdict,
    RouteMode,
    ToolContext,
    ToolDispatcher,
    argument_signature,
    decide_route,
)
from ragcore.errors import ToolExecutionError
from ragcore.models.acl import Classification, Principal
from ragcore.models.retrieval import MetadataFilter, RetrievalResult
from ragcore.models.tool import RestToolSpec, ToolAuth, ToolKind, ToolSpec
from ragcore.pii import PIIFinding, PIIReport
from ragcore.settings import Settings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tools.example.yaml"


# --------------------------------------------------------------------- helpers


class StubDetector:
    """Deterministic stand-in for the Presidio-backed detector.

    Loading Presidio costs tens of seconds and pulls a spaCy model, which makes it
    unusable in a unit test. This stub implements the two methods the tool layer
    calls and flags anything containing an ``@`` as an email address.
    """

    def analyze(self, text: str, *, language: str = "en") -> PIIReport:
        del language
        index = text.find("@")
        if index < 0:
            return PIIReport.empty()
        return PIIReport.from_findings(
            [
                PIIFinding(
                    entity_type="EMAIL_ADDRESS",
                    start=0,
                    end=len(text),
                    score=0.95,
                    snippet="***",
                )
            ]
        )

    def scan_and_redact(
        self, text: str, *, mode: str | None = None, language: str = "en"
    ) -> tuple[str, PIIReport]:
        del mode, language
        report = self.analyze(text)
        return ("<EMAIL_ADDRESS>" if report.has_pii else text), report


@pytest.fixture
def settings() -> Settings:
    return Settings(tool_mcp_enabled=True)


@pytest.fixture
def registry(settings: Settings):
    return build_registry(
        settings,
        document=load_tool_document(CONFIG_PATH),
        extra_tools=builtin_tools(settings),
    )


@pytest.fixture
def detector() -> StubDetector:
    return StubDetector()


def principal(
    *,
    tenant: str = "contoso",
    roles: list[str] | None = None,
    clearance: Classification = Classification.INTERNAL,
) -> Principal:
    return Principal(
        user_id=f"user-{tenant}",
        tenant_id=tenant,
        roles=roles or [],
        groups=[],
        max_classification=clearance,
    )


def rest_tool(
    settings: Settings,
    *,
    name: str = "echo",
    method: str = "GET",
    url: str = "https://api.example.com/things/{thing_id}",
    json_path: str | None = None,
    policy: ToolPolicy | None = None,
    max_result_chars: int = 8000,
    **spec_kwargs: Any,
):
    spec = ToolSpec(
        name=name,
        kind=ToolKind.REST,
        description="Echo a thing.",
        input_schema={
            "type": "object",
            "properties": {
                "thing_id": {"type": "string", "pattern": "^[a-z0-9-]+$"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            },
            "required": ["thing_id"],
            "additionalProperties": False,
        },
        rest=RestToolSpec(
            method=method,
            url_template=url,
            auth=ToolAuth.NONE,
            response_json_path=json_path,
        ),
        max_result_chars=max_result_chars,
        **spec_kwargs,
    )
    return register_tool(spec, policy=policy, settings=settings)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------- registry: filtering


def test_shipped_example_config_loads(registry) -> None:
    names = {tool.name for tool in registry.all_tools()}
    assert {"order_status", "inventory_search", "create_support_ticket"} <= names
    assert SEARCH_TOOL_NAME in names
    assert CONTEXT_TOOL_NAME in names
    assert {
        server.name
        for server in registry.local_mcp_servers_for(
            principal(roles=["finance.analyst"])
        )
    } <= {"vnet_incidents", "finance_ledger"}


def test_role_filtering_excludes_unauthorised_tools(registry) -> None:
    plain = principal()
    agent = principal(roles=["support.agent"])

    plain_names = {tool.name for tool in registry.tools_for(plain)}
    agent_names = {tool.name for tool in registry.tools_for(agent)}

    assert "create_support_ticket" not in plain_names
    assert "knowledge_ops" not in plain_names
    assert "create_support_ticket" in agent_names
    # order_status declares no allowed_roles, so every role sees it.
    assert "order_status" in plain_names


def test_tenant_filtering_beats_every_role(registry) -> None:
    other_tenant_admin = principal(
        tenant="fabrikam", roles=["rag.admin", "support.agent", "ops.engineer"]
    )
    names = {tool.name for tool in registry.tools_for(other_tenant_admin)}
    assert "create_support_ticket" not in names, "tenant-pinned tool leaked"
    assert "order_status" in names
    assert registry.get("create_support_ticket", other_tenant_admin) is None


def test_local_mcp_servers_are_tenant_filtered(registry) -> None:
    fabrikam = principal(tenant="fabrikam", roles=["finance.analyst"])
    names = {s.name for s in registry.local_mcp_servers_for(fabrikam)}
    assert "finance_ledger" not in names


def test_require_raises_for_denied_tool(registry) -> None:
    with pytest.raises(ToolExecutionError):
        registry.require("create_support_ticket", principal())


def test_anthropic_tools_exclude_remote_mcp(registry) -> None:
    admin = principal(roles=["rag.admin"])
    definitions = registry.anthropic_tools(admin)
    assert {"knowledge_ops"}.isdisjoint({d["name"] for d in definitions})
    assert all({"name", "description", "input_schema"} <= set(d) for d in definitions)


def test_restricted_content_never_forwarded(settings: Settings) -> None:
    greedy = ToolPolicy(max_classification=Classification.RESTRICTED)
    tool = rest_tool(settings, policy=greedy)
    assert tool.max_classification is Classification.CONFIDENTIAL
    assert tool.may_receive(Classification.CONFIDENTIAL) is True
    assert tool.may_receive(NEVER_FORWARD) is False

    search = next(t for t in builtin_tools(settings) if t.name == SEARCH_TOOL_NAME)
    assert search.may_receive(NEVER_FORWARD) is True


def test_registry_is_role_filtered_for_admin_only_policy(settings: Settings) -> None:
    tool = rest_tool(settings, name="danger", policy=ToolPolicy(require_admin=True))
    reg = build_registry(settings, document={}, extra_tools=[tool])
    assert reg.get("danger", principal()) is None
    assert reg.get("danger", principal(roles=["rag.admin"])) is not None


async def test_rate_limiter_is_tenant_scoped() -> None:
    limiter = RateLimiter()
    assert await limiter.acquire(tenant_id="a", tool_name="t", per_minute=1) is True
    assert await limiter.acquire(tenant_id="a", tool_name="t", per_minute=1) is False
    # A second tenant has its own bucket.
    assert await limiter.acquire(tenant_id="b", tool_name="t", per_minute=1) is True


def test_redact_arguments_masks_secrets_and_pii(
    settings: Settings, detector: StubDetector
) -> None:
    redacted = redact_arguments(
        {
            "api_key": "sk-live-123",
            "email": "jane@example.com",
            "count": 3,
            "nested": {"authorization": "Bearer abc", "note": "plain"},
            "items": ["a@b.c", "safe"],
        },
        detector=detector,
        settings=settings,
    )
    assert redacted["api_key"] == "«redacted»"
    assert redacted["nested"]["authorization"] == "«redacted»"
    assert redacted["email"] == "<EMAIL_ADDRESS>"
    assert redacted["items"] == ["<EMAIL_ADDRESS>", "safe"]
    assert redacted["count"] == 3
    assert redacted["nested"]["note"] == "plain"


# ------------------------------------------------------- REST: arg validation


def test_validate_arguments_applies_defaults_and_coercion() -> None:
    schema = {
        "type": "object",
        "properties": {
            "thing_id": {"type": "string"},
            "limit": {"type": "integer", "default": 2},
            "flag": {"type": "boolean"},
        },
        "required": ["thing_id"],
        "additionalProperties": False,
    }
    out = validate_arguments(
        schema, {"thing_id": "abc", "limit": "7", "flag": "true"}, tool_name="t"
    )
    assert out == {"thing_id": "abc", "limit": 7, "flag": True}


def test_validate_arguments_rejects_missing_required() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    with pytest.raises(InvalidToolArgumentsError):
        validate_arguments(schema, {}, tool_name="t")


def test_validate_arguments_rejects_undeclared_properties() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises(InvalidToolArgumentsError) as excinfo:
        validate_arguments(schema, {"a": "x", "sneaky": "../../admin"}, tool_name="t")
    assert "sneaky" in str(excinfo.value)


def test_validate_arguments_enforces_enum_and_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]},
            "n": {"type": "integer", "minimum": 1, "maximum": 3},
        },
    }
    with pytest.raises(InvalidToolArgumentsError):
        validate_arguments(schema, {"mode": "c"}, tool_name="t")
    with pytest.raises(InvalidToolArgumentsError):
        validate_arguments(schema, {"n": 9}, tool_name="t")


def test_render_template_percent_encodes_path_values() -> None:
    rendered = render_template(
        "https://api/orders/{order_id}",
        {"order_id": "../admin?x=1"},
        quote_values=True,
    )
    assert rendered == "https://api/orders/..%2Fadmin%3Fx%3D1"


def test_render_template_strict_reports_missing() -> None:
    with pytest.raises(InvalidToolArgumentsError):
        render_template("https://api/{a}", {})
    assert render_template("https://api/{a}", {}, strict=False) == ""


# --------------------------------------------------------- REST: projection


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (None, {"data": {"items": [{"n": 1}, {"n": 2}]}}),
        ("data.items", [{"n": 1}, {"n": 2}]),
        ("data.items[0]", {"n": 1}),
        ("data.items[-1]", {"n": 2}),
        ("data.items[*]", [{"n": 1}, {"n": 2}]),
        ("data.nope", None),
        ("data.items[9]", None),
    ],
)
def test_project_response(path: str | None, expected: Any) -> None:
    payload = {"data": {"items": [{"n": 1}, {"n": 2}]}}
    assert project_response(payload, path) == expected


def test_project_response_caps_list_length() -> None:
    payload = {"rows": list(range(50))}
    assert project_response(payload, "rows", max_items=3) == [0, 1, 2]


# ---------------------------------------------------------- REST: execution


async def test_rest_execute_projects_and_returns_content(
    settings: Settings, detector: StubDetector
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200, json={"data": {"order": {"id": "ord-1", "state": "shipped"}}}
        )

    tool = rest_tool(settings, json_path="data.order")
    executor = RestExecutor(
        settings=settings, client=mock_client(handler), detector=detector
    )
    result = await executor.execute(
        tool, tool_call_id="tc-1", arguments={"thing_id": "ord-1"}
    )

    assert result.is_error is False
    assert result.http_status == 200
    assert json.loads(result.content) == {"id": "ord-1", "state": "shipped"}
    assert result.structured == {"id": "ord-1", "state": "shipped"}
    assert seen["url"] == "https://api.example.com/things/ord-1"


async def test_rest_execute_rejects_invalid_arguments(
    settings: Settings, detector: StubDetector
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    tool = rest_tool(settings)
    executor = RestExecutor(
        settings=settings, client=mock_client(handler), detector=detector
    )
    result = await executor.execute(
        tool, tool_call_id="tc-2", arguments={"thing_id": "NOT VALID"}
    )
    assert result.is_error is True
    assert "invalid arguments" in (result.error_message or "")


async def test_rest_execute_truncates_and_scrubs(
    settings: Settings, detector: StubDetector
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"note": "reach me at jane@example.com"})

    tool = rest_tool(settings, max_result_chars=12)
    executor = RestExecutor(
        settings=settings, client=mock_client(handler), detector=detector
    )
    result = await executor.execute(
        tool, tool_call_id="tc-3", arguments={"thing_id": "x"}
    )
    assert result.truncated is True
    assert "jane@example.com" not in result.content


async def test_rest_execute_reports_upstream_failure(
    settings: Settings, detector: StubDetector
) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "down"})

    tool = rest_tool(settings, policy=ToolPolicy(retry_attempts=1))
    executor = RestExecutor(
        settings=settings, client=mock_client(handler), detector=detector
    )
    result = await executor.execute(
        tool, tool_call_id="tc-4", arguments={"thing_id": "x"}
    )
    assert result.is_error is True
    assert result.http_status == 503
    assert calls["n"] == 2, "the retry budget must be used"


def test_circuit_breaker_opens_and_half_opens() -> None:
    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=0.0)
    assert breaker.allows("t") is True
    breaker.record_failure("t")
    assert breaker.state("t") is CircuitState.CLOSED
    breaker.record_failure("t")
    assert breaker.state("t") is CircuitState.HALF_OPEN  # reset_seconds elapsed
    breaker.record_success("t")
    assert breaker.state("t") is CircuitState.CLOSED


# ------------------------------------------------------------------- built-ins


async def test_builtin_context_tool_returns_tenant_and_clock(
    settings: Settings,
) -> None:
    tool = next(t for t in builtin_tools(settings) if t.name == CONTEXT_TOOL_NAME)
    result = await BuiltinExecutor(settings=settings).execute(
        tool,
        tool_call_id="tc-5",
        arguments={"timezone": "Europe/Berlin"},
        principal=principal(roles=["rag.admin"]),
    )
    payload = json.loads(result.content)
    assert payload["tenant_id"] == "contoso"
    assert payload["timezone"] == "Europe/Berlin"
    assert payload["is_admin"] is True
    assert payload["utc_now"].endswith("+00:00")


async def test_builtin_search_uses_injected_retriever(settings: Settings) -> None:
    captured: dict[str, Any] = {}

    async def fake_retrieve(*, query, principal, filters, top_n) -> RetrievalResult:
        captured.update(
            query=query, tenant=principal.tenant_id, filters=filters, top_n=top_n
        )
        return RetrievalResult(chunks=[], queries_used=[query])

    tool = next(t for t in builtin_tools(settings) if t.name == SEARCH_TOOL_NAME)
    result = await BuiltinExecutor(settings=settings).execute(
        tool,
        tool_call_id="tc-6",
        arguments={"query": "refund window", "doc_types": ["policy"], "top_n": 3},
        principal=principal(),
        base_filter=MetadataFilter(languages=["en"]),
        retrieve=fake_retrieve,
    )
    assert result.is_error is False
    assert captured["query"] == "refund window"
    assert captured["tenant"] == "contoso"
    assert captured["top_n"] == 3
    assert captured["filters"].doc_types == ["policy"]
    assert captured["filters"].languages == ["en"]
    assert "No chunks" in result.content


async def test_builtin_search_survives_retriever_failure(settings: Settings) -> None:
    async def boom(*, query, principal, filters, top_n) -> RetrievalResult:
        raise RuntimeError("qdrant down")

    tool = next(t for t in builtin_tools(settings) if t.name == SEARCH_TOOL_NAME)
    result = await BuiltinExecutor(settings=settings).execute(
        tool,
        tool_call_id="tc-7",
        arguments={"query": "x"},
        principal=principal(),
        retrieve=boom,
    )
    assert result.is_error is True
    assert "RuntimeError" in (result.error_message or "")


def test_filter_from_arguments_only_narrows() -> None:
    base = MetadataFilter(doc_types=["policy", "faq"], languages=["en"])
    merged = filter_from_arguments({"doc_types": ["policy", "spec"]}, base=base)
    assert merged is not None
    assert set(merged.doc_types or []) == {"policy"}
    assert merged.languages == ["en"]


def test_render_chunks_marks_empty_result() -> None:
    text = render_chunks(RetrievalResult(), snippet_chars=100)
    assert "No chunks" in text


# ---------------------------------------------------------------- loop control


def test_argument_signature_is_key_order_stable() -> None:
    assert argument_signature("t", {"a": 1, "b": 2}) == argument_signature(
        "t", {"b": 2, "a": 1}
    )


def test_loop_guard_detects_repeat_calls() -> None:
    guard = LoopGuard(max_iterations=6, repeat_limit=2)
    args = {"query": "same"}
    assert guard.check("search_corpus", args) is LoopVerdict.ALLOW
    assert guard.check("search_corpus", args) is LoopVerdict.REPEAT
    assert guard.check("search_corpus", args) is LoopVerdict.BLOCKED
    assert guard.broken_reason == "repeat_loop"
    # A different argument set is not a loop.
    assert guard.check("search_corpus", {"query": "other"}) is LoopVerdict.ALLOW


def test_loop_guard_caps_iterations() -> None:
    guard = LoopGuard(max_iterations=2)
    assert guard.begin_iteration() is True
    assert guard.begin_iteration() is True
    assert guard.begin_iteration() is False
    assert guard.broken_reason == "max_iterations"
    assert guard.check("any", {}) is LoopVerdict.EXHAUSTED


def test_loop_guard_from_config(settings: Settings) -> None:
    guard = LoopGuard.from_config(tool_config(settings))
    assert guard.max_iterations == settings.tool_max_iterations


# --------------------------------------------------------------------- routing


def transformed(**kwargs: Any) -> TransformedQuery:
    kwargs.setdefault("rewritten", "what is the refund window")
    return TransformedQuery(**kwargs)


def test_route_is_retrieval_only_without_tools(settings: Settings) -> None:
    decision = decide_route(transformed(), tools=(), settings=settings)
    assert decision.mode is RouteMode.RETRIEVAL_ONLY
    assert decision.use_tools is False
    assert decision.max_iterations == 0


def test_route_honours_allow_tools_false(settings: Settings, registry) -> None:
    tools = registry.tools_for(principal())
    decision = decide_route(
        transformed(needs_tools=True), tools=tools, allow_tools=False, settings=settings
    )
    assert decision.use_tools is False
    assert "disabled by request" in decision.reason


def test_route_uses_tools_when_transform_asks(settings: Settings, registry) -> None:
    tools = registry.tools_for(principal())
    decision = decide_route(
        transformed(needs_tools=True, tool_hints=["rest"]),
        tools=tools,
        settings=settings,
    )
    assert decision.mode is RouteMode.BOTH
    assert decision.use_tools is True
    assert decision.max_iterations == settings.tool_max_iterations
    assert decision.tool_hints == ("rest",)


def test_route_uses_tools_when_retrieval_confidence_is_low(
    settings: Settings, registry
) -> None:
    tools = registry.tools_for(principal())
    weak = RetrievalResult(chunks=[])
    decision = decide_route(
        transformed(), tools=tools, retrieval=weak, settings=settings
    )
    assert decision.use_tools is True
    assert "confidence" in decision.reason
    assert decision.as_metadata()["mode"] == "both"


def test_route_tools_only_when_retrieval_not_needed(
    settings: Settings, registry
) -> None:
    tools = registry.tools_for(principal())
    decision = decide_route(
        transformed(needs_retrieval=False, needs_tools=True),
        tools=tools,
        settings=settings,
    )
    assert decision.mode is RouteMode.TOOLS_ONLY
    assert decision.use_retrieval is False


# -------------------------------------------------------------------- dispatch


def dispatcher(settings: Settings, **kwargs: Any) -> ToolDispatcher:
    kwargs.setdefault(
        "rest",
        RestExecutor(
            settings=settings,
            client=mock_client(lambda request: httpx.Response(200, json={"ok": True})),
        ),
    )
    return ToolDispatcher(settings=settings, **kwargs)


def context(settings: Settings, registry, detector: StubDetector, **kwargs: Any):
    ctx = ToolContext.build(
        kwargs.pop("principal", principal()),
        settings=settings,
        registry=registry,
        detector=detector,
        **kwargs,
    )
    return ctx


async def test_dispatch_denies_a_tool_the_principal_may_not_call(
    settings: Settings, registry, detector: StubDetector
) -> None:
    ctx = context(settings, registry, detector)
    ctx.exposed = {}
    result = await dispatcher(settings).dispatch(
        ctx, tool_call_id="tc-8", tool_name="create_support_ticket", arguments={}
    )
    assert result.is_error is True
    assert "not available" in (result.error_message or "")


async def test_dispatch_breaks_a_repeat_loop(
    settings: Settings, registry, detector: StubDetector
) -> None:
    tool = next(t for t in builtin_tools(settings) if t.name == CONTEXT_TOOL_NAME)
    ctx = context(settings, registry, detector)
    ctx.exposed = {tool.name: tool}
    ctx.guard = LoopGuard(max_iterations=6, repeat_limit=2)
    disp = dispatcher(settings)

    first = await disp.dispatch(
        ctx, tool_call_id="a", tool_name=tool.name, arguments={}
    )
    second = await disp.dispatch(
        ctx, tool_call_id="b", tool_name=tool.name, arguments={}
    )

    assert first.is_error is False
    assert second.is_error is True
    assert "already called" in (second.error_message or "")
    assert ctx.guard.broken_reason == "repeat_loop"


async def test_dispatch_enforces_the_rate_limit(
    settings: Settings, registry, detector: StubDetector
) -> None:
    tool = next(t for t in builtin_tools(settings) if t.name == CONTEXT_TOOL_NAME)
    limited = register_tool(
        tool.spec, policy=ToolPolicy(rate_limit_per_minute=1), settings=settings
    )
    ctx = context(settings, registry, detector)
    ctx.exposed = {limited.name: limited}
    disp = dispatcher(settings)

    ok = await disp.dispatch(
        ctx, tool_call_id="a", tool_name=limited.name, arguments={}
    )
    throttled = await disp.dispatch(
        ctx, tool_call_id="b", tool_name=limited.name, arguments={}
    )
    assert ok.is_error is False
    assert throttled.is_error is True
    assert "rate limit" in (throttled.error_message or "")


async def test_dispatch_refuses_restricted_context_for_an_external_tool(
    settings: Settings, registry, detector: StubDetector
) -> None:
    tool = rest_tool(settings, name="external")
    ctx = context(
        settings,
        registry,
        detector,
        context_classification=Classification.RESTRICTED,
    )
    ctx.exposed = {tool.name: tool}
    result = await dispatcher(settings).dispatch(
        ctx, tool_call_id="tc-9", tool_name="external", arguments={"thing_id": "x"}
    )
    assert result.is_error is True
    assert "restricted" in (result.error_message or "").lower()


async def test_dispatch_blocks_pii_arguments_unless_opted_in(
    settings: Settings, registry, detector: StubDetector
) -> None:
    tool = rest_tool(settings, name="lookup")
    ctx = context(settings, registry, detector)
    ctx.exposed = {tool.name: tool}
    result = await dispatcher(settings).dispatch(
        ctx,
        tool_call_id="tc-10",
        tool_name="lookup",
        arguments={"thing_id": "jane@example.com"},
    )
    assert result.is_error is True
    assert "personal data" in (result.error_message or "")


async def test_dispatch_persists_a_redacted_invocation(
    settings: Settings, registry, detector: StubDetector
) -> None:
    written: list[dict[str, Any]] = []

    class FakeSession:
        pass

    async def fake_write(session, **fields):
        written.append(fields)
        return None

    import app.rag.tools.router as router_module

    original = router_module.write_tool_invocation
    router_module.write_tool_invocation = fake_write
    try:
        tool = next(t for t in builtin_tools(settings) if t.name == CONTEXT_TOOL_NAME)
        ctx = context(settings, registry, detector, session=FakeSession())
        ctx.exposed = {tool.name: tool}
        await dispatcher(settings).dispatch(
            ctx,
            tool_call_id="tc-11",
            tool_name=tool.name,
            arguments={"timezone": "UTC", "note": "mail jane@example.com"},
        )
    finally:
        router_module.write_tool_invocation = original

    assert len(written) == 1
    record = written[0]
    assert record["tenant_id"] == "contoso"
    assert record["kind"] == ToolKind.RETRIEVAL.value
    assert record["arguments"]["note"] == "<EMAIL_ADDRESS>"
    assert record["is_error"] is False
