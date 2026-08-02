"""Tests for both MCP paths: the remote connector and the self-hosted SDK client."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.tools import mcp_client as mcp_module
from app.rag.tools.mcp_client import (
    ConnectorRequest,
    LocalMcpClient,
    LocalMcpTool,
    RemoteMcpConnector,
    build_connector_request,
    local_tool_name,
    render_mcp_content,
    split_local_tool_name,
    translate_mcp_tool,
)
from app.rag.tools.registry import build_registry, tool_config
from app.rag.tools.router import ToolPlan
from ragcore.models.acl import Classification, Principal
from ragcore.models.tool import McpServerSpec
from ragcore.settings import Settings

BETA = "mcp-client-2025-11-20"


# --------------------------------------------------------------------- helpers


@pytest.fixture
def settings() -> Settings:
    return Settings(tool_mcp_enabled=True)


def principal(*, tenant: str = "contoso", roles: list[str] | None = None) -> Principal:
    return Principal(
        user_id="u1",
        tenant_id=tenant,
        roles=roles or [],
        max_classification=Classification.INTERNAL,
    )


class FakeTool:
    """Stands in for ``mcp.types.Tool``."""

    def __init__(
        self, name: str, description: str = "", schema: dict[str, Any] | None = None
    ) -> None:
        """Build a fake discovered tool."""
        self.name = name
        self.description = description
        self.inputSchema = schema or {
            "type": "object",
            "properties": {"id": {"type": "string"}},
        }


class FakeListing:
    def __init__(self, tools: list[FakeTool]) -> None:
        """Build a fake ``list_tools`` response."""
        self.tools = tools


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        """Build a fake text content block."""
        self.type = "text"
        self.text = text


class FakeCallResult:
    def __init__(
        self,
        blocks: list[Any],
        *,
        is_error: bool = False,
        structured: Any = None,
    ) -> None:
        """Build a fake ``call_tool`` result."""
        self.content = blocks
        self.isError = is_error
        self.structuredContent = structured


class FakeSession:
    def __init__(self, tools: list[FakeTool], result: FakeCallResult | None = None):
        """Build a fake initialised MCP client session."""
        self._tools = tools
        self._result = result or FakeCallResult([FakeTextBlock("ok")])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> FakeListing:
        return FakeListing(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeCallResult:
        self.calls.append((name, arguments))
        return self._result


class StubLocalClient(LocalMcpClient):
    """A :class:`LocalMcpClient` whose transport is a fake session."""

    def __init__(self, session: FakeSession | None, **kwargs: Any) -> None:
        """Build a client that hands out ``session`` instead of dialling out."""
        super().__init__(**kwargs)
        self.session = session
        self.opened = 0

    async def _open_session(self, stack: Any, spec: Any) -> Any:
        self.opened += 1
        if self.session is None:
            msg = "connection refused"
            raise OSError(msg)
        return self.session


def vnet_spec(registry):
    """Return the ``vnet_incidents`` local server spec from a registry."""
    servers = registry.local_mcp_servers_for(principal())
    return next(s for s in servers if s.name == "vnet_incidents")


LOCAL_DOC: dict[str, Any] = {
    "mcp_servers": [
        {
            "name": "vnet_incidents",
            "transport": "streamable_http",
            "url": "https://mcp.internal.example.com/mcp",
            "allowed_tools": ["incident_by_id"],
            "policy": {"max_classification": "confidential"},
        },
        {
            "name": "finance_ledger",
            "transport": "stdio",
            "command": "finance-ledger-mcp",
            "tenant_id": "contoso",
            "allowed_roles": ["finance.analyst"],
        },
    ]
}


# ------------------------------------------------- (a) remote connector shape


def test_build_connector_request_emits_the_exact_pair() -> None:
    spec = McpServerSpec(name="svc", url="https://example.com/mcp")
    request = build_connector_request([spec], tokens={"svc": "tok-123"})

    assert request.betas == [BETA]
    assert request.mcp_servers == [
        {
            "type": "url",
            "name": "svc",
            "url": "https://example.com/mcp",
            "authorization_token": "tok-123",
        }
    ]
    assert request.tools == [{"type": "mcp_toolset", "mcp_server_name": "svc"}]


def test_connector_request_omits_token_when_absent() -> None:
    spec = McpServerSpec(name="svc", url="https://example.com/mcp")
    request = build_connector_request([spec])
    assert request.mcp_servers == [
        {"type": "url", "name": "svc", "url": "https://example.com/mcp"}
    ]
    assert "authorization_token" not in request.mcp_servers[0]


def test_connector_request_renders_an_allowlist_as_toolset_configs() -> None:
    spec = McpServerSpec(
        name="svc", url="https://example.com/mcp", allowed_tools=["a", "b"]
    )
    request = build_connector_request([spec])
    assert request.tools == [
        {
            "type": "mcp_toolset",
            "mcp_server_name": "svc",
            "default_config": {"enabled": False},
            "configs": [{"name": "a", "enabled": True}, {"name": "b", "enabled": True}],
        }
    ]


def test_connector_request_is_empty_without_servers() -> None:
    request = build_connector_request([])
    assert request.is_empty is True
    assert request.as_kwargs() == {}


def test_connector_request_never_yields_servers_without_a_toolset() -> None:
    specs = [
        McpServerSpec(name="a", url="https://a.example.com/mcp"),
        McpServerSpec(name="b", url="https://b.example.com/mcp"),
    ]
    kwargs = build_connector_request(specs).as_kwargs()
    assert [entry["name"] for entry in kwargs["mcp_servers"]] == ["a", "b"]
    assert [entry["mcp_server_name"] for entry in kwargs["tools"]] == ["a", "b"]
    assert kwargs["betas"] == [BETA]


async def test_remote_connector_resolves_tokens_and_filters_by_role(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOOL_SECRET_OPS_TOKEN", "kv-secret")
    document = {
        "mcp_servers": [
            {
                "name": "ops",
                "transport": "remote",
                "url": "https://ops.example.com/mcp",
                "authorization_token_ref": "ops-token",
                "allowed_roles": ["ops.engineer"],
            }
        ]
    }
    registry = build_registry(settings, document=document)
    connector = RemoteMcpConnector(settings=settings, config=tool_config(settings))

    denied = await connector.build(registry, principal())
    assert denied.is_empty is True

    allowed = await connector.build(registry, principal(roles=["ops.engineer"]))
    assert allowed.mcp_servers == [
        {
            "type": "url",
            "name": "ops",
            "url": "https://ops.example.com/mcp",
            "authorization_token": "kv-secret",
        }
    ]
    assert allowed.tools == [{"type": "mcp_toolset", "mcp_server_name": "ops"}]
    assert allowed.betas == [settings.tool_mcp_beta_flag]


async def test_remote_connector_is_empty_when_mcp_is_disabled() -> None:
    settings = Settings(tool_mcp_enabled=False)
    document = {
        "mcp_servers": [
            {
                "name": "ops",
                "transport": "remote",
                "url": "https://ops.example.com/mcp",
            }
        ]
    }
    registry = build_registry(settings, document=document)
    connector = RemoteMcpConnector(settings=settings, config=tool_config(settings))
    assert (await connector.build(registry, principal())).is_empty is True


async def test_plan_kwargs_are_what_the_beta_endpoint_receives() -> None:
    """The full call shape: betas + mcp_servers + a matching mcp_toolset entry."""
    captured: dict[str, Any] = {}

    class FakeBetaMessages:
        async def create(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    connector = build_connector_request(
        [McpServerSpec(name="svc", url="https://example.com/mcp")],
        tokens={"svc": "tok"},
    )
    plan = ToolPlan(
        tools=[{"name": "search_corpus", "description": "d", "input_schema": {}}],
        connector=connector,
        exposed={},
    )
    await FakeBetaMessages().create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        **plan.request_kwargs(),
    )

    assert captured["betas"] == [BETA]
    assert captured["mcp_servers"] == [
        {
            "type": "url",
            "name": "svc",
            "url": "https://example.com/mcp",
            "authorization_token": "tok",
        }
    ]
    assert captured["tools"] == [
        {"name": "search_corpus", "description": "d", "input_schema": {}},
        {"type": "mcp_toolset", "mcp_server_name": "svc"},
    ]


def test_plan_kwargs_omit_mcp_members_without_a_server() -> None:
    plan = ToolPlan(
        tools=[{"name": "t", "description": "d", "input_schema": {}}],
        connector=ConnectorRequest(betas=[], mcp_servers=[], tools=[]),
        exposed={},
    )
    kwargs = plan.request_kwargs()
    assert set(kwargs) == {"tools"}


# -------------------------------------------------------- (b) local MCP client


def test_tool_name_round_trip() -> None:
    name = local_tool_name("vnet_incidents", "incident.by-id")
    assert name == "vnet_incidents__incident_by-id"
    assert split_local_tool_name(name) == ("vnet_incidents", "incident_by-id")
    assert split_local_tool_name("bare") == ("", "bare")


def test_translate_mcp_tool_produces_an_anthropic_definition() -> None:
    definition = translate_mcp_tool("srv", FakeTool("lookup", "Find a thing."))
    assert definition == {
        "name": "srv__lookup",
        "description": "Find a thing.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
    }


def test_translate_mcp_tool_supplies_a_description_and_schema() -> None:
    definition = translate_mcp_tool("srv", FakeTool("bare", "", schema=None))
    assert "bare" in definition["description"]
    assert definition["input_schema"]["type"] == "object"


async def test_discover_caches_the_listing(settings: Settings) -> None:
    spec = vnet_spec(build_registry(settings, document=LOCAL_DOC))
    client = StubLocalClient(
        FakeSession([FakeTool("incident_by_id"), FakeTool("other")]),
        settings=settings,
        config=tool_config(settings),
    )

    first = await client.discover(spec)
    second = await client.discover(spec)

    assert client.opened == 1, "the second discovery must come from the cache"
    assert [t["name"] for t in first.tools] == ["vnet_incidents__incident_by_id"]
    assert second.tools == first.tools


async def test_discover_disables_an_unreachable_server(settings: Settings) -> None:
    spec = vnet_spec(build_registry(settings, document=LOCAL_DOC))
    client = StubLocalClient(None, settings=settings, config=tool_config(settings))

    listing = await client.discover(spec)

    assert listing.tools == []
    assert listing.error == "OSError"
    assert client.is_disabled(spec.name) is True


async def test_discovered_tools_are_tenant_and_role_filtered(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_module, "mcp_sdk_available", lambda: True)
    registry = build_registry(settings, document=LOCAL_DOC)
    client = StubLocalClient(
        FakeSession([FakeTool("incident_by_id")]),
        settings=settings,
        config=tool_config(settings),
    )

    outsider = await client.discovered_tools(
        registry, principal(tenant="fabrikam", roles=["finance.analyst"])
    )
    assert {tool.server_name for tool in outsider} == {"vnet_incidents"}, (
        "finance_ledger is pinned to contoso and must not reach another tenant"
    )

    insider = await client.discovered_tools(
        registry, principal(roles=["finance.analyst"])
    )
    assert {tool.server_name for tool in insider} == {
        "vnet_incidents",
        "finance_ledger",
    }


async def test_discovered_tool_carries_policy_and_never_takes_restricted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_module, "mcp_sdk_available", lambda: True)
    registry = build_registry(settings, document=LOCAL_DOC)
    client = StubLocalClient(
        FakeSession([FakeTool("incident_by_id")]),
        settings=settings,
        config=tool_config(settings),
    )
    tools = await client.discovered_tools(registry, principal())
    tool = next(t for t in tools if t.server_name == "vnet_incidents")

    assert isinstance(tool, LocalMcpTool)
    assert tool.max_classification is Classification.CONFIDENTIAL
    assert tool.may_receive(Classification.CONFIDENTIAL) is True
    assert tool.may_receive(Classification.RESTRICTED) is False
    assert tool.to_anthropic_tool()["name"] == "vnet_incidents__incident_by_id"
    assert tool.timeout_seconds > 0


async def test_call_invokes_the_unnamespaced_tool(settings: Settings) -> None:
    spec = vnet_spec(build_registry(settings, document=LOCAL_DOC))
    session = FakeSession(
        [FakeTool("incident_by_id")],
        FakeCallResult([FakeTextBlock("INC-9 is open")], structured={"id": "INC-9"}),
    )
    client = StubLocalClient(session, settings=settings, config=tool_config(settings))

    result = await client.call(
        spec,
        tool_call_id="tc-1",
        tool_name="vnet_incidents__incident_by_id",
        arguments={"id": "INC-9"},
        max_result_chars=1000,
    )

    assert session.calls == [("incident_by_id", {"id": "INC-9"})]
    assert result.is_error is False
    assert result.content == "INC-9 is open"
    assert result.structured == {"id": "INC-9"}
    assert result.kind.value == "mcp"


async def test_call_rejects_a_tool_outside_the_allowlist(settings: Settings) -> None:
    spec = vnet_spec(build_registry(settings, document=LOCAL_DOC))
    client = StubLocalClient(
        FakeSession([]), settings=settings, config=tool_config(settings)
    )
    result = await client.call(
        spec,
        tool_call_id="tc-2",
        tool_name="vnet_incidents__delete_everything",
        arguments={},
        max_result_chars=1000,
    )
    assert result.is_error is True
    assert "not exposed" in (result.error_message or "")


async def test_call_on_an_unreachable_server_errors_without_raising(
    settings: Settings,
) -> None:
    spec = vnet_spec(build_registry(settings, document=LOCAL_DOC))
    client = StubLocalClient(None, settings=settings, config=tool_config(settings))

    first = await client.call(
        spec,
        tool_call_id="tc-3",
        tool_name="vnet_incidents__incident_by_id",
        arguments={"id": "x"},
        max_result_chars=1000,
    )
    assert first.is_error is True
    assert client.is_disabled(spec.name) is True

    second = await client.call(
        spec,
        tool_call_id="tc-4",
        tool_name="vnet_incidents__incident_by_id",
        arguments={"id": "x"},
        max_result_chars=1000,
    )
    assert second.is_error is True
    assert "temporarily unavailable" in (second.error_message or "")
    assert client.opened == 1, "a disabled server must not be dialled again"


def test_render_mcp_content_flattens_blocks() -> None:
    result = FakeCallResult(
        [FakeTextBlock("line one"), FakeTextBlock("line two")],
        structured={"n": 2},
    )
    text, structured = render_mcp_content(result)
    assert text == "line one\nline two"
    assert structured == {"n": 2}


def test_render_mcp_content_describes_non_text_blocks() -> None:
    class Blob:
        type = "resource"
        uri = "file:///x.pdf"
        mimeType = "application/pdf"  # noqa: N815 - the SDK spells it this way

    text, structured = render_mcp_content(FakeCallResult([Blob()]))
    assert "resource" in text
    assert "file:///x.pdf" in text
    assert structured is None
