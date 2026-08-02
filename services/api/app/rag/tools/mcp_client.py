"""MCP integration — two real paths, chosen by where the server lives.

**(a) Remote, server-side.** Anthropic connects to the MCP server itself. That needs
three things together on the beta endpoint, exactly as ``LLM_FACTS`` specifies::

    client.beta.messages.create(
        betas=["mcp-client-2025-11-20"],
        mcp_servers=[{"type": "url", "name": "svc", "url": "https://…/mcp",
                      "authorization_token": "…"}],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "svc"}],
        ...)

Passing ``mcp_servers`` without a matching ``mcp_toolset`` entry is a validation
error, so :func:`build_connector_request` always emits the pair together — it never
returns one half. :meth:`RemoteMcpConnector.build` resolves the authorization tokens
from Key Vault first.

**(b) Local / self-hosted.** A server inside the VNet is unreachable from Anthropic's
side, so the platform speaks MCP itself over the official ``mcp`` Python SDK — stdio
for a co-located child process, streamable HTTP for an in-network endpoint. Tools are
discovered, translated into ordinary Anthropic client-side tool definitions, cached
for ``tool_mcp_discovery_ttl_seconds``, and invoked through a short-lived session.

An unavailable server disables itself for ``tool_mcp_disable_seconds`` instead of
failing the turn: the model simply does not see those tools.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import anyio

from app.rag.tools.registry import (
    LocalMcpServerSpec,
    ToolConfig,
    ToolPolicy,
    ToolRegistry,
    tool_config,
)
from app.rag.tools.rest_tool import SecretResolver
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.models.tool import McpServerSpec, ToolKind, ToolResult, ToolSpec
from ragcore.settings import Settings, get_settings

__all__ = [
    "LOCAL_NAME_SEPARATOR",
    "ConnectorRequest",
    "LocalMcpClient",
    "LocalMcpTool",
    "McpToolListing",
    "RemoteMcpConnector",
    "build_connector_request",
    "local_tool_name",
    "mcp_sdk_available",
    "render_mcp_content",
    "split_local_tool_name",
    "translate_mcp_tool",
]

_log = get_logger(__name__)

_MS = 1000.0

#: Separator joining an MCP server name to one of its tool names. Chosen because
#: ``ToolSpec`` only accepts ``[A-Za-z0-9_-]`` and a double underscore is unlikely to
#: collide with a server-side tool name.
LOCAL_NAME_SEPARATOR = "__"


def mcp_sdk_available() -> bool:
    """Whether the official ``mcp`` Python SDK is importable.

    Returns:
        True when local/self-hosted MCP can be used. The remote connector path does
        not need the SDK at all — Anthropic makes that connection.
    """
    try:
        import mcp  # noqa: F401 - probe only
    except ImportError:
        return False
    return True


# ------------------------------------------------------- (a) remote connector


@dataclass(frozen=True, slots=True)
class ConnectorRequest:
    """The three request members the MCP connector needs, produced together."""

    betas: list[str]
    mcp_servers: list[dict[str, Any]]
    tools: list[dict[str, Any]]

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to send.

        Returns:
            True when no server survived filtering.
        """
        return not self.mcp_servers

    def as_kwargs(self) -> dict[str, Any]:
        """Render the request members as keyword arguments.

        Returns:
            ``{}`` when empty, otherwise ``betas``, ``mcp_servers`` and ``tools``
            ready to merge into a ``client.beta.messages.create`` call. ``tools``
            must be concatenated with the client-side tool definitions, not replace
            them.
        """
        if self.is_empty:
            return {}
        return {
            "betas": list(self.betas),
            "mcp_servers": list(self.mcp_servers),
            "tools": list(self.tools),
        }


def build_connector_request(
    specs: Sequence[McpServerSpec],
    *,
    tokens: Mapping[str, str] | None = None,
    beta_flag: str | None = None,
) -> ConnectorRequest:
    """Build the ``mcp_servers`` + ``mcp_toolset`` + ``betas`` triple.

    Args:
        specs: Remote MCP server specs the principal may reach.
        tokens: Resolved authorization tokens keyed by server name. A server with no
            entry is sent without a token, which is correct for a public server.
        beta_flag: Beta flag to require. Defaults to each spec's own ``beta_flag``,
            which itself defaults to ``mcp-client-2025-11-20``.

    Returns:
        A :class:`ConnectorRequest`. Both halves are always produced together, so a
        caller cannot send ``mcp_servers`` without the matching toolset entry.
    """
    servers: list[dict[str, Any]] = []
    toolsets: list[dict[str, Any]] = []
    betas: list[str] = []
    resolved = dict(tokens or {})
    for spec in specs:
        server, toolset = spec.to_connector_entries(resolved.get(spec.name))
        servers.append(server)
        toolsets.append(toolset)
        flag = beta_flag or spec.beta_flag
        if flag and flag not in betas:
            betas.append(flag)
    return ConnectorRequest(betas=betas, mcp_servers=servers, tools=toolsets)


class RemoteMcpConnector:
    """Resolves tokens and renders the connector request for one principal."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        config: ToolConfig | None = None,
        secrets_resolver: SecretResolver | None = None,
    ) -> None:
        """Initialise the connector.

        Args:
            settings: Platform settings; defaults to :func:`get_settings`.
            config: Resolved tool config; derived from settings when omitted.
            secrets_resolver: Secret resolver; a fresh one is built when omitted.
        """
        self.settings = settings or get_settings()
        self.config = config or tool_config(self.settings)
        self.secrets = secrets_resolver or SecretResolver(config=self.config)

    async def build(
        self, registry: ToolRegistry, principal: Principal
    ) -> ConnectorRequest:
        """Build the connector request for the servers a principal may reach.

        Args:
            registry: The tool registry.
            principal: The caller. Tenant and role filtering happen in the registry,
                so a tenant can never reach another tenant's MCP server.

        Returns:
            The connector request, empty when MCP is disabled or no server applies.
        """
        if not (self.config.enabled and self.config.mcp_enabled):
            return ConnectorRequest(betas=[], mcp_servers=[], tools=[])
        specs = registry.mcp_specs_for(principal)
        tokens: dict[str, str] = {}
        for spec in specs:
            token = await self.secrets.resolve(spec.authorization_token_ref)
            if token:
                tokens[spec.name] = token
            elif spec.authorization_token_ref:
                _log.warning(
                    "mcp_token_unresolved",
                    server=spec.name,
                    ref=spec.authorization_token_ref,
                )
        return build_connector_request(
            specs, tokens=tokens, beta_flag=self.config.mcp_beta_flag
        )


# --------------------------------------------------------- (b) local MCP client


def local_tool_name(server: str, tool: str) -> str:
    """Namespace a server-side tool name for the Anthropic tool list.

    Args:
        server: MCP server name.
        tool: Tool name as the server advertises it.

    Returns:
        ``"<server>__<tool>"`` with any character outside ``[A-Za-z0-9_-]`` replaced
        by an underscore, so the name passes ``ToolSpec`` validation.
    """
    raw = f"{server}{LOCAL_NAME_SEPARATOR}{tool}"
    return "".join(char if (char.isalnum() or char in "_-") else "_" for char in raw)


def split_local_tool_name(name: str) -> tuple[str, str]:
    """Split a namespaced tool name back into server and tool.

    Args:
        name: A name produced by :func:`local_tool_name`.

    Returns:
        A ``(server, tool)`` pair. When the name carries no separator the server part
        is empty and the whole name is the tool.
    """
    server, separator, tool = name.partition(LOCAL_NAME_SEPARATOR)
    if not separator:
        return "", name
    return server, tool


def translate_mcp_tool(server: str, tool: Any) -> dict[str, Any]:
    """Translate one discovered MCP tool into an Anthropic tool definition.

    Args:
        server: MCP server name.
        tool: A ``mcp.types.Tool`` (or any object exposing ``name``, ``description``
            and ``inputSchema``).

    Returns:
        A mapping with ``name``, ``description`` and ``input_schema``.
    """
    raw_name = str(getattr(tool, "name", "") or "")
    description = str(getattr(tool, "description", "") or "")
    schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return {
        "name": local_tool_name(server, raw_name),
        "description": description or f"{raw_name} on the {server} MCP server.",
        "input_schema": schema,
    }


@dataclass(frozen=True, slots=True)
class LocalMcpTool:
    """One tool discovered on a self-hosted MCP server.

    A local MCP tool is **not** a :class:`~ragcore.models.tool.ToolSpec`: a ToolSpec
    of ``kind='mcp'`` describes a *remote* server the Anthropic connector dials, and
    requires an https URL that a stdio child process does not have. This type carries
    the same surface the tool loop needs (name, limits, policy, egress ceiling) so the
    router can treat both uniformly, and it still reports ``kind='mcp'`` on
    ``tool_invocations``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server: LocalMcpServerSpec
    policy: ToolPolicy
    timeout_seconds: float
    rate_limit_per_minute: int
    max_classification: Classification
    max_result_chars: int

    @property
    def kind(self) -> ToolKind:
        """Execution kind.

        Returns:
            Always ``mcp``.
        """
        return ToolKind.MCP

    @property
    def server_name(self) -> str:
        """Name of the owning MCP server.

        Returns:
            The server name.
        """
        return self.server.name

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Render the client-side tool definition sent in ``tools``.

        Local MCP tools are ordinary client-side tools: the platform holds the
        connection, so the model calls them the same way it calls a REST tool.

        Returns:
            A mapping with ``name``, ``description`` and ``input_schema``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }

    def is_allowed_for(self, principal: Principal) -> bool:
        """Tenant- and role-filter this tool for a principal.

        Args:
            principal: The caller.

        Returns:
            True when the owning server is exposed to this principal.
        """
        if not self.server.is_allowed_for(principal):
            return False
        return not (self.policy.require_admin and not principal.is_admin())

    def may_receive(self, classification: Classification) -> bool:
        """Whether content of a given classification may be sent to this tool.

        Args:
            classification: Highest classification present in what would be
                forwarded.

        Returns:
            True when the content is at or below the tool's ceiling. A self-hosted
            server is still outside the platform boundary, so ``RESTRICTED`` never
            passes.
        """
        return bool(classification <= self.max_classification)


@dataclass(slots=True)
class McpToolListing:
    """A cached tool listing for one local MCP server."""

    server: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str | None = None

    def is_fresh(self, ttl_seconds: float, *, now: float | None = None) -> bool:
        """Whether this listing may still be used.

        Args:
            ttl_seconds: Cache lifetime.
            now: Monotonic clock override, for tests.

        Returns:
            True while the listing is inside its TTL.
        """
        moment = time.monotonic() if now is None else now
        return moment - self.fetched_at < ttl_seconds


class LocalMcpClient:
    """Speaks MCP directly to self-hosted servers with the official ``mcp`` SDK.

    Discovery results are cached per server; a server that cannot be reached is
    disabled for ``tool_mcp_disable_seconds`` so one broken server costs one attempt
    per cool-down rather than one attempt per turn.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        config: ToolConfig | None = None,
        secrets_resolver: SecretResolver | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            settings: Platform settings; defaults to :func:`get_settings`.
            config: Resolved tool config; derived from settings when omitted.
            secrets_resolver: Secret resolver used for bearer tokens.
        """
        self.settings = settings or get_settings()
        self.config = config or tool_config(self.settings)
        self.secrets = secrets_resolver or SecretResolver(config=self.config)
        self._listings: dict[str, McpToolListing] = {}
        self._disabled_until: dict[str, float] = {}
        self._lock = anyio.Lock()

    # ------------------------------------------------------------- availability

    def is_disabled(self, server: str) -> bool:
        """Whether a server is currently in its failure cool-down.

        Args:
            server: Server name.

        Returns:
            True while the server is disabled for this turn.
        """
        until = self._disabled_until.get(server)
        if until is None:
            return False
        if until <= time.monotonic():
            self._disabled_until.pop(server, None)
            return False
        return True

    def disable(self, server: str, *, reason: str) -> None:
        """Disable a server for the configured cool-down.

        Args:
            server: Server name.
            reason: Redacted reason, logged once per transition.
        """
        self._disabled_until[server] = (
            time.monotonic() + self.config.mcp_disable_seconds
        )
        _log.warning("mcp_server_disabled", server=server, reason=reason)

    def reset(self) -> None:
        """Drop cached listings and cool-downs. Test helper."""
        self._listings.clear()
        self._disabled_until.clear()

    # ---------------------------------------------------------------- sessions

    async def _open_session(
        self, stack: AsyncExitStack, spec: LocalMcpServerSpec
    ) -> Any:
        """Open an initialised MCP client session on the given exit stack.

        Args:
            stack: Exit stack owning the transport and the session.
            spec: The server to connect to.

        Returns:
            An initialised ``mcp.ClientSession``.

        Raises:
            RuntimeError: When the ``mcp`` SDK is not installed.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import (
                streamablehttp_client,
            )
        except ImportError as exc:  # pragma: no cover - depends on the environment
            msg = "the 'mcp' package is required for self-hosted MCP servers"
            raise RuntimeError(msg) from exc

        if spec.transport == "stdio":
            params = StdioServerParameters(
                command=str(spec.command),
                args=list(spec.args),
                env=dict(spec.env) or None,
                cwd=spec.cwd,
            )
            read, write, *_ = await stack.enter_async_context(stdio_client(params))
        else:
            headers = dict(spec.headers)
            token = await self.secrets.resolve(spec.authorization_token_ref)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            url = str(spec.url)
            if url.startswith("http://") and not self.config.allow_insecure_http:
                msg = f"MCP server {spec.name!r} uses plain http; refusing"
                raise RuntimeError(msg)
            read, write, *_ = await stack.enter_async_context(
                streamablehttp_client(url, headers=headers or None)
            )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # --------------------------------------------------------------- discovery

    async def discover(
        self, spec: LocalMcpServerSpec, *, force: bool = False
    ) -> McpToolListing:
        """List a local server's tools, using the cache when it is fresh.

        Args:
            spec: The server to query.
            force: Ignore the cache and re-query.

        Returns:
            The listing. On failure the listing carries ``error`` and an empty tool
            list, and the server is disabled for the cool-down — discovery never
            raises into the turn.
        """
        cached = self._listings.get(spec.name)
        if (
            not force
            and cached is not None
            and cached.is_fresh(self.config.mcp_discovery_ttl_seconds)
        ):
            return cached
        if self.is_disabled(spec.name):
            return McpToolListing(
                server=spec.name, fetched_at=time.monotonic(), error="disabled"
            )

        async with self._lock:
            try:
                with anyio.fail_after(self.config.mcp_connect_timeout_seconds):
                    async with AsyncExitStack() as stack:
                        session = await self._open_session(stack, spec)
                        listed = await session.list_tools()
            except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                self.disable(spec.name, reason=type(exc).__name__)
                listing = McpToolListing(
                    server=spec.name,
                    fetched_at=time.monotonic(),
                    error=type(exc).__name__,
                )
                self._listings[spec.name] = listing
                return listing
            except Exception as exc:
                self.disable(spec.name, reason=type(exc).__name__)
                listing = McpToolListing(
                    server=spec.name,
                    fetched_at=time.monotonic(),
                    error=type(exc).__name__,
                )
                self._listings[spec.name] = listing
                return listing

        allow = set(spec.allowed_tools)
        tools = [
            translate_mcp_tool(spec.name, tool)
            for tool in getattr(listed, "tools", [])
            if not allow or str(getattr(tool, "name", "")) in allow
        ]
        listing = McpToolListing(
            server=spec.name, tools=tools, fetched_at=time.monotonic()
        )
        self._listings[spec.name] = listing
        _log.info("mcp_tools_discovered", server=spec.name, tools=len(tools))
        return listing

    async def discovered_tools(
        self, registry: ToolRegistry, principal: Principal
    ) -> list[LocalMcpTool]:
        """Discover every local MCP tool this principal may call.

        Args:
            registry: The tool registry, which owns tenant and role filtering.
            principal: The caller.

        Returns:
            One :class:`LocalMcpTool` per discovered tool, policy-resolved with the
            owning server's policy. Empty when the SDK is missing, local MCP is
            disabled, or every server is unavailable — an unreachable server costs
            the turn nothing beyond the tools it would have offered.
        """
        servers = registry.local_mcp_servers_for(principal)
        if not servers:
            return []
        if not mcp_sdk_available():
            _log.warning("mcp_sdk_missing", servers=len(servers))
            return []
        tools: list[LocalMcpTool] = []
        for spec in servers:
            listing = await self.discover(spec)
            tools.extend(
                self._tool_for(registry, spec, definition)
                for definition in listing.tools
            )
        return tools

    def _tool_for(
        self,
        registry: ToolRegistry,
        spec: LocalMcpServerSpec,
        definition: Mapping[str, Any],
    ) -> LocalMcpTool:
        """Resolve the registry's policy defaults onto one discovered tool.

        Args:
            registry: The registry supplying policy defaults.
            spec: The owning server.
            definition: The translated Anthropic tool definition.

        Returns:
            The policy-resolved local tool.
        """
        probe = ToolSpec(
            name=str(definition["name"]),
            kind=ToolKind.RETRIEVAL,
            description=str(definition["description"]),
            input_schema=dict(definition["input_schema"]),
            tenant_id=spec.tenant_id,
            allowed_roles=list(spec.allowed_roles),
            max_result_chars=registry.config.max_result_chars,
        )
        resolved = registry.resolve_policy(probe, spec.policy)
        ceiling = spec.policy.resolved_max_classification(
            kind=ToolKind.MCP, default=registry.config.max_classification
        )
        return LocalMcpTool(
            name=probe.name,
            description=probe.description,
            input_schema=probe.input_schema,
            server=spec,
            policy=resolved.policy,
            timeout_seconds=float(spec.timeout_seconds or resolved.timeout_seconds),
            rate_limit_per_minute=resolved.rate_limit_per_minute,
            max_classification=ceiling,
            max_result_chars=resolved.max_result_chars,
        )

    # -------------------------------------------------------------- invocation

    async def call(
        self,
        spec: LocalMcpServerSpec,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        max_result_chars: int,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Invoke one tool on a self-hosted MCP server.

        Args:
            spec: The owning server.
            tool_call_id: The model's ``tool_use`` block id.
            tool_name: Namespaced tool name as the model called it.
            arguments: Arguments the model supplied.
            max_result_chars: Truncation budget for the rendered result.
            timeout_seconds: Per-call timeout; falls back to the server's own or the
                tool-layer default.

        Returns:
            The tool result. A transport failure returns ``is_error`` and disables the
            server for the cool-down rather than raising.
        """
        _, remote_name = split_local_tool_name(tool_name)
        started = time.perf_counter()
        budget = float(
            timeout_seconds or spec.timeout_seconds or self.config.timeout_seconds
        )

        def elapsed() -> float:
            """Milliseconds since the call started.

            Returns:
                The elapsed time in milliseconds.
            """
            return (time.perf_counter() - started) * _MS

        if self.is_disabled(spec.name):
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                kind=ToolKind.MCP,
                message=f"MCP server {spec.name!r} is temporarily unavailable",
                latency_ms=elapsed(),
            )
        if spec.allowed_tools and remote_name not in spec.allowed_tools:
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                kind=ToolKind.MCP,
                message=f"tool {remote_name!r} is not exposed by {spec.name!r}",
                latency_ms=elapsed(),
            )

        try:
            with anyio.fail_after(budget):
                async with AsyncExitStack() as stack:
                    session = await self._open_session(stack, spec)
                    outcome = await session.call_tool(remote_name, dict(arguments))
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            self.disable(spec.name, reason=type(exc).__name__)
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                kind=ToolKind.MCP,
                message=f"{type(exc).__name__} calling {spec.name!r}",
                latency_ms=elapsed(),
            )
        except Exception as exc:
            self.disable(spec.name, reason=type(exc).__name__)
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                kind=ToolKind.MCP,
                message=f"{type(exc).__name__} calling {spec.name!r}",
                latency_ms=elapsed(),
            )

        text, structured = render_mcp_content(outcome)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            kind=ToolKind.MCP,
            content=text,
            structured=structured,
            is_error=bool(getattr(outcome, "isError", False)),
            error_message="the MCP server reported an error"
            if getattr(outcome, "isError", False)
            else None,
            latency_ms=elapsed(),
        ).truncate(max_result_chars)


def render_mcp_content(outcome: Any) -> tuple[str, dict[str, Any] | None]:
    """Flatten an MCP ``CallToolResult`` into text plus a structured payload.

    Args:
        outcome: The SDK result object, or anything exposing ``content`` and
            ``structuredContent``.

    Returns:
        A ``(text, structured)`` pair. Text blocks are concatenated; every other
        block kind is rendered as a short JSON descriptor so the model still knows
        something was returned.
    """
    structured = getattr(outcome, "structuredContent", None)
    parts: list[str] = []
    for block in getattr(outcome, "content", None) or []:
        kind = str(getattr(block, "type", "") or "")
        if kind == "text":
            parts.append(str(getattr(block, "text", "")))
            continue
        descriptor: dict[str, Any] = {"type": kind or "unknown"}
        for attribute in ("uri", "name", "mimeType"):
            value = getattr(block, attribute, None)
            if value is not None:
                descriptor[attribute] = str(value)
        parts.append(json.dumps(descriptor, ensure_ascii=False))
    text = "\n".join(part for part in parts if part)
    if not text and structured is not None:
        text = json.dumps(structured, ensure_ascii=False, default=str)
    if structured is not None and not isinstance(structured, dict):
        structured = {"result": structured}
    return text, structured
