"""Requirement #4 — calling APIs and MCP tools when the answer is not in the index.

Layering, in dependency order:

``registry``
    Declarative :class:`~ragcore.models.tool.ToolSpec` loading, per-principal
    filtering (**tenant first, then role**), per-tool policy — timeout, rate limit and
    the egress classification ceiling.
``rest_tool``
    The declarative HTTP executor: schema-validated arguments, templated URL/query/
    body, credentials from Key Vault, retry, circuit breaker, response projection and
    a PII scan before anything reaches the prompt.
``mcp_client``
    Remote MCP through Anthropic's server-side connector (``mcp_servers`` +
    ``mcp_toolset`` + the ``mcp-client-2025-11-20`` beta), and self-hosted MCP spoken
    directly over the official ``mcp`` SDK for servers inside the VNet.
``builtin``
    The corpus-search tool the model can re-query with different filters, plus the
    clock / tenant-context tool.
``router``
    Retrieval-versus-tool routing, the iteration cap, loop detection, and the single
    dispatch chokepoint that screens, executes, traces and persists every call.

Each submodule imports only from the ones above it in that list, so there is no
cycle regardless of the order these re-exports are written in.
"""

from __future__ import annotations

from app.rag.tools.builtin import (
    CONTEXT_TOOL_NAME,
    SEARCH_TOOL_NAME,
    BuiltinExecutor,
    RetrieveFn,
    builtin_tools,
    context_tool_spec,
    qdrant_retrieve,
    search_tool_spec,
)
from app.rag.tools.mcp_client import (
    ConnectorRequest,
    LocalMcpClient,
    LocalMcpTool,
    McpToolListing,
    RemoteMcpConnector,
    build_connector_request,
    local_tool_name,
    mcp_sdk_available,
    split_local_tool_name,
    translate_mcp_tool,
)
from app.rag.tools.registry import (
    NEVER_FORWARD,
    LocalMcpServerSpec,
    RateLimiter,
    RegisteredTool,
    ToolConfig,
    ToolPolicy,
    ToolRegistry,
    build_registry,
    get_tool_registry,
    load_tool_document,
    redact_arguments,
    register_tool,
    reset_tool_registry_cache,
    tool_config,
)
from app.rag.tools.rest_tool import (
    CircuitBreaker,
    CircuitState,
    InvalidToolArgumentsError,
    PreparedRequest,
    RestExecutor,
    SecretResolver,
    TokenProvider,
    get_rest_executor,
    project_response,
    render_template,
    reset_rest_executor,
    validate_arguments,
)
from app.rag.tools.router import (
    ExposedTool,
    LoopGuard,
    LoopVerdict,
    RouteDecision,
    RouteMode,
    ToolContext,
    ToolDispatcher,
    ToolPlan,
    argument_signature,
    decide_route,
)

__all__ = [
    "CONTEXT_TOOL_NAME",
    "NEVER_FORWARD",
    "SEARCH_TOOL_NAME",
    "BuiltinExecutor",
    "CircuitBreaker",
    "CircuitState",
    "ConnectorRequest",
    "ExposedTool",
    "InvalidToolArgumentsError",
    "LocalMcpClient",
    "LocalMcpServerSpec",
    "LocalMcpTool",
    "LoopGuard",
    "LoopVerdict",
    "McpToolListing",
    "PreparedRequest",
    "RateLimiter",
    "RegisteredTool",
    "RemoteMcpConnector",
    "RestExecutor",
    "RetrieveFn",
    "RouteDecision",
    "RouteMode",
    "SecretResolver",
    "TokenProvider",
    "ToolConfig",
    "ToolContext",
    "ToolDispatcher",
    "ToolPlan",
    "ToolPolicy",
    "ToolRegistry",
    "argument_signature",
    "build_connector_request",
    "build_registry",
    "builtin_tools",
    "context_tool_spec",
    "decide_route",
    "get_rest_executor",
    "get_tool_registry",
    "load_tool_document",
    "local_tool_name",
    "mcp_sdk_available",
    "project_response",
    "qdrant_retrieve",
    "redact_arguments",
    "register_tool",
    "render_template",
    "reset_rest_executor",
    "reset_tool_registry_cache",
    "search_tool_spec",
    "split_local_tool_name",
    "tool_config",
    "translate_mcp_tool",
    "validate_arguments",
]
