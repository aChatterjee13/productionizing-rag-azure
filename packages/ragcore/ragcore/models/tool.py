"""Tool-calling models for requirement #4: reaching data that is not indexed.

Three kinds of tool reach the model, all described by one :class:`ToolSpec`:

``retrieval``
    The built-in corpus search tool. Always ACL-filtered through
    :func:`ragcore.vectorstore.filters.build_acl_filter`.
``rest``
    A declarative HTTP call described by :class:`RestToolSpec` — no per-tool Python.
``mcp``
    A remote MCP server described by :class:`McpServerSpec`, invoked server-side via
    the Anthropic MCP connector. The connector needs **both** an ``mcp_servers`` entry
    and a matching ``{"type": "mcp_toolset", "mcp_server_name": ...}`` tool entry; use
    :meth:`McpServerSpec.to_connector_entries` so the pair can never drift.

Secrets are never stored in a spec. ``*_secret_ref`` fields name a Key Vault secret
that the tool executor resolves at call time.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragcore.models.acl import Principal

__all__ = [
    "McpServerSpec",
    "RestToolSpec",
    "ToolAuth",
    "ToolKind",
    "ToolResult",
    "ToolSpec",
]


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class ToolKind(StrEnum):
    """How a tool is executed.

    Matches the ``kind`` field on :class:`ragcore.models.chat.ToolCall` and the
    ``tool_invocations.kind`` column.
    """

    RETRIEVAL = "retrieval"
    REST = "rest"
    MCP = "mcp"


class ToolAuth(StrEnum):
    """How a REST tool authenticates to its upstream."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    BASIC = "basic"
    ENTRA_OBO = "entra_obo"
    MANAGED_IDENTITY = "managed_identity"


class RestToolSpec(BaseModel):
    """A declarative HTTP tool: everything needed to call it, and nothing more.

    Templates use ``{name}`` placeholders filled from the model-supplied arguments
    after they have been validated against :attr:`ToolSpec.input_schema`. Only names
    declared in the schema may appear, so the model cannot inject arbitrary URL parts.
    """

    model_config = ConfigDict(extra="forbid")

    method: str = Field(default="GET", description="HTTP method.")
    url_template: str = Field(
        description="Absolute URL with {placeholder} segments, e.g. "
        "'https://api.example.com/orders/{order_id}'."
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Static headers sent on every call."
    )
    query_template: dict[str, str] = Field(
        default_factory=dict, description="Query parameters, values may be templates."
    )
    body_template: dict[str, Any] | None = Field(
        default=None, description="JSON body template for POST/PUT/PATCH."
    )
    auth: ToolAuth = Field(default=ToolAuth.NONE, description="Authentication scheme.")
    auth_secret_ref: str | None = Field(
        default=None,
        description=(
            "Key Vault secret name holding the token or key. Never the secret value."
        ),
    )
    auth_header_name: str = Field(
        default="Authorization",
        description="Header the credential is sent in (api_key schemes often differ).",
    )
    auth_scope: str | None = Field(
        default=None, description="Scope requested for entra_obo / managed_identity."
    )
    timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        description="Per-call timeout; defaults to tool_timeout_seconds.",
    )
    response_json_path: str | None = Field(
        default=None,
        description=(
            "Dotted path into the JSON response to extract, e.g. 'data.items'. "
            "None passes the whole body to the model."
        ),
    )
    success_status_codes: list[int] = Field(
        default_factory=lambda: [200, 201, 202, 204],
        description="Statuses treated as success; anything else is a tool error.",
    )
    verify_tls: bool = Field(
        default=True, description="Verify the upstream TLS certificate. Keep true."
    )

    @field_validator("method")
    @classmethod
    def _upper_method(cls, value: str) -> str:
        """Normalise and validate the HTTP method.

        Args:
            value: Candidate method.

        Returns:
            The upper-cased method.

        Raises:
            ValueError: If the method is not one of the supported verbs.
        """
        method = value.upper()
        allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
        if method not in allowed:
            msg = f"unsupported HTTP method {value!r}"
            raise ValueError(msg)
        return method

    @model_validator(mode="after")
    def _validate_auth(self) -> RestToolSpec:
        """Ensure credential-bearing schemes name a secret to resolve.

        Returns:
            The validated spec.

        Raises:
            ValueError: If a scheme that needs a secret has no ``auth_secret_ref``,
                or if an OBO scheme has no scope.
        """
        needs_secret = {ToolAuth.BEARER, ToolAuth.API_KEY, ToolAuth.BASIC}
        if self.auth in needs_secret and not self.auth_secret_ref:
            msg = f"auth={self.auth.value} requires auth_secret_ref"
            raise ValueError(msg)
        if (
            self.auth in {ToolAuth.ENTRA_OBO, ToolAuth.MANAGED_IDENTITY}
            and not self.auth_scope
        ):
            msg = f"auth={self.auth.value} requires auth_scope"
            raise ValueError(msg)
        return self

    @property
    def placeholders(self) -> set[str]:
        """Placeholder names referenced anywhere in this spec's templates.

        Returns:
            The set of ``{name}`` identifiers found in the URL, query and body.
        """
        found: set[str] = set()
        candidates = [self.url_template, *self.query_template.values()]
        if self.body_template is not None:
            candidates.extend(_string_leaves(self.body_template))
        for template in candidates:
            for _, field_name, _, _ in string.Formatter().parse(template):
                if field_name:
                    found.add(field_name.split(".")[0].split("[")[0])
        return found


def _string_leaves(value: Any) -> list[str]:
    """Collect every string leaf in a nested JSON-like structure.

    Args:
        value: A mapping, sequence or scalar.

    Returns:
        All string values found, depth-first.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _string_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item)]
    return []


class McpServerSpec(BaseModel):
    """A remote MCP server reachable through the Anthropic MCP connector."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Server name; referenced by the matching mcp_toolset entry."
    )
    url: str = Field(description="Streamable HTTP MCP endpoint URL.")
    authorization_token_ref: str | None = Field(
        default=None,
        description=(
            "Key Vault secret holding the bearer token. Never the token itself."
        ),
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Allowlist of tool names exposed to the model. Empty exposes every tool "
            "the server advertises."
        ),
    )
    beta_flag: str = Field(
        default="mcp-client-2025-11-20",
        description="Beta flag the connector requires on the beta endpoint.",
    )

    @field_validator("url")
    @classmethod
    def _require_https(cls, value: str) -> str:
        """Reject non-HTTPS MCP endpoints.

        Args:
            value: Candidate URL.

        Returns:
            The validated URL.

        Raises:
            ValueError: If the URL is not HTTPS.
        """
        if not value.startswith("https://"):
            msg = f"MCP server url must be https, got {value!r}"
            raise ValueError(msg)
        return value

    def to_connector_entries(
        self, authorization_token: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the matched ``mcp_servers`` and ``mcp_toolset`` pair.

        Passing ``mcp_servers`` without a matching ``mcp_toolset`` entry is a
        validation error at the API, so both halves are always produced together.

        Args:
            authorization_token: Resolved bearer token, when the server needs one.

        Returns:
            A ``(server_entry, toolset_entry)`` pair ready to place in the request's
            ``mcp_servers`` and ``tools`` lists respectively.
        """
        server: dict[str, Any] = {"type": "url", "name": self.name, "url": self.url}
        if authorization_token:
            server["authorization_token"] = authorization_token
        toolset: dict[str, Any] = {
            "type": "mcp_toolset",
            "mcp_server_name": self.name,
        }
        if self.allowed_tools:
            toolset["default_config"] = {"enabled": False}
            toolset["configs"] = [
                {"name": tool, "enabled": True} for tool in self.allowed_tools
            ]
        return server, toolset


class ToolSpec(BaseModel):
    """A tool the model may call, filtered by tenant and role before exposure."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool name as the model sees it.")
    kind: ToolKind = Field(description="retrieval | rest | mcp.")
    description: str = Field(
        description=(
            "When and why to call this tool. Be prescriptive about the trigger "
            "condition, not just what the tool does."
        )
    )
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the tool arguments.",
    )

    tenant_id: str | None = Field(
        default=None,
        description="Restrict to one tenant. None exposes the tool to every tenant.",
    )
    allowed_roles: list[str] = Field(
        default_factory=list,
        description="App roles permitted to use this tool; empty means all roles.",
    )
    enabled: bool = Field(default=True, description="Skip this tool when false.")

    rest: RestToolSpec | None = Field(
        default=None, description="Transport details when kind is 'rest'."
    )
    mcp: McpServerSpec | None = Field(
        default=None, description="Server details when kind is 'mcp'."
    )

    max_result_chars: int = Field(
        default=8_000,
        gt=0,
        description="Truncate results longer than this before returning to the model.",
    )
    cacheable: bool = Field(
        default=False,
        description="Whether identical arguments may reuse a recent result.",
    )
    cache_ttl_seconds: int = Field(
        default=60, gt=0, description="TTL when cacheable is true."
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Require a name the Anthropic API accepts.

        Args:
            value: Candidate tool name.

        Returns:
            The validated name.

        Raises:
            ValueError: If the name is empty or contains characters outside
                ``[A-Za-z0-9_-]``.
        """
        if not value:
            msg = "tool name must not be empty"
            raise ValueError(msg)
        if not all(char.isalnum() or char in {"_", "-"} for char in value):
            msg = f"tool name {value!r} may only contain letters, digits, '_' and '-'"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_transport(self) -> Self:
        """Ensure the transport block matches the declared kind.

        Returns:
            The validated spec.

        Raises:
            ValueError: If the required transport block is missing, or a block for a
                different kind is present.
        """
        if self.kind is ToolKind.REST and self.rest is None:
            msg = f"tool {self.name!r} has kind='rest' but no rest spec"
            raise ValueError(msg)
        if self.kind is ToolKind.MCP and self.mcp is None:
            msg = f"tool {self.name!r} has kind='mcp' but no mcp spec"
            raise ValueError(msg)
        if self.kind is not ToolKind.REST and self.rest is not None:
            msg = f"tool {self.name!r} has a rest spec but kind={self.kind.value}"
            raise ValueError(msg)
        if self.kind is not ToolKind.MCP and self.mcp is not None:
            msg = f"tool {self.name!r} has an mcp spec but kind={self.kind.value}"
            raise ValueError(msg)
        if self.rest is not None:
            declared = set(self.input_schema.get("properties", {}))
            missing = self.rest.placeholders - declared
            if missing:
                msg = (
                    f"tool {self.name!r} templates reference undeclared arguments: "
                    f"{sorted(missing)}"
                )
                raise ValueError(msg)
        return self

    def is_allowed_for(self, principal: Principal) -> bool:
        """Whether this tool may be exposed to a principal.

        Args:
            principal: The caller.

        Returns:
            True when the tool is enabled, the tenant matches (or the tool is
            tenant-agnostic), and the principal holds a permitted role.
        """
        if not self.enabled:
            return False
        if self.tenant_id is not None and self.tenant_id != principal.tenant_id:
            return False
        if not self.allowed_roles:
            return True
        return not set(self.allowed_roles).isdisjoint(principal.roles)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Render the client-side tool definition sent in ``tools``.

        MCP tools are not client-side; they are exposed through the connector's
        ``mcp_toolset`` entry instead.

        Returns:
            A mapping with ``name``, ``description`` and ``input_schema``.

        Raises:
            ValueError: If called on an MCP tool.
        """
        if self.kind is ToolKind.MCP:
            msg = (
                f"tool {self.name!r} is an MCP tool: use "
                "McpServerSpec.to_connector_entries() instead"
            )
            raise ValueError(msg)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolResult(BaseModel):
    """The outcome of one tool invocation, as returned to the model and persisted.

    ``content`` is what the model sees; ``structured`` keeps the parsed payload for
    lineage and for citation building when a tool returns citable records.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(description="Matches the tool_use block id.")
    tool_name: str = Field(description="Name of the invoked tool.")
    kind: ToolKind = Field(description="retrieval | rest | mcp.")
    content: str = Field(
        default="", description="Rendered result text returned to the model."
    )
    structured: dict[str, Any] | None = Field(
        default=None, description="Parsed result payload, when the tool returned JSON."
    )
    is_error: bool = Field(
        default=False,
        description=(
            "True when the invocation failed. The loop still returns a tool_result "
            "with is_error so the model can recover rather than stalling."
        ),
    )
    error_message: str | None = Field(
        default=None, description="Redacted failure description."
    )
    http_status: int | None = Field(
        default=None, description="Upstream HTTP status, for REST tools."
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Invocation latency.")
    truncated: bool = Field(
        default=False, description="True when content hit max_result_chars."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="When the invocation completed."
    )

    @classmethod
    def failure(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        kind: ToolKind,
        message: str,
        http_status: int | None = None,
        latency_ms: float = 0.0,
    ) -> Self:
        """Build an error result for a failed invocation.

        Args:
            tool_call_id: Matches the tool_use block id.
            tool_name: Name of the failing tool.
            kind: Tool kind.
            message: Redacted failure description shown to the model.
            http_status: Upstream status, when the tool was an HTTP call.
            latency_ms: Time spent before failing.

        Returns:
            A :class:`ToolResult` with ``is_error`` set.
        """
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            kind=kind,
            content=f"Tool error: {message}",
            is_error=True,
            error_message=message,
            http_status=http_status,
            latency_ms=latency_ms,
        )

    def truncate(self, limit: int) -> Self:
        """Return a copy whose content fits a character budget.

        Args:
            limit: Maximum characters of content to keep.

        Returns:
            This result when already short enough, otherwise a truncated copy with
            ``truncated`` set and an explicit marker appended.
        """
        if len(self.content) <= limit:
            return self
        marker = "\n… [truncated]"
        keep = max(0, limit - len(marker))
        return self.model_copy(
            update={"content": self.content[:keep] + marker, "truncated": True}
        )

    def to_result_summary(self, limit: int = 500) -> str:
        """Build the short summary persisted on ``tool_invocations``.

        Args:
            limit: Maximum summary length.

        Returns:
            The error message when the call failed, otherwise a clipped preview of
            the content.
        """
        if self.is_error:
            return (self.error_message or "error")[:limit]
        return self.content[:limit]

    def to_tool_result_block(self) -> dict[str, Any]:
        """Render the ``tool_result`` content block sent back to the model.

        Returns:
            A mapping with ``type``, ``tool_use_id``, ``content`` and ``is_error``.
        """
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }
