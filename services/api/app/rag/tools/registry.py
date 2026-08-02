"""Tool registry: load, filter and render the tools the model may call.

Requirement #4 needs a set of tools that is different for every principal. This
module owns that decision:

* declarative specs are loaded from a YAML/JSON file plus ``ragcore.settings``;
* :meth:`ToolRegistry.tools_for` filters by **tenant first, then role** — a tenant
  must never see another tenant's tool, let alone call it;
* every tool carries a :class:`ToolPolicy` with a timeout, a rate limit and an
  allow/deny classification ceiling describing what content it may receive.

``RESTRICTED`` content never leaves the platform: :data:`NEVER_FORWARD` is enforced
on every non-retrieval tool regardless of what the configuration file asks for.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import anyio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragcore.errors import ConfigError, ToolExecutionError
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.models.tool import (
    McpServerSpec,
    ToolKind,
    ToolSpec,
)
from ragcore.pii import PIIDetector, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "DEFAULT_CONFIG_FILENAMES",
    "NEVER_FORWARD",
    "LocalMcpServerSpec",
    "RateLimiter",
    "RegisteredTool",
    "ToolConfig",
    "ToolPolicy",
    "ToolRegistry",
    "build_registry",
    "config_search_paths",
    "get_tool_registry",
    "load_tool_document",
    "redact_arguments",
    "register_tool",
    "reset_tool_registry_cache",
    "tool_config",
]

_log = get_logger(__name__)

#: Classification that may never be forwarded to anything outside the platform.
NEVER_FORWARD: Classification = Classification.RESTRICTED

#: Config files looked for under ``services/api/config`` when no path is configured.
DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("tools.yaml", "tools.example.yaml")

_SECRETISH_KEY = re.compile(
    r"(?:^|[_-])(secret|password|passwd|token|api[_-]?key|credential|authorization"
    r"|cookie|bearer|signature)(?:$|[_-])",
    re.IGNORECASE,
)
_REDACTED = "«redacted»"
_MAX_REDACTED_VALUE_CHARS = 512


# --------------------------------------------------------------------------- config


class ToolTuning(BaseSettings):
    """Tool-layer tunables that ``ragcore.settings`` does not carry yet.

    Read from the same ``RAG_*`` environment and ``.env`` file as
    :class:`ragcore.settings.Settings`, so an operator configures them the same
    way. Any field that later appears on :class:`~ragcore.settings.Settings` wins
    over the value here — see :func:`tool_config`.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tool_retry_attempts: int = Field(
        default=2, ge=0, description="Retries after the first REST attempt."
    )
    tool_retry_backoff_seconds: float = Field(
        default=0.25, gt=0, description="Base delay for jittered REST backoff."
    )
    tool_retry_max_backoff_seconds: float = Field(
        default=4.0, gt=0, description="Ceiling for one REST backoff sleep."
    )
    tool_circuit_failure_threshold: int = Field(
        default=5, gt=0, description="Consecutive failures that open the breaker."
    )
    tool_circuit_reset_seconds: float = Field(
        default=30.0, gt=0, description="Cool-down before the breaker half-opens."
    )
    tool_rate_limit_per_minute: int = Field(
        default=30, gt=0, description="Default per-tenant, per-tool call budget."
    )
    tool_max_classification: str = Field(
        default=Classification.INTERNAL.value,
        description="Default ceiling of content a tool may receive.",
    )
    tool_max_response_bytes: int = Field(
        default=2_000_000, gt=0, description="Hard cap on a REST response body."
    )
    tool_max_projected_items: int = Field(
        default=25, gt=0, description="Items kept when a projection yields a list."
    )
    tool_response_pii_scan: bool = Field(
        default=True, description="Redact PII in a tool response before the prompt."
    )
    tool_secret_cache_seconds: float = Field(
        default=300.0, gt=0, description="TTL of a resolved Key Vault secret."
    )
    tool_oauth_expiry_skew_seconds: float = Field(
        default=60.0, ge=0, description="Refresh an OAuth token this early."
    )
    tool_loop_repeat_limit: int = Field(
        default=2,
        gt=1,
        description="Identical (tool, arguments) calls tolerated before breaking.",
    )
    tool_router_min_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence below which tools are attempted.",
    )
    tool_mcp_local_enabled: bool = Field(
        default=True,
        description="Allow self-hosted MCP servers over the official mcp SDK.",
    )
    tool_mcp_discovery_ttl_seconds: float = Field(
        default=300.0, gt=0, description="TTL of a cached local MCP tool listing."
    )
    tool_mcp_connect_timeout_seconds: float = Field(
        default=10.0, gt=0, description="Connect timeout for a local MCP session."
    )
    tool_mcp_disable_seconds: float = Field(
        default=120.0,
        gt=0,
        description="How long an unreachable MCP server stays disabled.",
    )
    tool_result_cache_max_entries: int = Field(
        default=256, gt=0, description="Entries kept in the cacheable-tool cache."
    )
    tool_result_summary_chars: int = Field(
        default=500,
        gt=0,
        description="Length of the redacted preview stored on tool_invocations.",
    )
    tool_secret_env_prefix: str = Field(
        default="RAG_TOOL_SECRET_",
        description="Environment fallback prefix when Key Vault is unavailable.",
    )
    tool_oauth_token_url_template: str = Field(
        default="{authority}/{tenant}/oauth2/v2.0/token",
        description="OAuth2 client-credentials token endpoint template.",
    )
    tool_oauth_client_secret_ref: str | None = Field(
        default=None,
        description="Key Vault secret name holding the OAuth2 client secret.",
    )
    tool_managed_identity_endpoint: str = Field(
        default="http://169.254.169.254/metadata/identity/oauth2/token",
        description="Azure IMDS token endpoint used by managed-identity auth.",
    )
    tool_managed_identity_api_version: str = Field(
        default="2018-02-01", description="IMDS API version query parameter."
    )
    tool_user_agent: str = Field(
        default="productionizing-rag-tools/0.1",
        description="User-Agent header sent on every outbound REST tool call.",
    )
    tool_follow_redirects: bool = Field(
        default=False,
        description="Follow upstream redirects. Off by default: a redirect can "
        "move a credential-bearing request to another host.",
    )


@dataclass(frozen=True, slots=True)
class ToolConfig:
    """Resolved tool-layer configuration for one :class:`Settings` instance."""

    enabled: bool
    max_iterations: int
    timeout_seconds: float
    max_result_chars: int
    registry_path: str | None
    mcp_enabled: bool
    mcp_beta_flag: str
    allow_insecure_http: bool
    retry_attempts: int
    retry_backoff_seconds: float
    retry_max_backoff_seconds: float
    circuit_failure_threshold: int
    circuit_reset_seconds: float
    rate_limit_per_minute: int
    max_classification: Classification
    max_response_bytes: int
    max_projected_items: int
    response_pii_scan: bool
    secret_cache_seconds: float
    oauth_expiry_skew_seconds: float
    loop_repeat_limit: int
    router_min_confidence: float
    mcp_local_enabled: bool
    mcp_discovery_ttl_seconds: float
    mcp_connect_timeout_seconds: float
    mcp_disable_seconds: float
    result_cache_max_entries: int
    result_summary_chars: int
    secret_env_prefix: str
    oauth_token_url_template: str
    oauth_client_secret_ref: str | None
    managed_identity_endpoint: str
    managed_identity_api_version: str
    user_agent: str
    follow_redirects: bool
    authority_host: str
    entra_tenant_id: str | None
    entra_client_id: str | None
    key_vault_url: str | None


_tuning: ToolTuning | None = None


def _tuning_singleton() -> ToolTuning:
    """Return the process-wide :class:`ToolTuning` instance.

    Returns:
        The lazily constructed tuning object.
    """
    global _tuning
    if _tuning is None:
        _tuning = ToolTuning()
    return _tuning


def tool_config(settings: Settings | None = None) -> ToolConfig:
    """Resolve every tool-layer tunable, preferring ``ragcore.settings``.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The merged :class:`ToolConfig`. No caller ever hard-codes one of these
        values.
    """
    cfg = settings or get_settings()
    tune = _tuning_singleton()

    def pick(name: str) -> Any:
        """Read a field from Settings when it exists, else from ToolTuning.

        Args:
            name: Field name, identical on both objects.

        Returns:
            The resolved value.
        """
        value = getattr(cfg, name, None)
        return getattr(tune, name) if value is None else value

    ceiling = Classification(str(pick("tool_max_classification")))
    return ToolConfig(
        enabled=cfg.tool_enabled,
        max_iterations=cfg.tool_max_iterations,
        timeout_seconds=cfg.tool_timeout_seconds,
        max_result_chars=cfg.tool_max_result_chars,
        registry_path=cfg.tool_registry_path,
        mcp_enabled=cfg.tool_mcp_enabled,
        mcp_beta_flag=cfg.tool_mcp_beta_flag,
        allow_insecure_http=cfg.tool_allow_insecure_http,
        retry_attempts=int(pick("tool_retry_attempts")),
        retry_backoff_seconds=float(pick("tool_retry_backoff_seconds")),
        retry_max_backoff_seconds=float(pick("tool_retry_max_backoff_seconds")),
        circuit_failure_threshold=int(pick("tool_circuit_failure_threshold")),
        circuit_reset_seconds=float(pick("tool_circuit_reset_seconds")),
        rate_limit_per_minute=int(pick("tool_rate_limit_per_minute")),
        max_classification=ceiling,
        max_response_bytes=int(pick("tool_max_response_bytes")),
        max_projected_items=int(pick("tool_max_projected_items")),
        response_pii_scan=bool(pick("tool_response_pii_scan")),
        secret_cache_seconds=float(pick("tool_secret_cache_seconds")),
        oauth_expiry_skew_seconds=float(pick("tool_oauth_expiry_skew_seconds")),
        loop_repeat_limit=int(pick("tool_loop_repeat_limit")),
        router_min_confidence=float(pick("tool_router_min_confidence")),
        mcp_local_enabled=bool(pick("tool_mcp_local_enabled")),
        mcp_discovery_ttl_seconds=float(pick("tool_mcp_discovery_ttl_seconds")),
        mcp_connect_timeout_seconds=float(pick("tool_mcp_connect_timeout_seconds")),
        mcp_disable_seconds=float(pick("tool_mcp_disable_seconds")),
        result_cache_max_entries=int(pick("tool_result_cache_max_entries")),
        result_summary_chars=int(pick("tool_result_summary_chars")),
        secret_env_prefix=str(pick("tool_secret_env_prefix")),
        oauth_token_url_template=str(pick("tool_oauth_token_url_template")),
        oauth_client_secret_ref=pick("tool_oauth_client_secret_ref"),
        managed_identity_endpoint=str(pick("tool_managed_identity_endpoint")),
        managed_identity_api_version=str(pick("tool_managed_identity_api_version")),
        user_agent=str(pick("tool_user_agent")),
        follow_redirects=bool(pick("tool_follow_redirects")),
        authority_host=str(cfg.entra_authority_host),
        entra_tenant_id=cfg.entra_tenant_id,
        entra_client_id=cfg.azure_client_id or cfg.entra_client_id,
        key_vault_url=cfg.azure_key_vault_url,
    )


# ---------------------------------------------------------------------- policy


class ToolPolicy(BaseModel):
    """Operational limits and the egress classification gate for one tool."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float | None = Field(
        default=None, gt=0, description="Overrides tool_timeout_seconds."
    )
    rate_limit_per_minute: int | None = Field(
        default=None, gt=0, description="Overrides tool_rate_limit_per_minute."
    )
    retry_attempts: int | None = Field(
        default=None, ge=0, description="Overrides tool_retry_attempts."
    )
    circuit_failure_threshold: int | None = Field(
        default=None, gt=0, description="Overrides tool_circuit_failure_threshold."
    )
    circuit_reset_seconds: float | None = Field(
        default=None, gt=0, description="Overrides tool_circuit_reset_seconds."
    )
    max_classification: Classification | None = Field(
        default=None,
        description=(
            "Highest classification of context content this tool may receive. "
            "Clamped below RESTRICTED for every tool that leaves the platform."
        ),
    )
    allow_pii_in_arguments: bool = Field(
        default=False,
        description=(
            "Whether detected PII may be forwarded in the arguments. False blocks "
            "the call; set true only for a tool whose whole job is a lookup by a "
            "personal identifier."
        ),
    )
    response_pii_scan: bool | None = Field(
        default=None, description="Overrides tool_response_pii_scan."
    )
    require_admin: bool = Field(
        default=False, description="Additionally require the platform admin role."
    )

    def resolved_max_classification(
        self, *, kind: ToolKind, default: Classification
    ) -> Classification:
        """Return the effective egress ceiling for this tool.

        Args:
            kind: The owning tool's kind. Only ``retrieval`` stays in-process.
            default: Fallback ceiling from :class:`ToolConfig`.

        Returns:
            The configured ceiling, never above :data:`NEVER_FORWARD` and never
            equal to it for a tool that calls out of the platform.
        """
        ceiling = self.max_classification or default
        if kind is ToolKind.RETRIEVAL:
            return ceiling
        if ceiling >= NEVER_FORWARD:
            return Classification.CONFIDENTIAL
        return ceiling


class LocalMcpServerSpec(BaseModel):
    """A self-hosted MCP server spoken to directly with the official ``mcp`` SDK.

    Used when the server lives inside the VNet and Anthropic's server-side
    connector cannot reach it. ``stdio`` launches a child process; ``streamable_http``
    connects to an HTTP endpoint that may be plain ``http://`` inside the VNet when
    ``tool_allow_insecure_http`` is set.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Server name; prefixes every discovered tool.")
    transport: str = Field(default="stdio", description="'stdio' or 'streamable_http'.")
    description: str = Field(
        default="", description="Shown to the model when a tool has none."
    )
    command: str | None = Field(default=None, description="Executable for stdio.")
    args: list[str] = Field(default_factory=list, description="Arguments for stdio.")
    env: dict[str, str] = Field(
        default_factory=dict, description="Extra environment for the child process."
    )
    cwd: str | None = Field(default=None, description="Working directory for stdio.")
    url: str | None = Field(
        default=None, description="Endpoint for streamable_http transports."
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Static headers for streamable_http."
    )
    authorization_token_ref: str | None = Field(
        default=None, description="Key Vault secret name holding a bearer token."
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Allowlist of server-side tool names; empty exposes all.",
    )
    tenant_id: str | None = Field(
        default=None, description="Restrict to one tenant. None means every tenant."
    )
    allowed_roles: list[str] = Field(
        default_factory=list, description="App roles permitted; empty means all."
    )
    enabled: bool = Field(default=True, description="Skip this server when false.")
    timeout_seconds: float | None = Field(
        default=None, gt=0, description="Overrides tool_timeout_seconds."
    )
    policy: ToolPolicy = Field(
        default_factory=ToolPolicy, description="Applied to every discovered tool."
    )

    @field_validator("transport")
    @classmethod
    def _known_transport(cls, value: str) -> str:
        """Validate the transport name.

        Args:
            value: Candidate transport.

        Returns:
            The normalised transport.

        Raises:
            ValueError: If the transport is not supported locally.
        """
        normalised = value.strip().lower()
        if normalised not in {"stdio", "streamable_http"}:
            msg = (
                f"local MCP transport must be 'stdio' or 'streamable_http', "
                f"got {value!r}"
            )
            raise ValueError(msg)
        return normalised

    @model_validator(mode="after")
    def _require_transport_fields(self) -> Self:
        """Ensure the fields the chosen transport needs are present.

        Returns:
            The validated spec.

        Raises:
            ValueError: If a stdio server has no command or an HTTP server no URL.
        """
        if self.transport == "stdio" and not self.command:
            msg = f"local MCP server {self.name!r} uses stdio but has no command"
            raise ValueError(msg)
        if self.transport == "streamable_http" and not self.url:
            msg = f"local MCP server {self.name!r} uses HTTP but has no url"
            raise ValueError(msg)
        return self

    def is_allowed_for(self, principal: Principal) -> bool:
        """Whether this server's tools may be exposed to a principal.

        Args:
            principal: The caller.

        Returns:
            True when enabled, the tenant matches and a permitted role is held.
        """
        if not self.enabled:
            return False
        if self.tenant_id is not None and self.tenant_id != principal.tenant_id:
            return False
        if not self.allowed_roles:
            return True
        return not set(self.allowed_roles).isdisjoint(principal.roles)


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A :class:`ToolSpec` plus the resolved policy the executors enforce."""

    spec: ToolSpec
    policy: ToolPolicy
    timeout_seconds: float
    rate_limit_per_minute: int
    max_classification: Classification
    max_result_chars: int
    server_name: str | None = None

    @property
    def name(self) -> str:
        """Tool name as the model sees it.

        Returns:
            The spec's name.
        """
        return self.spec.name

    @property
    def kind(self) -> ToolKind:
        """Execution kind.

        Returns:
            ``retrieval``, ``rest`` or ``mcp``.
        """
        return self.spec.kind

    def may_receive(self, classification: Classification) -> bool:
        """Whether content of a given classification may be sent to this tool.

        Args:
            classification: Highest classification present in what would be
                forwarded (arguments plus any quoted context).

        Returns:
            True when the content is at or below the tool's ceiling. ``RESTRICTED``
            is always False for a tool that leaves the platform.
        """
        if self.kind is not ToolKind.RETRIEVAL and classification >= NEVER_FORWARD:
            return False
        return classification <= self.max_classification

    def is_allowed_for(self, principal: Principal) -> bool:
        """Tenant- and role-filter this tool for a principal.

        Args:
            principal: The caller.

        Returns:
            True when the tool may be exposed.
        """
        if not self.spec.is_allowed_for(principal):
            return False
        return not (self.policy.require_admin and not principal.is_admin())


# -------------------------------------------------------------- file loading


def _parse_scalar(raw: str) -> Any:
    """Convert a YAML-ish scalar token into a Python value.

    Args:
        raw: The token, already stripped of surrounding whitespace.

    Returns:
        A str, int, float, bool or None.
    """
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"null", "~", ""}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except ValueError:
            return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_lines(text: str) -> list[tuple[int, str]]:
    """Return significant lines with their indentation depth.

    Args:
        text: Raw document text.

    Returns:
        ``(indent, content)`` pairs, comments and blank lines removed.
    """
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        without_tabs = raw.replace("\t", "  ")
        stripped = without_tabs.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((len(without_tabs) - len(without_tabs.lstrip(" ")), stripped))
    return out


def _strip_comment(value: str) -> str:
    """Remove a trailing ``#`` comment that is outside quotes.

    Args:
        value: A scalar token possibly followed by a comment.

    Returns:
        The token without its trailing comment.
    """
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "#" and index and value[index - 1] == " ":
            return value[:index].rstrip()
    return value.rstrip()


def _parse_block(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[Any, int]:
    """Parse one indented block of the restricted YAML subset.

    Args:
        lines: Output of :func:`_split_lines`.
        start: Index of the first line of the block.
        indent: Indentation depth owning the block.

    Returns:
        A ``(value, next_index)`` pair.
    """
    if start < len(lines) and _is_item(lines[start][1]):
        return _parse_sequence(lines, start, indent)
    return _parse_mapping(lines, start, indent)


def _is_item(content: str) -> bool:
    """Whether a line opens a block-sequence item.

    Args:
        content: The stripped line.

    Returns:
        True for ``-`` on its own or a ``- `` prefix.
    """
    return content == "-" or content.startswith("- ")


def _parse_sequence(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[list[Any], int]:
    """Parse a block sequence.

    Args:
        lines: All significant lines.
        start: Index of the first ``- `` line.
        indent: Indentation of the sequence.

    Returns:
        A ``(list, next_index)`` pair.
    """
    items: list[Any] = []
    index = start
    while index < len(lines):
        depth, content = lines[index]
        if depth != indent or not _is_item(content):
            break
        body = content[1:].lstrip()
        index += 1
        if not body:
            if index < len(lines) and lines[index][0] > depth:
                value, index = _parse_block(lines, index, lines[index][0])
                items.append(value)
            else:
                items.append(None)
            continue
        if ":" in body and not body.startswith(("'", '"')):
            child_indent = depth + 2
            synthetic: list[tuple[int, str]] = [(child_indent, body)]
            nested = index
            while nested < len(lines) and lines[nested][0] > depth:
                synthetic.append(lines[nested])
                nested += 1
            value, _ = _parse_mapping(synthetic, 0, child_indent)
            items.append(value)
            index = nested
            continue
        items.append(_parse_scalar(_strip_comment(body)))
    return items, index


def _parse_mapping(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[dict[str, Any], int]:
    """Parse a block mapping.

    Args:
        lines: All significant lines.
        start: Index of the first key line.
        indent: Indentation of the mapping.

    Returns:
        A ``(dict, next_index)`` pair.

    Raises:
        ConfigError: If a line is neither ``key:`` nor ``key: value``.
    """
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        depth, content = lines[index]
        if depth < indent:
            break
        if depth > indent:
            msg = f"unexpected indentation in tool config near {content[:40]!r}"
            raise ConfigError(msg)
        if _is_item(content):
            break
        if ":" not in content:
            msg = f"expected 'key: value' in tool config, got {content[:40]!r}"
            raise ConfigError(msg)
        key, _, rest = content.partition(":")
        key = key.strip().strip("'\"")
        rest = _strip_comment(rest.strip())
        index += 1
        if rest:
            mapping[key] = _parse_scalar(rest)
            continue
        if index < len(lines) and lines[index][0] > indent:
            child_indent = lines[index][0]
            value, index = _parse_block(lines, index, child_indent)
            mapping[key] = value
        elif index < len(lines) and _is_item(lines[index][1]):
            value, index = _parse_sequence(lines, index, lines[index][0])
            mapping[key] = value
        else:
            mapping[key] = None
    return mapping, index


def _parse_restricted_yaml(text: str) -> dict[str, Any]:
    """Parse the YAML subset the shipped tool config uses, without PyYAML.

    Supported: block mappings, block sequences, quoted and bare scalars, inline
    JSON for flow collections, and ``#`` comments. Anchors, multi-document files
    and block scalars are not supported — install PyYAML for those.

    Args:
        text: Document text.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the document is not a mapping at the top level.
    """
    lines = _split_lines(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    if not isinstance(value, dict):
        msg = "tool config must be a mapping at the top level"
        raise ConfigError(msg)
    return value


def load_tool_document(path: str | Path) -> dict[str, Any]:
    """Load a tool configuration document from YAML or JSON.

    PyYAML is used when installed; otherwise a restricted YAML subset parser
    handles the shipped format so a missing optional dependency cannot take the
    tool layer down.

    Args:
        path: File to read.

    Returns:
        The parsed document, or an empty mapping when the file is empty.

    Raises:
        ConfigError: If the file is missing or does not parse to a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"tool config not found: {file_path}"
        raise ConfigError(msg)
    text = file_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        document = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            document = _parse_restricted_yaml(text)
        else:
            document = yaml.safe_load(text) or {}
    if document is None:
        return {}
    if not isinstance(document, dict):
        msg = f"tool config {file_path} must be a mapping at the top level"
        raise ConfigError(msg)
    return document


def config_search_paths(settings: Settings | None = None) -> list[Path]:
    """Ordered candidate paths for the tool configuration file.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        ``tool_registry_path`` first when set, then ``services/api/config/tools.yaml``
        and finally the checked-in ``tools.example.yaml``.
    """
    cfg = settings or get_settings()
    candidates: list[Path] = []
    if cfg.tool_registry_path:
        candidates.append(Path(cfg.tool_registry_path))
    config_dir = Path(__file__).resolve().parents[3] / "config"
    candidates.extend(config_dir / name for name in DEFAULT_CONFIG_FILENAMES)
    return candidates


# --------------------------------------------------------------- rate limiting


class RateLimiter:
    """Per-``(tenant, tool)`` token bucket refilled at a per-minute rate."""

    def __init__(self) -> None:
        """Initialise an empty limiter."""
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = anyio.Lock()

    async def acquire(self, *, tenant_id: str, tool_name: str, per_minute: int) -> bool:
        """Consume one token for a tenant's use of a tool.

        Args:
            tenant_id: Owning tenant — the bucket is tenant-scoped so one tenant
                can never exhaust another's budget.
            tool_name: Tool being called.
            per_minute: Budget for this tool.

        Returns:
            True when the call may proceed, False when the budget is exhausted.
        """
        key = (tenant_id, tool_name)
        now = time.monotonic()
        capacity = float(per_minute)
        refill_per_second = capacity / 60.0
        async with self._lock:
            tokens, last = self._buckets.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * refill_per_second)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True

    def reset(self) -> None:
        """Drop every bucket. Test helper."""
        self._buckets.clear()


# ------------------------------------------------------------------- redaction


def redact_arguments(
    arguments: Mapping[str, Any],
    *,
    detector: PIIDetector | None = None,
    settings: Settings | None = None,
    max_chars: int = _MAX_REDACTED_VALUE_CHARS,
) -> dict[str, Any]:
    """Redact tool arguments before they are logged, traced or persisted.

    Two passes: keys that look like credentials are replaced outright, and every
    remaining string is run through the PII detector. Nothing else in this package
    persists an argument value.

    Args:
        arguments: Raw arguments the model produced.
        detector: PII detector; defaults to the cached one.
        settings: Platform settings; defaults to :func:`get_settings`.
        max_chars: Longest string value kept.

    Returns:
        A JSON-serialisable copy safe to persist.
    """
    cfg = settings or get_settings()
    pii = detector if detector is not None else get_pii_detector(cfg)

    def scrub(key: str | None, value: Any) -> Any:
        """Redact one value.

        Args:
            key: Owning key, when the value came from a mapping.
            value: The value to redact.

        Returns:
            The redacted value.
        """
        if key and _SECRETISH_KEY.search(key):
            return _REDACTED
        if isinstance(value, str):
            clipped = value[:max_chars]
            try:
                redacted, _ = pii.scan_and_redact(clipped)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                return _REDACTED
            return redacted
        if isinstance(value, Mapping):
            return {str(k): scrub(str(k), v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(None, item) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return scrub(None, str(value))

    return {str(k): scrub(str(k), v) for k, v in arguments.items()}


# -------------------------------------------------------------------- registry


class ToolRegistry:
    """The set of tools the platform knows about, filtered per principal."""

    def __init__(
        self,
        tools: Sequence[RegisteredTool],
        *,
        local_mcp_servers: Sequence[LocalMcpServerSpec] = (),
        settings: Settings | None = None,
        config: ToolConfig | None = None,
        source: str | None = None,
    ) -> None:
        """Build a registry.

        Args:
            tools: Registered tools, already policy-resolved.
            local_mcp_servers: Self-hosted MCP servers reached with the mcp SDK.
            settings: Platform settings; defaults to :func:`get_settings`.
            config: Pre-resolved tool config; derived from settings when omitted.
            source: Path the document came from, for diagnostics.

        Raises:
            ConfigError: If two tools share a name.
        """
        self.settings = settings or get_settings()
        self.config = config or tool_config(self.settings)
        self.source = source
        self.rate_limiter = RateLimiter()
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                msg = f"duplicate tool name {tool.name!r} in the registry"
                raise ConfigError(msg)
            self._tools[tool.name] = tool
        self._local_servers: dict[str, LocalMcpServerSpec] = {}
        for server in local_mcp_servers:
            if server.name in self._local_servers:
                msg = f"duplicate MCP server name {server.name!r}"
                raise ConfigError(msg)
            self._local_servers[server.name] = server

    def __len__(self) -> int:
        """Number of registered tools.

        Returns:
            The tool count.
        """
        return len(self._tools)

    def all_tools(self) -> list[RegisteredTool]:
        """Every registered tool, unfiltered.

        Returns:
            Tools in registration order. Never pass this to a model.
        """
        return list(self._tools.values())

    def tools_for(self, principal: Principal) -> list[RegisteredTool]:
        """Tools this principal may see and call.

        Tenant is checked before role: a tool pinned to another tenant is invisible
        even to a principal holding every role.

        Args:
            principal: The caller.

        Returns:
            The permitted tools, in registration order.
        """
        if not self.config.enabled:
            return []
        allowed = [t for t in self._tools.values() if t.is_allowed_for(principal)]
        if not self.config.mcp_enabled:
            allowed = [t for t in allowed if t.kind is not ToolKind.MCP]
        return allowed

    def get(self, name: str, principal: Principal) -> RegisteredTool | None:
        """Look a tool up, enforcing the same filter as :meth:`tools_for`.

        Args:
            name: Tool name the model used.
            principal: The caller.

        Returns:
            The tool, or None when it does not exist or is not permitted.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        if not tool.is_allowed_for(principal):
            _log.warning(
                "tool_access_denied",
                tool=name,
                tenant_id=principal.tenant_id,
                reason="tenant_or_role",
            )
            return None
        if tool.kind is ToolKind.MCP and not self.config.mcp_enabled:
            return None
        return tool

    def require(self, name: str, principal: Principal) -> RegisteredTool:
        """Look a tool up or raise.

        Args:
            name: Tool name the model used.
            principal: The caller.

        Returns:
            The permitted tool.

        Raises:
            ToolExecutionError: When the tool is unknown or not permitted. The
                message is identical either way so it cannot be used to probe
                another tenant's tool names.
        """
        tool = self.get(name, principal)
        if tool is None:
            raise ToolExecutionError(
                f"tool {name!r} is not available to this caller",
                tool_name=name,
            )
        return tool

    def anthropic_tools(self, principal: Principal) -> list[dict[str, Any]]:
        """Client-side tool definitions for the Anthropic request.

        MCP tools are excluded: they reach the model through the connector's
        ``mcp_toolset`` entry instead.

        Args:
            principal: The caller.

        Returns:
            One mapping per non-MCP tool.
        """
        return [
            tool.spec.to_anthropic_tool()
            for tool in self.tools_for(principal)
            if tool.kind is not ToolKind.MCP
        ]

    def mcp_specs_for(self, principal: Principal) -> list[McpServerSpec]:
        """Remote MCP server specs this principal may reach.

        Args:
            principal: The caller.

        Returns:
            The :class:`McpServerSpec` of every permitted MCP tool.
        """
        return [
            tool.spec.mcp
            for tool in self.tools_for(principal)
            if tool.kind is ToolKind.MCP and tool.spec.mcp is not None
        ]

    def local_mcp_servers_for(self, principal: Principal) -> list[LocalMcpServerSpec]:
        """Self-hosted MCP servers this principal may reach.

        Args:
            principal: The caller.

        Returns:
            Permitted server specs; empty when local MCP is disabled.
        """
        if not (self.config.enabled and self.config.mcp_local_enabled):
            return []
        return [s for s in self._local_servers.values() if s.is_allowed_for(principal)]

    def register(self, tool: RegisteredTool, *, replace_existing: bool = True) -> None:
        """Add a tool at runtime (the built-ins use this).

        Args:
            tool: The tool to add.
            replace_existing: Overwrite a same-named tool instead of raising.

        Raises:
            ConfigError: If the name is taken and ``replace_existing`` is False.
        """
        if tool.name in self._tools and not replace_existing:
            msg = f"duplicate tool name {tool.name!r} in the registry"
            raise ConfigError(msg)
        self._tools[tool.name] = tool

    def resolve_policy(
        self, spec: ToolSpec, policy: ToolPolicy | None = None
    ) -> RegisteredTool:
        """Apply this registry's defaults to a spec.

        Args:
            spec: The tool spec.
            policy: Optional policy overrides.

        Returns:
            The registered tool with every limit resolved.
        """
        return _register(spec, policy or ToolPolicy(), self.config)


def register_tool(
    spec: ToolSpec,
    *,
    policy: ToolPolicy | None = None,
    config: ToolConfig | None = None,
    settings: Settings | None = None,
) -> RegisteredTool:
    """Resolve a spec plus optional policy into a :class:`RegisteredTool`.

    The public entry point the built-ins and any runtime registration use, so the
    limit-resolution rules live in exactly one place.

    Args:
        spec: The tool spec.
        policy: Policy overrides; an all-defaults policy when omitted.
        config: Resolved tool-layer defaults; derived from ``settings`` when omitted.
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The registered tool with every limit resolved.
    """
    resolved = config or tool_config(settings)
    return _register(spec, policy or ToolPolicy(), resolved)


def _register(spec: ToolSpec, policy: ToolPolicy, config: ToolConfig) -> RegisteredTool:
    """Resolve a spec plus policy into a :class:`RegisteredTool`.

    Args:
        spec: The tool spec.
        policy: Policy overrides from the config file.
        config: Resolved tool-layer defaults.

    Returns:
        The registered tool.
    """
    timeout = policy.timeout_seconds
    if timeout is None and spec.rest is not None:
        timeout = spec.rest.timeout_seconds
    return RegisteredTool(
        spec=spec,
        policy=policy,
        timeout_seconds=float(timeout or config.timeout_seconds),
        rate_limit_per_minute=int(
            policy.rate_limit_per_minute or config.rate_limit_per_minute
        ),
        max_classification=policy.resolved_max_classification(
            kind=spec.kind, default=config.max_classification
        ),
        max_result_chars=min(spec.max_result_chars, config.max_result_chars),
        server_name=spec.mcp.name if spec.mcp is not None else None,
    )


def _apply_defaults(
    entry: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge document-level defaults under one entry.

    Args:
        entry: The tool or server entry.
        defaults: The document's ``defaults`` block.

    Returns:
        A new mapping with defaults filled in for absent keys.
    """
    merged: dict[str, Any] = {k: v for k, v in defaults.items() if k != "policy"}
    merged.update({k: v for k, v in entry.items() if k != "policy"})
    policy: dict[str, Any] = dict(defaults.get("policy") or {})
    policy.update(dict(entry.get("policy") or {}))
    if policy:
        merged["policy"] = policy
    return merged


def _tool_from_entry(
    entry: Mapping[str, Any], defaults: Mapping[str, Any], config: ToolConfig
) -> RegisteredTool:
    """Build one registered tool from a config entry.

    Args:
        entry: The ``tools:`` list item.
        defaults: Document-level defaults.
        config: Resolved tool-layer defaults.

    Returns:
        The registered tool.

    Raises:
        ConfigError: If the entry is not a valid :class:`ToolSpec`.
    """
    merged = _apply_defaults(entry, defaults)
    policy_data = merged.pop("policy", {}) or {}
    merged.pop("transport", None)
    try:
        policy = ToolPolicy.model_validate(policy_data)
        spec = ToolSpec.model_validate(merged)
    except Exception as exc:
        name = merged.get("name", "<unnamed>")
        msg = f"invalid tool definition {name!r}: {exc}"
        raise ConfigError(msg) from exc
    return _register(spec, policy, config)


def _mcp_entry(
    entry: Mapping[str, Any], defaults: Mapping[str, Any], config: ToolConfig
) -> tuple[RegisteredTool | None, LocalMcpServerSpec | None]:
    """Build either a remote-connector tool or a local MCP server spec.

    Args:
        entry: The ``mcp_servers:`` list item.
        defaults: Document-level defaults.
        config: Resolved tool-layer defaults.

    Returns:
        ``(registered_tool, None)`` for a remote server, ``(None, local_spec)``
        for a self-hosted one.

    Raises:
        ConfigError: If the entry is invalid.
    """
    merged = _apply_defaults(entry, defaults)
    transport = str(merged.get("transport", "remote")).strip().lower()
    policy_data = merged.pop("policy", {}) or {}
    if transport != "remote":
        try:
            merged["transport"] = transport
            merged["policy"] = policy_data
            return None, LocalMcpServerSpec.model_validate(merged)
        except Exception as exc:
            name = merged.get("name", "<unnamed>")
            msg = f"invalid local MCP server {name!r}: {exc}"
            raise ConfigError(msg) from exc

    merged.pop("transport", None)
    description = str(
        merged.pop("description", "")
        or f"Tools exposed by the {merged.get('name')} MCP server."
    )
    tenant_id = merged.pop("tenant_id", None)
    allowed_roles = list(merged.pop("allowed_roles", []) or [])
    enabled = bool(merged.pop("enabled", True))
    max_result_chars = int(merged.pop("max_result_chars", config.max_result_chars))
    merged.setdefault("beta_flag", config.mcp_beta_flag)
    try:
        server = McpServerSpec.model_validate(merged)
        spec = ToolSpec(
            name=str(server.name),
            kind=ToolKind.MCP,
            description=description,
            tenant_id=tenant_id,
            allowed_roles=allowed_roles,
            enabled=enabled,
            mcp=server,
            max_result_chars=max_result_chars,
        )
        policy = ToolPolicy.model_validate(policy_data)
    except Exception as exc:
        name = merged.get("name", "<unnamed>")
        msg = f"invalid MCP server {name!r}: {exc}"
        raise ConfigError(msg) from exc
    return _register(spec, policy, config), None


def build_registry(
    settings: Settings | None = None,
    *,
    document: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    extra_tools: Iterable[RegisteredTool] = (),
) -> ToolRegistry:
    """Load the registry from a document, an explicit path or the search paths.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.
        document: Pre-parsed document, which wins over ``path``.
        path: Explicit config path, which wins over the search paths.
        extra_tools: Tools registered on top of the document (the built-ins).

    Returns:
        The populated registry. A missing config file is not an error — the
        built-ins alone are a valid registry.

    Raises:
        ConfigError: If the document is malformed or an explicit path is missing.
    """
    cfg = settings or get_settings()
    config = tool_config(cfg)
    source: str | None = None

    if document is None:
        chosen: Path | None = None
        if path is not None:
            chosen = Path(path)
            if not chosen.is_file():
                msg = f"tool config not found: {chosen}"
                raise ConfigError(msg)
        else:
            chosen = next(
                (p for p in config_search_paths(cfg) if p.is_file()),
                None,
            )
        if chosen is None:
            _log.info("tool_registry_no_config", searched=len(config_search_paths(cfg)))
            document = {}
        else:
            document = load_tool_document(chosen)
            source = str(chosen)

    defaults = dict(document.get("defaults") or {})
    tools: list[RegisteredTool] = []
    for entry in document.get("tools") or []:
        if not isinstance(entry, Mapping):
            msg = "each item under 'tools' must be a mapping"
            raise ConfigError(msg)
        tools.append(_tool_from_entry(entry, defaults, config))

    local_servers: list[LocalMcpServerSpec] = []
    for entry in document.get("mcp_servers") or []:
        if not isinstance(entry, Mapping):
            msg = "each item under 'mcp_servers' must be a mapping"
            raise ConfigError(msg)
        remote, local = _mcp_entry(entry, defaults, config)
        if remote is not None:
            tools.append(remote)
        if local is not None:
            local_servers.append(local)

    for tool in extra_tools:
        tools.append(replace(tool))

    registry = ToolRegistry(
        tools,
        local_mcp_servers=local_servers,
        settings=cfg,
        config=config,
        source=source,
    )
    _log.info(
        "tool_registry_loaded",
        source=source,
        tools=len(registry),
        local_mcp_servers=len(local_servers),
    )
    return registry


_registry_cache: dict[tuple[Any, ...], ToolRegistry] = {}


def get_tool_registry(settings: Settings | None = None) -> ToolRegistry:
    """Return the cached registry for a settings instance.

    The built-in retrieval and context tools are always registered, so a
    deployment with no config file still gives the model a way to re-query the
    corpus.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The shared :class:`ToolRegistry`.
    """
    cfg = settings or get_settings()
    key = (
        cfg.tool_registry_path,
        cfg.tool_enabled,
        cfg.tool_mcp_enabled,
        cfg.tool_max_result_chars,
        cfg.tool_timeout_seconds,
    )
    cached = _registry_cache.get(key)
    if cached is not None:
        return cached
    # Imported here: builtin imports the registry for its policy types.
    from app.rag.tools.builtin import builtin_tools

    registry = build_registry(cfg, extra_tools=builtin_tools(cfg))
    _registry_cache[key] = registry
    return registry


def reset_tool_registry_cache() -> None:
    """Drop the cached registries and the cached tuning. Test helper."""
    global _tuning
    _registry_cache.clear()
    _tuning = None
