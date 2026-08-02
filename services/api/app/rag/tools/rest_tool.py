"""Declarative REST tool executor.

A REST tool is data, not code: :class:`~ragcore.models.tool.RestToolSpec` describes
the method, a URL template, query and body templates, an authentication scheme and a
response projection. This module turns that description into a real HTTP call with:

* JSON-Schema validation of the model-supplied arguments **before** anything is
  interpolated, so an undeclared placeholder can never reach the URL;
* credentials resolved at call time from Key Vault (or the environment fallback),
  never stored in the spec — bearer, API key, basic, and OAuth2 client-credentials
  with token caching for the Entra-backed schemes;
* timeout, jittered retry and a per-tool circuit breaker;
* a response size cap plus a JMESPath-style projection, so a 5 MB payload does not
  flood the context window;
* a PII scan of whatever survives, because the projection is about to be pasted into
  a prompt.

Failures never escape as exceptions in the pipeline's happy path: they come back as
``ToolResult(is_error=True)`` so the model can recover.
"""

from __future__ import annotations

import base64
import json
import re
import secrets
import string
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import quote

import anyio
import httpx

from app.rag.tools.registry import (
    RegisteredTool,
    ToolConfig,
    tool_config,
)
from ragcore.errors import ToolExecutionError
from ragcore.logging import get_logger
from ragcore.models.tool import RestToolSpec, ToolAuth, ToolKind, ToolResult
from ragcore.pii import PIIDetector, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "InvalidToolArgumentsError",
    "PreparedRequest",
    "RestExecutor",
    "SecretResolver",
    "TokenProvider",
    "coerce_arguments",
    "get_rest_executor",
    "project_response",
    "render_template",
    "reset_rest_executor",
    "validate_arguments",
]

_log = get_logger(__name__)

_MS = 1000.0
_JSON_MEDIA = re.compile(r"\bjson\b", re.IGNORECASE)
_PATH_SAFE = ""


class InvalidToolArgumentsError(ToolExecutionError):
    """The model supplied arguments that do not satisfy the tool's input schema.

    Raised before any interpolation happens, so a bad argument can never become part
    of a URL, a header or a body.
    """

    status_code = 400
    code = "tool_invalid_arguments"


# --------------------------------------------------------------- schema validation


def _schema_type_names(schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the declared JSON Schema types of a node.

    Args:
        schema: A JSON Schema node.

    Returns:
        A tuple of type names; empty when the node declares no ``type``.
    """
    declared = schema.get("type")
    if declared is None:
        return ()
    if isinstance(declared, str):
        return (declared,)
    return tuple(str(item) for item in declared)


def _coerce_scalar(value: Any, types: Sequence[str]) -> Any:
    """Coerce a model-supplied scalar into a declared JSON Schema type.

    Claude occasionally renders a number or a boolean as a string. Coercing here is
    safer than rejecting the call, and strictly narrower than accepting anything:
    a value that cannot be coerced is left alone and fails validation.

    Args:
        value: The supplied value.
        types: Declared type names for this node.

    Returns:
        The coerced value, or the original when no coercion applies.
    """
    if not types or not isinstance(value, str):
        return value
    text = value.strip()
    if "boolean" in types and text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if "integer" in types:
        try:
            return int(text)
        except ValueError:
            return value
    if "number" in types:
        try:
            return float(text)
        except ValueError:
            return value
    return value


def coerce_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply schema defaults and light scalar coercion to raw model arguments.

    Args:
        schema: The tool's ``input_schema``.
        arguments: Arguments exactly as the model produced them.

    Returns:
        A new mapping with declared defaults filled in and scalars coerced to their
        declared types. Unknown keys are preserved so validation can reject them.
    """
    properties: Mapping[str, Any] = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name, node in properties.items():
        if not isinstance(node, Mapping):
            continue
        if name in arguments:
            out[name] = _coerce_scalar(arguments[name], _schema_type_names(node))
        elif "default" in node:
            out[name] = node["default"]
    for name, value in arguments.items():
        if name not in out:
            out[name] = value
    return out


def _matches_type(value: Any, types: Sequence[str]) -> bool:
    """Whether a value satisfies at least one declared JSON Schema type.

    Args:
        value: The value to check.
        types: Declared type names.

    Returns:
        True when the value matches, or when nothing was declared.
    """
    if not types:
        return True
    checks: dict[str, Any] = {
        "string": str,
        "boolean": bool,
        "array": (list, tuple),
        "object": Mapping,
    }
    for name in types:
        if name == "null" and value is None:
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if (
            name == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return True
        expected = checks.get(name)
        if expected is not None and isinstance(value, expected):
            if name == "string" and isinstance(value, bool):
                continue
            return True
    return False


def _validate_node(
    value: Any, schema: Mapping[str, Any], path: str, errors: list[str]
) -> None:
    """Validate one value against one JSON Schema node, collecting errors.

    Supports the keyword subset a tool argument schema legitimately needs: types,
    ``enum``, ``required``, ``properties``, ``additionalProperties``, ``items``,
    numeric bounds, string bounds and ``pattern``.

    Args:
        value: The value to validate.
        schema: The schema node.
        path: Dotted path used in error messages.
        errors: Accumulator the caller inspects.
    """
    types = _schema_type_names(schema)
    if not _matches_type(value, types):
        errors.append(f"{path}: expected {'/'.join(types)}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of {sorted(map(str, schema['enum']))}")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path}: shorter than minLength {minimum}")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path}: longer than maxLength {maximum}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not re.search(pattern, value):
            errors.append(f"{path}: does not match the required pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low = schema.get("minimum")
        high = schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            errors.append(f"{path}: below minimum {low}")
        if isinstance(high, (int, float)) and value > high:
            errors.append(f"{path}: above maximum {high}")
    if isinstance(value, (list, tuple)):
        item_schema = schema.get("items")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: fewer than minItems {min_items}")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path}: more than maxItems {max_items}")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_node(item, item_schema, f"{path}[{index}]", errors)
    if isinstance(value, Mapping):
        _validate_object(value, schema, path, errors)


def _validate_object(
    value: Mapping[str, Any], schema: Mapping[str, Any], path: str, errors: list[str]
) -> None:
    """Validate an object node: required keys, properties and unknown keys.

    Args:
        value: The object to validate.
        schema: The schema node.
        path: Dotted path used in error messages.
        errors: Accumulator the caller inspects.
    """
    properties: Mapping[str, Any] = schema.get("properties") or {}
    for name in schema.get("required") or []:
        if name not in value:
            errors.append(f"{path}.{name}: required" if path else f"{name}: required")
    for name, node in properties.items():
        if name in value and isinstance(node, Mapping):
            child = f"{path}.{name}" if path else name
            _validate_node(value[name], node, child, errors)
    if schema.get("additionalProperties") is True:
        return
    unknown = sorted(set(value) - set(properties))
    if unknown:
        errors.append(f"unexpected argument(s): {', '.join(unknown)}")


def validate_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    """Validate and normalise model-supplied tool arguments.

    ``jsonschema`` is used for the full check when it is installed; the built-in
    validator runs either way, and it always rejects undeclared properties, so
    behaviour does not change with an optional dependency.

    Args:
        schema: The tool's ``input_schema``.
        arguments: Arguments exactly as the model produced them.
        tool_name: Tool name, used in the error message.

    Returns:
        The coerced, default-filled arguments.

    Raises:
        InvalidToolArgumentsError: When the arguments do not satisfy the schema.
    """
    prepared = coerce_arguments(schema, arguments)
    errors: list[str] = []
    try:
        import jsonschema
    except ImportError:
        pass
    else:  # pragma: no cover - exercised only where jsonschema is installed
        validator = jsonschema.Draft202012Validator(dict(schema))
        errors.extend(
            f"{'.'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in validator.iter_errors(prepared)
        )
    _validate_node(prepared, schema, "", errors)
    if errors:
        msg = f"invalid arguments for {tool_name!r}: {'; '.join(sorted(set(errors)))}"
        raise InvalidToolArgumentsError(msg, tool_name=tool_name, kind="rest")
    return prepared


# ------------------------------------------------------------ template rendering


class _SafeArguments(dict[str, Any]):
    """Formatting mapping that reports a missing placeholder instead of KeyError."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Wrap the validated arguments.

        Args:
            values: The validated arguments.
        """
        super().__init__(values)
        self.missing: set[str] = set()

    def __missing__(self, key: str) -> str:
        """Record an unresolved placeholder.

        Args:
            key: The placeholder name.

        Returns:
            An empty string, so rendering can continue and report every miss at once.
        """
        self.missing.add(key)
        return ""


def render_template(
    template: str,
    arguments: Mapping[str, Any],
    *,
    quote_values: bool = False,
    strict: bool = True,
) -> str:
    """Fill ``{name}`` placeholders from validated arguments.

    Args:
        template: The template string.
        arguments: Validated arguments.
        quote_values: Percent-encode each substituted value. Set for URL paths so a
            value can never introduce a new path segment or query string.
        strict: Raise when a placeholder has no argument. A URL path must be strict —
            a hole in a path is a different resource. A query parameter must not be:
            an optional argument the model omitted simply drops out.

    Returns:
        The rendered string.

    Raises:
        InvalidToolArgumentsError: When ``strict`` and a placeholder is unresolved.
    """
    values: dict[str, Any] = dict(arguments)
    if quote_values:
        values = {
            key: quote(str(value), safe=_PATH_SAFE) if value is not None else ""
            for key, value in values.items()
        }
    mapping = _SafeArguments(values)
    rendered = string.Formatter().vformat(template, (), mapping)
    if mapping.missing and strict:
        msg = f"missing template argument(s): {', '.join(sorted(mapping.missing))}"
        raise InvalidToolArgumentsError(msg, tool_name="<template>", kind="rest")
    return "" if mapping.missing else rendered


class _Missing:
    """Sentinel for a body key whose template referenced an absent argument."""


_MISSING = _Missing()


def _render_body(node: Any, arguments: Mapping[str, Any]) -> Any:
    """Render a JSON body template, preserving types for whole-value placeholders.

    ``{"limit": "{limit}"}`` with ``limit=10`` produces ``{"limit": 10}`` rather than
    the string ``"10"``, because a body template is JSON and upstreams are typed. A
    key whose template references an argument the model omitted is dropped from the
    body rather than sent as an empty string.

    Args:
        node: The template node.
        arguments: Validated arguments.

    Returns:
        The rendered node, or the missing sentinel.
    """
    if isinstance(node, str):
        stripped = node.strip()
        whole = (
            stripped.startswith("{")
            and stripped.endswith("}")
            and stripped.count("{") == 1
        )
        if whole:
            key = stripped[1:-1].split(".")[0].split("[")[0]
            return arguments.get(key, _MISSING)
        rendered = render_template(node, arguments, strict=False)
        return rendered if rendered != "" else _MISSING
    if isinstance(node, Mapping):
        rendered_map = {str(k): _render_body(v, arguments) for k, v in node.items()}
        return {k: v for k, v in rendered_map.items() if not isinstance(v, _Missing)}
    if isinstance(node, list):
        items = [_render_body(item, arguments) for item in node]
        return [item for item in items if not isinstance(item, _Missing)]
    return node


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """Everything needed to issue one REST tool call."""

    method: str
    url: str
    headers: dict[str, str]
    params: dict[str, str]
    json_body: Any | None
    timeout_seconds: float
    verify_tls: bool


# ------------------------------------------------------------ response projection


_INDEX = re.compile(r"^(?P<name>[^\[\]]*)\[(?P<index>\*|-?\d+)\]$")


def project_response(
    payload: Any, path: str | None, *, max_items: int | None = None
) -> Any:
    """Pick a subtree out of a JSON response.

    Supports the JMESPath subset a tool projection actually needs: dotted keys,
    ``[n]`` indexing (negative allowed) and ``[*]`` mapping over a list. A path that
    does not resolve yields ``None`` rather than raising, because a shape change
    upstream must degrade the tool, not the turn.

    Args:
        payload: The parsed JSON body.
        path: Dotted projection path, e.g. ``"data.items[*].name"``. None returns the
            payload unchanged.
        max_items: Cap on the length of a list result. None leaves it uncapped.

    Returns:
        The projected value, truncated to ``max_items`` when it is a list.
    """
    current: Any = payload
    for raw_segment in (path or "").split("."):
        segment = raw_segment.strip()
        if not segment:
            continue
        match = _INDEX.match(segment)
        name = match.group("name") if match else segment
        if name:
            if not isinstance(current, Mapping) or name not in current:
                return None
            current = current[name]
        if match is None:
            continue
        index = match.group("index")
        if index == "*":
            if not isinstance(current, list):
                return None
            continue
        if not isinstance(current, list):
            return None
        position = int(index)
        if not -len(current) <= position < len(current):
            return None
        current = current[position]
    if isinstance(current, list) and max_items is not None and len(current) > max_items:
        return current[:max_items]
    return current


# ------------------------------------------------------------------- credentials


class SecretResolver:
    """Resolve ``*_secret_ref`` names to values, with a short-lived cache.

    Azure Key Vault is used when ``azure_key_vault_url`` is configured and the SDK is
    installed. Otherwise the value is read from the environment, which is what local
    development and CI use. A spec never contains a secret value.
    """

    def __init__(self, *, config: ToolConfig) -> None:
        """Initialise the resolver.

        Args:
            config: Resolved tool-layer configuration.
        """
        self._config = config
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = anyio.Lock()
        self._client: Any | None = None
        self._credential: Any | None = None

    def _env_names(self, ref: str) -> tuple[str, ...]:
        """Environment variable names checked for a secret reference.

        Args:
            ref: The Key Vault secret name.

        Returns:
            Candidate environment variable names, most specific first.
        """
        normalised = ref.replace("-", "_").replace(".", "_").upper()
        return (f"{self._config.secret_env_prefix}{normalised}", normalised, ref)

    async def _from_key_vault(self, ref: str) -> str | None:
        """Read a secret from Azure Key Vault.

        Args:
            ref: The secret name.

        Returns:
            The secret value, or None when Key Vault is not configured or the SDK is
            absent.
        """
        if not self._config.key_vault_url:
            return None
        if self._client is None:
            try:
                from azure.identity.aio import (
                    DefaultAzureCredential,
                )
                from azure.keyvault.secrets.aio import (
                    SecretClient,
                )
            except ImportError:
                _log.warning("tool_key_vault_sdk_missing", ref=ref)
                return None
            self._credential = DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=self._config.key_vault_url, credential=self._credential
            )
        try:
            secret = await self._client.get_secret(ref)
        except Exception as exc:
            _log.warning(
                "tool_key_vault_read_failed", ref=ref, error=type(exc).__name__
            )
            return None
        return str(secret.value) if secret.value is not None else None

    async def resolve(self, ref: str | None) -> str | None:
        """Resolve one secret reference.

        Args:
            ref: The Key Vault secret name, or None.

        Returns:
            The secret value, or None when it cannot be resolved.
        """
        if not ref:
            return None
        now = time.monotonic()
        async with self._lock:
            cached = self._cache.get(ref)
            if cached is not None and cached[1] > now:
                return cached[0]
        value = await self._from_key_vault(ref)
        if value is None:
            import os

            for name in self._env_names(ref):
                candidate = os.environ.get(name)
                if candidate:
                    value = candidate
                    break
        if value is None:
            return None
        async with self._lock:
            self._cache[ref] = (value, now + self._config.secret_cache_seconds)
        return value

    async def aclose(self) -> None:
        """Close the Key Vault client and credential, if any."""
        for closeable in (self._client, self._credential):
            if closeable is None:
                continue
            close = getattr(closeable, "close", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                _log.debug("tool_secret_close_failed", error=type(exc).__name__)
        self._client = None
        self._credential = None


class TokenProvider:
    """OAuth2 client-credentials tokens with per-scope caching.

    Used by the ``entra_obo`` and ``managed_identity`` schemes. Managed identity goes
    to the IMDS endpoint; everything else performs a standard client-credentials
    grant against the configured token endpoint. Tokens are cached until
    ``tool_oauth_expiry_skew_seconds`` before expiry.
    """

    def __init__(self, *, config: ToolConfig, secrets_resolver: SecretResolver) -> None:
        """Initialise the provider.

        Args:
            config: Resolved tool-layer configuration.
            secrets_resolver: Resolver for the client secret.
        """
        self._config = config
        self._secrets = secrets_resolver
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = anyio.Lock()

    def _token_url(self) -> str:
        """Build the client-credentials token endpoint URL.

        Returns:
            The rendered token endpoint.

        Raises:
            ToolExecutionError: When no Entra tenant is configured.
        """
        if not self._config.entra_tenant_id:
            msg = "oauth2 client-credentials needs entra_tenant_id"
            raise ToolExecutionError(msg, tool_name="<oauth>", kind="rest")
        return self._config.oauth_token_url_template.format(
            authority=self._config.authority_host.rstrip("/"),
            tenant=self._config.entra_tenant_id,
        )

    async def _fetch_client_credentials(
        self, client: httpx.AsyncClient, scope: str
    ) -> tuple[str, float]:
        """Perform the client-credentials grant.

        Args:
            client: HTTP client to use.
            scope: Requested scope.

        Returns:
            An ``(access_token, expires_in_seconds)`` pair.

        Raises:
            ToolExecutionError: When the client id or secret is missing, or the token
                endpoint rejects the request.
        """
        client_id = self._config.entra_client_id
        secret = await self._secrets.resolve(self._config.oauth_client_secret_ref)
        if not client_id or not secret:
            msg = (
                "oauth2 client-credentials needs azure_client_id/entra_client_id and "
                "tool_oauth_client_secret_ref"
            )
            raise ToolExecutionError(msg, tool_name="<oauth>", kind="rest")
        response = await client.post(
            self._token_url(),
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": secret,
                "scope": scope,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != httpx.codes.OK:
            msg = f"oauth2 token endpoint returned {response.status_code}"
            raise ToolExecutionError(
                msg, tool_name="<oauth>", kind="rest", status=response.status_code
            )
        body = response.json()
        return str(body["access_token"]), float(body.get("expires_in", 3600))

    async def _fetch_managed_identity(
        self, client: httpx.AsyncClient, scope: str
    ) -> tuple[str, float]:
        """Fetch a token from the Azure instance metadata service.

        Args:
            client: HTTP client to use.
            scope: Requested resource/scope.

        Returns:
            An ``(access_token, expires_in_seconds)`` pair.

        Raises:
            ToolExecutionError: When IMDS rejects the request.
        """
        params = {
            "api-version": self._config.managed_identity_api_version,
            "resource": scope,
        }
        if self._config.entra_client_id:
            params["client_id"] = self._config.entra_client_id
        response = await client.get(
            self._config.managed_identity_endpoint,
            params=params,
            headers={"Metadata": "true"},
        )
        if response.status_code != httpx.codes.OK:
            msg = f"managed identity endpoint returned {response.status_code}"
            raise ToolExecutionError(
                msg, tool_name="<oauth>", kind="rest", status=response.status_code
            )
        body = response.json()
        expires_in = float(body.get("expires_in", 3600))
        return str(body["access_token"]), expires_in

    async def token(
        self, client: httpx.AsyncClient, *, auth: ToolAuth, scope: str
    ) -> str:
        """Return a cached or freshly minted access token.

        Args:
            client: HTTP client used for the token request.
            auth: The scheme asking for a token.
            scope: Requested scope or resource.

        Returns:
            The bearer token value.
        """
        key = f"{auth.value}:{scope}"
        now = time.monotonic()
        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached[1] > now:
                return cached[0]
        if auth is ToolAuth.MANAGED_IDENTITY:
            value, expires_in = await self._fetch_managed_identity(client, scope)
        else:
            value, expires_in = await self._fetch_client_credentials(client, scope)
        expiry = now + max(0.0, expires_in - self._config.oauth_expiry_skew_seconds)
        async with self._lock:
            self._cache[key] = (value, expiry)
        return value

    def reset(self) -> None:
        """Drop every cached token. Test helper."""
        self._cache.clear()


# --------------------------------------------------------------- circuit breaker


class CircuitState(StrEnum):
    """States of a per-tool circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class _Circuit:
    """Mutable breaker state for one tool."""

    failures: int = 0
    opened_at: float = 0.0
    state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """Per-tool circuit breaker.

    ``failure_threshold`` consecutive failures open the circuit; after
    ``reset_seconds`` it half-opens and lets one probe through. A success closes it.
    """

    def __init__(self, *, failure_threshold: int, reset_seconds: float) -> None:
        """Initialise the breaker.

        Args:
            failure_threshold: Consecutive failures that open a circuit.
            reset_seconds: Cool-down before a circuit half-opens.
        """
        self._threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._circuits: dict[str, _Circuit] = {}

    def state(self, name: str) -> CircuitState:
        """Current state of one tool's circuit.

        Args:
            name: Tool name.

        Returns:
            The circuit state, transitioning ``open`` to ``half_open`` once the
            cool-down has elapsed.
        """
        circuit = self._circuits.get(name)
        if circuit is None:
            return CircuitState.CLOSED
        if (
            circuit.state is CircuitState.OPEN
            and time.monotonic() - circuit.opened_at >= self._reset_seconds
        ):
            circuit.state = CircuitState.HALF_OPEN
        return circuit.state

    def allows(self, name: str) -> bool:
        """Whether a call to this tool may proceed.

        Args:
            name: Tool name.

        Returns:
            False only while the circuit is fully open.
        """
        return self.state(name) is not CircuitState.OPEN

    def record_success(self, name: str) -> None:
        """Close the circuit after a successful call.

        Args:
            name: Tool name.
        """
        self._circuits.pop(name, None)

    def record_failure(self, name: str, *, threshold: int | None = None) -> None:
        """Count a failure and open the circuit when the threshold is reached.

        Args:
            name: Tool name.
            threshold: Per-tool override of the default failure threshold.
        """
        circuit = self._circuits.setdefault(name, _Circuit())
        circuit.failures += 1
        if circuit.failures >= (threshold or self._threshold):
            circuit.state = CircuitState.OPEN
            circuit.opened_at = time.monotonic()
            _log.warning("tool_circuit_open", tool=name, failures=circuit.failures)

    def reset(self) -> None:
        """Clear every circuit. Test helper."""
        self._circuits.clear()


# ------------------------------------------------------------------ the executor


@dataclass(slots=True)
class _CacheEntry:
    """One cached tool result."""

    result: ToolResult
    expires_at: float


class RestExecutor:
    """Executes declarative REST tools.

    One executor is shared per process: it owns the HTTP connection pool, the
    circuit breakers, the secret cache and the OAuth token cache.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        config: ToolConfig | None = None,
        client: httpx.AsyncClient | None = None,
        detector: PIIDetector | None = None,
    ) -> None:
        """Initialise the executor.

        Args:
            settings: Platform settings; defaults to :func:`get_settings`.
            config: Resolved tool config; derived from settings when omitted.
            client: Injected HTTP client. Tests pass a transport-mounted client.
            detector: PII detector; defaults to the cached one.
        """
        self.settings = settings or get_settings()
        self.config = config or tool_config(self.settings)
        self._client = client
        self._owns_client = client is None
        self._detector = detector
        self.breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            reset_seconds=self.config.circuit_reset_seconds,
        )
        self._cache: dict[str, _CacheEntry] = {}
        self.secrets = SecretResolver(config=self.config)
        self.tokens = TokenProvider(config=self.config, secrets_resolver=self.secrets)
        self._rng = secrets.SystemRandom()

    @property
    def detector(self) -> PIIDetector:
        """PII detector used to scrub responses.

        Returns:
            The detector, resolved lazily so importing this module is cheap.
        """
        if self._detector is None:
            self._detector = get_pii_detector(self.settings)
        return self._detector

    def _http(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it on first use.

        Returns:
            The client.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                follow_redirects=self.config.follow_redirects,
                headers={"user-agent": self.config.user_agent},
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client and any credential clients this executor owns."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        await self.secrets.aclose()

    # ------------------------------------------------------------- preparation

    def _check_url(self, url: str, *, tool_name: str) -> None:
        """Reject a URL scheme the deployment does not permit.

        Args:
            url: The rendered URL.
            tool_name: Tool name, for the error message.

        Raises:
            ToolExecutionError: When the URL is not absolute, or is plain HTTP while
                ``tool_allow_insecure_http`` is false.
        """
        if url.startswith("https://"):
            return
        if url.startswith("http://") and self.config.allow_insecure_http:
            _log.warning("tool_insecure_http", tool=tool_name)
            return
        msg = f"tool {tool_name!r} resolved to a non-https URL"
        raise ToolExecutionError(msg, tool_name=tool_name, kind="rest")

    async def _auth_headers(
        self, spec: RestToolSpec, *, tool_name: str
    ) -> dict[str, str]:
        """Resolve the authentication headers for one call.

        Args:
            spec: The REST spec.
            tool_name: Tool name, for error messages.

        Returns:
            Headers to merge into the request.

        Raises:
            ToolExecutionError: When a required credential cannot be resolved.
        """
        if spec.auth is ToolAuth.NONE:
            return {}
        if spec.auth in {ToolAuth.ENTRA_OBO, ToolAuth.MANAGED_IDENTITY}:
            token = await self.tokens.token(
                self._http(), auth=spec.auth, scope=spec.auth_scope or ""
            )
            return {spec.auth_header_name: f"Bearer {token}"}
        value = await self.secrets.resolve(spec.auth_secret_ref)
        if not value:
            msg = f"secret {spec.auth_secret_ref!r} could not be resolved"
            raise ToolExecutionError(msg, tool_name=tool_name, kind="rest")
        if spec.auth is ToolAuth.BEARER:
            return {spec.auth_header_name: f"Bearer {value}"}
        if spec.auth is ToolAuth.BASIC:
            encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
            return {spec.auth_header_name: f"Basic {encoded}"}
        return {spec.auth_header_name: value}

    async def prepare(
        self, tool: RegisteredTool, arguments: Mapping[str, Any]
    ) -> PreparedRequest:
        """Validate arguments and render the HTTP request for a tool.

        Args:
            tool: The registered REST tool.
            arguments: Raw model arguments.

        Returns:
            The prepared request.

        Raises:
            ToolExecutionError: When the tool is not a REST tool, an argument is
                invalid, or the resolved URL is not permitted.
        """
        spec = tool.spec.rest
        if tool.kind is not ToolKind.REST or spec is None:
            msg = f"tool {tool.name!r} is not a REST tool"
            raise ToolExecutionError(msg, tool_name=tool.name, kind=tool.kind.value)
        validated = validate_arguments(
            tool.spec.input_schema, arguments, tool_name=tool.name
        )
        url = render_template(spec.url_template, validated, quote_values=True)
        self._check_url(url, tool_name=tool.name)
        params = {
            key: render_template(value, validated, strict=False)
            for key, value in spec.query_template.items()
        }
        body = (
            _render_body(spec.body_template, validated)
            if spec.body_template is not None
            else None
        )
        headers = {str(k): str(v) for k, v in spec.headers.items()}
        headers.update(await self._auth_headers(spec, tool_name=tool.name))
        headers.setdefault("accept", "application/json")
        return PreparedRequest(
            method=spec.method,
            url=url,
            headers=headers,
            params={k: v for k, v in params.items() if v != ""},
            json_body=body,
            timeout_seconds=tool.timeout_seconds,
            verify_tls=spec.verify_tls,
        )

    # ---------------------------------------------------------------- transport

    async def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read a response body up to ``tool_max_response_bytes``.

        Args:
            response: A streaming response.

        Returns:
            A ``(body, truncated)`` pair.
        """
        cap = self.config.max_response_bytes
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= cap:
                return b"".join(chunks)[:cap], True
        return b"".join(chunks), False

    async def _send_once(
        self, request: PreparedRequest
    ) -> tuple[int, bytes, str, bool]:
        """Issue one HTTP attempt.

        Args:
            request: The prepared request.

        Returns:
            A ``(status_code, body, content_type, truncated)`` tuple.
        """
        client = self._http()
        async with client.stream(
            request.method,
            request.url,
            headers=request.headers,
            params=request.params or None,
            json=request.json_body,
            timeout=httpx.Timeout(request.timeout_seconds),
        ) as response:
            body, truncated = await self._read_capped(response)
            content_type = response.headers.get("content-type", "")
            return response.status_code, body, content_type, truncated

    async def _sleep_backoff(self, attempt: int) -> None:
        """Sleep for a jittered exponential backoff interval.

        Args:
            attempt: Zero-based attempt index that just failed.
        """
        base = self.config.retry_backoff_seconds * (2**attempt)
        delay = min(base, self.config.retry_max_backoff_seconds)
        await anyio.sleep(self._rng.uniform(delay / 2, delay))

    # ------------------------------------------------------------------ results

    def _render_content(
        self, body: bytes, content_type: str, spec: RestToolSpec
    ) -> tuple[str, dict[str, Any] | None]:
        """Decode, project and serialise a response body for the model.

        Args:
            body: Raw response bytes.
            content_type: Upstream ``content-type`` header.
            spec: The REST spec, carrying ``response_json_path``.

        Returns:
            A ``(text, structured)`` pair. ``structured`` is None for non-JSON bodies.
        """
        text = body.decode("utf-8", errors="replace")
        if not _JSON_MEDIA.search(content_type):
            return text, None
        try:
            parsed = json.loads(text) if text.strip() else None
        except ValueError:
            return text, None
        projected = project_response(
            parsed, spec.response_json_path, max_items=self.config.max_projected_items
        )
        rendered = json.dumps(projected, ensure_ascii=False, default=str)
        structured = projected if isinstance(projected, dict) else {"result": projected}
        return rendered, structured

    def _scrub(self, tool: RegisteredTool, text: str) -> str:
        """Redact PII in a tool response before it enters the prompt.

        Args:
            tool: The tool that produced the response.
            text: The rendered response text.

        Returns:
            The redacted text, or the input when scanning is disabled.
        """
        enabled = tool.policy.response_pii_scan
        if enabled is None:
            enabled = self.config.response_pii_scan
        if not enabled or not text:
            return text
        redacted, report = self.detector.scan_and_redact(text)
        if report.has_pii:
            _log.info(
                "tool_response_pii_redacted",
                tool=tool.name,
                entity_types=sorted(report.entity_types),
                findings=len(report.findings),
            )
        return redacted

    def _cache_key(self, tool: RegisteredTool, arguments: Mapping[str, Any]) -> str:
        """Build the cache key for a cacheable tool call.

        Args:
            tool: The tool.
            arguments: Validated arguments.

        Returns:
            A stable string key.
        """
        payload = json.dumps(arguments, sort_keys=True, default=str)
        return f"{tool.name}:{payload}"

    def _cached(self, key: str) -> ToolResult | None:
        """Look a cacheable result up.

        Args:
            key: Cache key.

        Returns:
            The cached result, or None when absent or expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._cache.pop(key, None)
            return None
        return entry.result

    def _store(self, key: str, result: ToolResult, ttl_seconds: int) -> None:
        """Store a cacheable result, evicting the oldest entry when full.

        Args:
            key: Cache key.
            result: The result to cache.
            ttl_seconds: Entry lifetime.
        """
        if len(self._cache) >= self.config.result_cache_max_entries:
            oldest = min(self._cache, key=lambda k: self._cache[k].expires_at)
            self._cache.pop(oldest, None)
        self._cache[key] = _CacheEntry(
            result=result, expires_at=time.monotonic() + ttl_seconds
        )

    async def execute(
        self,
        tool: RegisteredTool,
        *,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Run one REST tool call end to end.

        Never raises for an upstream failure: a bad status, a timeout, an open
        circuit or an invalid argument all come back as ``is_error`` results so the
        model can correct itself inside the loop.

        Args:
            tool: The registered REST tool.
            tool_call_id: The model's ``tool_use`` block id.
            arguments: Raw model arguments.

        Returns:
            The tool result, already truncated to ``max_result_chars``.
        """
        started = time.perf_counter()

        def elapsed() -> float:
            """Milliseconds since the call started.

            Returns:
                The elapsed time in milliseconds.
            """
            return (time.perf_counter() - started) * _MS

        def fail(message: str, *, status: int | None = None) -> ToolResult:
            """Build a failure result.

            Args:
                message: Redacted failure description.
                status: Upstream HTTP status, when known.

            Returns:
                The error result.
            """
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                kind=ToolKind.REST,
                message=message,
                http_status=status,
                latency_ms=elapsed(),
            )

        if not self.breaker.allows(tool.name):
            return fail("upstream is unavailable (circuit open); do not retry it")

        try:
            request = await self.prepare(tool, arguments)
        except ToolExecutionError as exc:
            return fail(exc.message)

        spec = tool.spec.rest
        assert spec is not None  # noqa: S101 - prepare() already proved this

        cache_key = ""
        if tool.spec.cacheable:
            cache_key = self._cache_key(tool, {"url": request.url, **request.params})
            cached = self._cached(cache_key)
            if cached is not None:
                return cached.model_copy(update={"tool_call_id": tool_call_id})

        attempts = (tool.policy.retry_attempts or self.config.retry_attempts) + 1
        last_error = "upstream did not respond"
        last_status: int | None = None
        for attempt in range(attempts):
            try:
                status, body, content_type, truncated = await self._send_once(request)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__} calling the upstream"
                self.breaker.record_failure(
                    tool.name, threshold=tool.policy.circuit_failure_threshold
                )
                if attempt + 1 < attempts:
                    await self._sleep_backoff(attempt)
                    continue
                return fail(last_error)

            last_status = status
            if status in spec.success_status_codes:
                self.breaker.record_success(tool.name)
                text, structured = self._render_content(body, content_type, spec)
                result = ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool.name,
                    kind=ToolKind.REST,
                    content=self._scrub(tool, text),
                    structured=structured,
                    http_status=status,
                    latency_ms=elapsed(),
                    truncated=truncated,
                ).truncate(tool.max_result_chars)
                if tool.spec.cacheable:
                    self._store(cache_key, result, tool.spec.cache_ttl_seconds)
                return result

            last_error = f"upstream returned HTTP {status}"
            retryable = status == httpx.codes.TOO_MANY_REQUESTS or status >= 500
            self.breaker.record_failure(
                tool.name, threshold=tool.policy.circuit_failure_threshold
            )
            if retryable and attempt + 1 < attempts:
                await self._sleep_backoff(attempt)
                continue
            break

        return fail(last_error, status=last_status)


_executor: RestExecutor | None = None
_executor_key: tuple[Any, ...] | None = None


def get_rest_executor(settings: Settings | None = None) -> RestExecutor:
    """Return the process-wide REST executor.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The shared :class:`RestExecutor`.
    """
    global _executor, _executor_key
    cfg = settings or get_settings()
    key = (cfg.tool_timeout_seconds, cfg.tool_max_result_chars, cfg.env)
    if _executor is None or _executor_key != key:
        _executor = RestExecutor(settings=cfg)
        _executor_key = key
    return _executor


async def reset_rest_executor() -> None:
    """Close and drop the shared executor. Shutdown hook and test helper."""
    global _executor, _executor_key
    if _executor is not None:
        await _executor.aclose()
    _executor = None
    _executor_key = None
