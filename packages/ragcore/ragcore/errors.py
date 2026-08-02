"""Exception hierarchy shared by every service, with HTTP status mapping.

The API layer converts any :class:`RagError` into a JSON problem response using
:meth:`RagError.to_dict` and :attr:`RagError.status_code`, so handlers never need to
map exception types to status codes by hand. Anything that is not a ``RagError``
is a genuine bug and becomes a 500.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AuthError",
    "ConfigError",
    "GuardrailBlocked",
    "RagError",
    "RetrievalError",
    "TenantMismatchError",
    "ToolExecutionError",
    "http_status_for",
]


class RagError(Exception):
    """Base class for every expected, reportable failure in the platform.

    Attributes:
        status_code: HTTP status the API returns for this failure.
        code: Stable machine-readable error code for clients and dashboards.
    """

    status_code: int = 500
    code: str = "rag_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description. Must not contain raw user content
                that has not passed PII redaction.
            code: Override for the class-level machine-readable code.
            status_code: Override for the class-level HTTP status.
            detail: Extra structured context (ids, counts, stage names). Never put
                secrets or unredacted user text here — it is logged and returned.
        """
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.detail: dict[str, Any] = detail or {}

    def to_dict(self) -> dict[str, Any]:
        """Render the error as a JSON-serialisable problem body.

        Returns:
            A mapping with ``error``, ``code``, ``message`` and ``detail`` keys.
        """
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        """Return the human-readable message.

        Returns:
            The message passed at construction time.
        """
        return self.message


class AuthError(RagError):
    """Authentication or authorisation failure.

    Used for a missing, malformed, expired or untrusted token, and for a principal
    that lacks a required app role. Defaults to 401; pass ``status_code=403`` when
    the caller is authenticated but not permitted.
    """

    status_code = 401
    code = "auth_error"


class TenantMismatchError(RagError):
    """A request tried to reach data belonging to a different tenant.

    This is a security-boundary violation, not a user error: it means a tenant id
    derived from the token disagreed with a tenant id on a resource or filter.
    Always audit-log it.
    """

    status_code = 403
    code = "tenant_mismatch"

    def __init__(
        self,
        message: str = "tenant mismatch",
        *,
        expected: str | None = None,
        actual: str | None = None,
        resource: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the mismatch error.

        Args:
            message: Human-readable description.
            expected: Tenant id the principal is scoped to.
            actual: Tenant id found on the resource or filter.
            resource: Optional resource identifier (table, collection, id).
            detail: Additional structured context.
        """
        merged: dict[str, Any] = dict(detail or {})
        if expected is not None:
            merged["expected_tenant_id"] = expected
        if actual is not None:
            merged["actual_tenant_id"] = actual
        if resource is not None:
            merged["resource"] = resource
        super().__init__(message, detail=merged)


class RetrievalError(RagError):
    """Retrieval failed: Qdrant, the embedder or the reranker was unavailable.

    Distinct from an empty result set, which is a normal out-of-domain outcome
    handled by the OOD gate rather than an error.
    """

    status_code = 503
    code = "retrieval_error"


class GuardrailBlocked(RagError):
    """A guardrail short-circuited the pipeline.

    Raised when a stage decides the request or response must not proceed: prompt
    injection over the block threshold, an oversized turn, a classification
    violation, or PII egress when ``pii_block_on_egress`` is enabled.

    Attributes:
        stage: Pipeline stage that blocked, one of ``input``/``retrieval``/``output``.
        kind: Guardrail kind, e.g. ``pii``, ``injection``, ``classification``.
    """

    status_code = 422
    code = "guardrail_blocked"

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        kind: str,
        entities: list[str] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the guardrail failure.

        Args:
            message: Redacted description of why the request was blocked.
            stage: Pipeline stage: ``input``, ``retrieval`` or ``output``.
            kind: Guardrail kind: ``pii``, ``injection``, ``ood``, ``contradiction``,
                ``classification`` or ``groundedness``.
            entities: Entity type names involved (never entity values).
            detail: Additional structured context.
        """
        merged: dict[str, Any] = dict(detail or {})
        merged["stage"] = stage
        merged["kind"] = kind
        if entities:
            merged["entities"] = entities
        super().__init__(message, detail=merged)
        self.stage = stage
        self.kind = kind


class ToolExecutionError(RagError):
    """A REST or MCP tool invocation failed.

    The tool loop catches this, records the failed invocation, and returns an
    ``is_error`` tool result to the model rather than aborting the turn.

    Attributes:
        tool_name: Name of the tool that failed.
        kind: Tool kind, one of ``rest``/``mcp``/``retrieval``.
    """

    status_code = 502
    code = "tool_execution_error"

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        kind: str = "rest",
        status: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the tool failure.

        Args:
            message: Redacted description of the failure.
            tool_name: Name of the failing tool.
            kind: Tool kind: ``rest``, ``mcp`` or ``retrieval``.
            status: Upstream HTTP status, when the tool was an HTTP call.
            detail: Additional structured context.
        """
        merged: dict[str, Any] = dict(detail or {})
        merged["tool_name"] = tool_name
        merged["tool_kind"] = kind
        if status is not None:
            merged["upstream_status"] = status
        super().__init__(message, detail=merged)
        self.tool_name = tool_name
        self.kind = kind


class ConfigError(RagError):
    """Configuration is missing, contradictory or unsafe.

    Raised at startup (a required Entra setting absent, a source config that names
    an unknown connector) so a misconfigured process fails fast instead of serving
    wrong answers.
    """

    status_code = 500
    code = "config_error"


def http_status_for(exc: BaseException) -> int:
    """Map any exception onto an HTTP status code.

    Args:
        exc: The exception to classify.

    Returns:
        :attr:`RagError.status_code` for platform errors, 504 for timeouts,
        499 for a client-cancelled request, and 500 for anything unexpected.
    """
    if isinstance(exc, RagError):
        return exc.status_code
    if isinstance(exc, TimeoutError):
        return 504
    if isinstance(exc, PermissionError):
        return 403
    return 500
