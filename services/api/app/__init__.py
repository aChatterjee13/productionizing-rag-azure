"""The RAG HTTP service (``services/api``), importable as ``app``.

Layout:

* :mod:`app.main` -- the ASGI application, lifespan and OpenAPI metadata.
* :mod:`app.deps` -- request-scoped dependencies: principal, database session,
  rate limiting, role gates.
* :mod:`app.middleware` -- request id, structlog binding, CORS, GZip and the
  RFC 7807 exception handlers.
* :mod:`app.auth` -- Entra ID token validation and claim-to-:class:`Principal`
  mapping.
* :mod:`app.routers` -- the HTTP surface from `docs/CONTRACTS.md`.
* :mod:`app.rag` -- the ordered pipeline; :mod:`app.rag.orchestrator` wires it.

Importing ``app`` itself must stay cheap: it pulls neither FastAPI nor
``qdrant-client``. Only :data:`API_SETTING_DEFAULTS` and :func:`api_setting` live
here, because both the middleware and the dependencies need them and neither may
import the other.

**Tunables.** Same pattern as :func:`app.rag.rag_setting`,
:func:`app.rag.guardrails.guardrail_setting` and
:func:`app.rag.memory.optional_setting`: every knob is read through
:func:`api_setting`, which prefers the real :class:`ragcore.settings.Settings`
field and falls back to the documented default until that field is declared. Keys
are real ``RAG_``-prefixed setting names, so ``RAG_API_GZIP_MIN_BYTES=2048`` starts
working the moment the field exists. Nothing in this package writes a threshold,
limit, header name or URL at a call site. See "Addendum P" of `docs/CONTRACTS.md`.
"""

from __future__ import annotations

from typing import Any

from ragcore.settings import Settings

__all__ = [
    "API_SETTING_DEFAULTS",
    "SERVICE_VERSION",
    "api_setting",
]

#: Version reported by ``GET /health`` and in the OpenAPI document. Mirrors
#: ``services/api/pyproject.toml``'s ``project.version``.
SERVICE_VERSION = "0.1.0"


#: Tunables the HTTP layer needs that ``ragcore.settings`` does not declare yet,
#: with the value that applies until it does.
API_SETTING_DEFAULTS: dict[str, Any] = {
    # ------------------------------------------------------------- correlation
    # Request-correlation header. Echoed on every response and bound into
    # structlog, so a log line, a trace and a browser network entry can be joined.
    "api_request_id_header": "x-request-id",
    # Response header carrying the Langfuse trace id, when a trace is open.
    "api_trace_id_header": "x-trace-id",
    # ------------------------------------------------------------------ CORS
    # Browsers only send credentials cross-origin when the server opts in. MSAL
    # tokens travel in an Authorization header rather than a cookie, but the web
    # app also reads the request-id and trace-id headers, which must be exposed.
    "api_cors_allow_credentials": True,
    "api_cors_allow_headers": ["*"],
    "api_cors_expose_headers": ["x-request-id", "x-trace-id", "retry-after"],
    "api_cors_max_age_seconds": 600,
    # ------------------------------------------------------------------ GZip
    # Below this, compression costs more than it saves. SSE responses are never
    # compressed: GZipMiddleware buffers, which would defeat token streaming.
    "api_gzip_min_bytes": 1024,
    # --------------------------------------------------------- rate limiting
    "api_rate_limit_enabled": True,
    # Bucket depth. A chat turn is one request but a page load is several, so the
    # burst allowance is deliberately larger than one.
    "api_rate_limit_burst": 20,
    # Rate limiting is per (tenant, user). The tenant prefix keeps one tenant's
    # traffic from evicting another's counters in a shared Redis.
    "api_rate_limit_prefix": "rag:ratelimit:",
    # ------------------------------------------------------------------- SSE
    # `retry:` field sent once at the start of a stream, in milliseconds.
    "api_sse_retry_ms": 2000,
    # Comment heartbeat text. A bare ":" comment keeps proxies from timing the
    # connection out without producing a client-visible event.
    "api_sse_heartbeat_comment": "heartbeat",
    # ---------------------------------------------------------------- errors
    # Base URI for RFC 7807 `type` members. A stable, dereferenceable-looking URI
    # is what makes a problem type machine-identifiable.
    "api_problem_type_base": "https://productionizing-rag.dev/problems",
    # ------------------------------------------------------------------ docs
    # Interactive docs are served unless this is False. `main` additionally
    # refuses them in production unless explicitly re-enabled.
    "api_docs_enabled": True,
    # ---------------------------------------------------------------- listing
    # Default and maximum `limit` on list endpoints.
    "api_default_page_size": 50,
    "api_max_page_size": 200,
    # -------------------------------------------------------------- lifespan
    # Load the embedding and rerank models during startup rather than on the
    # first turn. Disable for a fast-booting CI container.
    "api_warm_models": True,
    # Create the Qdrant collections during startup. Idempotent; disable when a
    # separate bootstrap job owns the schema.
    "api_ensure_collections": True,
    # Seconds the readiness probe waits on each dependency.
    "api_readiness_timeout_seconds": 5.0,
}


def api_setting(settings: Settings, name: str) -> Any:
    """Read an HTTP-layer tunable, preferring the settings field over the default.

    Args:
        settings: Active settings object.
        name: A key of :data:`API_SETTING_DEFAULTS`, e.g. ``"api_gzip_min_bytes"``.

    Returns:
        ``getattr(settings, name)`` when the field exists and is not None,
        otherwise the documented default.

    Raises:
        KeyError: If ``name`` is not a declared tunable. That is a typo at the call
            site, not a configuration problem, so it fails loudly.
    """
    if name not in API_SETTING_DEFAULTS:
        msg = f"unknown api tunable {name!r}"
        raise KeyError(msg)
    value = getattr(settings, name, None)
    return API_SETTING_DEFAULTS[name] if value is None else value
