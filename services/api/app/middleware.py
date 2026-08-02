"""Cross-cutting HTTP concerns: correlation, CORS, compression and errors.

The request-context middleware is written as **pure ASGI** rather than as a
``BaseHTTPMiddleware`` subclass on purpose: ``BaseHTTPMiddleware`` pumps the
response through an anyio memory stream, which buffers a ``StreamingResponse``
and would turn the chat SSE stream into one delivery at the end of the turn.

Error handling maps the :mod:`ragcore.errors` hierarchy onto RFC 7807
``application/problem+json``. Every failure the API returns — validation, auth,
guardrail, tool, retrieval, unexpected — has that one shape, so a client needs one
error parser and an operator can join a problem body to a log line through
``request_id``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import api_setting
from app.schemas.responses import ProblemDetail
from ragcore.errors import RagError, http_status_for
from ragcore.logging import bind_request_context, clear_request_context, get_logger
from ragcore.observability import get_current_trace_id, observe_http_request
from ragcore.settings import Settings

__all__ = [
    "PROBLEM_CONTENT_TYPE",
    "RequestContextMiddleware",
    "install_exception_handlers",
    "install_middleware",
    "problem_response",
]

_log = get_logger(__name__)

#: Media type every error response carries.
PROBLEM_CONTENT_TYPE = "application/problem+json"


class RequestContextMiddleware:
    """Assigns a request id, binds log context and records HTTP metrics."""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        """Initialise the middleware.

        Args:
            app: The wrapped ASGI application.
            settings: Active settings supplying the header names.
        """
        self.app = app
        self._settings = settings
        self._request_header = str(api_setting(settings, "api_request_id_header"))
        self._trace_header = str(api_setting(settings, "api_trace_id_header"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event stream.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_id = headers.get(self._request_header) or uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id

        bind_request_context(request_id=request_id, route=scope.get("path"))
        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            """Stamp correlation headers on the response start event.

            Args:
                message: The outgoing ASGI message.
            """
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                raw = list(message.get("headers") or [])
                raw.append(
                    (
                        self._request_header.encode("latin-1"),
                        request_id.encode("latin-1"),
                    )
                )
                trace_id = get_current_trace_id()
                if trace_id:
                    raw.append(
                        (
                            self._trace_header.encode("latin-1"),
                            trace_id.encode("latin-1"),
                        )
                    )
                message = {**message, "headers": raw}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            template = getattr(route, "path", None) or scope.get("path", "unknown")
            observe_http_request(
                route=str(template),
                method=str(scope.get("method", "GET")),
                status_code=status_holder["status"],
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            clear_request_context()


def install_middleware(app: FastAPI, settings: Settings) -> None:
    """Attach CORS, compression and request context to the application.

    Starlette applies middleware in reverse registration order, so the request
    context is added **last** and therefore runs **first** — every other layer,
    including a CORS pre-flight rejection, is already correlated and measured.

    Args:
        app: The FastAPI application.
        settings: Active settings.
    """
    # GZip is registered first (so it runs innermost). Its `minimum_size` keeps it
    # off small JSON bodies, and Starlette's implementation leaves a streaming
    # response with no content-length alone, which is what keeps SSE unbuffered.
    app.add_middleware(
        GZipMiddleware, minimum_size=int(api_setting(settings, "api_gzip_min_bytes"))
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api_cors_origins),
        allow_credentials=bool(api_setting(settings, "api_cors_allow_credentials")),
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=list(api_setting(settings, "api_cors_allow_headers")),
        expose_headers=list(api_setting(settings, "api_cors_expose_headers")),
        max_age=int(api_setting(settings, "api_cors_max_age_seconds")),
    )
    app.add_middleware(RequestContextMiddleware, settings=settings)


# ------------------------------------------------------------------- problems


def problem_response(
    request: Request,
    *,
    status: int,
    title: str,
    detail: str,
    code: str,
    settings: Settings,
    context: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render an RFC 7807 problem document.

    Args:
        request: The request that failed, for ``instance`` and the request id.
        status: HTTP status code.
        title: Short summary of the problem type.
        detail: Explanation of this occurrence. Must already be redacted.
        code: Stable machine-readable error code.
        settings: Active settings, supplying the problem type base URI.
        context: Structured context from the error.
        errors: Field-level validation problems.
        headers: Extra response headers, e.g. ``Retry-After``.

    Returns:
        A ``application/problem+json`` response.
    """
    base = str(api_setting(settings, "api_problem_type_base")).rstrip("/")
    body = ProblemDetail(
        type=f"{base}/{code}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
        code=code,
        request_id=getattr(request.state, "request_id", None),
        trace_id=get_current_trace_id(),
        errors=errors or [],
        context=context or {},
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(mode="json"),
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


def install_exception_handlers(app: FastAPI, settings: Settings) -> None:
    """Register the handlers that turn every failure into problem+json.

    Args:
        app: The FastAPI application.
        settings: Active settings.
    """

    @app.exception_handler(RagError)
    async def _rag_error(request: Request, exc: Exception) -> JSONResponse:
        """Map the platform error hierarchy onto a problem document.

        Args:
            request: The failing request.
            exc: A :class:`~ragcore.errors.RagError`.

        Returns:
            The problem response.
        """
        error = exc if isinstance(exc, RagError) else RagError(str(exc))
        status = error.status_code
        headers: dict[str, str] | None = None
        if status == 401:
            # RFC 6750: a 401 must say how to authenticate.
            headers = {"WWW-Authenticate": "Bearer"}
        log = _log.warning if status < 500 else _log.error
        log(
            "request_failed",
            code=error.code,
            status=status,
            path=request.url.path,
            error=type(error).__name__,
        )
        return problem_response(
            request,
            status=status,
            title=_title_for(type(error).__name__),
            detail=error.message,
            code=error.code,
            settings=settings,
            context=error.detail,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
        """Report a malformed request body or query.

        Args:
            request: The failing request.
            exc: The validation error.

        Returns:
            The problem response. Pydantic's ``input`` member is stripped: it
            echoes the offending value, which for ``POST /chat`` is the user's
            turn and has not passed PII redaction.
        """
        raw = exc.errors() if isinstance(exc, RequestValidationError) else []
        errors = [
            {
                "loc": [str(part) for part in item.get("loc", ())],
                "type": str(item.get("type", "")),
                "msg": str(item.get("msg", "")),
            }
            for item in raw
        ]
        return problem_response(
            request,
            status=422,
            title="Request Validation Failed",
            detail="the request body or query parameters are not valid",
            code="validation_error",
            settings=settings,
            errors=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: Exception) -> JSONResponse:
        """Render a routing-level failure as a problem document.

        Args:
            request: The failing request.
            exc: The HTTP exception.

        Returns:
            The problem response.
        """
        error = exc if isinstance(exc, StarletteHTTPException) else None
        status = error.status_code if error else 500
        detail = str(error.detail) if error and error.detail else ""
        headers = dict(error.headers or {}) if error and error.headers else None
        return problem_response(
            request,
            status=status,
            title=_title_for_status(status),
            detail=detail,
            code=f"http_{status}",
            settings=settings,
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Contain an unexpected failure.

        Args:
            request: The failing request.
            exc: Any exception that is not a :class:`~ragcore.errors.RagError`.

        Returns:
            A 500 problem response whose ``detail`` names the exception class and
            nothing else — an exception raised while handling user content can
            quote that content in its message.
        """
        _log.exception("unhandled_error", path=request.url.path)
        return problem_response(
            request,
            status=http_status_for(exc),
            title="Internal Server Error",
            detail="the request could not be completed",
            code="internal_error",
            settings=settings,
            context={"error": type(exc).__name__},
        )


def _title_for(class_name: str) -> str:
    """Turn an exception class name into a human-readable title.

    Args:
        class_name: e.g. ``"TenantMismatchError"``.

    Returns:
        e.g. ``"Tenant Mismatch"``.
    """
    trimmed = class_name.removesuffix("Error")
    words: list[str] = []
    current = ""
    for char in trimmed:
        if char.isupper() and current:
            words.append(current)
            current = char
        else:
            current += char
    if current:
        words.append(current)
    return " ".join(words) or class_name


_STATUS_TITLES: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def _title_for_status(status: int) -> str:
    """Human-readable title for an HTTP status.

    Args:
        status: The status code.

    Returns:
        The reason phrase, or a generic label.
    """
    return _STATUS_TITLES.get(status, f"HTTP {status}")
