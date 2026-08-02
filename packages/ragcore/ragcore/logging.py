"""Structured JSON logging with request-context binding and secret redaction.

Call :func:`configure_logging` once per process (API startup, Functions host,
eval harness). After that, ``structlog.get_logger(__name__)`` anywhere returns a
logger that emits newline-delimited JSON with the bound request context attached.

Two safety rules are enforced by processors rather than by convention:

* :func:`redact_secrets` drops or masks any event key that looks like a credential,
  so a stray ``api_key=`` never reaches Log Analytics.
* :func:`guard_raw_content` refuses keys reserved for user content unless the
  caller has explicitly marked them redacted, protecting the "never log raw user
  content that has not passed PII redaction" rule.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    get_contextvars,
    unbind_contextvars,
)
from structlog.types import EventDict, WrappedLogger

from ragcore.settings import Settings, get_settings

__all__ = [
    "SECRET_KEY_PATTERN",
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "guard_raw_content",
    "redact_secrets",
    "redact_value",
    "request_context",
]

#: Event keys whose values are replaced with a placeholder before emission.
SECRET_KEY_PATTERN = re.compile(
    r"(?:^|_)("
    r"secret|password|passwd|token|api_key|apikey|access_key|private_key|"
    r"client_secret|connection_string|sas|credential|authorization|auth_header|"
    r"cookie|session_key|signing_key|jwt|bearer"
    r")(?:$|_)",
    re.IGNORECASE,
)

#: Event keys that carry user or document text. They are only allowed through when
#: the same event sets ``pii_redacted=True``.
RAW_CONTENT_KEYS: frozenset[str] = frozenset(
    {
        "text",
        "content",
        "message",
        "query",
        "question",
        "answer",
        "prompt",
        "chunk_text",
        "user_input",
        "document_text",
        "snippet",
    }
)

_REDACTED = "***redacted***"
_DROPPED = "***omitted-unredacted***"

_configured = False


def redact_value(value: object, *, keep: int = 4) -> str:
    """Mask a secret-ish value, keeping a short suffix for correlation.

    Args:
        value: The value to mask.
        keep: Number of trailing characters to preserve. Values shorter than
            ``keep * 2`` are masked entirely.

    Returns:
        A masked string safe to log.
    """
    text = str(value)
    if len(text) <= keep * 2:
        return _REDACTED
    return f"{_REDACTED}{text[-keep:]}"


def redact_secrets(
    _logger: WrappedLogger, _method: str, event_dict: EventDict
) -> EventDict:
    """Mask event values whose key names look like credentials.

    Nested mappings are walked recursively so a ``detail={"api_key": ...}`` payload
    is masked too.

    Args:
        _logger: Unused; required by the structlog processor protocol.
        _method: Unused; required by the structlog processor protocol.
        event_dict: The event being processed.

    Returns:
        The event dict with secret-looking values replaced.
    """
    return _redact_mapping(event_dict)


def _redact_mapping(mapping: MutableMapping[str, Any]) -> Any:
    for key, value in list(mapping.items()):
        if isinstance(key, str) and SECRET_KEY_PATTERN.search(key):
            mapping[key] = redact_value(value)
            continue
        if isinstance(value, MutableMapping):
            _redact_mapping(value)
        elif isinstance(value, list):
            mapping[key] = [
                _redact_mapping(item) if isinstance(item, MutableMapping) else item
                for item in value
            ]
    return mapping


def guard_raw_content(
    _logger: WrappedLogger, _method: str, event_dict: EventDict
) -> EventDict:
    """Drop user-content keys that have not been marked as PII-redacted.

    An event may log ``text=`` only when it also sets ``pii_redacted=True``. This
    makes the redact-before-log rule a mechanical property of the logging pipeline
    instead of something each call site has to remember.

    Args:
        _logger: Unused; required by the structlog processor protocol.
        _method: Unused; required by the structlog processor protocol.
        event_dict: The event being processed.

    Returns:
        The event dict with unredacted content keys replaced by a marker.
    """
    if event_dict.get("pii_redacted") is True:
        return event_dict
    for key in RAW_CONTENT_KEYS:
        if key in event_dict and key != "event":
            event_dict[key] = _DROPPED
    return event_dict


def _add_service_context(settings: Settings) -> Any:
    """Build a processor that stamps static service identity onto every event.

    Args:
        settings: Settings providing the service name and environment.

    Returns:
        A structlog processor.
    """

    def processor(
        _logger: WrappedLogger, _method: str, event_dict: EventDict
    ) -> EventDict:
        event_dict.setdefault("service", settings.service_name)
        event_dict.setdefault("env", settings.env)
        if settings.langfuse_release:
            event_dict.setdefault("release", settings.langfuse_release)
        return event_dict

    return processor


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib logging bridge for this process.

    Idempotent: repeated calls are ignored unless ``force`` is set, so importing a
    module that configures logging cannot clobber an existing setup.

    Args:
        settings: Settings to read log level and format from. Defaults to
            :func:`ragcore.settings.get_settings`.
        force: Reconfigure even if logging was already configured.
    """
    global _configured
    if _configured and not force:
        return
    cfg = settings or get_settings()

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_service_context(cfg),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        guard_raw_content,
        redact_secrets,
    ]

    if cfg.log_json:
        renderer: Any = structlog.processors.JSONRenderer()
        shared.append(structlog.processors.format_exc_info)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        shared.append(structlog.processors.ExceptionPrettyPrinter())

    level = logging.getLevelNamesMapping()[cfg.log_level]

    # Route structlog through stdlib logging. This is what makes third-party output
    # (uvicorn, sqlalchemy, httpx, the Azure SDK) render with the same processors —
    # including secret redaction — instead of escaping as unstructured text.
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for noisy, level in (
        ("uvicorn.access", logging.WARNING),
        ("sqlalchemy.engine", logging.WARNING),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("azure", logging.WARNING),
        ("azure.core.pipeline.policies.http_logging_policy", logging.WARNING),
        ("fastembed", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(level)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A bound logger that emits through the configured processor chain.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name if name is not None else "ragcore")


def bind_request_context(
    *,
    request_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    route: str | None = None,
    **extra: Any,
) -> dict[str, str]:
    """Bind request-scoped identifiers onto every subsequent log event.

    Values propagate through ``contextvars``, so nothing has to thread a logger
    through call signatures. Keys with a None value are skipped.

    Args:
        request_id: Correlation id for this request. Generated when omitted.
        tenant_id: Tenant the request is scoped to.
        user_id: Entra object id (``oid``) of the caller.
        session_id: Chat session id, when applicable.
        trace_id: Langfuse trace id, when a trace is open.
        route: Logical route or pipeline stage name.
        **extra: Additional low-cardinality context to bind.

    Returns:
        The mapping that was bound, including the resolved ``request_id``.
    """
    bound: dict[str, str] = {"request_id": request_id or uuid.uuid4().hex}
    candidates: dict[str, Any] = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "route": route,
        **extra,
    }
    for key, value in candidates.items():
        if value is not None:
            bound[key] = str(value)
    bind_contextvars(**bound)
    return bound


def clear_request_context(*keys: str) -> None:
    """Remove request-scoped context.

    Args:
        *keys: Specific keys to unbind. When omitted, all context is cleared.
    """
    if keys:
        unbind_contextvars(*keys)
    else:
        clear_contextvars()


@contextmanager
def request_context(**kwargs: Any) -> Iterator[dict[str, str]]:
    """Bind request context for the duration of a block, then restore it.

    Args:
        **kwargs: Same keyword arguments as :func:`bind_request_context`.

    Yields:
        The mapping that was bound, including the resolved ``request_id``.
    """
    # Snapshot the whole context, not just the keys we are about to set: this block
    # also generates a request_id, and anything bound inside it must not leak out.
    snapshot = dict(get_contextvars())
    bound = bind_request_context(**kwargs)
    try:
        yield bound
    finally:
        clear_contextvars()
        if snapshot:
            bind_contextvars(**snapshot)
