"""Langfuse tracing that can never break the request path.

Three properties matter here:

1. **Optional.** `langfuse` is an optional runtime dependency and Langfuse itself
   is a remote service. When the package is missing, ``langfuse_enabled`` is
   false, either key is absent, or any SDK call raises, the tracer degrades to
   :class:`NoopTracer` and the pipeline keeps running.
2. **Version tolerant.** The Langfuse SDK changed shape between v2 (stateful
   ``trace()``/``span()`` clients) and v3 (OpenTelemetry-style
   ``start_span()``). Every call site probes both spellings and gives up quietly
   rather than pinning us to one.
3. **Contextvar propagation.** Trace and span identity flows through
   ``contextvars``, so nothing has to thread a handle through call signatures
   and :func:`traced` can decorate any async function.

Content policy: handles accept ``input``/``output`` payloads, but callers must
only pass **PII-redacted or structural** data. :class:`ragcore.llm.LLMClient`
deliberately sends message shape and token counts, never raw turns.
"""

from __future__ import annotations

import functools
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from ragcore.logging import get_logger
from ragcore.settings import Settings, get_settings

__all__ = [
    "LangfuseTracer",
    "NoopTracer",
    "SpanHandle",
    "TraceHandle",
    "Tracer",
    "current_span_id",
    "flush_tracer",
    "get_current_trace_id",
    "get_tracer",
    "reset_tracer_cache",
    "shutdown_tracer",
    "traced",
]

P = ParamSpec("P")
T = TypeVar("T")

_log = get_logger(__name__)

_current_trace_id: ContextVar[str | None] = ContextVar("ragcore_trace_id", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("ragcore_span_id", default=None)
_current_observation: ContextVar[Any] = ContextVar("ragcore_observation", default=None)

#: Consecutive SDK failures after which a tracer stops trying for this process.
FAILURE_BUDGET = 20


def get_current_trace_id() -> str | None:
    """Return the trace id bound to the current async context.

    Returns:
        The active trace id, or None when no trace is open.
    """
    return _current_trace_id.get()


def current_span_id() -> str | None:
    """Return the span id bound to the current async context.

    Returns:
        The active span id, or None when no span is open.
    """
    return _current_span_id.get()


@dataclass(slots=True)
class TraceHandle:
    """A handle onto one open trace.

    Attributes:
        trace_id: Langfuse trace id, or None when tracing is disabled.
        name: Trace name.
    """

    trace_id: str | None
    name: str
    _tracer: Tracer | None = None
    _obj: Any = None

    def update(self, **fields: Any) -> None:
        """Attach or overwrite trace-level fields.

        Args:
            **fields: Langfuse trace fields (``output``, ``metadata``, ``tags``,
                ``level``, ``status_message``). Must contain no unredacted user
                content.
        """
        if self._tracer is not None:
            self._tracer.update_observation(self._obj, **fields)

    def score(self, *, name: str, value: float, comment: str | None = None) -> None:
        """Attach a score to this trace.

        Args:
            name: Score name, e.g. ``"faithfulness"`` or ``"user_feedback"``.
            value: Numeric score.
            comment: Optional redacted comment.
        """
        if self._tracer is not None:
            self._tracer.score(
                name=name, value=value, comment=comment, trace_id=self.trace_id
            )


@dataclass(slots=True)
class SpanHandle:
    """A handle onto one open span.

    Attributes:
        span_id: Langfuse observation id, or None when tracing is disabled.
        trace_id: Owning trace id, or None when tracing is disabled.
        name: Span name.
    """

    span_id: str | None
    trace_id: str | None
    name: str
    _tracer: Tracer | None = None
    _obj: Any = None

    def update(self, **fields: Any) -> None:
        """Attach or overwrite span-level fields.

        Args:
            **fields: Langfuse observation fields (``output``, ``metadata``,
                ``level``, ``status_message``). Must contain no unredacted user
                content.
        """
        if self._tracer is not None:
            self._tracer.update_observation(self._obj, **fields)


@dataclass(slots=True)
class Tracer:
    """No-op tracer: the base class and the safe default.

    Every method is a working no-op, so callers never branch on whether tracing
    is configured. :class:`LangfuseTracer` overrides the emitting parts.

    Attributes:
        enabled: Whether spans are actually shipped anywhere.
    """

    enabled: bool = False
    _failures: int = field(default=0, repr=False)

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[TraceHandle]:
        """Open a trace for the duration of the block.

        Args:
            name: Trace name, e.g. ``"chat.turn"``.
            user_id: Entra object id of the caller.
            session_id: Chat session id.
            tenant_id: Owning tenant id, recorded in metadata and as a tag.
            tags: Additional low-cardinality tags.
            metadata: Structural metadata. Never unredacted user content.

        Yields:
            A :class:`TraceHandle`.
        """
        del user_id, session_id, tenant_id, tags, metadata
        yield TraceHandle(trace_id=None, name=name)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[SpanHandle]:
        """Open a span nested under the current trace or span.

        Args:
            name: Span name, e.g. ``"retrieve"``.
            input: Redacted or structural input payload.
            metadata: Structural metadata.

        Yields:
            A :class:`SpanHandle`.
        """
        del input, metadata
        yield SpanHandle(span_id=None, trace_id=self.current_trace_id(), name=name)

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one model call as a generation observation.

        Args:
            name: Logical call site, e.g. ``"llm.complete"``.
            model: Exact Anthropic model id.
            input: Structural request description. Never raw prompt text.
            output: Structural response description. Never raw answer text.
            usage: Token counts and cost.
            metadata: Additional structural metadata.
        """

    def score(
        self,
        *,
        name: str,
        value: float,
        comment: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Attach a score to a trace.

        Args:
            name: Score name.
            value: Numeric score.
            comment: Optional redacted comment.
            trace_id: Target trace; defaults to the current trace.
        """

    def event(self, name: str, **fields: Any) -> None:
        """Record a point-in-time event on the current observation.

        Args:
            name: Event name.
            **fields: Structural fields.
        """

    def update_observation(self, obj: Any, **fields: Any) -> None:
        """Update an SDK observation object in place.

        Args:
            obj: The SDK object returned when the observation was created.
            **fields: Fields to set.
        """

    def current_trace_id(self) -> str | None:
        """Return the trace id bound to the current async context.

        Returns:
            The active trace id, or None.
        """
        return _current_trace_id.get()

    def flush(self) -> None:
        """Flush buffered observations. No-op unless tracing is enabled."""

    def shutdown(self) -> None:
        """Flush and release SDK resources. No-op unless tracing is enabled."""


@dataclass(slots=True)
class NoopTracer(Tracer):
    """Explicit no-op tracer, selected when Langfuse is disabled or unconfigured."""

    enabled: bool = False


@dataclass(slots=True)
class LangfuseTracer(Tracer):
    """Tracer backed by the Langfuse SDK.

    Attributes:
        enabled: True while the SDK is usable; flipped to False after
            :data:`FAILURE_BUDGET` consecutive failures.
    """

    enabled: bool = True
    _client: Any = None
    _sample_rate: float = 1.0

    # ---------------------------------------------------------------- internals
    def _safe(self, action: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Call an SDK function, swallowing and counting any failure.

        Args:
            action: Short label used in the warning log.
            fn: SDK callable.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            The SDK return value, or None on failure.
        """
        if not self.enabled or fn is None:
            return None
        try:
            result = fn(**kwargs)
        except Exception:
            self._failures += 1
            _log.warning(
                "langfuse_call_failed",
                action=action,
                failures=self._failures,
                exc_info=True,
            )
            if self._failures >= FAILURE_BUDGET:
                self.enabled = False
                _log.error("langfuse_disabled_after_failures", failures=self._failures)
            return None
        self._failures = 0
        return result

    def _should_sample(self) -> bool:
        """Decide whether to record this trace.

        Returns:
            True when the trace falls inside the configured sample rate.
        """
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        return random.random() < self._sample_rate  # noqa: S311 - not security

    @staticmethod
    def _ids(obj: Any) -> tuple[str | None, str | None]:
        """Extract ``(trace_id, observation_id)`` from an SDK object.

        Args:
            obj: SDK trace, span or generation object.

        Returns:
            A ``(trace_id, observation_id)`` pair; either may be None.
        """
        if obj is None:
            return None, None
        trace_id = getattr(obj, "trace_id", None)
        obs_id = getattr(obj, "id", None) or getattr(obj, "observation_id", None)
        if trace_id is None:
            trace_id = obs_id
        return (str(trace_id) if trace_id else None, str(obs_id) if obs_id else None)

    def _end(self, obj: Any) -> None:
        """End an SDK observation when the SDK version supports it.

        Args:
            obj: SDK observation object.
        """
        ender = getattr(obj, "end", None)
        if callable(ender):
            self._safe("end", ender)

    # ------------------------------------------------------------------ tracing
    @asynccontextmanager
    async def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[TraceHandle]:
        """Open a Langfuse trace for the duration of the block.

        Args:
            name: Trace name.
            user_id: Entra object id of the caller.
            session_id: Chat session id.
            tenant_id: Owning tenant id.
            tags: Additional low-cardinality tags.
            metadata: Structural metadata only.

        Yields:
            A :class:`TraceHandle`; ``trace_id`` is None when the trace was
            sampled out or the SDK call failed.
        """
        if not self._should_sample():
            async with Tracer.trace(
                self,
                name,
                user_id=user_id,
                session_id=session_id,
                tenant_id=tenant_id,
                tags=tags,
                metadata=metadata,
            ) as handle:
                yield handle
            return

        merged: dict[str, Any] = dict(metadata or {})
        if tenant_id:
            merged.setdefault("tenant_id", tenant_id)
        all_tags = list(tags or [])
        if tenant_id:
            all_tags.append(f"tenant:{tenant_id}")

        obj = self._safe(
            "trace",
            getattr(self._client, "trace", None),
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=merged,
            tags=all_tags or None,
        )
        if obj is None:
            obj = self._safe(
                "start_span",
                getattr(self._client, "start_span", None),
                name=name,
                metadata=merged,
            )
            if obj is not None:
                self._safe(
                    "update_current_trace",
                    getattr(self._client, "update_current_trace", None),
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=merged,
                    tags=all_tags or None,
                )

        trace_id, obs_id = self._ids(obj)
        handle = TraceHandle(trace_id=trace_id, name=name, _tracer=self, _obj=obj)
        trace_token = _current_trace_id.set(trace_id)
        span_token = _current_span_id.set(obs_id)
        obs_token = _current_observation.set(obj)
        try:
            yield handle
        finally:
            self._end(obj)
            _current_observation.reset(obs_token)
            _current_span_id.reset(span_token)
            _current_trace_id.reset(trace_token)
            self.flush()

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[SpanHandle]:
        """Open a Langfuse span nested under the current observation.

        Args:
            name: Span name.
            input: Redacted or structural input payload.
            metadata: Structural metadata only.

        Yields:
            A :class:`SpanHandle`.
        """
        parent = _current_observation.get()
        target = parent if parent is not None else self._client
        obj = self._safe(
            "span",
            getattr(target, "span", None),
            name=name,
            input=input,
            metadata=dict(metadata or {}),
        )
        if obj is None:
            obj = self._safe(
                "start_span",
                getattr(target, "start_span", None),
                name=name,
                input=input,
                metadata=dict(metadata or {}),
            )

        trace_id, obs_id = self._ids(obj)
        handle = SpanHandle(
            span_id=obs_id,
            trace_id=trace_id or _current_trace_id.get(),
            name=name,
            _tracer=self,
            _obj=obj,
        )
        span_token = _current_span_id.set(handle.span_id)
        obs_token = _current_observation.set(obj if obj is not None else parent)
        try:
            yield handle
        finally:
            self._end(obj)
            _current_observation.reset(obs_token)
            _current_span_id.reset(span_token)

    def generation(
        self,
        name: str,
        *,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one model call as a Langfuse generation.

        Args:
            name: Logical call site, e.g. ``"llm.complete"``.
            model: Exact Anthropic model id.
            input: Structural request description. Never raw prompt text.
            output: Structural response description. Never raw answer text.
            usage: Token counts and cost.
            metadata: Additional structural metadata.
        """
        parent = _current_observation.get()
        target = parent if parent is not None else self._client
        payload: dict[str, Any] = {
            "name": name,
            "model": model,
            "input": input,
            "output": output,
            "metadata": dict(metadata or {}),
        }
        obj = self._safe(
            "generation",
            getattr(target, "generation", None),
            usage=dict(usage or {}),
            **payload,
        )
        if obj is None:
            obj = self._safe(
                "start_generation",
                getattr(target, "start_generation", None),
                usage_details=dict(usage or {}),
                **payload,
            )
        self._end(obj)

    def score(
        self,
        *,
        name: str,
        value: float,
        comment: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Attach a score to a trace.

        Args:
            name: Score name.
            value: Numeric score.
            comment: Optional redacted comment.
            trace_id: Target trace; defaults to the current trace.
        """
        target_id = trace_id or _current_trace_id.get()
        if target_id is None:
            return
        payload = {
            "name": name,
            "value": value,
            "comment": comment,
            "trace_id": target_id,
        }
        obj = self._safe("score", getattr(self._client, "score", None), **payload)
        if obj is None:
            self._safe(
                "create_score", getattr(self._client, "create_score", None), **payload
            )

    def event(self, name: str, **fields: Any) -> None:
        """Record a point-in-time event on the current observation.

        Args:
            name: Event name.
            **fields: Structural fields.
        """
        parent = _current_observation.get()
        target = parent if parent is not None else self._client
        obj = self._safe("event", getattr(target, "event", None), name=name, **fields)
        if obj is None:
            self._safe(
                "create_event",
                getattr(target, "create_event", None),
                name=name,
                **fields,
            )

    def update_observation(self, obj: Any, **fields: Any) -> None:
        """Update an SDK observation object in place.

        Args:
            obj: The SDK object returned when the observation was created.
            **fields: Fields to set.
        """
        if obj is None:
            return
        self._safe("update", getattr(obj, "update", None), **fields)

    def flush(self) -> None:
        """Flush buffered observations to Langfuse."""
        self._safe("flush", getattr(self._client, "flush", None))

    def shutdown(self) -> None:
        """Flush and release SDK resources."""
        self.flush()
        self._safe("shutdown", getattr(self._client, "shutdown", None))


def _build_client(settings: Settings) -> Any:
    """Construct a Langfuse SDK client, tolerating v2 and v3 signatures.

    Args:
        settings: Settings supplying host, keys and SDK options.

    Returns:
        A Langfuse client, or None when the package is missing or construction
        failed.
    """
    try:
        from langfuse import Langfuse
    except ImportError:
        _log.warning("langfuse_not_installed", host=settings.langfuse_host)
        return None

    full: dict[str, Any] = {
        "public_key": settings.langfuse_public_key,
        "secret_key": settings.langfuse_secret_key,
        "host": settings.langfuse_host,
        "debug": settings.langfuse_debug,
        "release": settings.langfuse_release,
        "flush_interval": settings.langfuse_flush_interval_seconds,
    }
    minimal = {
        "public_key": settings.langfuse_public_key,
        "secret_key": settings.langfuse_secret_key,
        "host": settings.langfuse_host,
    }
    for kwargs in (full, minimal):
        try:
            return Langfuse(**kwargs)
        except TypeError:
            continue
        except Exception:
            _log.warning("langfuse_client_init_failed", exc_info=True)
            return None
    _log.warning("langfuse_client_signature_unsupported")
    return None


_TRACERS: dict[tuple[Any, ...], Tracer] = {}


def get_tracer(settings: Settings | None = None) -> Tracer:
    """Return the process-wide tracer for these settings.

    Args:
        settings: Settings to read Langfuse configuration from. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        A :class:`LangfuseTracer` when tracing is enabled and both keys are
        present and the SDK is importable, else a :class:`NoopTracer`.
    """
    cfg = settings or get_settings()
    key: tuple[Any, ...] = (
        cfg.langfuse_ready,
        cfg.langfuse_host,
        cfg.langfuse_public_key,
        cfg.langfuse_sample_rate,
    )
    cached = _TRACERS.get(key)
    if cached is not None:
        return cached

    if not cfg.langfuse_ready:
        tracer: Tracer = NoopTracer()
        _log.info(
            "tracing_disabled",
            langfuse_enabled=cfg.langfuse_enabled,
            keys_present=bool(cfg.langfuse_public_key and cfg.langfuse_secret_key),
        )
    else:
        client = _build_client(cfg)
        if client is None:
            tracer = NoopTracer()
        else:
            tracer = LangfuseTracer(
                enabled=True,
                _client=client,
                _sample_rate=cfg.langfuse_sample_rate,
            )
            _log.info("tracing_enabled", host=cfg.langfuse_host)
    _TRACERS[key] = tracer
    return tracer


def reset_tracer_cache() -> None:
    """Drop cached tracers. Tests call this after mutating settings."""
    _TRACERS.clear()


def flush_tracer(settings: Settings | None = None) -> None:
    """Flush the tracer for these settings.

    Args:
        settings: Settings selecting the tracer. Defaults to the process
            settings.
    """
    get_tracer(settings).flush()


def shutdown_tracer(settings: Settings | None = None) -> None:
    """Flush and shut down the tracer for these settings.

    Args:
        settings: Settings selecting the tracer. Defaults to the process
            settings.
    """
    get_tracer(settings).shutdown()


def traced(
    name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Wrap an async function in a span.

    Args:
        name: Span name.

    Returns:
        A decorator that opens a span around each call. The wrapped function's
        arguments are never sent to Langfuse, because they may contain
        unredacted user content.
    """

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_tracer()
            async with tracer.span(name, metadata={"function": fn.__qualname__}):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
