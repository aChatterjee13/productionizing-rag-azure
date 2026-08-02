"""Prometheus instrumentation for the RAG platform.

Every metric lives in a dedicated :class:`~prometheus_client.CollectorRegistry`
(:data:`REGISTRY`) so ``GET /metrics`` renders exactly what this module declares
and nothing a third-party library happens to have registered globally.

`prometheus_client` is imported defensively. If it is absent the module still
imports and every ``observe_*`` helper becomes a no-op, because losing telemetry
must never take the request path down. :func:`render_metrics` then returns a
single comment line so ``/metrics`` still answers with valid (empty) exposition.

Latency is accepted in **milliseconds** at the call sites (that is what the
pipeline measures) and recorded in **seconds**, which is the Prometheus
convention.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "METRICS_CONTENT_TYPE",
    "PROMETHEUS_AVAILABLE",
    "REGISTRY",
    "cache_lookups",
    "guardrail_events",
    "http_request_latency",
    "http_requests",
    "ingest_documents",
    "ingest_runs",
    "llm_calls",
    "llm_cost_usd",
    "llm_latency",
    "llm_tokens",
    "observe_cache_lookup",
    "observe_guardrail",
    "observe_http_request",
    "observe_ingest_documents",
    "observe_ingest_run",
    "observe_llm_call",
    "observe_pipeline_stage",
    "observe_retrieval_stage",
    "observe_tool_invocation",
    "pipeline_stage_latency",
    "render_metrics",
    "retrieval_candidates",
    "retrieval_stage_latency",
    "set_build_info",
    "tool_invocations",
]

_MS_PER_SECOND = 1000.0

#: Latency buckets (seconds) for retrieval and pipeline stages.
_STAGE_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
#: Latency buckets (seconds) for model calls, which can run many minutes.
_LLM_BUCKETS = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
#: Latency buckets (seconds) for HTTP requests.
_HTTP_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
#: Cost buckets (USD) for a single model call.
_COST_BUCKETS = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


class _NoopMetric:
    """Stand-in for a Prometheus collector when the client library is absent."""

    def labels(self, *_args: Any, **_kwargs: Any) -> _NoopMetric:
        """Return self so chained ``.labels(...).inc()`` calls keep working.

        Returns:
            This same instance.
        """
        return self

    def inc(self, _amount: float = 1.0) -> None:
        """Discard a counter increment."""

    def observe(self, _amount: float) -> None:
        """Discard a histogram observation."""

    def set(self, _value: float) -> None:
        """Discard a gauge assignment."""

    def info(self, _value: dict[str, str]) -> None:
        """Discard an info-metric assignment."""


class _NoopRegistry:
    """Stand-in for a Prometheus registry when the client library is absent."""


try:  # pragma: no cover - depends on the installed extras
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dependency
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def _noop_factory(*_args: Any, **_kwargs: Any) -> _NoopMetric:
        """Build a no-op collector.

        Returns:
            A :class:`_NoopMetric`.
        """
        return _NoopMetric()

    CollectorRegistry = _NoopRegistry  # type: ignore[assignment,misc]
    Counter = _noop_factory  # type: ignore[assignment]
    Gauge = _noop_factory  # type: ignore[assignment]
    Histogram = _noop_factory  # type: ignore[assignment]

    def generate_latest(_registry: Any = None) -> bytes:  # type: ignore[misc]
        """Render an empty exposition payload.

        Returns:
            A single comment line explaining why no metrics are present.
        """
        return b"# prometheus_client is not installed; metrics are disabled\n"


#: Content type for the ``/metrics`` response body.
METRICS_CONTENT_TYPE: str = CONTENT_TYPE_LATEST

#: Dedicated registry holding every platform metric.
REGISTRY = CollectorRegistry()

build_info = Gauge(
    "rag_build_info",
    "Static build and deployment identity (always 1).",
    ("service", "env", "release"),
    registry=REGISTRY,
)

http_requests = Counter(
    "rag_http_requests_total",
    "HTTP requests handled by the API, by route, method and status class.",
    ("route", "method", "status"),
    registry=REGISTRY,
)
http_request_latency = Histogram(
    "rag_http_request_duration_seconds",
    "End-to-end HTTP request latency.",
    ("route", "method"),
    buckets=_HTTP_BUCKETS,
    registry=REGISTRY,
)

pipeline_stage_latency = Histogram(
    "rag_pipeline_stage_duration_seconds",
    "Latency of one ordered RAG pipeline stage (orchestrator.py).",
    ("stage",),
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)
retrieval_stage_latency = Histogram(
    "rag_retrieval_stage_duration_seconds",
    "Retrieval latency by stage: dense|sparse|fusion|rerank|cache|tool|mmr|dedupe.",
    ("stage",),
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)
retrieval_candidates = Histogram(
    "rag_retrieval_candidates",
    "Candidate counts recorded per retrieval phase.",
    ("phase",),
    buckets=(1, 2, 5, 10, 20, 30, 50, 100, 200, 500),
    registry=REGISTRY,
)

llm_calls = Counter(
    "rag_llm_calls_total",
    "Anthropic Messages API calls, by model, operation and outcome.",
    ("model", "operation", "outcome"),
    registry=REGISTRY,
)
llm_tokens = Counter(
    "rag_llm_tokens_total",
    "Token usage by model and kind: input|output|cache_read|cache_write.",
    ("model", "kind"),
    registry=REGISTRY,
)
llm_cost_usd = Counter(
    "rag_llm_cost_usd_total",
    "Accumulated model spend in USD, by model and operation.",
    ("model", "operation"),
    registry=REGISTRY,
)
llm_call_cost_usd = Histogram(
    "rag_llm_call_cost_usd",
    "Distribution of per-call model spend in USD.",
    ("model",),
    buckets=_COST_BUCKETS,
    registry=REGISTRY,
)
llm_latency = Histogram(
    "rag_llm_call_duration_seconds",
    "Latency of one Anthropic API call.",
    ("model", "operation"),
    buckets=_LLM_BUCKETS,
    registry=REGISTRY,
)

cache_lookups = Counter(
    "rag_cache_lookups_total",
    "Cache probes by cache name and result (hit|miss); ratio gives the hit rate.",
    ("cache", "result"),
    registry=REGISTRY,
)

guardrail_events = Counter(
    "rag_guardrail_events_total",
    "Guardrail decisions by stage, kind and action; action=block counts blocks.",
    ("stage", "kind", "action"),
    registry=REGISTRY,
)

tool_invocations = Counter(
    "rag_tool_invocations_total",
    "Tool-loop invocations by tool, kind and outcome.",
    ("tool", "kind", "outcome"),
    registry=REGISTRY,
)
tool_latency = Histogram(
    "rag_tool_invocation_duration_seconds",
    "Latency of one tool invocation.",
    ("tool", "kind"),
    buckets=_STAGE_BUCKETS,
    registry=REGISTRY,
)

ingest_runs = Counter(
    "rag_ingest_runs_total",
    "Ingestion runs by trigger and status.",
    ("trigger", "status"),
    registry=REGISTRY,
)
ingest_documents = Counter(
    "rag_ingest_documents_total",
    "Documents processed by ingestion, by source type and action.",
    ("source_type", "action"),
    registry=REGISTRY,
)


def set_build_info(*, service: str, env: str, release: str | None = None) -> None:
    """Publish static build identity as a gauge sample.

    Args:
        service: Logical service name.
        env: Deployment environment.
        release: Release identifier (git SHA); empty when unset.
    """
    build_info.labels(service=service, env=env, release=release or "").set(1)


def observe_http_request(
    *, route: str, method: str, status_code: int, latency_ms: float
) -> None:
    """Record one handled HTTP request.

    Args:
        route: Route template, e.g. ``"/api/v1/chat"``. Never a rendered path
            with ids in it, which would explode label cardinality.
        method: HTTP method.
        status_code: Response status code.
        latency_ms: Wall-clock handling time in milliseconds.
    """
    http_requests.labels(
        route=route, method=method.upper(), status=str(status_code)
    ).inc()
    http_request_latency.labels(route=route, method=method.upper()).observe(
        latency_ms / _MS_PER_SECOND
    )


def observe_pipeline_stage(*, stage: str, latency_ms: float) -> None:
    """Record the duration of one RAG pipeline stage.

    Args:
        stage: Stage name, e.g. ``"input_guard"`` or ``"generate"``.
        latency_ms: Duration in milliseconds.
    """
    pipeline_stage_latency.labels(stage=stage).observe(latency_ms / _MS_PER_SECOND)


def observe_retrieval_stage(
    *, stage: str, latency_ms: float, candidates: int | None = None
) -> None:
    """Record retrieval latency, and optionally the candidate count, for a stage.

    Args:
        stage: One of ``dense``, ``sparse``, ``fusion``, ``dedupe``, ``rerank``,
            ``mmr``, ``cache`` or ``tool``.
        latency_ms: Duration in milliseconds.
        candidates: Number of candidates the stage emitted, when meaningful.
    """
    retrieval_stage_latency.labels(stage=stage).observe(latency_ms / _MS_PER_SECOND)
    if candidates is not None:
        retrieval_candidates.labels(phase=stage).observe(candidates)


def observe_llm_call(
    *,
    model: str,
    operation: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    outcome: str = "ok",
) -> None:
    """Record tokens, cost and latency for one model call.

    Also feeds the prompt-cache hit rate: a call with ``cache_read_tokens > 0``
    counts as a hit on the ``prompt`` cache, otherwise a miss.

    Args:
        model: Exact Anthropic model id.
        operation: Logical call site, e.g. ``"generate"`` or ``"classify"``.
        input_tokens: Uncached input tokens.
        output_tokens: Generated tokens.
        cache_read_tokens: Tokens served from the prompt cache.
        cache_write_tokens: Tokens written to the prompt cache.
        cost_usd: Computed cost of the call.
        latency_ms: Wall-clock duration in milliseconds.
        outcome: ``ok``, ``refused`` or ``error``.
    """
    llm_calls.labels(model=model, operation=operation, outcome=outcome).inc()
    for kind, amount in (
        ("input", input_tokens),
        ("output", output_tokens),
        ("cache_read", cache_read_tokens),
        ("cache_write", cache_write_tokens),
    ):
        if amount:
            llm_tokens.labels(model=model, kind=kind).inc(amount)
    if cost_usd:
        llm_cost_usd.labels(model=model, operation=operation).inc(cost_usd)
        llm_call_cost_usd.labels(model=model).observe(cost_usd)
    if latency_ms:
        llm_latency.labels(model=model, operation=operation).observe(
            latency_ms / _MS_PER_SECOND
        )
    observe_cache_lookup(cache="prompt", hit=cache_read_tokens > 0)


def observe_cache_lookup(*, cache: str, hit: bool) -> None:
    """Record a cache probe.

    Args:
        cache: Cache name, e.g. ``prompt``, ``semantic`` or ``tool``.
        hit: Whether the probe was served from cache.
    """
    cache_lookups.labels(cache=cache, result="hit" if hit else "miss").inc()


def observe_guardrail(*, stage: str, kind: str, action: str) -> None:
    """Record one guardrail decision.

    Args:
        stage: ``input``, ``retrieval`` or ``output``.
        kind: ``pii``, ``injection``, ``ood``, ``contradiction``,
            ``classification``, ``groundedness`` or ``size``.
        action: ``allow``, ``redact``, ``block``, ``warn`` or ``clarify``.
    """
    guardrail_events.labels(stage=stage, kind=kind, action=action).inc()


def observe_tool_invocation(
    *, tool: str, kind: str, latency_ms: float, is_error: bool = False
) -> None:
    """Record one tool invocation from the agentic loop.

    Args:
        tool: Tool name.
        kind: ``retrieval``, ``rest`` or ``mcp``.
        latency_ms: Duration in milliseconds.
        is_error: Whether the tool returned an error result.
    """
    tool_invocations.labels(
        tool=tool, kind=kind, outcome="error" if is_error else "ok"
    ).inc()
    tool_latency.labels(tool=tool, kind=kind).observe(latency_ms / _MS_PER_SECOND)


def observe_ingest_run(*, trigger: str, status: str) -> None:
    """Record the completion of an ingestion run.

    Args:
        trigger: ``timer``, ``queue``, ``http``, ``manual``, ``reindex`` or
            ``upload``.
        status: ``running``, ``succeeded``, ``failed``, ``partial`` or
            ``skipped``.
    """
    ingest_runs.labels(trigger=trigger, status=status).inc()


def observe_ingest_documents(*, source_type: str, action: str, count: int = 1) -> None:
    """Record documents processed by an ingestion run.

    Args:
        source_type: ``blob``, ``sharepoint``, ``http``, ``sql``, ``upload`` or
            ``local``.
        action: ``create``, ``update``, ``delete``, ``skip`` or ``acl_only``.
        count: Number of documents.
    """
    if count:
        ingest_documents.labels(source_type=source_type, action=action).inc(count)


def render_metrics() -> bytes:
    """Render the registry in Prometheus text exposition format.

    Returns:
        The response body for ``GET /metrics``. Serve it with
        :data:`METRICS_CONTENT_TYPE`.
    """
    return generate_latest(REGISTRY)
