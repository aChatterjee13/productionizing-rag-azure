"""Tracing, lineage and metrics.

Three independent surfaces, all designed so a failure here can never take the
request path down:

* :mod:`ragcore.observability.langfuse` -- :class:`Tracer` with contextvar
  propagation, degrading to :class:`NoopTracer` when Langfuse is disabled,
  unconfigured, missing or erroring.
* :mod:`ragcore.observability.lineage` -- :class:`LineageRecord` plus the
  tenant-scoped provenance walk behind ``GET /documents/{id}/lineage``.
* :mod:`ragcore.observability.metrics` -- Prometheus collectors and
  :func:`render_metrics` for ``GET /metrics``, no-oping when
  ``prometheus_client`` is absent.
"""

from __future__ import annotations

from ragcore.observability.langfuse import (
    LangfuseTracer,
    NoopTracer,
    SpanHandle,
    TraceHandle,
    Tracer,
    current_span_id,
    flush_tracer,
    get_current_trace_id,
    get_tracer,
    reset_tracer_cache,
    shutdown_tracer,
    traced,
)
from ragcore.observability.lineage import (
    DEFAULT_PROVENANCE_DEPTH,
    LineageKind,
    LineageRecord,
    document_provenance,
    record_lineage,
    subject_provenance,
)
from ragcore.observability.metrics import (
    METRICS_CONTENT_TYPE,
    PROMETHEUS_AVAILABLE,
    REGISTRY,
    observe_cache_lookup,
    observe_guardrail,
    observe_http_request,
    observe_ingest_documents,
    observe_ingest_run,
    observe_llm_call,
    observe_pipeline_stage,
    observe_retrieval_stage,
    observe_tool_invocation,
    render_metrics,
    set_build_info,
)

__all__ = [
    "DEFAULT_PROVENANCE_DEPTH",
    "METRICS_CONTENT_TYPE",
    "PROMETHEUS_AVAILABLE",
    "REGISTRY",
    "LangfuseTracer",
    "LineageKind",
    "LineageRecord",
    "NoopTracer",
    "SpanHandle",
    "TraceHandle",
    "Tracer",
    "current_span_id",
    "document_provenance",
    "flush_tracer",
    "get_current_trace_id",
    "get_tracer",
    "observe_cache_lookup",
    "observe_guardrail",
    "observe_http_request",
    "observe_ingest_documents",
    "observe_ingest_run",
    "observe_llm_call",
    "observe_pipeline_stage",
    "observe_retrieval_stage",
    "observe_tool_invocation",
    "record_lineage",
    "render_metrics",
    "reset_tracer_cache",
    "set_build_info",
    "shutdown_tracer",
    "subject_provenance",
    "traced",
]
