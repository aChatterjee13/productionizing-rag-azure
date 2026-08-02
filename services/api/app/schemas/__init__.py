"""Wire shapes owned by the HTTP layer.

Anything `docs/CONTRACTS.md` already models lives in :mod:`ragcore.models` and is
re-exported here rather than redefined, so the API and the pipeline can never
drift into two versions of the same shape.
"""

from __future__ import annotations

from app.schemas.requests import (
    ChatRequest,
    EvalRunRequest,
    FeedbackRequest,
    IngestTriggerRequest,
    MemoryConsentRequest,
    ProfileUpdateRequest,
    SearchRequest,
)
from app.schemas.responses import (
    ChatResponse,
    CompactionResponse,
    DocumentSummary,
    HealthResponse,
    LineageResponse,
    ProblemDetail,
    ReadinessResponse,
    ScheduleResponse,
    SessionSummary,
    SourceSummary,
    TenantSummary,
    UsagePayload,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CompactionResponse",
    "DocumentSummary",
    "EvalRunRequest",
    "FeedbackRequest",
    "HealthResponse",
    "IngestTriggerRequest",
    "LineageResponse",
    "MemoryConsentRequest",
    "ProblemDetail",
    "ProfileUpdateRequest",
    "ReadinessResponse",
    "ScheduleResponse",
    "SearchRequest",
    "SessionSummary",
    "SourceSummary",
    "TenantSummary",
    "UsagePayload",
]
