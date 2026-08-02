"""Request bodies for the HTTP surface.

Only shapes `docs/CONTRACTS.md` leaves to this layer live here. Anything the
contract already models — :class:`~ragcore.models.chat.ChatRequest`,
:class:`~ragcore.models.retrieval.MetadataFilter` — is re-used rather than
re-declared, so there is exactly one definition of each wire shape.

**Tenant and user never come from a body.** They come from the resolved
:class:`~ragcore.models.acl.Principal`. A field that would let a client name a
tenant is not merely absent: it is refused, because ``extra="forbid"`` turns an
attempt into a 422 rather than a silently ignored parameter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ragcore.models.chat import ChatRequest
from ragcore.models.retrieval import MetadataFilter

__all__ = [
    "ChatRequest",
    "EvalRunRequest",
    "FeedbackRequest",
    "IngestTriggerRequest",
    "MemoryConsentRequest",
    "ProfileUpdateRequest",
    "SearchRequest",
]


class SearchRequest(BaseModel):
    """Body of ``POST /search`` — retrieval only, no generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="The question to retrieve for.")
    filters: MetadataFilter | None = Field(
        default=None, description="Facets to narrow by. Intersected with the ACL."
    )
    top_n: int | None = Field(
        default=None,
        gt=0,
        description="Final chunk count. Defaults to settings.retrieval_top_n.",
    )


class FeedbackRequest(BaseModel):
    """Body of ``POST /feedback``: a thumb, optionally with a comment."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None, description="Session the feedback is about."
    )
    message_id: str | None = Field(
        default=None, description="Assistant message the feedback is about."
    )
    rating: int = Field(
        description="+1 or -1. Persisted into the integer `feedback.rating` column."
    )
    comment: str | None = Field(
        default=None, description="Free text. PII-redacted before it is stored."
    )
    tags: list[str] = Field(
        default_factory=list, description="Low-cardinality labels for triage."
    )


class MemoryConsentRequest(BaseModel):
    """Body of ``PUT /memory/consent``."""

    model_config = ConfigDict(extra="forbid")

    memory_consent: bool = Field(
        description=(
            "False switches long-term memory off and soft-deletes what is stored."
        )
    )


class ProfileUpdateRequest(BaseModel):
    """Body of ``PUT /memory/profile`` (Addendum W).

    Every field is optional and only the supplied ones are written, so a client
    can change the preferred language without clobbering the model-maintained
    rolling summary.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str | None = Field(
        default=None,
        description="Rolling persona summary. Normally model-maintained.",
    )
    preferred_style: str | None = Field(
        default=None, description="Preferred answer style, e.g. 'bullet points'."
    )
    preferred_language: str | None = Field(
        default=None, description="Preferred answer language, ISO 639-1."
    )
    top_topics: list[str] | None = Field(
        default=None, description="Topics the user cares about."
    )


class EvalRunRequest(BaseModel):
    """Body of ``POST /eval/runs``."""

    model_config = ConfigDict(extra="forbid")

    golden_set_path: str | None = Field(
        default=None, description="Golden set to run. Defaults to eval_golden_path."
    )
    sample_size: int | None = Field(
        default=None, gt=0, description="Evaluate only the first N items."
    )
    notes: str | None = Field(default=None, description="Operator note for the run.")


class IngestTriggerRequest(BaseModel):
    """Body of ``POST /admin/ingest/trigger``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str | None = Field(
        default=None, description="Run exactly this source. Omit to run all of them."
    )
    force: bool = Field(
        default=False,
        description="Override the working-hours guard. Never overrides 'disabled'.",
    )
    full_scan: bool = Field(
        default=False,
        description=(
            "Clear the delta cursor so enumeration is complete and deletion "
            "detection becomes sound again."
        ),
    )
