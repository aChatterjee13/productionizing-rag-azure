"""``POST /feedback`` — a thumb, optionally with a comment.

The comment is free text a user typed, so it goes through the PII detector before
it is written: ``repositories.write_feedback`` refuses content that has not, and
that refusal is a feature — it makes "we forgot to redact" a failed write rather
than a quiet leak into the feedback table.

The same rating is mirrored into Langfuse as a score, so a thumbs-down lands next
to the trace that produced the answer.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.deps import CurrentPrincipal, DbSession, RateLimit, SettingsDep
from app.schemas.requests import FeedbackRequest
from ragcore.db import repositories as repo
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.observability import get_tracer
from ragcore.pii import get_pii_detector

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(tags=["feedback"])

#: Ratings the contract allows. Anything else is a client bug, not a preference.
_ALLOWED_RATINGS = frozenset({1, -1})


class InvalidRatingError(RagError):
    """The rating was not +1 or -1."""

    status_code = 422
    code = "invalid_rating"


@router.post(
    "/feedback", status_code=status.HTTP_204_NO_CONTENT, summary="Rate an answer"
)
async def submit_feedback(
    body: FeedbackRequest,
    principal: CurrentPrincipal,
    session: DbSession,
    settings: SettingsDep,
    _limited: RateLimit,
) -> None:
    """Record a rating against a turn.

    Args:
        body: The feedback.
        principal: The authenticated caller. Tenant and user come from here; a
            client cannot file feedback on another tenant's session.
        session: Database session.
        settings: Active settings.
        _limited: Rate-limit gate.

    Raises:
        InvalidRatingError: When the rating is not +1 or -1.
        RagError: 404 when the named session does not belong to the caller.
    """
    if body.rating not in _ALLOWED_RATINGS:
        raise InvalidRatingError("rating must be +1 or -1")

    if body.session_id:
        # Verifies ownership; the repository raises TenantMismatchError when the
        # session exists under another tenant, which is auditable.
        row = await repo.get_session_row(
            session,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            session_id=body.session_id,
        )
        if row is None:
            raise RagError("no such session", code="session_not_found", status_code=404)

    comment = body.comment or ""
    redacted: str | None = None
    if comment.strip():
        detector = get_pii_detector(settings)
        redacted, _report = detector.scan_and_redact(comment)

    trace_id = get_tracer(settings).current_trace_id()
    await repo.write_feedback(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        rating=body.rating,
        pii_redacted=True,
        session_id=body.session_id,
        message_id=body.message_id,
        comment=redacted,
        tags=body.tags,
        trace_id=trace_id,
    )
    await session.commit()

    # The score is attached to the *answer's* trace when the client sent one, not
    # to this request's trace, so it lands beside the generation it grades.
    get_tracer(settings).score(
        name="user_feedback",
        value=float(body.rating),
        comment=redacted,
        trace_id=body.message_id or trace_id,
    )
    _log.info(
        "feedback_recorded",
        tenant_id=principal.tenant_id,
        rating=body.rating,
        has_comment=bool(redacted),
    )
