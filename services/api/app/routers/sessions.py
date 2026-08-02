"""Chat session management.

Every handler here is scoped by ``(tenant_id, user_id)`` from the principal. The
session id in the path is a *filter*, never an authorisation: reaching a session
owned by another tenant raises
:class:`~ragcore.errors.TenantMismatchError` from the repository, which the
problem handler renders as 403 and which is auditable, rather than a 404 that
would quietly hide a cross-tenant probe.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, status

from app.deps import CurrentPrincipal, DbSession, PageLimit, RateLimit, SettingsDep
from app.rag.context import get_context_assembler
from app.rag.memory.short_term import get_short_term_memory
from app.schemas.responses import CompactionResponse, SessionSummary
from ragcore.db import repositories as repo
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.chat import Message

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

_SESSION_ID = Path(description="Session id, scoped to the caller's tenant.")


class SessionNotFoundError(RagError):
    """The session does not exist for this caller."""

    status_code = 404
    code = "session_not_found"


@router.get("", response_model=list[SessionSummary], summary="List sessions")
async def list_sessions(
    principal: CurrentPrincipal,
    session: DbSession,
    limit: PageLimit,
    offset: int = Query(default=0, ge=0, description="Rows to skip."),
    include_archived: bool = Query(
        default=False, description="Include archived sessions."
    ),
) -> list[SessionSummary]:
    """List the caller's sessions, newest first.

    Args:
        principal: The authenticated caller.
        session: Database session.
        limit: Page size, clamped by ``api_max_page_size``.
        offset: Rows to skip.
        include_archived: Include archived sessions.

    Returns:
        Session summaries.
    """
    rows = await repo.list_sessions(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )
    return [SessionSummary.from_row(row) for row in rows]


@router.get("/{session_id}", response_model=SessionSummary, summary="Fetch a session")
async def get_session(
    principal: CurrentPrincipal,
    session: DbSession,
    session_id: str = _SESSION_ID,
) -> SessionSummary:
    """Fetch one session.

    Args:
        principal: The authenticated caller.
        session: Database session.
        session_id: Session to fetch.

    Returns:
        The session summary.

    Raises:
        SessionNotFoundError: When no such session exists for this caller.
    """
    row = await repo.get_session_row(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=session_id,
    )
    if row is None:
        raise SessionNotFoundError("no such session")
    return SessionSummary.from_row(row)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(
    principal: CurrentPrincipal,
    session: DbSession,
    session_id: str = _SESSION_ID,
) -> None:
    """Delete a session, its messages and its cached window.

    Args:
        principal: The authenticated caller.
        session: Database session.
        session_id: Session to delete.

    Raises:
        SessionNotFoundError: When no such session exists for this caller.
    """
    deleted = await repo.delete_session(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=session_id,
    )
    if not deleted:
        raise SessionNotFoundError("no such session")
    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="session.delete",
        resource_type="chat_session",
        resource_id=session_id,
    )
    await session.commit()
    # The Redis window is a cache of the rows just removed; leaving it behind
    # would resurrect the conversation on the next turn.
    await get_short_term_memory().clear(principal=principal, session_id=session_id)


@router.get(
    "/{session_id}/messages",
    response_model=list[Message],
    summary="List a session's messages",
)
async def list_messages(
    principal: CurrentPrincipal,
    session: DbSession,
    limit: PageLimit,
    session_id: str = _SESSION_ID,
    include_suppressed: bool = Query(
        default=True,
        description="Include turns folded into the rolling summary.",
    ),
) -> list[Message]:
    """List a session's messages in chronological order.

    Args:
        principal: The authenticated caller.
        session: Database session.
        limit: Page size.
        session_id: Session to read.
        include_suppressed: Include suppressed turns, flagged as such, so the UI
            can show that context management happened rather than silently losing
            history.

    Returns:
        The messages.

    Raises:
        SessionNotFoundError: When no such session exists for this caller.
    """
    row = await repo.get_session_row(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=session_id,
    )
    if row is None:
        raise SessionNotFoundError("no such session")
    return await repo.list_session_messages(
        session,
        tenant_id=principal.tenant_id,
        session_id=session_id,
        limit=limit,
        include_suppressed=include_suppressed,
        ascending=True,
    )


@router.post(
    "/{session_id}/compact",
    response_model=CompactionResponse,
    summary="Force context compaction",
)
async def compact_session(
    principal: CurrentPrincipal,
    session: DbSession,
    settings: SettingsDep,
    _limited: RateLimit,
    session_id: str = _SESSION_ID,
) -> CompactionResponse:
    """Fold the oldest non-pinned turns into the rolling summary now.

    Compaction is normally proactive and periodic; this is the manual trigger the
    UI's context meter offers when a user wants the window trimmed before asking
    a long question.

    Args:
        principal: The authenticated caller.
        session: Database session.
        settings: Active settings.
        _limited: Rate-limit gate — summarisation is a model call.
        session_id: Session to compact.

    Returns:
        What changed, including the fresh :class:`ContextStats` so the meter can
        be redrawn without another round trip.

    Raises:
        SessionNotFoundError: When no such session exists for this caller.
    """
    row = await repo.get_session_row(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=session_id,
    )
    if row is None:
        raise SessionNotFoundError("no such session")

    short_term = get_short_term_memory(settings)
    window = await short_term.load(
        principal=principal, session_id=session_id, db_session=session
    )
    outcome = await short_term.compact(window, reason="forced", persist=True)
    suppressed = await short_term.persist_compaction(
        window, outcome, db_session=session
    )
    await session.commit()

    # Re-assemble with no new question so the meter reflects the state the next
    # turn will actually start from, rather than the pre-compaction one.
    assembler = get_context_assembler(settings)
    assembled = await assembler.assemble(
        principal=principal,
        window=window,
        question="",
        db_session=session,
    )
    await session.commit()
    _log.info(
        "session_compacted",
        tenant_id=principal.tenant_id,
        session_id=session_id,
        retired=len(outcome.retired),
        suppressed_rows=suppressed,
    )
    return CompactionResponse(
        session_id=session_id,
        messages_suppressed=len(outcome.retired),
        compaction_events=window.compaction_events,
        summary_tokens=outcome.summary_tokens,
        context_stats=assembled.stats,
    )
