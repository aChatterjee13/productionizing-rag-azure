"""Long-term memory: profile, items and consent.

Requirement #2's user-facing half. Everything is scoped to
``(tenant_id, user_id)`` from the principal — a memory belongs to a person, and
there is no path here that reads or writes another person's.

Turning consent off is destructive on purpose: it soft-deletes the caller's stored
memories rather than merely pausing recall, because "stop remembering me" that
leaves the data in place is not consent management.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, status

from app.deps import CurrentPrincipal, DbSession, PageLimit, SettingsDep
from app.rag.memory.long_term import get_long_term_memory
from app.schemas.requests import MemoryConsentRequest, ProfileUpdateRequest
from ragcore.db import repositories as repo
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.memory import LongTermMemory, MemoryKind, UserProfile

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])

_MEMORY_ID = Path(description="Memory id owned by the caller.")


class MemoryNotFoundError(RagError):
    """The memory does not exist, or does not belong to this caller."""

    status_code = 404
    code = "memory_not_found"


@router.get("/profile", response_model=UserProfile, summary="Fetch the user profile")
async def get_profile(principal: CurrentPrincipal, session: DbSession) -> UserProfile:
    """Return the caller's rolling profile, creating it on first read.

    Args:
        principal: The authenticated caller.
        session: Database session.

    Returns:
        The profile.
    """
    profile = await repo.get_or_create_profile(
        session, tenant_id=principal.tenant_id, user_id=principal.user_id
    )
    await session.commit()
    return profile


@router.put("/profile", response_model=UserProfile, summary="Edit the user profile")
async def update_profile(
    body: ProfileUpdateRequest,
    principal: CurrentPrincipal,
    session: DbSession,
) -> UserProfile:
    """Apply the user-editable subset of the profile.

    Only the fields present in the body are written, so setting a preferred
    language cannot clobber the model-maintained rolling summary.

    Args:
        body: The requested changes.
        principal: The authenticated caller. Tenant and user come from here and
            never from the body.
        session: Database session.

    Returns:
        The updated profile.
    """
    profile = await repo.get_or_create_profile(
        session, tenant_id=principal.tenant_id, user_id=principal.user_id
    )
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    updated = profile.model_copy(update=changes) if changes else profile
    await repo.save_profile(session, updated)
    await session.commit()
    _log.info(
        "profile_updated",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        fields=sorted(changes),
    )
    return updated


@router.get(
    "/items", response_model=list[LongTermMemory], summary="List stored memories"
)
async def list_memories(
    principal: CurrentPrincipal,
    session: DbSession,
    limit: PageLimit,
    kind: MemoryKind | None = Query(default=None, description="Filter by kind."),
) -> list[LongTermMemory]:
    """List what the platform remembers about the caller.

    Args:
        principal: The authenticated caller.
        session: Database session.
        limit: Page size.
        kind: Optional kind filter.

    Returns:
        The stored memories, most salient first. The relational mirror is the
        source of truth for this listing: it is the one a user can be shown
        without a vector search, and it carries the same text the vector store
        holds.
    """
    rows = await repo.list_memories(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        kinds=[kind.value] if kind else None,
        limit=limit,
    )
    return [
        LongTermMemory(
            memory_id=row.memory_id,
            user_id=row.user_id,
            tenant_id=row.tenant_id,
            kind=MemoryKind(row.kind),
            text=row.text,
            salience=float(row.salience),
            source_session_id=row.source_session_id,
            supersedes=row.supersedes,
            hit_count=int(row.hit_count or 0),
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            expires_at=row.expires_at,
            pii_redacted=bool(row.pii_redacted),
        )
        for row in rows
    ]


@router.delete(
    "/items/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Forget one memory",
)
async def delete_memory(
    principal: CurrentPrincipal,
    session: DbSession,
    settings: SettingsDep,
    memory_id: str = _MEMORY_ID,
) -> None:
    """Delete one memory from both stores.

    Args:
        principal: The authenticated caller.
        session: Database session.
        settings: Active settings.
        memory_id: Memory to forget.

    Raises:
        MemoryNotFoundError: When the caller owns no such memory. The vector
            point is removed first and is ownership-checked there too, because a
            Qdrant point id carries no tenant.
    """
    store = get_long_term_memory(settings)
    removed_vector = await store.forget(principal, memory_id, db_session=None)
    removed_row = await repo.delete_memory(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        memory_id=memory_id,
    )
    if not (removed_vector or removed_row):
        raise MemoryNotFoundError("no such memory")
    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="memory.delete",
        resource_type="user_memory",
        resource_id=memory_id,
    )
    await session.commit()


@router.put("/consent", response_model=UserProfile, summary="Toggle long-term memory")
async def set_consent(
    body: MemoryConsentRequest,
    principal: CurrentPrincipal,
    session: DbSession,
) -> UserProfile:
    """Switch long-term memory on or off for the caller.

    Args:
        body: The desired consent state.
        principal: The authenticated caller.
        session: Database session.

    Returns:
        The updated profile. Switching consent off soft-deletes the caller's
        memories and makes stage 13 skip entirely on subsequent turns.
    """
    row = await repo.set_memory_consent(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        consent=body.memory_consent,
    )
    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="memory.consent",
        outcome="allow" if body.memory_consent else "revoke",
        resource_type="user_profile",
        resource_id=principal.user_id,
        detail={"memory_consent": body.memory_consent},
    )
    await session.commit()
    _log.info(
        "memory_consent_changed",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        consent=body.memory_consent,
    )
    return UserProfile(
        user_id=row.user_id,
        tenant_id=row.tenant_id,
        summary=row.summary or "",
        preferred_style=row.preferred_style,
        preferred_language=row.preferred_language,
        top_topics=list(row.top_topics or []),
        memory_consent=bool(row.memory_consent),
        updated_at=row.updated_at,
    )
