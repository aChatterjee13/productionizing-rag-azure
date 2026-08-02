"""``POST /search`` and ``GET /me``.

Search is retrieval without generation: the same stage 5 the chat pipeline runs,
so an operator debugging a bad answer sees exactly the candidate set the model
saw — including the audited drops, which is what makes a recall problem
distinguishable from a ranking problem.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import CurrentPrincipal, RateLimit, SettingsDep
from app.rag.retriever import retrieve
from app.schemas.requests import SearchRequest
from ragcore.logging import get_logger
from ragcore.models.acl import Principal
from ragcore.models.retrieval import RetrievalResult
from ragcore.observability import get_tracer

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(tags=["search"])


@router.get("/me", response_model=Principal, summary="Echo the resolved principal")
async def me(principal: CurrentPrincipal) -> Principal:
    """Return the caller's resolved identity.

    The web app uses this to render the clearance badge and to decide which admin
    controls to show, so it echoes exactly what the API will enforce rather than
    what the token happened to contain.

    Args:
        principal: The authenticated caller.

    Returns:
        The principal, including the derived clearance ceiling.
    """
    return principal


@router.post(
    "/search", response_model=RetrievalResult, summary="Retrieve without generating"
)
async def search(
    body: SearchRequest,
    principal: CurrentPrincipal,
    settings: SettingsDep,
    _limited: RateLimit,
) -> RetrievalResult:
    """Run stage 5 for one query.

    Args:
        body: The search request.
        principal: The authenticated caller. Every Qdrant request this makes is
            scoped by ``build_acl_filter`` derived from this principal — there is
            no unscoped search path.
        settings: Active settings.
        _limited: Rate-limit gate.

    Returns:
        The full :class:`~ragcore.models.retrieval.RetrievalResult`: ordered
        chunks, the probes issued, the serialised filter, the three stage counters
        and every dropped candidate with its reason.
    """
    tracer = get_tracer(settings)
    async with tracer.trace(
        "search",
        user_id=principal.user_id,
        tenant_id=principal.tenant_id,
        tags=["search"],
        metadata={"query_chars": len(body.query), "top_n": body.top_n},
    ):
        result = await retrieve(
            principal,
            [body.query],
            body.filters,
            top_n=body.top_n,
            settings=settings,
        )
    _log.info(
        "search_completed",
        tenant_id=principal.tenant_id,
        chunks=len(result.chunks),
        dropped=len(result.dropped),
        max_score=round(result.max_score, 4),
    )
    return result
