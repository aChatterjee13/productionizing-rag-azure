"""Administrative surface, gated on the ``rag.admin`` app role.

The role gate is applied at the **router** level rather than per handler, so
adding an endpoint here cannot accidentally ship without it. Every handler is
still tenant-scoped on top of that: an administrator administers their own
directory, not the deployment. ``GET /admin/tenants`` is the one endpoint that
could span tenants, and it deliberately returns only the caller's own row.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import (
    DbSession,
    PageLimit,
    RateLimit,
    SettingsDep,
    TenantAdmin,
    require_roles,
)
from app.schemas.requests import IngestTriggerRequest
from app.schemas.responses import ScheduleResponse, SourceSummary, TenantSummary
from ragcore.db import repositories as repo
from ragcore.db.models import IngestRun, Tenant
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.document import IngestRunSummary, IngestStatus, IngestTrigger
from ragcore.settings import Settings

__all__ = ["router"]

_log = get_logger(__name__)

#: Environment variable holding the ingestion Function App's HTTP trigger, e.g.
#: ``https://<app>.azurewebsites.net/api/ingest/trigger``. Read from the process
#: environment until ``ragcore.settings`` declares an ``ingest_function_url``
#: field, which then wins — the same "settings field beats documented default"
#: indirection the rest of this service uses. **Note for the settings owner:**
#: add ``ingest_function_url`` / ``ingest_function_key`` and this reads them
#: with no change here.
INGEST_FUNCTION_URL_ENV = "RAG_INGEST_FUNCTION_URL"

#: Optional Azure Functions host key, sent as ``x-functions-key``. Omit for an
#: anonymous trigger or when the call is authorised by network policy.
INGEST_FUNCTION_KEY_ENV = "RAG_INGEST_FUNCTION_KEY"

#: The dispatch POST returns as soon as the Durable orchestrator is started, so
#: this bounds a *handshake*, never a crawl.
DISPATCH_TIMEOUT_SECONDS = 10.0

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_roles())],
)


class IngestionUnavailableError(RagError):
    """The ingestion package is not installed in this deployment."""

    status_code = 503
    code = "ingestion_unavailable"


@router.get("/tenants", response_model=list[TenantSummary], summary="List tenants")
async def list_tenants(
    principal: TenantAdmin, session: DbSession
) -> list[TenantSummary]:
    """Return the caller's own tenant.

    Deliberately not a directory listing. A tenant administrator has no business
    enumerating the other tenants on a shared deployment, and returning a list of
    one keeps the response shape stable for the UI.

    Args:
        principal: An administrator.
        session: Database session.

    Returns:
        Zero or one tenant summary.
    """
    row = (
        (
            await session.execute(
                select(Tenant).where(Tenant.tenant_id == principal.tenant_id)
            )
        )
        .scalars()
        .first()
    )
    return [TenantSummary.from_row(row)] if row is not None else []


@router.get("/sources", response_model=list[SourceSummary], summary="List sources")
async def list_sources(
    principal: TenantAdmin,
    session: DbSession,
    enabled_only: bool = Query(default=False, description="Hide disabled sources."),
) -> list[SourceSummary]:
    """List the tenant's configured ingestion sources.

    Args:
        principal: An administrator.
        session: Database session.
        enabled_only: Hide disabled sources.

    Returns:
        Source summaries. Connector ``options`` and the delta ``cursor`` are
        omitted: an option can name a secret and a cursor can embed a Graph
        deltaLink token.
    """
    rows = await repo.list_source_configs(
        session, tenant_id=principal.tenant_id, enabled_only=enabled_only
    )
    return [SourceSummary.from_row(row) for row in rows]


@router.get(
    "/schedule", response_model=ScheduleResponse, summary="Ingestion schedule state"
)
async def get_schedule(
    principal: TenantAdmin, settings: SettingsDep
) -> ScheduleResponse:
    """Report the delta-refresh schedule and whether a run may start now.

    Args:
        principal: An administrator.
        settings: Active settings.

    Returns:
        The schedule, the working-hours verdict and the guard's reason.
    """
    del principal
    now = datetime.now(UTC)
    may_start, reason = settings.may_start_scheduled_ingest(now)
    return ScheduleResponse(
        ingest_cron=settings.ingest_cron,
        ingest_timezone=settings.ingest_timezone,
        ingest_enabled=settings.ingest_enabled,
        ingest_working_hours_start=settings.ingest_working_hours_start,
        ingest_working_hours_end=settings.ingest_working_hours_end,
        within_working_hours=settings.is_within_working_hours(now),
        may_start=may_start,
        reason=reason,
        next_run_at=_next_run_at(settings, now),
    )


@router.post(
    "/ingest/trigger",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a delta refresh",
)
async def trigger_ingest(
    body: IngestTriggerRequest,
    principal: TenantAdmin,
    session: DbSession,
    settings: SettingsDep,
    background: BackgroundTasks,
    _limited: RateLimit,
) -> IngestRunSummary:
    """Accept an ingestion run for the caller's tenant and dispatch it.

    A crawl can run for minutes. Running it inside the request handler pins an
    API worker for its whole duration, so one administrator pressing "Run now"
    would take capacity away from the chat path. The work is therefore handed
    off and the response is immediate:

    * **Deployed** — when :data:`INGEST_FUNCTION_URL_ENV` names the ingestion
      Function App's HTTP trigger, the request is forwarded there with
      ``wait=false`` so the Durable orchestrator owns the run. Only the
      handshake is awaited.
    * **Local** — otherwise the pipeline is imported in-process and driven by a
      FastAPI background task, which starts after this response is written.

    The returned ``run_id`` correlates the dispatch in the audit log and the
    service logs; the per-source rows the run itself writes appear in
    ``GET /admin/ingest/runs`` as it progresses.

    Args:
        body: Which source, whether to force past the working-hours guard, and
            whether to clear the delta cursor first.
        principal: An administrator.
        session: Database session, used for source validation and the audit row.
        settings: Active settings.
        background: FastAPI's background-task queue, used by the local path.
        _limited: Rate-limit gate — a full scan is expensive.

    Returns:
        A ``running`` :class:`~ragcore.models.document.IngestRunSummary` carrying
        the run id, returned with **202 Accepted**.

    Raises:
        IngestionUnavailableError: When no function URL is configured and the
            ingestion package is absent, so there is nothing to dispatch to.
        RagError: 404 when a named source does not exist for this tenant; 502
            when the Function App could not be reached.
    """
    await _require_known_source(session, principal.tenant_id, body.source_id)

    now = datetime.now(UTC)
    run_id = repo.new_id()
    function_url = _function_setting(settings, "ingest_function_url")

    if function_url:
        instance_id = await _dispatch_to_function(
            url=function_url,
            key=_function_setting(settings, "ingest_function_key"),
            tenant_id=principal.tenant_id,
            body=body,
        )
        dispatch = "function"
    else:
        pipeline = _ingestion_pipeline()
        instance_id = None
        dispatch = "background"
        background.add_task(
            _run_in_background,
            pipeline=pipeline,
            run_id=run_id,
            tenant_id=principal.tenant_id,
            body=body,
            settings=settings,
        )

    await repo.write_audit(
        session,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        action="ingest.trigger",
        resource_type="source_config",
        resource_id=body.source_id,
        detail={
            "force": body.force,
            "full_scan": body.full_scan,
            "dispatch": dispatch,
            "run_id": run_id,
        },
    )
    await session.commit()
    _log.info(
        "ingest_dispatched",
        tenant_id=principal.tenant_id,
        source_id=body.source_id,
        run_id=run_id,
        dispatch=dispatch,
        instance_id=instance_id,
    )
    return IngestRunSummary(
        run_id=run_id,
        tenant_id=principal.tenant_id,
        source_id=body.source_id,
        trigger=IngestTrigger.MANUAL,
        status=IngestStatus.RUNNING,
        started_at=now,
        forced=body.force,
        within_working_hours=settings.is_within_working_hours(now),
    )


@router.get("/ingest/runs", summary="Recent ingestion runs")
async def list_ingest_runs(
    principal: TenantAdmin,
    session: DbSession,
    limit: PageLimit,
    source_id: str | None = Query(default=None, description="Filter by source."),
) -> list[dict[str, Any]]:
    """List the tenant's recent ingestion runs, newest first.

    Args:
        principal: An administrator.
        session: Database session.
        limit: Page size.
        source_id: Optional source filter.

    Returns:
        Run rows projected into the contract's ``IngestRunSummary`` shape.
    """
    statement = (
        select(IngestRun)
        .where(IngestRun.tenant_id == principal.tenant_id)
        .order_by(IngestRun.started_at.desc())
        .limit(limit)
    )
    if source_id:
        statement = statement.where(IngestRun.source_id == source_id)
    rows = (await session.execute(statement)).scalars().all()
    return [_run_payload(row) for row in rows]


# ------------------------------------------------------------------- helpers


def _function_setting(settings: Settings, name: str) -> str:
    """Resolve a Function App connection value.

    A real ``ragcore.settings`` field wins when it exists; the process
    environment supplies it until then. Same indirection as
    :func:`app.api_setting`, kept local because the tunable belongs to the
    ingestion deployment rather than to the HTTP layer.

    Args:
        settings: Active settings.
        name: ``"ingest_function_url"`` or ``"ingest_function_key"``.

    Returns:
        The configured value, stripped, or ``""`` when unset.
    """
    env_var = {
        "ingest_function_url": INGEST_FUNCTION_URL_ENV,
        "ingest_function_key": INGEST_FUNCTION_KEY_ENV,
    }[name]
    value = getattr(settings, name, None)
    if value is None:
        value = os.environ.get(env_var, "")
    return str(value).strip()


async def _require_known_source(
    session: AsyncSession, tenant_id: str, source_id: str | None
) -> None:
    """Reject a named source the tenant does not own.

    Dispatching is asynchronous, so the run's outcome is no longer knowable in
    the response — but whether the request names a real source still is, and
    that was the old handler's 404.

    Args:
        session: Database session.
        tenant_id: Caller's tenant.
        source_id: Requested source, or None for an all-sources sweep.

    Raises:
        RagError: 404 when the source does not exist for this tenant.
    """
    if source_id is None:
        return
    rows = await repo.list_source_configs(
        session, tenant_id=tenant_id, enabled_only=False
    )
    if any(row.source_id == source_id for row in rows):
        return
    raise RagError(
        "no ingestion source matched the request",
        code="no_source_matched",
        status_code=404,
    )


async def _dispatch_to_function(
    *, url: str, key: str, tenant_id: str, body: IngestTriggerRequest
) -> str | None:
    """Hand the run to the ingestion Function App's HTTP trigger.

    ``wait=false`` makes the function start the Durable orchestrator and answer
    with its status URLs, so this call costs a handshake rather than a crawl.
    ``enforce_schedule=false`` because an administrator pressing "Run now" has
    already made the working-hours decision; ``force`` still travels so the
    guard inside the pipeline behaves identically on both paths.

    Args:
        url: The trigger URL.
        key: Azure Functions host key, or ``""`` for an anonymous trigger.
        tenant_id: Tenant the run belongs to.
        body: The validated request.

    Returns:
        The Durable instance id when the function reports one, else None.

    Raises:
        RagError: 502 when the function is unreachable or answers non-2xx.
    """
    payload = {
        "tenant_id": tenant_id,
        "source_id": body.source_id,
        "force": body.force,
        "full_scan": body.full_scan,
        "wait": False,
        "enforce_schedule": False,
    }
    headers = {"x-functions-key": key} if key else {}
    try:
        async with httpx.AsyncClient(timeout=DISPATCH_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        _log.exception("ingest_dispatch_failed", tenant_id=tenant_id)
        raise RagError(
            "the ingestion run could not be started",
            code="ingest_trigger_failed",
            status_code=502,
            detail={"error": type(exc).__name__},
        ) from exc

    try:
        reported = response.json()
    except ValueError:
        return None
    return str(reported.get("id")) if isinstance(reported, dict) else None


async def _run_in_background(
    *,
    pipeline: Any,
    run_id: str,
    tenant_id: str,
    body: IngestTriggerRequest,
    settings: Settings,
) -> None:
    """Drive the pipeline after the response has been written.

    This is the local-development path. Failures are logged rather than raised:
    the client already has its 202, and the run's own row in ``ingest_runs``
    carries the outcome.

    Args:
        pipeline: The :mod:`ingestion.pipeline` module.
        run_id: Correlation id reported to the caller.
        tenant_id: Tenant the run belongs to.
        body: The validated request.
        settings: Active settings.
    """
    try:
        summaries = await pipeline.run_ingest(
            tenant_id=tenant_id,
            source_id=body.source_id,
            trigger=IngestTrigger.MANUAL,
            force=body.force,
            full_scan=body.full_scan,
            settings=settings,
        )
    except Exception:
        _log.exception("ingest_background_failed", tenant_id=tenant_id, run_id=run_id)
        return
    _log.info(
        "ingest_background_finished",
        tenant_id=tenant_id,
        run_id=run_id,
        runs=len(summaries),
    )


def _ingestion_pipeline() -> Any:
    """Import the ingestion pipeline lazily.

    Returns:
        The :mod:`ingestion.pipeline` module.

    Raises:
        IngestionUnavailableError: When ``rag-ingestion`` is not installed.
    """
    try:
        from ingestion import pipeline
    except ImportError as exc:
        _log.warning("ingestion_package_missing")
        raise IngestionUnavailableError(
            "ingestion runs in the ingestion service, which is not installed in "
            "this deployment; trigger it there or install the 'rag-ingestion' "
            "workspace package"
        ) from exc
    return pipeline


def _run_payload(row: IngestRun) -> dict[str, Any]:
    """Project an ``ingest_runs`` row onto the wire shape.

    Args:
        row: The row.

    Returns:
        A JSON-serialisable mapping matching
        :class:`~ragcore.models.document.IngestRunSummary`.
    """
    return {
        "run_id": row.run_id,
        "tenant_id": row.tenant_id,
        "source_id": row.source_id,
        "trigger": row.trigger,
        "status": row.status,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "documents_seen": int(row.documents_seen or 0),
        "documents_created": int(row.documents_created or 0),
        "documents_updated": int(row.documents_updated or 0),
        "documents_deleted": int(row.documents_deleted or 0),
        "documents_skipped": int(row.documents_skipped or 0),
        "documents_failed": int(row.documents_failed or 0),
        "chunks_upserted": int(row.chunks_upserted or 0),
        "chunks_deleted": int(row.chunks_deleted or 0),
        "tokens_embedded": int(row.tokens_embedded or 0),
        "duplicates_dropped": int(row.duplicates_dropped or 0),
        "pii_documents": int(row.pii_documents or 0),
        "forced": bool(row.forced),
        "within_working_hours": bool(row.within_working_hours),
        "skip_reason": row.skip_reason,
        "error_message": row.error_message,
        "metrics": dict(row.metrics or {}),
    }


def _next_run_at(settings: Settings, now: datetime) -> datetime | None:
    """Compute the next scheduled ingestion time from the NCRONTAB expression.

    Only the common shape is interpreted — fixed second, minute and hour with
    wildcard day fields, which is what ``ingest_cron``'s default and every
    realistic override look like. Anything else returns None rather than a wrong
    answer, and the UI then simply shows the cron string.

    Args:
        settings: Active settings.
        now: Reference moment.

    Returns:
        The next run time in the configured timezone, or None when the expression
        is not a simple daily schedule.
    """
    fields = settings.ingest_cron.split()
    expected_fields = 6
    if len(fields) != expected_fields:
        return None
    second, minute, hour, day, month, weekday = fields
    if not {day, month, weekday} <= {"*"}:
        return None
    try:
        parts = (int(second), int(minute), int(hour))
    except ValueError:
        return None

    tz = ZoneInfo(settings.ingest_timezone)
    local = now.astimezone(tz)
    candidate = local.replace(
        hour=parts[2], minute=parts[1], second=parts[0], microsecond=0
    )
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate
