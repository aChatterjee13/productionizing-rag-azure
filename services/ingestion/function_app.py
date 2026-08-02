"""Azure Functions adapter (Python v2 programming model + Durable Functions).

This file contains **no ingestion logic**. Every trigger unpacks its input, calls
:mod:`ingestion.pipeline`, and serialises the result — so the nightly Azure run, a
manual admin trigger and a local CLI invocation all execute exactly the same code.

Triggers:

``ingest_timer``
    NCRONTAB timer on ``%RAG_INGEST_CRON%`` (six fields, as the Functions host
    requires — the same environment variable ``ragcore.settings.ingest_cron`` reads).
    Refuses to run inside working hours unless ``RAG_INGEST_FORCE`` is set, recording
    a ``skipped`` run with ``skip_reason`` so the refusal is auditable rather than
    invisible.

``ingest_http``
    ``POST /api/ingest/trigger`` for manual and admin runs. ``wait=true`` runs inline
    and returns the summaries; otherwise it starts the orchestrator and returns the
    Durable status URLs.

``ingest_orchestrator``
    Durable orchestrator: fan out one activity per document (in waves of
    ``ingest_batch_size`` so parallelism stays bounded), fan in one run summary per
    source.

``ingest_blob``
    Blob trigger for immediate single-document ingestion — a document dropped in the
    container is searchable in seconds instead of waiting for the nightly pass.

``ingest_retry``
    Storage-queue trigger. Accepts either a full ``DocumentTask`` (retry one document
    inside a run) or ``{tenant_id, source_id, source_uri}`` (reindex one document).

Orchestrator determinism: the orchestrator only iterates its input and calls
activities. Every clock read, configuration read and I/O operation happens inside an
activity, and the batch size travels in the plan payload rather than being read from
the environment during replay.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from datetime import datetime
from typing import Any

import azure.durable_functions as df
import azure.functions as func

from ingestion.pipeline import (
    DocumentOutcome,
    DocumentTask,
    SourcePlan,
    finalize_run,
    guard_scheduled_run,
    ingest_single_document,
    load_source_configs,
    plan_source,
    process_document,
    run_ingest,
    run_ingestion,
    start_run,
)
from ragcore.logging import configure_logging, get_logger
from ragcore.models.document import IngestTrigger, SourceConfig
from ragcore.settings import Settings, get_settings

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

_log = get_logger("ingestion.function_app")
_configured = False

#: Activity names. Declared once so the orchestrator and the activities cannot drift.
ACTIVITY_PLAN = "ingest_plan_activity"
ACTIVITY_DOCUMENT = "ingest_document_activity"
ACTIVITY_FINALIZE = "ingest_finalize_activity"
ORCHESTRATOR = "ingest_orchestrator"


def _bootstrap() -> Settings:
    """Resolve settings and configure structured logging once per worker.

    Returns:
        The process settings singleton.
    """
    global _configured
    settings = get_settings()
    if not _configured:
        configure_logging(settings)
        _configured = True
    return settings


def _json_response(payload: Any, status_code: int = 200) -> func.HttpResponse:
    """Build a JSON HTTP response.

    Args:
        payload: JSON-serialisable body.
        status_code: HTTP status code.

    Returns:
        The response.
    """
    return func.HttpResponse(
        json.dumps(payload, default=str),
        status_code=status_code,
        mimetype="application/json",
    )


# ------------------------------------------------------------------- timer entry
@app.function_name(name="ingest_timer")
@app.timer_trigger(
    schedule="%RAG_INGEST_CRON%",
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True,
)
@app.durable_client_input(client_name="client")
async def ingest_timer(
    timer: func.TimerRequest, client: df.DurableOrchestrationClient
) -> None:
    """Nightly delta refresh across every active tenant.

    Args:
        timer: Timer metadata from the host.
        client: Durable client used to start the orchestrator.
    """
    settings = _bootstrap()
    allowed, reason = guard_scheduled_run(settings, forced=settings.ingest_force)
    if not allowed:
        # Record the refusal through the pipeline so ingest_runs shows *why* nothing
        # happened. run_ingestion with enforce_schedule=True returns a skipped summary.
        summaries = await run_ingestion(
            trigger=IngestTrigger.TIMER,
            settings=settings,
            enforce_schedule=True,
            forced=settings.ingest_force,
        )
        _log.warning(
            "timer.refused",
            reason=reason,
            past_due=bool(getattr(timer, "past_due", False)),
            runs=len(summaries),
        )
        return

    instance_id = await client.start_new(
        ORCHESTRATOR,
        client_input={
            "trigger": IngestTrigger.TIMER.value,
            "forced": settings.ingest_force or reason == "forced",
            "enforce_schedule": True,
        },
    )
    _log.info(
        "timer.started",
        instance_id=instance_id,
        reason=reason,
        past_due=bool(getattr(timer, "past_due", False)),
    )


# -------------------------------------------------------------------- http entry
@app.function_name(name="ingest_http")
@app.route(route="ingest/trigger", methods=["POST"])
@app.durable_client_input(client_name="client")
async def ingest_http(
    req: func.HttpRequest, client: df.DurableOrchestrationClient
) -> func.HttpResponse:
    """Trigger ingestion manually.

    Body (all optional): ``tenant_id``, ``source_id`` or ``source_ids`` (list),
    ``force``, ``full_scan``, ``dry_run``, ``wait``, ``enforce_schedule`` — the
    ``/admin/ingest/trigger`` body of Addendum W plus the Durable-specific flags.

    A single-tenant inline run goes through
    :func:`ingestion.pipeline.run_ingest`, which always evaluates the working-hours
    guard and reports a refusal as a skipped summary. A multi-tenant inline run uses
    :func:`ingestion.pipeline.run_ingestion`, which applies that guard only when
    ``enforce_schedule`` is true — an operator asking for a cross-tenant sweep has
    already made that decision.

    Args:
        req: The HTTP request.
        client: Durable client used to start the orchestrator.

    Returns:
        The run summaries when ``wait`` is true, otherwise the Durable status URLs.
    """
    settings = _bootstrap()
    try:
        body = req.get_json() if req.get_body() else {}
    except ValueError:
        return _json_response({"error": "body must be JSON"}, status_code=400)
    if not isinstance(body, dict):
        return _json_response({"error": "body must be a JSON object"}, status_code=400)

    tenant_id = body.get("tenant_id")
    source_ids = body.get("source_ids")
    source_id = body.get("source_id")
    if source_id and not source_ids:
        source_ids = [source_id]
    forced = bool(body.get("force", False))
    enforce_schedule = bool(body.get("enforce_schedule", False))
    full_scan = bool(body.get("full_scan", False))
    dry_run = bool(body.get("dry_run", False))

    if bool(body.get("wait", False)):
        if tenant_id:
            # Single-tenant runs go through the documented operational entry point so
            # the admin trigger, the CLI and a local run behave identically.
            summaries = await run_ingest(
                tenant_id=str(tenant_id),
                source_id=str(source_id) if source_id else None,
                trigger=IngestTrigger.HTTP,
                force=forced,
                dry_run=dry_run,
                full_scan=full_scan,
                settings=settings,
            )
        else:
            summaries = await run_ingestion(
                tenant_id=None,
                source_ids=source_ids,
                trigger=IngestTrigger.HTTP,
                forced=forced,
                settings=settings,
                enforce_schedule=enforce_schedule,
            )
        return _json_response(
            {"runs": [summary.model_dump(mode="json") for summary in summaries]}
        )

    instance_id = await client.start_new(
        ORCHESTRATOR,
        client_input={
            "tenant_id": tenant_id,
            "source_ids": source_ids,
            "trigger": IngestTrigger.HTTP.value,
            "forced": forced,
            "enforce_schedule": enforce_schedule,
        },
    )
    _log.info("http.started", instance_id=instance_id, tenant_id=tenant_id or "*")
    return client.create_check_status_response(req, instance_id)


# ----------------------------------------------------------------- orchestration
@app.function_name(name=ORCHESTRATOR)
@app.orchestration_trigger(context_name="context")
def ingest_orchestrator(
    context: df.DurableOrchestrationContext,
) -> Generator[Any, Any, dict[str, Any]]:
    """Fan out one activity per document, fan in one summary per source.

    Args:
        context: Durable orchestration context.

    Yields:
        Durable task awaitables for the plan, document and finalise activities.

    Returns:
        ``{"runs": [IngestRunSummary, ...]}``.
    """
    request = context.get_input() or {}
    envelopes = yield context.call_activity(ACTIVITY_PLAN, request)

    summaries: list[dict[str, Any]] = []
    for envelope in envelopes:
        tasks: list[dict[str, Any]] = envelope.get("tasks", [])
        batch_size = max(1, int(envelope.get("batch_size", 1)))
        outcomes: list[dict[str, Any]] = []
        for start in range(0, len(tasks), batch_size):
            wave = tasks[start : start + batch_size]
            results = yield context.task_all(
                [context.call_activity(ACTIVITY_DOCUMENT, task) for task in wave]
            )
            outcomes.extend(results)
        summary = yield context.call_activity(
            ACTIVITY_FINALIZE, {**envelope, "outcomes": outcomes}
        )
        summaries.append(summary)
    return {"runs": summaries}


# --------------------------------------------------------------------- activities
@app.function_name(name=ACTIVITY_PLAN)
@app.activity_trigger(input_name="request")
async def ingest_plan_activity(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Plan every source in scope and return one envelope per source.

    Args:
        request: ``{tenant_id?, source_ids?, trigger, forced, enforce_schedule}``.

    Returns:
        Envelopes carrying the serialised plan, its source config, the batch size and
        the run's start time. Returns an empty list when the schedule guard refuses.
    """
    settings = _bootstrap()
    trigger = IngestTrigger(request.get("trigger") or IngestTrigger.TIMER.value)
    forced = bool(request.get("forced", False))

    if bool(request.get("enforce_schedule", False)):
        allowed, reason = guard_scheduled_run(settings, forced=forced)
        if not allowed:
            _log.warning("plan_activity.refused", reason=reason)
            return []
        forced = forced or reason == "forced"

    sources = await load_source_configs(
        settings=settings,
        tenant_id=request.get("tenant_id"),
        source_ids=request.get("source_ids"),
    )
    envelopes: list[dict[str, Any]] = []
    for source in sources:
        run_id, started_at = await start_run(
            source, trigger=trigger, forced=forced, settings=settings
        )
        plan = await plan_source(
            source,
            run_id=run_id,
            settings=settings,
            trigger=trigger,
            forced=forced,
        )
        envelopes.append(
            {
                "plan": plan.model_dump(mode="json"),
                "source": source.model_dump(mode="json"),
                "tasks": [task.model_dump(mode="json") for task in plan.tasks],
                "batch_size": settings.ingest_batch_size,
                "started_at": started_at.isoformat(),
                "within_working_hours": settings.is_within_working_hours(started_at),
            }
        )
    return envelopes


@app.function_name(name=ACTIVITY_DOCUMENT)
@app.activity_trigger(input_name="task")
async def ingest_document_activity(task: dict[str, Any]) -> dict[str, Any]:
    """Process one document.

    Idempotent by construction: chunk point ids are deterministic and the document row
    is an upsert, so a Durable retry of this activity cannot duplicate anything.

    Args:
        task: Serialised :class:`~ingestion.pipeline.DocumentTask`.

    Returns:
        Serialised :class:`~ingestion.pipeline.DocumentOutcome`.
    """
    settings = _bootstrap()
    outcome = await process_document(
        DocumentTask.model_validate(task), settings=settings
    )
    return outcome.model_dump(mode="json")


@app.function_name(name=ACTIVITY_FINALIZE)
@app.activity_trigger(input_name="envelope")
async def ingest_finalize_activity(envelope: dict[str, Any]) -> dict[str, Any]:
    """Fold outcomes into the manifest and write the run summary.

    Args:
        envelope: The plan envelope plus ``outcomes``.

    Returns:
        Serialised :class:`~ragcore.models.document.IngestRunSummary`.
    """
    settings = _bootstrap()
    plan = SourcePlan.model_validate(envelope["plan"])
    source = SourceConfig.model_validate(envelope["source"])
    outcomes = [
        DocumentOutcome.model_validate(item) for item in envelope.get("outcomes", [])
    ]
    started_raw = envelope.get("started_at")
    started_at = datetime.fromisoformat(started_raw) if started_raw else None

    summary = await finalize_run(
        plan,
        outcomes,
        source=source,
        settings=settings,
        started_at=started_at,
        within_working_hours=bool(envelope.get("within_working_hours", False)),
    )
    return summary.model_dump(mode="json")


# ------------------------------------------------------------------- blob trigger
@app.function_name(name="ingest_blob")
@app.blob_trigger(
    arg_name="blob",
    path="%RAG_AZURE_BLOB_CONTAINER%/{name}",
    connection="AzureWebJobsStorage",
)
async def ingest_blob(blob: func.InputStream) -> None:
    """Index a single document as soon as it lands in the container.

    The owning source is resolved by matching the blob's container and prefix against
    the configured blob sources, so no naming convention is imposed on the container
    and a document can never be attributed to the wrong tenant by accident.

    Args:
        blob: The triggering blob.
    """
    settings = _bootstrap()
    name = (blob.name or "").split("/", 1)[-1]
    if not name or name.endswith(".acl.json"):
        return

    source = await _source_for_blob(settings, name)
    if source is None:
        _log.warning(
            "blob_trigger.no_matching_source",
            container=settings.azure_blob_container,
        )
        return

    account = (settings.azure_blob_account_url or "").rstrip("/")
    container = str(source.option("container") or settings.azure_blob_container)
    source_uri = (
        f"{account}/{container}/{name}" if account else f"blob://{container}/{name}"
    )
    try:
        summary = await ingest_single_document(
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            source_uri=source_uri,
            trigger=IngestTrigger.UPLOAD,
            settings=settings,
        )
    except Exception as exc:
        _log.error(
            "blob_trigger.failed",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            error=type(exc).__name__,
        )
        raise
    _log.info(
        "blob_trigger.indexed",
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        run_id=summary.run_id,
        status=summary.status.value,
        chunks=summary.chunks_upserted,
    )


async def _source_for_blob(settings: Settings, name: str) -> SourceConfig | None:
    """Find the blob source that owns a blob name.

    Args:
        settings: Process settings.
        name: Blob name within the triggering container.

    Returns:
        The best-matching :class:`~ragcore.models.document.SourceConfig` (longest
        matching prefix wins), or None when no configured source covers it.
    """
    sources = await load_source_configs(settings=settings)
    container = settings.azure_blob_container
    candidates = [
        source
        for source in sources
        if source.source_type.value == "blob"
        and str(source.option("container") or container) == container
        and name.startswith(str(source.option("prefix") or ""))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda source: len(str(source.option("prefix") or "")))


# ------------------------------------------------------------------ queue trigger
@app.function_name(name="ingest_retry")
@app.queue_trigger(
    arg_name="message",
    queue_name="%RAG_AZURE_STORAGE_QUEUE_NAME%",
    connection="AzureWebJobsStorage",
)
async def ingest_retry(message: func.QueueMessage) -> None:
    """Retry one document, or reindex one on demand.

    Accepted message shapes:

    * a full serialised :class:`~ingestion.pipeline.DocumentTask` — used by the
      orchestrator's poison path to retry a single document without replanning; and
    * ``{"tenant_id": ..., "source_id": ..., "source_uri": ..., "force": false}`` —
      used by ``POST /documents/{id}/reindex``.

    Args:
        message: The queue message.
    """
    settings = _bootstrap()
    try:
        payload = json.loads(message.get_body().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        _log.error("queue.unparseable_message", message_id=message.id)
        return
    if not isinstance(payload, dict):
        _log.error("queue.unexpected_payload", message_id=message.id)
        return

    if "descriptor" in payload and "source" in payload:
        outcome = await process_document(
            DocumentTask.model_validate(payload), settings=settings
        )
        _log.info(
            "queue.document_processed",
            document_id=outcome.document_id,
            action=outcome.action.value,
            status=outcome.status.value,
        )
        return

    required = ("tenant_id", "source_id", "source_uri")
    if not all(payload.get(key) for key in required):
        _log.error("queue.missing_fields", message_id=message.id)
        return
    summary = await ingest_single_document(
        tenant_id=str(payload["tenant_id"]),
        source_id=str(payload["source_id"]),
        source_uri=str(payload["source_uri"]),
        trigger=IngestTrigger.QUEUE,
        settings=settings,
        forced=bool(payload.get("force", False)),
    )
    _log.info(
        "queue.reindexed",
        tenant_id=summary.tenant_id,
        run_id=summary.run_id,
        status=summary.status.value,
    )
