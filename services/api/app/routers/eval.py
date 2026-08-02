"""Evaluation runs (requirement #8).

The harness itself lives in ``services/eval``, a separate workspace member that is
not a dependency of this service. This router therefore does two things:

* **Reads** ``eval_runs``/``eval_results`` straight from PostgreSQL, tenant-scoped
  — which works in every deployment, because the harness writes there through the
  same repositories.
* **Starts** a run by importing the harness lazily, answering 503 with an
  actionable code when it is not installed rather than pretending the endpoint
  does not exist.

``GET /eval/runs/{id}`` includes each result's ``category``, so a dashboard can
show pass/fail per golden-item category and an ``acl_negative`` regression is
distinguishable from a quality dip.

A full golden-set run is minutes of model calls, so ``POST /eval/runs`` **starts**
one and answers immediately with the run id it allocated. The client then polls
``GET /eval/runs/{id}``, which is why that route reads from PostgreSQL: the row the
harness writes when it finishes is the same row the poll finds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Path, status
from sqlalchemy import select

from app.deps import CurrentPrincipal, DbSession, PageLimit, SettingsDep, TenantAdmin
from app.schemas.requests import EvalRunRequest
from ragcore.db.models import EvalResultRow, EvalRunRow
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.settings import Settings

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])

_RUN_ID = Path(description="Evaluation run id, scoped to the caller's tenant.")


class EvalRunNotFoundError(RagError):
    """No such evaluation run for this tenant."""

    status_code = 404
    code = "eval_run_not_found"


class EvalUnavailableError(RagError):
    """The evaluation harness is not installed in this deployment."""

    status_code = 503
    code = "eval_unavailable"


class EvalDisabledError(RagError):
    """Evaluation runs are switched off by configuration."""

    status_code = 409
    code = "eval_disabled"


@router.get("/runs", summary="List evaluation runs")
async def list_runs(
    principal: CurrentPrincipal, session: DbSession, limit: PageLimit
) -> list[dict[str, Any]]:
    """List the tenant's evaluation runs, newest first.

    Args:
        principal: The authenticated caller.
        session: Database session.
        limit: Page size.

    Returns:
        Run summaries without their per-item results.
    """
    statement = (
        select(EvalRunRow)
        .where(EvalRunRow.tenant_id == principal.tenant_id)
        .order_by(EvalRunRow.started_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(statement)).scalars().all()
    return [_run_payload(row) for row in rows]


@router.get("/runs/{run_id}", summary="Fetch one evaluation run")
async def get_run(
    principal: CurrentPrincipal,
    session: DbSession,
    run_id: str = _RUN_ID,
) -> dict[str, Any]:
    """Fetch a run with every per-item result.

    Args:
        principal: The authenticated caller.
        session: Database session.
        run_id: Run to fetch.

    Returns:
        The run payload, including ``results``.

    Raises:
        EvalRunNotFoundError: When the caller's tenant owns no such run.
    """
    statement = (
        select(EvalRunRow)
        .where(EvalRunRow.run_id == run_id)
        .where(EvalRunRow.tenant_id == principal.tenant_id)
    )
    row = (await session.execute(statement)).scalars().first()
    if row is None:
        raise EvalRunNotFoundError("no such evaluation run")

    results = (
        (
            await session.execute(
                select(EvalResultRow)
                .where(EvalResultRow.run_id == run_id)
                .where(EvalResultRow.tenant_id == principal.tenant_id)
                .order_by(EvalResultRow.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    payload = _run_payload(row)
    payload["results"] = [_result_payload(result) for result in results]
    return payload


@router.post("/runs", status_code=status.HTTP_201_CREATED, summary="Start a run")
async def start_run(
    body: EvalRunRequest,
    principal: TenantAdmin,
    session: DbSession,
    settings: SettingsDep,
    background: BackgroundTasks,
) -> dict[str, Any]:
    """Start an evaluation run against the golden set.

    The run is executed in the background and this returns as soon as its id is
    allocated: a golden set is dozens of full pipeline turns, and holding an HTTP
    request open for it would time out at every proxy in between. Poll
    ``GET /eval/runs/{run_id}`` for the results, which the harness writes as it
    finishes.

    Args:
        body: Which golden set, how many items, and an operator note.
        principal: An administrator of the tenant being evaluated.
        session: Database session.
        settings: Active settings.
        background: FastAPI's background-task queue, which runs the harness once
            this response has been sent.

    Returns:
        The started run: its id and start time, with empty results until it
        finishes.

    Raises:
        EvalDisabledError: When ``eval_enabled`` is off.
        EvalUnavailableError: When the harness package is not installed.
    """
    if not settings.eval_enabled:
        raise EvalDisabledError("evaluation runs are disabled by configuration")
    del session
    harness = _harness()
    run_id = harness.new_run_id()
    background.add_task(
        _execute_run,
        harness=harness,
        run_id=run_id,
        tenant_id=principal.tenant_id,
        golden_set_path=body.golden_set_path or settings.eval_golden_path,
        sample_size=body.sample_size or settings.eval_sample_size,
        notes=body.notes,
        settings=settings,
    )
    _log.info(
        "eval_run_started",
        tenant_id=principal.tenant_id,
        run_id=run_id,
        golden_set_path=body.golden_set_path or settings.eval_golden_path,
        sample_size=body.sample_size or settings.eval_sample_size,
    )
    return {
        "run_id": run_id,
        "tenant_id": principal.tenant_id,
        "started_at": datetime.now(UTC),
        "finished_at": None,
        "git_sha": None,
        "config_fingerprint": "",
        "golden_set_path": body.golden_set_path or settings.eval_golden_path,
        "aggregate": {},
        "gate_passed": False,
        "item_count": 0,
        "passed_count": 0,
        "failed_count": 0,
        "total_cost_usd": 0.0,
        "notes": body.notes,
        "results": [],
    }


async def _execute_run(
    *,
    harness: Any,
    run_id: str,
    tenant_id: str,
    golden_set_path: str,
    sample_size: int | None,
    notes: str | None,
    settings: Settings,
) -> None:
    """Run the golden set after the response has been sent.

    A failure here is logged, never raised: there is no request left to answer, and
    the absence of the run's rows is what the poller sees.

    Args:
        harness: The imported ``eval.harness`` module.
        run_id: Id already handed to the caller; the harness files its rows under it.
        tenant_id: Tenant being evaluated.
        golden_set_path: Golden file to run.
        sample_size: Item cap, or None for the whole set.
        notes: Operator note stored with the run.
        settings: Active settings.
    """
    try:
        run = await harness.run_evaluation(
            tenant_id=tenant_id,
            golden_set_path=golden_set_path,
            sample_size=sample_size,
            notes=notes,
            settings=settings,
            run_id=run_id,
        )
    except Exception:
        _log.exception("eval_run_failed", tenant_id=tenant_id, run_id=run_id)
        return
    _log.info(
        "eval_run_completed",
        tenant_id=tenant_id,
        run_id=getattr(run, "run_id", run_id),
        gate_passed=getattr(run, "gate_passed", None),
        item_count=getattr(run, "item_count", None),
    )


#: Entry points ``eval.harness`` must expose for this router to drive it.
_HARNESS_ENTRY_POINTS = ("run_evaluation", "new_run_id")


def _harness() -> Any:
    """Import the evaluation harness lazily.

    Returns:
        The harness module exposing :data:`_HARNESS_ENTRY_POINTS`.

    Raises:
        EvalUnavailableError: When ``services/eval`` is not installed here, or
            does not expose the entry points.
    """
    try:
        from eval import harness
    except ImportError as exc:
        _log.warning("eval_package_missing")
        raise EvalUnavailableError(
            "the evaluation harness is not installed in this deployment; run it "
            "from services/eval or install the workspace package"
        ) from exc
    missing = [name for name in _HARNESS_ENTRY_POINTS if not hasattr(harness, name)]
    if missing:
        raise EvalUnavailableError(
            "the installed evaluation harness exposes no "
            f"{' / '.join(missing)} entry point"
        )
    return harness


def _run_payload(row: EvalRunRow) -> dict[str, Any]:
    """Project an ``eval_runs`` row.

    Args:
        row: The row.

    Returns:
        A JSON-serialisable mapping in the shape the dashboard expects.
    """
    return {
        "run_id": row.run_id,
        "tenant_id": row.tenant_id,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "git_sha": row.git_sha,
        "config_fingerprint": row.config_fingerprint,
        "golden_set_path": row.golden_set_path,
        "aggregate": dict(row.aggregate or {}),
        "gate_passed": bool(row.gate_passed),
        "item_count": int(row.item_count or 0),
        "passed_count": int(row.passed_count or 0),
        "failed_count": int(row.failed_count or 0),
        "total_cost_usd": float(row.total_cost_usd or 0.0),
        "notes": row.notes,
        "results": [],
    }


def _result_payload(row: EvalResultRow) -> dict[str, Any]:
    """Project an ``eval_results`` row.

    Args:
        row: The row.

    Returns:
        A JSON-serialisable mapping including ``category``.
    """
    return {
        "result_id": row.result_id,
        "run_id": row.run_id,
        "item_id": row.item_id,
        "category": row.category,
        "question": row.question,
        "answer": row.answer,
        "retrieved_chunk_ids": list(row.retrieved_chunk_ids or []),
        "scores": dict(row.scores or {}),
        "passed": bool(row.passed),
        "failures": list(row.failures or []),
        "latency_ms": float(row.latency_ms or 0.0),
        "cost_usd": float(row.cost_usd or 0.0),
        "trace_id": row.trace_id,
    }
