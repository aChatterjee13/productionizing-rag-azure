"""Liveness, readiness and Prometheus metrics.

Mounted **outside** the versioned prefix and outside authentication: a container
orchestrator's probe must not depend on the identity provider being reachable, and
a scrape must not need a token. None of the three endpoints touches tenant data.
"""

from __future__ import annotations

from typing import Any

import anyio
from fastapi import APIRouter, Response

from app import SERVICE_VERSION, api_setting
from app.deps import SettingsDep
from app.schemas.responses import HealthResponse, ReadinessResponse
from ragcore.db import check_database
from ragcore.logging import get_logger
from ragcore.observability import METRICS_CONTENT_TYPE, render_metrics
from ragcore.vectorstore.client import check_qdrant

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is up.

    Deliberately probes nothing: a liveness check that fails because a downstream
    dependency is slow gets the container killed and makes an outage worse.

    Args:
        settings: Active settings.

    Returns:
        The service identity and environment.
    """
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=SERVICE_VERSION,
        env=settings.env,
    )


@router.get("/livez", response_model=HealthResponse, include_in_schema=False)
async def livez(settings: SettingsDep) -> HealthResponse:
    """Alias of :func:`health` for orchestrators that expect ``/livez``.

    Args:
        settings: Active settings.

    Returns:
        The same body as ``GET /health``.
    """
    return await health(settings)


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(settings: SettingsDep, response: Response) -> ReadinessResponse:
    """Probe every dependency the request path needs.

    Args:
        settings: Active settings.
        response: The response, whose status is set to 503 when a probe fails.

    Returns:
        One boolean per dependency. Probes run concurrently and each is bounded by
        ``api_readiness_timeout_seconds``, so a hung dependency cannot hang the
        probe itself.
    """
    timeout = float(api_setting(settings, "api_readiness_timeout_seconds"))
    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    async def probe(name: str, coroutine_factory: Any) -> None:
        """Run one dependency probe under a timeout.

        Args:
            name: Dependency name used as the result key.
            coroutine_factory: Zero-argument callable returning the probe
                coroutine.
        """
        try:
            with anyio.fail_after(timeout):
                checks[name] = bool(await coroutine_factory())
        except TimeoutError:
            checks[name] = False
            detail[name] = "timeout"
        except Exception as exc:
            checks[name] = False
            detail[name] = type(exc).__name__

    async with anyio.create_task_group() as group:
        group.start_soon(probe, "qdrant", lambda: check_qdrant(settings=settings))
        group.start_soon(probe, "database", lambda: check_database(settings))

    ready = all(checks.values())
    if not ready:
        response.status_code = 503
        _log.warning("readiness_failed", checks=checks, detail=detail)
    return ReadinessResponse(
        status="ok" if ready else "degraded", checks=checks, detail=detail
    )


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics(settings: SettingsDep) -> Response:
    """Serve the Prometheus text exposition.

    Args:
        settings: Active settings.

    Returns:
        The rendered metrics, or 404 when ``api_metrics_enabled`` is off.
    """
    if not settings.api_metrics_enabled:
        return Response(status_code=404)
    return Response(content=render_metrics(), media_type=METRICS_CONTENT_TYPE)
