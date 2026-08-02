"""`POST /admin/ingest/trigger` dispatches instead of crawling inline.

A full crawl can run for minutes. Running it inside the request handler pins an
API worker for the whole of it, so on Azure a single administrator pressing
"Run now" removes capacity from the chat path. The route must therefore hand the
work off — to the ingestion Function App when its HTTP trigger is configured,
and to a FastAPI background task for a local checkout — and answer **202** with
the run id either way.

The response body stays :class:`~ragcore.models.document.IngestRunSummary`-shaped
(Addendum W), in the ``running`` state, so the existing web client keeps working.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import pytest
import respx
from conftest import TENANT_A, auth_headers
from sqlalchemy.ext.asyncio import AsyncSession

FUNCTION_URL = "https://rag-ingest.azurewebsites.net/api/ingest/trigger"
URL_ENV = "RAG_INGEST_FUNCTION_URL"
KEY_ENV = "RAG_INGEST_FUNCTION_KEY"
TRIGGER_PATH = "/api/v1/admin/ingest/trigger"

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture
async def source_row(db_session: AsyncSession) -> str:
    """Give the tenant one registered source, so `src-1` is a real id.

    Returns:
        The source id the trigger tests name.
    """
    from ragcore.db.models import SourceConfigRow

    db_session.add(
        SourceConfigRow(
            source_id="src-1",
            tenant_id=TENANT_A,
            source_type="local",
            name="local corpus",
        )
    )
    await db_session.commit()
    return "src-1"


@pytest.fixture
def pipeline_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `ingestion.pipeline.run_ingest` with a recorder.

    Returns:
        A mapping that gains a ``calls`` list entry per invocation. The fake
        returns a summary whose run id is deliberately *not* the one the route
        answers with, so a test can tell "the route waited for the pipeline"
        apart from "the route dispatched and minted its own id".
    """
    from ingestion import pipeline
    from ragcore.models.document import IngestRunSummary, IngestStatus

    recorded: dict[str, Any] = {"calls": []}

    async def fake_run_ingest(**kwargs: Any) -> list[IngestRunSummary]:
        recorded["calls"].append(kwargs)
        return [
            IngestRunSummary(
                run_id="f" * 32,
                tenant_id=kwargs["tenant_id"],
                status=IngestStatus.SUCCEEDED,
            )
        ]

    monkeypatch.setattr(pipeline, "run_ingest", fake_run_ingest)
    return recorded


async def test_local_trigger_returns_202_and_runs_in_the_background(
    api: httpx.AsyncClient,
    acme_admin: Any,
    pipeline_spy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no function URL the run is handed to a background task."""
    monkeypatch.delenv(URL_ENV, raising=False)

    response = await api.post(
        TRIGGER_PATH,
        json={"source_id": None, "force": True, "full_scan": True},
        headers=auth_headers(acme_admin),
    )

    assert response.status_code == httpx.codes.ACCEPTED
    body = response.json()
    assert _HEX32.match(body["run_id"])
    assert body["status"] == "running"
    assert body["tenant_id"] == TENANT_A
    assert body["forced"] is True
    # The handler answered without waiting for the pipeline's own summary.
    assert body["run_id"] != "f" * 32

    assert len(pipeline_spy["calls"]) == 1
    call = pipeline_spy["calls"][0]
    assert call["tenant_id"] == TENANT_A
    assert call["force"] is True
    assert call["full_scan"] is True


@respx.mock
async def test_trigger_posts_to_the_function_app_when_configured(
    api: httpx.AsyncClient,
    acme_admin: Any,
    pipeline_spy: dict[str, Any],
    source_row: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured function URL takes the crawl out of the API process."""
    del source_row
    monkeypatch.setenv(URL_ENV, FUNCTION_URL)
    monkeypatch.setenv(KEY_ENV, "s3cret")
    route = respx.post(FUNCTION_URL).mock(
        return_value=httpx.Response(202, json={"id": "durable-instance-1"})
    )

    response = await api.post(
        TRIGGER_PATH,
        json={"source_id": "src-1", "force": False, "full_scan": False},
        headers=auth_headers(acme_admin),
    )

    assert response.status_code == httpx.codes.ACCEPTED
    body = response.json()
    assert _HEX32.match(body["run_id"])
    assert body["status"] == "running"
    assert body["source_id"] == "src-1"

    assert route.called
    request = route.calls.last.request
    assert request.headers["x-functions-key"] == "s3cret"
    import json as _json

    sent = _json.loads(request.content)
    assert sent["tenant_id"] == TENANT_A
    assert sent["source_id"] == "src-1"
    assert sent["wait"] is False
    assert sent["enforce_schedule"] is False

    # Nothing was crawled in this process.
    assert pipeline_spy["calls"] == []


async def test_unknown_source_is_still_a_404(
    api: httpx.AsyncClient,
    acme_admin: Any,
    pipeline_spy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatching hides the run's outcome, not a bad request.

    The old inline handler answered 404 ``no_source_matched`` when nothing ran.
    Whether the named source exists is still knowable before dispatch, so that
    error survives; nothing is queued for it.
    """
    monkeypatch.delenv(URL_ENV, raising=False)

    response = await api.post(
        TRIGGER_PATH,
        json={"source_id": "src-does-not-exist"},
        headers=auth_headers(acme_admin),
    )

    assert response.status_code == httpx.codes.NOT_FOUND
    assert response.json()["code"] == "no_source_matched"
    assert pipeline_spy["calls"] == []


@respx.mock
async def test_unreachable_function_app_is_reported_as_502(
    api: httpx.AsyncClient,
    acme_admin: Any,
    pipeline_spy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch failure is a gateway error, not a silent success."""
    monkeypatch.setenv(URL_ENV, FUNCTION_URL)
    monkeypatch.delenv(KEY_ENV, raising=False)
    respx.post(FUNCTION_URL).mock(return_value=httpx.Response(500, text="boom"))

    response = await api.post(
        TRIGGER_PATH,
        json={},
        headers=auth_headers(acme_admin),
    )

    assert response.status_code == httpx.codes.BAD_GATEWAY
    assert response.json()["code"] == "ingest_trigger_failed"
    assert pipeline_spy["calls"] == []
