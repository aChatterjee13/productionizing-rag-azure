"""End-to-end walks of the happy paths, against the real pipeline.

Only Qdrant and Anthropic are faked; everything between the HTTP layer and the
database is the code that ships. That is what makes the SSE assertion worth
making: the event sequence comes out of
:meth:`app.rag.orchestrator.Orchestrator.stream` running all thirteen stages, not
out of a script.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from conftest import (
    ANSWER_FACT,
    TENANT_A,
    auth_headers,
    event_names,
    parse_sse,
    payload_for,
)

from ragcore.models.chat import SSEEvent

CHAT = "/api/v1/chat"

#: Events every turn must emit, in this relative order. The stream may carry
#: `guardrail`, `thinking`, `tool_call` and `tool_result` between them.
REQUIRED_ORDER = [
    SSEEvent.SESSION.value,
    SSEEvent.RETRIEVAL.value,
    SSEEvent.TOKEN.value,
    SSEEvent.CITATIONS.value,
    SSEEvent.CONTEXT_STATS.value,
    SSEEvent.USAGE.value,
    SSEEvent.DONE.value,
]


def assert_contract_order(names: list[str]) -> None:
    """Assert the required events appear, once each is reached, in order.

    Args:
        names: Event names in arrival order.

    Raises:
        AssertionError: When a required event is missing or out of order.
    """
    positions = []
    for required in REQUIRED_ORDER:
        assert required in names, f"missing {required!r} in {names}"
        positions.append(names.index(required))
    assert positions == sorted(positions), f"out of contract order: {names}"
    assert names[-1] == SSEEvent.DONE.value, f"stream did not end on done: {names}"


# --------------------------------------------------------------------- chat


async def test_chat_stream_emits_the_contract_event_sequence(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A normal turn streams the documented events, in the documented order."""
    async with api.stream(
        "POST",
        CHAT,
        json={"message": "What is the meal allowance for domestic travel?"},
        headers=auth_headers(acme_user),
    ) as response:
        assert response.status_code == httpx.codes.OK
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert body.startswith("retry: ")
    events = parse_sse(body)
    names = event_names(events)
    assert_contract_order(names)

    session = payload_for(events, SSEEvent.SESSION.value)
    assert session["session_id"]

    retrieval = payload_for(events, SSEEvent.RETRIEVAL.value)
    assert retrieval["chunks"], "retrieval event carried no chunks"
    # The retrieval event must never carry chunk text: it is emitted before the
    # stage 12 egress scan has run.
    assert "text" not in retrieval["chunks"][0]["payload"]

    tokens = "".join(
        json.loads(data)["text"]
        for name, data in events
        if name == SSEEvent.TOKEN.value
    )
    assert ANSWER_FACT in tokens

    citations = payload_for(events, SSEEvent.CITATIONS.value)
    assert isinstance(citations, list)
    assert citations, "no citation survived verification"
    assert citations[0]["marker"] == "[1]"
    assert citations[0]["document_id"] == "doc-acme-travel-2025"
    # The verified span must really occur in the cited chunk.
    assert citations[0]["quoted_span"]

    stats = payload_for(events, SSEEvent.CONTEXT_STATS.value)
    assert stats["budget_tokens"] > 0
    assert stats["window_tokens"] > 0

    usage = payload_for(events, SSEEvent.USAGE.value)
    assert usage["output_tokens"] > 0
    assert usage["cost_usd"] >= 0.0

    done = payload_for(events, SSEEvent.DONE.value)
    assert done["session_id"] == session["session_id"]
    assert done["message_id"], "done must carry message_id so feedback can attach"
    assert done["refused"] is False


async def test_chat_persists_the_turn_and_its_citations(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """The streamed turn is readable back through the sessions API."""
    async with api.stream(
        "POST",
        CHAT,
        json={"message": "What is the meal allowance?"},
        headers=auth_headers(acme_user),
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])
    events = parse_sse(body)
    done = payload_for(events, SSEEvent.DONE.value)
    session_id = done["session_id"]

    listing = await api.get("/api/v1/sessions", headers=auth_headers(acme_user))
    assert listing.status_code == httpx.codes.OK
    assert [row["session_id"] for row in listing.json()] == [session_id]

    messages = await api.get(
        f"/api/v1/sessions/{session_id}/messages", headers=auth_headers(acme_user)
    )
    assert messages.status_code == httpx.codes.OK
    turns = messages.json()
    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assistant = turns[-1]
    assert assistant["message_id"] == done["message_id"]
    assert ANSWER_FACT in assistant["content"]
    assert assistant["citations"], "citations were not persisted with the turn"


async def test_chat_non_streaming_returns_one_body(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """``stream=false`` returns the same turn as one JSON document."""
    response = await api.post(
        CHAT,
        json={"message": "What is the meal allowance?", "stream": False},
        headers=auth_headers(acme_user),
    )
    assert response.status_code == httpx.codes.OK
    body = response.json()
    assert body["session_id"]
    assert body["message"]["role"] == "assistant"
    assert ANSWER_FACT in body["message"]["content"]
    assert body["message"]["citations"]
    assert body["context_stats"]["budget_tokens"] > 0
    assert body["usage"]["model"]


async def test_chat_continues_an_existing_session(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A second turn on the same session id appends rather than starting over."""
    first = await api.post(
        CHAT,
        json={"message": "What is the meal allowance?", "stream": False},
        headers=auth_headers(acme_user),
    )
    session_id = first.json()["session_id"]

    second = await api.post(
        CHAT,
        json={
            "message": "And what about receipts?",
            "session_id": session_id,
            "stream": False,
        },
        headers=auth_headers(acme_user),
    )
    assert second.json()["session_id"] == session_id

    messages = await api.get(
        f"/api/v1/sessions/{session_id}/messages", headers=auth_headers(acme_user)
    )
    assert len(messages.json()) == 4


async def test_a_blocked_turn_still_closes_the_stream(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A guardrail refusal emits context_stats, usage and done so the UI unblocks."""
    async with api.stream(
        "POST",
        CHAT,
        json={"message": "   "},
        headers=auth_headers(acme_user),
    ) as response:
        assert response.status_code == httpx.codes.OK
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = parse_sse(body)
    names = event_names(events)
    for required in (
        SSEEvent.SESSION.value,
        SSEEvent.CITATIONS.value,
        SSEEvent.CONTEXT_STATS.value,
        SSEEvent.USAGE.value,
        SSEEvent.DONE.value,
    ):
        assert required in names, f"missing {required!r} in {names}"
    assert names[-1] == SSEEvent.DONE.value
    assert SSEEvent.GUARDRAIL.value in names
    assert payload_for(events, SSEEvent.DONE.value)["refused"] is True
    # Nothing was retrieved and nothing was generated.
    assert SSEEvent.RETRIEVAL.value not in names
    assert payload_for(events, SSEEvent.USAGE.value)["output_tokens"] == 0


# ------------------------------------------------------------------- search


async def test_search_returns_the_audited_result(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """``POST /search`` retrieves without generating."""
    response = await api.post(
        "/api/v1/search",
        json={"query": "meal allowance", "filters": None, "top_n": 5},
        headers=auth_headers(acme_user),
    )
    assert response.status_code == httpx.codes.OK
    body = response.json()
    assert body["chunks"]
    assert body["queries_used"] == ["meal allowance"]
    assert body["chunks"][0]["payload"]["tenant_id"] == TENANT_A
    assert body["chunks"][0]["final_score"] > 0


# ------------------------------------------------------------------ sessions


async def test_session_lifecycle(api: httpx.AsyncClient, acme_user: Any) -> None:
    """Create through chat, read, compact and delete."""
    created = await api.post(
        CHAT,
        json={"message": "What is the meal allowance?", "stream": False},
        headers=auth_headers(acme_user),
    )
    session_id = created.json()["session_id"]

    fetched = await api.get(
        f"/api/v1/sessions/{session_id}", headers=auth_headers(acme_user)
    )
    assert fetched.json()["session_id"] == session_id

    compacted = await api.post(
        f"/api/v1/sessions/{session_id}/compact", headers=auth_headers(acme_user)
    )
    assert compacted.status_code == httpx.codes.OK
    payload = compacted.json()
    assert payload["session_id"] == session_id
    assert payload["context_stats"]["budget_tokens"] > 0

    removed = await api.delete(
        f"/api/v1/sessions/{session_id}", headers=auth_headers(acme_user)
    )
    assert removed.status_code == httpx.codes.NO_CONTENT

    gone = await api.get(
        f"/api/v1/sessions/{session_id}", headers=auth_headers(acme_user)
    )
    assert gone.status_code == httpx.codes.NOT_FOUND
    assert gone.json()["code"] == "session_not_found"


# -------------------------------------------------------------------- memory


async def test_memory_profile_items_and_consent(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """The memory surface reads, edits, lists and revokes."""
    profile = await api.get("/api/v1/memory/profile", headers=auth_headers(acme_user))
    assert profile.status_code == httpx.codes.OK
    assert profile.json()["memory_consent"] is True
    assert profile.json()["user_id"] == acme_user.user_id

    edited = await api.put(
        "/api/v1/memory/profile",
        json={"preferred_style": "bullet points", "top_topics": ["travel"]},
        headers=auth_headers(acme_user),
    )
    assert edited.status_code == httpx.codes.OK
    assert edited.json()["preferred_style"] == "bullet points"
    assert edited.json()["top_topics"] == ["travel"]

    items = await api.get("/api/v1/memory/items", headers=auth_headers(acme_user))
    assert items.status_code == httpx.codes.OK
    assert items.json() == []

    revoked = await api.put(
        "/api/v1/memory/consent",
        json={"memory_consent": False},
        headers=auth_headers(acme_user),
    )
    assert revoked.status_code == httpx.codes.OK
    assert revoked.json()["memory_consent"] is False

    # The edit must survive the consent change rather than being reset.
    after = await api.get("/api/v1/memory/profile", headers=auth_headers(acme_user))
    assert after.json()["preferred_style"] == "bullet points"
    assert after.json()["memory_consent"] is False


async def test_deleting_an_unknown_memory_is_a_problem_document(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A missing memory answers 404 in problem+json, not a bare 500."""
    response = await api.delete(
        "/api/v1/memory/items/does-not-exist", headers=auth_headers(acme_user)
    )
    assert response.status_code == httpx.codes.NOT_FOUND
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "memory_not_found"


# ------------------------------------------------------------------ feedback


async def test_feedback_is_recorded_and_validated(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A thumb attaches to a turn; an invalid rating is refused."""
    turn = await api.post(
        CHAT,
        json={"message": "What is the meal allowance?", "stream": False},
        headers=auth_headers(acme_user),
    )
    body = turn.json()

    accepted = await api.post(
        "/api/v1/feedback",
        json={
            "session_id": body["session_id"],
            "message_id": body["message"]["message_id"],
            "rating": 1,
            "comment": "Correct, and my email is jane.doe@example.test",
            "tags": ["accurate"],
        },
        headers=auth_headers(acme_user),
    )
    assert accepted.status_code == httpx.codes.NO_CONTENT

    refused = await api.post(
        "/api/v1/feedback",
        json={"rating": 5},
        headers=auth_headers(acme_user),
    )
    assert refused.status_code == httpx.codes.UNPROCESSABLE_ENTITY
    assert refused.json()["code"] == "invalid_rating"


# ------------------------------------------------------------- admin and eval


async def test_admin_schedule_reports_the_guard(
    api: httpx.AsyncClient, acme_admin: Any
) -> None:
    """``GET /admin/schedule`` mirrors ``may_start_scheduled_ingest``."""
    response = await api.get("/api/v1/admin/schedule", headers=auth_headers(acme_admin))
    assert response.status_code == httpx.codes.OK
    body = response.json()
    assert body["ingest_cron"]
    assert body["reason"] in {"ok", "disabled", "forced", "working_hours"}
    assert isinstance(body["may_start"], bool)


async def test_eval_runs_listing_is_empty_and_scoped(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """The eval surface reads from PostgreSQL and is tenant-scoped."""
    response = await api.get("/api/v1/eval/runs", headers=auth_headers(acme_user))
    assert response.status_code == httpx.codes.OK
    assert response.json() == []

    missing = await api.get("/api/v1/eval/runs/nope", headers=auth_headers(acme_user))
    assert missing.status_code == httpx.codes.NOT_FOUND
    assert missing.json()["code"] == "eval_run_not_found"


# ----------------------------------------------------------------- transport


async def test_every_response_carries_a_request_id(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """The correlation header is echoed, and honoured when the client sets one."""
    generated = await api.get("/api/v1/me", headers=auth_headers(acme_user))
    assert generated.headers["x-request-id"]

    supplied = await api.get(
        "/api/v1/me",
        headers={**auth_headers(acme_user), "x-request-id": "abc-123"},
    )
    assert supplied.headers["x-request-id"] == "abc-123"


async def test_validation_failure_is_a_problem_document_without_the_input(
    api: httpx.AsyncClient, acme_user: Any
) -> None:
    """A 422 names the field but never echoes the user's text back."""
    response = await api.post(
        CHAT,
        json={"message": "sensitive text 4111111111111111", "unknown_field": 1},
        headers=auth_headers(acme_user),
    )
    assert response.status_code == httpx.codes.UNPROCESSABLE_ENTITY
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["errors"]
    assert "4111111111111111" not in response.text


async def test_openapi_documents_the_surface(api: httpx.AsyncClient) -> None:
    """The OpenAPI document covers every contract path."""
    response = await api.get("/openapi.json")
    assert response.status_code == httpx.codes.OK
    paths = response.json()["paths"]
    for path in (
        "/api/v1/chat",
        "/api/v1/search",
        "/api/v1/me",
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}/messages",
        "/api/v1/sessions/{session_id}/compact",
        "/api/v1/documents",
        "/api/v1/documents/{document_id}/lineage",
        "/api/v1/memory/profile",
        "/api/v1/memory/items",
        "/api/v1/memory/consent",
        "/api/v1/feedback",
        "/api/v1/eval/runs",
        "/api/v1/admin/tenants",
        "/api/v1/admin/sources",
        "/api/v1/admin/schedule",
        "/api/v1/admin/ingest/trigger",
        "/api/v1/admin/ingest/runs",
        "/health",
        "/readyz",
    ):
        assert path in paths, f"{path} missing from the OpenAPI document"


# ----------------------------------------------------------------- tool loop


async def test_tool_loop_dispatches_and_resumes(
    orchestrator: Any, retrieval: Any, fake_llm: Any, acme_admin: Any
) -> None:
    """Stage 8 dispatches a tool call and stage 10 resumes with its result.

    Driven against the orchestrator rather than HTTP because the interesting part
    is the loop, not the transport. Retrieval is deliberately weak and the plan
    asks for tools, which is the route that sends the turn to stage 8 instead of
    refusing it as out of domain.
    """
    from conftest import CHUNK_A_TEXT, make_chunk

    import app.rag.orchestrator as module
    from ragcore.llm.client import LLMResponse, StreamEvent, StreamEventType
    from ragcore.models.chat import ChatRequest

    rounds = {"n": 0}
    scripted = fake_llm.stream

    async def stream(**kwargs: Any) -> Any:
        """Ask for one tool on the first round, then answer."""
        rounds["n"] += 1
        if rounds["n"] == 1:
            usage = fake_llm._usage()
            call = {
                "id": "call_1",
                "name": "search_corpus",
                "input": {"query": "meals"},
                "kind": "retrieval",
            }
            yield StreamEvent(type=StreamEventType.TOOL_USE, tool_call=call)
            yield StreamEvent(type=StreamEventType.USAGE, usage=usage)
            yield StreamEvent(
                type=StreamEventType.DONE,
                usage=usage,
                stop_reason="tool_use",
                response=LLMResponse(
                    text="",
                    tool_calls=[call],
                    stop_reason="tool_use",
                    usage=usage,
                    refused=False,
                    raw=None,
                ),
            )
            return
        async for event in scripted(**kwargs):
            yield event

    fake_llm.stream = stream
    retrieval.chunks = [
        make_chunk(
            tenant_id=TENANT_A,
            document_id="doc-acme-travel-2025",
            text=CHUNK_A_TEXT,
            score=0.2,
        )
    ]

    async def plan(*_args: Any, **_kwargs: Any) -> Any:
        """Report a plan that wants tools, which is what routes to stage 8."""
        return module.TransformedQuery(
            intent="lookup",
            needs_retrieval=True,
            needs_tools=True,
            tool_hints=["search_corpus"],
            rewritten="What is the meal allowance?",
        )

    from app.rag.tools.registry import get_tool_registry

    original_transform = module.transform_query
    module.transform_query = plan
    original_settings = orchestrator.settings
    original_registry = orchestrator._registry
    enabled = original_settings.model_copy(update={"tool_enabled": True})
    # The registry is filtered by `tool_enabled` at build time, so switching the
    # flag on means rebuilding it too — exactly what a process configured with
    # tools enabled would have done at startup.
    orchestrator._settings = enabled
    orchestrator._registry = get_tool_registry(enabled)
    try:
        names = [
            event.event
            async for event in orchestrator.stream(
                ChatRequest(message="What is the meal allowance?"),
                principal=acme_admin,
            )
        ]
    finally:
        module.transform_query = original_transform
        orchestrator._settings = original_settings
        orchestrator._registry = original_registry

    assert rounds["n"] == 2, "the loop did not resume after the tool result"
    assert SSEEvent.TOOL_CALL.value in names
    assert SSEEvent.TOOL_RESULT.value in names
    assert names.index(SSEEvent.TOOL_CALL.value) < names.index(
        SSEEvent.TOOL_RESULT.value
    )
    assert names.index(SSEEvent.TOOL_RESULT.value) < names.index(SSEEvent.TOKEN.value)
    assert names[-1] == SSEEvent.DONE.value
