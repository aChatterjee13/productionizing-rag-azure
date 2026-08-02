"""``POST /chat`` — the streaming and non-streaming turn.

Streaming is a raw ``StreamingResponse`` with ``media_type="text/event-stream"``
rather than a helper library, because three things have to be right at once and a
helper hides all of them:

* **Framing.** One ``event:`` line, one ``data:`` line, one blank line. A single
  ``retry:`` hint opens the stream so a browser reconnects on a transport drop.
* **Heartbeats.** A comment line every ``api_sse_keepalive_seconds`` keeps an
  idle proxy from closing the connection while the model is still thinking. The
  heartbeat runs as its own task and is merged with the pipeline's events through
  a memory object stream, so a slow stage still produces bytes on the wire.
* **Disconnects.** Starlette's ``StreamingResponse`` already listens for
  ``http.disconnect`` and cancels the body iterator, which cancels this task
  group, which aborts the model call — so nobody pays for tokens after the reader
  has gone. The body iterator deliberately does **not** poll
  ``Request.is_disconnected``: that would consume from the same ASGI receive
  channel Starlette is already watching and could swallow the disconnect message.

The turn itself is :class:`app.rag.orchestrator.Orchestrator`; this module is
transport only and contains no pipeline logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app import api_setting
from app.deps import CurrentPrincipal, DbSession, RateLimit, SettingsDep
from app.rag.orchestrator import Orchestrator, SSEMessage, get_orchestrator
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse, UsagePayload
from ragcore.logging import get_logger
from ragcore.models.acl import Principal
from ragcore.models.chat import SSEEvent
from ragcore.settings import Settings

__all__ = ["router"]

_log = get_logger(__name__)

router = APIRouter(tags=["chat"])

#: Headers that stop an intermediary from buffering or caching the stream.
#: ``X-Accel-Buffering`` is nginx-specific and is what keeps the reverse proxy in
#: `web/nginx.conf` from batching tokens into one delivery.
_STREAM_HEADERS = {
    "cache-control": "no-cache, no-transform",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}

#: Frames buffered between the producers and the wire. Deep enough that a burst
#: of tokens does not block the pipeline, shallow enough that a stalled reader
#: applies back-pressure instead of growing memory without bound.
_STREAM_BUFFER = 64


@router.post(
    "/chat",
    summary="Ask a question",
    response_model=None,
    responses={
        200: {
            "content": {
                "text/event-stream": {},
                "application/json": {"schema": ChatResponse.model_json_schema()},
            },
            "description": (
                "An SSE stream when `stream` is true, otherwise one JSON body."
            ),
        }
    },
)
async def chat(
    body: ChatRequest,
    principal: CurrentPrincipal,
    settings: SettingsDep,
    session: DbSession,
    _limited: RateLimit,
) -> StreamingResponse | ChatResponse:
    """Run one RAG turn for the authenticated caller.

    Args:
        body: The chat request.
        principal: The authenticated caller. Every retrieval and every SQL
            statement this turn issues is scoped by its tenant.
        settings: Active settings.
        session: Database session for the non-streaming path. The streaming path
            opens its own inside the pipeline, because persistence has to outlive
            the handler for as long as the body is still being produced.
        _limited: Rate-limit gate; raises before any work is done.

    Returns:
        A ``text/event-stream`` response, or a :class:`ChatResponse` when
        ``stream`` is false.
    """
    orchestrator = get_orchestrator(settings)
    if not body.stream:
        turn = await orchestrator.run(body, principal=principal, db_session=session)
        return ChatResponse(
            session_id=turn.session_id,
            message=turn.message,
            retrieval=turn.retrieval,
            context_stats=turn.context_stats,
            guardrails=turn.guardrails,
            usage=UsagePayload(**turn.usage) if turn.usage else None,
            trace_id=turn.trace_id,
        )

    return StreamingResponse(
        _sse_body(
            orchestrator=orchestrator,
            body=body,
            principal=principal,
            settings=settings,
        ),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


async def _sse_body(
    *,
    orchestrator: Orchestrator,
    body: ChatRequest,
    principal: Principal,
    settings: Settings,
) -> AsyncIterator[str]:
    """Produce the SSE wire format for one turn.

    Args:
        orchestrator: The pipeline.
        body: The chat request.
        principal: The authenticated caller.
        settings: Active settings.

    Yields:
        Encoded SSE frames: one ``retry:`` hint, then the pipeline's events
        interleaved with heartbeat comments, ending after ``done``.
    """
    yield f"retry: {int(api_setting(settings, 'api_sse_retry_ms'))}\n\n"

    send, receive = anyio.create_memory_object_stream[str | None](
        max_buffer_size=_STREAM_BUFFER
    )
    # The task group is entered and exited in this single task, and the frames
    # are consumed in the same task that yields them, which is the one shape
    # anyio cancel scopes and async generators safely combine in.
    async with anyio.create_task_group() as group:
        group.start_soon(_pump, orchestrator, body, principal, send.clone())
        group.start_soon(
            _heartbeat,
            send.clone(),
            float(settings.api_sse_keepalive_seconds),
            str(api_setting(settings, "api_sse_heartbeat_comment")),
        )
        await send.aclose()
        async with receive:
            async for frame in receive:
                if frame is None:
                    break
                yield frame
        group.cancel_scope.cancel()


async def _pump(
    orchestrator: Orchestrator,
    body: ChatRequest,
    principal: Principal,
    send: MemoryObjectSendStream[str | None],
) -> None:
    """Run the pipeline and push each event onto the merged stream.

    Args:
        orchestrator: The pipeline.
        body: The chat request.
        principal: The authenticated caller.
        send: The merged stream's send half; closed on completion. A ``None``
            sentinel is pushed last so the consumer stops without waiting for the
            heartbeat task to end.
    """
    async with send:
        try:
            async for message in orchestrator.stream(body, principal=principal):
                await send.send(message.encode())
                if message.event == SSEEvent.DONE.value:
                    break
        except anyio.get_cancelled_exc_class():
            _log.info("chat_stream_cancelled", session_id=body.session_id)
            raise
        except Exception as exc:
            _log.exception("chat_stream_failed")
            await send.send(
                SSEMessage(
                    event=SSEEvent.ERROR.value,
                    data={"detail": "internal error", "code": type(exc).__name__},
                ).encode()
            )
        finally:
            with anyio.CancelScope(shield=True), anyio.move_on_after(1.0):
                await send.send(None)


async def _heartbeat(
    send: MemoryObjectSendStream[str | None], interval: float, comment: str
) -> None:
    """Emit an SSE comment on a timer for as long as the stream is open.

    Args:
        send: The merged stream's send half.
        interval: Seconds between heartbeats.
        comment: Comment text. A line beginning with ``:`` is a comment in the SSE
            wire format and never surfaces as an event, so an idle connection is
            kept warm without the client seeing anything.
    """
    async with send:
        while True:
            await anyio.sleep(interval)
            try:
                send.send_nowait(f": {comment}\n\n")
            except anyio.WouldBlock:
                # The consumer is already behind; another heartbeat would only
                # make that worse.
                continue
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                return
