"""The 13 ordered pipeline stages, wired end to end.

`docs/CONTRACTS.md` specifies the stages; this module is the only place that runs
them, and both entry points share one code path:

* :meth:`Orchestrator.run` returns a :class:`ChatTurn` — the non-streaming
  ``POST /chat`` body.
* :meth:`Orchestrator.stream` yields :class:`SSEMessage` values in the contract's
  documented order — the streaming ``POST /chat`` body.

Both drive :meth:`Orchestrator._pipeline`, an async generator that emits every SSE
event and mutates a :class:`TurnState`. ``run`` drains it and reads the state;
``stream`` forwards the events. There is no second implementation of a stage, so a
behaviour that only appears in one mode is impossible by construction.

Two invariants the rest of the service depends on:

* **Every exit emits ``context_stats``, ``usage`` and ``done``.** A guardrail
  block, an out-of-domain refusal and a semantic-cache hit all short-circuit
  earlier stages, and all three still close the stream properly, so a UI waiting
  for ``done`` never hangs.
* **A non-essential stage never fails the turn.** Memory, the semantic cache,
  contradiction detection, memory write-back, persistence and tracing are wrapped
  by :meth:`Orchestrator._stage`, which converts a failure into a
  :class:`~ragcore.models.chat.GuardrailEvent` warning on the ``guardrail`` event
  and carries on. Retrieval and generation are essential and propagate.

Ordering note. The contract numbers the tool loop as stage 8 and generation as
stage 10, but an agentic loop cannot run before the prompt exists. Stage 8 here is
*planning*: the tool surface (``ToolDispatcher.plan``) and the route
(``decide_route``) are decided before assembly, and the tool **executions**
interleave with stage 10's streamed generation, each dispatch inside its own
``tool.<name>`` Langfuse span. Nothing else about the order changes.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.citations import (
    CitationReport,
    append_uncertainty_notice,
    extract_citations,
    format_marker,
)
from app.rag.context import AssembledContext, ContextAssembler, get_context_assembler
from app.rag.guardrails.contradiction import (
    ContradictionReport,
    check_contradictions,
    render_contradiction_notes,
)
from app.rag.guardrails.injection import scan_retrieved
from app.rag.guardrails.input_guard import InputDecision, run_input_guard
from app.rag.guardrails.ood import run_ood_gate
from app.rag.guardrails.output_guard import OutputDecision, run_output_guard
from app.rag.memory.long_term import LongTermMemoryStore, get_long_term_memory
from app.rag.memory.semantic_cache import SemanticCache, get_semantic_cache
from app.rag.memory.short_term import (
    SessionWindow,
    ShortTermMemory,
    get_short_term_memory,
)
from app.rag.query_transform import TransformedQuery, merge_filters, transform_query
from app.rag.retriever import retrieve, retrieve_by_ids
from app.rag.tools.registry import ToolRegistry, get_tool_registry, tool_config
from app.rag.tools.router import (
    LoopGuard,
    RouteDecision,
    ToolContext,
    ToolDispatcher,
    ToolPlan,
    decide_route,
)
from ragcore.db import repositories as repo
from ragcore.db import session_scope
from ragcore.errors import RagError
from ragcore.llm import LLMClient, StreamEventType, get_llm_client
from ragcore.llm.prompts import prompt_metadata
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.models.chat import (
    ChatRequest,
    ContextStats,
    GuardrailAction,
    GuardrailEvent,
    GuardrailKind,
    GuardrailStage,
    Message,
    Role,
    SSEEvent,
    ToolCall,
)
from ragcore.models.memory import UserProfile
from ragcore.models.retrieval import (
    Citation,
    MetadataFilter,
    RetrievalResult,
    RetrievedChunk,
)
from ragcore.models.tool import ToolResult
from ragcore.observability import get_tracer, observe_guardrail, observe_pipeline_stage
from ragcore.observability.langfuse import Tracer
from ragcore.pii import PIIDetector, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "STAGES",
    "ChatTurn",
    "Orchestrator",
    "SSEMessage",
    "TurnState",
    "get_orchestrator",
]

_log = get_logger(__name__)
_MS = 1000.0

#: The contract's ordered stage names, used as Langfuse span names and as the
#: ``stage`` label on ``rag_pipeline_stage_duration_seconds``.
STAGES: tuple[str, ...] = (
    "input_guard",
    "memory_load",
    "query_transform",
    "cache_probe",
    "retrieve",
    "ood_gate",
    "contradiction",
    "tool_plan",
    "context_assembly",
    "generate",
    "citations",
    "output_guard",
    "memory_write_back",
)


# --------------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class SSEMessage:
    """One semantic event on the chat stream.

    Attributes:
        event: An :class:`~ragcore.models.chat.SSEEvent` value.
        data: JSON-serialisable payload.
    """

    event: str
    data: Any

    def encode(self) -> str:
        """Render as an SSE frame.

        Returns:
            ``event:``/``data:`` lines terminated by a blank line. The payload is
            serialised without embedded newlines, so one frame is always one
            ``data:`` line and no client has to reassemble.
        """
        payload = json.dumps(self.data, default=str, ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n"


def _event(name: SSEEvent, data: Any) -> SSEMessage:
    """Build an :class:`SSEMessage`.

    Args:
        name: The event name.
        data: The payload.

    Returns:
        The message.
    """
    return SSEMessage(event=name.value, data=data)


# ---------------------------------------------------------------------- state


@dataclass(slots=True)
class TurnState:
    """Everything one turn accumulates, shared by both entry points."""

    principal: Principal
    session_id: str
    message_id: str
    started_at: float = field(default_factory=time.perf_counter)
    trace_id: str | None = None
    title: str = ""
    guardrails: list[GuardrailEvent] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    sources: list[RetrievedChunk] = field(default_factory=list)
    context_stats: ContextStats | None = None
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    stored_answer: str = ""
    pii_types: list[str] = field(default_factory=list)
    stop_reason: str = ""
    refused: bool = False
    model: str = ""
    pending: list[SSEMessage] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        """Wall-clock time since the turn started.

        Returns:
            Milliseconds elapsed.
        """
        return (time.perf_counter() - self.started_at) * _MS

    @property
    def cost_usd(self) -> float:
        """Cost accumulated across every model call this turn.

        Returns:
            USD, zero when no call was made.
        """
        return float(self.usage.get("cost_usd", 0.0))

    def guardrail(self, event: GuardrailEvent) -> None:
        """Record a guardrail decision and queue it for the stream.

        Args:
            event: The decision.
        """
        self.guardrails.append(event)
        observe_guardrail(stage=event.stage, kind=event.kind, action=event.action)
        self.pending.append(_event(SSEEvent.GUARDRAIL, event.model_dump(mode="json")))

    def drain(self) -> list[SSEMessage]:
        """Take everything queued since the last drain.

        Returns:
            The queued messages, oldest first.
        """
        queued = list(self.pending)
        self.pending.clear()
        return queued


@dataclass(slots=True)
class ChatTurn:
    """The finished turn — the ``stream=false`` response body."""

    session_id: str
    message: Message
    retrieval: RetrievalResult | None = None
    context_stats: ContextStats | None = None
    guardrails: list[GuardrailEvent] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    trace_id: str | None = None


# --------------------------------------------------------------- orchestrator


class Orchestrator:
    """Runs the ordered pipeline for one chat turn."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        tracer: Tracer | None = None,
        short_term: ShortTermMemory | None = None,
        assembler: ContextAssembler | None = None,
        long_term: LongTermMemoryStore | None = None,
        cache: SemanticCache | None = None,
        registry: ToolRegistry | None = None,
        dispatcher: ToolDispatcher | None = None,
        detector: PIIDetector | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Every collaborator is injectable so a test can drive the real pipeline
        against fakes rather than re-implementing it.

        Args:
            settings: Active settings. Defaults to the process settings.
            llm: Model client. Defaults to the cached one.
            tracer: Langfuse tracer. Defaults to the cached one, which is a working
                no-op when tracing is off.
            short_term: Session-window façade.
            assembler: Context assembler (stage 9).
            long_term: Long-term memory store (stages 2 and 13).
            cache: Semantic cache (stage 4).
            registry: Tool registry (stage 8).
            dispatcher: Tool dispatcher (stage 8).
            detector: PII detector, shared by stages 1 and 12.
        """
        self._settings = settings or get_settings()
        self._llm = llm or get_llm_client(self._settings)
        self._tracer = tracer or get_tracer(self._settings)
        self._short_term = short_term or get_short_term_memory(self._settings)
        self._assembler = assembler or get_context_assembler(self._settings)
        self._long_term = long_term or get_long_term_memory(self._settings)
        self._cache = cache or get_semantic_cache(self._settings)
        self._registry = registry or get_tool_registry(self._settings)
        self._dispatcher = dispatcher or ToolDispatcher(settings=self._settings)
        self._detector = detector or get_pii_detector(self._settings)

    @property
    def settings(self) -> Settings:
        """Settings this orchestrator was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def short_term(self) -> ShortTermMemory:
        """The session-window façade this turn uses.

        Returns:
            The bound façade, so ``POST /sessions/{id}/compact`` and the eval
            harness operate on the same window as a live turn.
        """
        return self._short_term

    @property
    def assembler(self) -> ContextAssembler:
        """The context assembler this turn uses.

        Returns:
            The bound assembler, sharing one token-count memo with the pipeline.
        """
        return self._assembler

    # ------------------------------------------------------------ entry points
    async def run(
        self,
        request: ChatRequest,
        *,
        principal: Principal,
        db_session: AsyncSession | None = None,
    ) -> ChatTurn:
        """Run one turn and return the finished answer.

        Args:
            request: The validated chat request.
            principal: The authenticated caller. Every retrieval and every SQL
                statement this turn issues is scoped by its tenant.
            db_session: Database session to persist through. One is opened for the
                duration of the turn when omitted.

        Returns:
            The :class:`ChatTurn`.

        Raises:
            RagError: When an essential stage fails. The HTTP layer maps it.
        """
        state = self._new_state(request, principal)
        async for _ in self._pipeline(
            request=request, principal=principal, state=state, db_session=db_session
        ):
            pass
        return self._finish(state)

    async def stream(
        self,
        request: ChatRequest,
        *,
        principal: Principal,
        db_session: AsyncSession | None = None,
    ) -> AsyncIterator[SSEMessage]:
        """Run one turn, yielding the contract's SSE events in order.

        Args:
            request: The validated chat request.
            principal: The authenticated caller.
            db_session: Database session to persist through.

        Yields:
            :class:`SSEMessage` values: ``session`` first, then ``retrieval``,
            ``guardrail``, ``thinking``, ``tool_call``, ``tool_result``, ``token``
            and ``citations`` as they happen, and always ``context_stats``,
            ``usage`` and ``done`` last. An essential failure is reported as an
            ``error`` event followed by ``done`` rather than propagating, because a
            half-open stream has no status code to carry it.
        """
        state = self._new_state(request, principal)
        try:
            async for message in self._pipeline(
                request=request, principal=principal, state=state, db_session=db_session
            ):
                yield message
        except RagError as exc:
            _log.warning(
                "chat_turn_failed",
                code=exc.code,
                status=exc.status_code,
                session_id=state.session_id,
            )
            yield _event(SSEEvent.ERROR, {"detail": exc.message, "code": exc.code})
            yield self._done_event(state)
        except Exception as exc:
            _log.exception("chat_turn_crashed", session_id=state.session_id)
            yield _event(
                SSEEvent.ERROR,
                {"detail": "internal error", "code": type(exc).__name__},
            )
            yield self._done_event(state)

    # ------------------------------------------------------------------ helpers
    def _new_state(self, request: ChatRequest, principal: Principal) -> TurnState:
        """Create the mutable state for one turn.

        Args:
            request: The chat request.
            principal: The caller.

        Returns:
            A fresh :class:`TurnState` with ids allocated up front, so the
            ``session`` event can be emitted before anything else runs.
        """
        return TurnState(
            principal=principal,
            session_id=request.session_id or repo.new_id(),
            message_id=repo.new_id(),
        )

    def _finish(self, state: TurnState) -> ChatTurn:
        """Project the accumulated state onto the response body.

        Args:
            state: The turn state.

        Returns:
            The :class:`ChatTurn`.
        """
        message = Message(
            message_id=state.message_id,
            session_id=state.session_id,
            role=Role.ASSISTANT,
            content=state.answer,
            citations=list(state.citations),
            tool_calls=list(state.tool_calls),
        )
        return ChatTurn(
            session_id=state.session_id,
            message=message,
            retrieval=state.retrieval,
            context_stats=state.context_stats,
            guardrails=list(state.guardrails),
            usage=dict(state.usage) if state.usage else None,
            trace_id=state.trace_id,
        )

    def _done_event(self, state: TurnState) -> SSEMessage:
        """Build the terminal ``done`` event.

        Args:
            state: The turn state.

        Returns:
            The message. ``message_id`` is always present so the UI can attach
            feedback to the persisted turn.
        """
        return _event(
            SSEEvent.DONE,
            {
                "session_id": state.session_id,
                "message_id": state.message_id,
                "stop_reason": state.stop_reason or "end_turn",
                "refused": state.refused,
                "trace_id": state.trace_id,
            },
        )

    @contextmanager
    def _stage(self, name: str, state: TurnState, *, essential: bool) -> Iterator[None]:
        """Time a stage, record its metric, and contain a non-essential failure.

        Args:
            name: Stage name from :data:`STAGES`.
            state: The turn state, used to report a degradation.
            essential: Whether a failure must abort the turn.

        Yields:
            None. The body runs inside the timer.

        Raises:
            Exception: Re-raised for an essential stage.
        """
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            observe_pipeline_stage(
                stage=name, latency_ms=(time.perf_counter() - started) * _MS
            )
            if essential:
                raise
            # The message names the exception class, never its text: an exception
            # raised while handling user content can quote that content, and this
            # value is streamed to the client and written to `chat_messages`.
            _log.warning(
                "pipeline_stage_degraded",
                stage=name,
                error=type(exc).__name__,
                session_id=state.session_id,
                tenant_id=state.principal.tenant_id,
            )
            state.guardrail(
                GuardrailEvent(
                    stage=GuardrailStage.RETRIEVAL.value,
                    kind=GuardrailKind.GROUNDEDNESS.value,
                    action=GuardrailAction.WARN.value,
                    detail=f"stage {name} degraded ({type(exc).__name__})",
                )
            )
            return
        observe_pipeline_stage(
            stage=name, latency_ms=(time.perf_counter() - started) * _MS
        )

    @asynccontextmanager
    async def _database(
        self, db_session: AsyncSession | None, state: TurnState
    ) -> AsyncIterator[AsyncSession | None]:
        """Provide the turn's database session.

        Args:
            db_session: A caller-supplied session, used as-is when present.
            state: The turn state, used to report an unreachable database.

        Yields:
            The session, or None when the database could not be reached. Every
            persistence call in this module tolerates None, so a database outage
            degrades a turn to "answered but not recorded" rather than failing it.
        """
        if db_session is not None:
            yield db_session
            return
        try:
            async with session_scope(self._settings) as session:
                yield session
        except Exception as exc:
            _log.error("database_unavailable", error=type(exc).__name__)
            state.guardrail(
                GuardrailEvent(
                    stage=GuardrailStage.OUTPUT.value,
                    kind=GuardrailKind.GROUNDEDNESS.value,
                    action=GuardrailAction.WARN.value,
                    detail=(
                        "conversation storage is unavailable; this turn was not saved"
                    ),
                )
            )
            yield None

    # ------------------------------------------------------------- the pipeline
    async def _pipeline(
        self,
        *,
        request: ChatRequest,
        principal: Principal,
        state: TurnState,
        db_session: AsyncSession | None,
    ) -> AsyncIterator[SSEMessage]:
        """Run every stage, emitting events as they happen.

        Args:
            request: The chat request.
            principal: The caller.
            state: Mutable turn state, populated as stages complete.
            db_session: Caller-supplied database session, or None.

        Yields:
            :class:`SSEMessage` values in the contract's order.

        Raises:
            RagError: From an essential stage.
        """
        cfg = self._settings
        async with (
            self._tracer.trace(
                "chat.turn",
                user_id=principal.user_id,
                session_id=state.session_id,
                tenant_id=principal.tenant_id,
                tags=["chat"],
                metadata={
                    "allow_tools": request.allow_tools,
                    "stream": request.stream,
                    "has_filters": request.filters is not None,
                    "message_chars": len(request.message),
                },
            ) as trace,
            self._database(db_session, state) as session,
        ):
            state.trace_id = trace.trace_id
            for message in state.drain():
                yield message
            yield _event(SSEEvent.SESSION, {"session_id": state.session_id})

            # ---------------------------------------------- stage 1: input guard
            guard = InputDecision(text=request.message, redacted_text=request.message)
            with self._stage("input_guard", state, essential=True):
                guard = await run_input_guard(
                    request.message,
                    principal=principal,
                    settings=cfg,
                    detector=self._detector,
                )
            for event in guard.events:
                state.guardrail(event)
            state.pii_types = list(guard.pii_types)
            state.title = _title_for(guard.redacted_text)
            for message in state.drain():
                yield message

            await self._ensure_session(session, state)
            await self._persist_user_turn(session, state, guard)

            if guard.blocked:
                async for message in self._short_circuit(
                    session, state, answer=guard.refusal, stop_reason="guardrail"
                ):
                    yield message
                return

            # ---------------------------------------------- stage 2: memory load
            window = SessionWindow.empty(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                session_id=state.session_id,
            )
            profile: UserProfile | None = None
            memories: list[Any] = []
            with self._stage("memory_load", state, essential=False):
                async with self._tracer.span("memory_load"):
                    window = await self._short_term.load(
                        principal=principal,
                        session_id=state.session_id,
                        db_session=session,
                    )
                    profile = await self._long_term.resolve_profile(
                        principal, db_session=session
                    )
                    recall = await self._long_term.recall(
                        principal,
                        guard.text,
                        profile=profile,
                        db_session=session,
                    )
                    memories = list(recall.memories)
            for message in state.drain():
                yield message

            # ------------------------------------------- stage 3: query transform
            plan = TransformedQuery(
                intent="answer",
                rewritten=guard.text,
                degraded=True,
                degraded_reason="not_run",
            )
            with self._stage("query_transform", state, essential=False):
                async with self._tracer.span("query_transform"):
                    plan = await transform_query(
                        guard.text,
                        history=window.history_pairs(cfg.qt_history_turns),
                        settings=cfg,
                        llm=self._llm,
                    )
            question = plan.rewritten or guard.text
            filters = merge_filters(request.filters, plan.metadata_filter)
            for message in state.drain():
                yield message

            # ------------------------------- stages 4 and 5: cache probe, retrieve
            result = await self._retrieve(
                state=state,
                principal=principal,
                guard=guard,
                plan=plan,
                filters=filters,
                session=session,
            )
            state.retrieval = result
            for message in state.drain():
                yield message
            yield _event(SSEEvent.RETRIEVAL, result.without_text())

            # ------------------------------------------------ stage 6: OOD gate
            tool_available = bool(
                request.allow_tools
                and cfg.tool_enabled
                and self._registry.tools_for(principal)
            )
            with self._stage("ood_gate", state, essential=False):
                async with self._tracer.span("ood_gate"):
                    verdict = await run_ood_gate(
                        question=question,
                        result=result,
                        principal=principal,
                        transformed=plan,
                        tool_available=tool_available,
                        settings=cfg,
                        llm=self._llm,
                    )
                for event in verdict.events:
                    state.guardrail(event)
                if verdict.is_out_of_domain:
                    for message in state.drain():
                        yield message
                    async for message in self._short_circuit(
                        session,
                        state,
                        answer=verdict.refusal,
                        stop_reason="out_of_domain",
                    ):
                        yield message
                    return
            for message in state.drain():
                yield message

            # -------------------------------------------- stage 7: contradiction
            report = ContradictionReport()
            with self._stage("contradiction", state, essential=False):
                async with self._tracer.span("contradiction"):
                    report = await check_contradictions(
                        result.chunks,
                        question=question,
                        markers=_markers_for(result.chunks),
                        settings=cfg,
                        llm=self._llm,
                    )
                for event in report.events:
                    state.guardrail(event)
            for message in state.drain():
                yield message

            # ------------------------------------------------ stage 8: tool plan
            ctx = ToolContext.build(
                principal,
                settings=cfg,
                registry=self._registry,
                session=session,
                session_id=state.session_id,
                message_id=state.message_id,
                base_filter=filters,
                retrieve=_retriever_for(cfg),
                context_classification=_highest_classification(result.chunks),
                tracer=self._tracer,
                detector=self._detector,
                guard=LoopGuard.from_config(tool_config(cfg)),
            )
            tool_plan = ToolPlan(tools=[], connector=_empty_connector(), exposed={})
            route = decide_route(
                plan, tools=(), retrieval=result, allow_tools=False, settings=cfg
            )
            if tool_available:
                with self._stage("tool_plan", state, essential=False):
                    async with self._tracer.span("tool_plan"):
                        tool_plan = await self._dispatcher.plan(ctx)
                        route = decide_route(
                            plan,
                            tools=list(ctx.exposed.values()),
                            retrieval=result,
                            allow_tools=request.allow_tools,
                            settings=cfg,
                        )
            tool_kwargs: dict[str, Any] = {}
            if route.use_tools:
                tool_kwargs = _tool_request_kwargs(tool_plan)
            for message in state.drain():
                yield message

            # ------------------------------------------ stage 9: context assembly
            assembled = await self._assemble(
                state=state,
                principal=principal,
                window=window,
                question=question,
                chunks=result.chunks,
                memories=memories,
                profile=profile,
                report=report,
                tools=tool_kwargs.get("tools"),
                session=session,
            )
            state.sources = list(assembled.sources)
            for chunk in assembled.dropped:
                result.dropped.append(chunk)
            for message in state.drain():
                yield message

            # ------------------------ stages 8/10: generation with the tool loop
            async for message in self._generate(
                state=state,
                ctx=ctx,
                assembled=assembled,
                route=route,
                tool_kwargs=tool_kwargs,
            ):
                yield message

            # ------------------------------------------------ stage 11: citations
            citation_report = CitationReport(cleaned_answer=state.answer)
            with self._stage("citations", state, essential=False):
                async with self._tracer.span("citations"):
                    citation_report = extract_citations(
                        state.answer, assembled.sources, settings=cfg
                    )
            answer = citation_report.cleaned_answer or state.answer
            answer = append_uncertainty_notice(answer, citation_report, settings=cfg)
            for message in state.drain():
                yield message

            # --------------------------------------------- stage 12: output guard
            output = OutputDecision(text=answer, redacted_text=answer)
            with self._stage("output_guard", state, essential=True):
                async with self._tracer.span("output_guard"):
                    output = await run_output_guard(
                        answer=answer,
                        citations=citation_report.citations,
                        chunks=assembled.sources,
                        principal=principal,
                        settings=cfg,
                        detector=self._detector,
                        groundedness=citation_report.citation_validity,
                    )
            for event in output.events:
                state.guardrail(event)
            state.answer = output.text
            state.stored_answer = output.redacted_text
            state.citations = list(output.citations)
            state.pii_types = sorted({*state.pii_types, *output.pii_types})
            if output.blocked:
                state.refused = True
                state.stop_reason = state.stop_reason or "guardrail"
            for message in state.drain():
                yield message
            yield _event(
                SSEEvent.CITATIONS,
                [citation.model_dump(mode="json") for citation in state.citations],
            )

            self._score(state, citation_report, output, result)
            await self._persist_assistant_turn(session, state, window=window)

            # ------------------------------------------ stage 13: memory write-back
            with self._stage("memory_write_back", state, essential=False):
                async with self._tracer.span("memory_write_back"):
                    await self._long_term.write_back(
                        principal=principal,
                        session_id=state.session_id,
                        user_text=guard.redacted_text,
                        assistant_text=state.stored_answer,
                        profile=profile,
                        db_session=session,
                    )
            for message in state.drain():
                yield message

            yield _event(
                SSEEvent.CONTEXT_STATS,
                (state.context_stats or _empty_stats(cfg)).model_dump(mode="json"),
            )
            yield _event(SSEEvent.USAGE, state.usage or _empty_usage(cfg))
            yield self._done_event(state)

    # -------------------------------------------------------- short-circuiting
    async def _short_circuit(
        self,
        session: AsyncSession | None,
        state: TurnState,
        *,
        answer: str,
        stop_reason: str,
    ) -> AsyncIterator[SSEMessage]:
        """Close a turn that never reached generation.

        A guardrail block and an out-of-domain refusal both produce a complete,
        user-facing answer without a model call. The answer is still streamed as
        ``token`` events so the UI renders it the same way, and ``citations``,
        ``context_stats``, ``usage`` and ``done`` are still emitted so nothing
        waiting on the stream hangs.

        Args:
            session: Database session, or None.
            state: The turn state.
            answer: The refusal text. Operator-authored, so it carries no user
                content and is safe to persist verbatim.
            stop_reason: Value for the ``done`` event.

        Yields:
            The closing events.
        """
        state.answer = answer
        state.stored_answer = answer
        state.refused = True
        state.stop_reason = stop_reason
        if answer:
            yield _event(SSEEvent.TOKEN, {"text": answer})
        yield _event(SSEEvent.CITATIONS, [])
        await self._persist_assistant_turn(session, state, window=None)
        self._tracer.score(name="refused", value=1.0, comment=stop_reason)
        yield _event(
            SSEEvent.CONTEXT_STATS,
            (state.context_stats or _empty_stats(self._settings)).model_dump(
                mode="json"
            ),
        )
        yield _event(SSEEvent.USAGE, state.usage or _empty_usage(self._settings))
        yield self._done_event(state)

    # ---------------------------------------------------- stages 4 and 5
    async def _retrieve(
        self,
        *,
        state: TurnState,
        principal: Principal,
        guard: InputDecision,
        plan: TransformedQuery,
        filters: MetadataFilter | None,
        session: AsyncSession | None,
    ) -> RetrievalResult:
        """Probe the semantic cache, then retrieve, then scan for injection.

        Args:
            state: The turn state.
            principal: The caller.
            guard: Stage 1's decision, supplying the prompt-safe query text.
            plan: Stage 3's plan.
            filters: The merged metadata filter.
            session: Database session, or None.

        Returns:
            The retrieval result. On a usable cache hit the chunks come from the
            cache's ACL re-fetch and ``cache_hit`` is True; otherwise a full hybrid
            search runs and the plan is written back to the cache.

        Raises:
            RetrievalError: When every probe failed against Qdrant.
        """
        cfg = self._settings
        empty = RetrievalResult(
            chunks=[], queries_used=list(plan.queries), filter_applied={}
        )
        if not plan.needs_retrieval:
            return empty

        # -------------------------------------------- stage 4: semantic cache
        probe = None
        with self._stage("cache_probe", state, essential=False):
            async with self._tracer.span("cache_probe"):
                probe = await self._cache.probe(
                    principal,
                    guard.text,
                    extra=filters,
                    db_session=session,
                    resolver=_cache_resolver(cfg),
                )

        result: RetrievalResult
        if probe is not None and probe.usable:
            entry_ids = probe.entry.chunk_ids if probe.entry else []
            result = RetrievalResult(
                chunks=list(probe.chunks),
                queries_used=list(probe.transformed_queries or plan.queries),
                filter_applied={"fingerprint": probe.fingerprint},
                total_candidates=len(entry_ids),
                after_dedupe=len(entry_ids),
                after_rerank=len(probe.chunks),
                latency_ms={"cache": probe.latency_ms, "total": probe.latency_ms},
                cache_hit=True,
            )
            _log.info(
                "semantic_cache_hit",
                tenant_id=principal.tenant_id,
                chunks=len(probe.chunks),
                revoked=len(probe.revoked_chunk_ids),
            )
        else:
            with self._stage("retrieve", state, essential=True):
                async with self._tracer.span("retrieve"):
                    result = await retrieve(
                        principal,
                        plan.queries,
                        filters,
                        hyde_passage=plan.hyde_passage,
                        rerank_query=plan.rewritten or guard.text,
                        settings=cfg,
                    )
            with self._stage("cache_probe", state, essential=False):
                await self._cache.store(
                    principal,
                    guard.text,
                    chunk_ids=result.chunk_ids,
                    transformed_queries=result.queries_used,
                    extra=filters,
                    db_session=session,
                )

        # ------------------------- stage 5 tail: indirect-injection quarantine
        with self._stage("retrieve", state, essential=False):
            scan = await scan_retrieved(result.chunks, settings=cfg, llm=self._llm)
            result.chunks = list(scan.kept)
            result.dropped.extend(scan.quarantined)
            for event in scan.events:
                state.guardrail(event)
        return result

    # ------------------------------------------------------------- stage 9
    async def _assemble(
        self,
        *,
        state: TurnState,
        principal: Principal,
        window: SessionWindow,
        question: str,
        chunks: Sequence[RetrievedChunk],
        memories: Sequence[Any],
        profile: UserProfile | None,
        report: ContradictionReport,
        tools: Sequence[Mapping[str, Any]] | None,
        session: AsyncSession | None,
    ) -> AssembledContext:
        """Pack the prompt to the measured budget.

        When stage 7 found a conflict and packing dropped one of the chunks it
        cited, the notes are re-rendered against the surviving order and the
        assembly is repeated once. Without that, the note would cite a marker
        number that now points at a different source — which is worse than not
        surfacing the conflict at all.

        Args:
            state: The turn state.
            principal: The caller.
            window: The live session window, mutated in place by suppression.
            question: The context-resolved question.
            chunks: Retrieved chunks, best first.
            memories: Long-term memories to inject.
            profile: The caller's rolling profile.
            report: Stage 7's contradiction report.
            tools: Tool definitions, which consume prompt tokens too.
            session: Database session, or None.

        Returns:
            The assembled context.

        Raises:
            RagError: Assembly is essential; a failure here means there is no
                prompt to send.
        """
        assembled: AssembledContext | None = None
        with self._stage("context_assembly", state, essential=True):
            async with self._tracer.span("context_assembly"):
                assembled = await self._assembler.assemble(
                    principal=principal,
                    window=window,
                    question=question,
                    chunks=chunks,
                    memories=memories,
                    profile=profile,
                    notes=report.notes,
                    tools=tools,
                    db_session=session,
                )
                if report.notes and _order_changed(chunks, assembled.sources):
                    realigned = _realign_notes(report, assembled.sources)
                    if realigned != report.notes:
                        assembled = await self._assembler.assemble(
                            principal=principal,
                            window=window,
                            question=question,
                            chunks=assembled.sources,
                            memories=memories,
                            profile=profile,
                            notes=realigned,
                            tools=tools,
                            db_session=session,
                        )
        if assembled is None:  # pragma: no cover - _stage re-raises when essential
            msg = "context assembly produced no prompt"
            raise RagError(msg, code="context_assembly_failed")
        state.context_stats = assembled.stats
        return assembled

    # ------------------------------------------------------ stages 8 and 10
    async def _generate(
        self,
        *,
        state: TurnState,
        ctx: ToolContext,
        assembled: AssembledContext,
        route: RouteDecision,
        tool_kwargs: Mapping[str, Any],
    ) -> AsyncIterator[SSEMessage]:
        """Stream the answer, dispatching tools until the model stops asking.

        Args:
            state: The turn state; text, usage and tool calls accumulate here.
            ctx: The turn's tool context.
            assembled: Stage 9's payload.
            route: Stage 8's routing decision, bounding the iterations.
            tool_kwargs: ``tools``/``mcp_servers`` for the model call.

        Yields:
            ``thinking``, ``token``, ``tool_call`` and ``tool_result`` events.

        Raises:
            RagError: Generation is essential.
        """
        cfg = self._settings
        request_kwargs = assembled.request_kwargs()
        messages: list[Any] = list(request_kwargs.pop("messages"))
        max_iterations = route.max_iterations if route.use_tools else 0
        metadata = {
            **prompt_metadata("answer"),
            "route": route.mode.value,
            "sources": len(assembled.sources),
        }

        with self._stage("generate", state, essential=True):
            for iteration in range(max_iterations + 1):
                pending: list[dict[str, Any]] = []
                response = None
                async with self._tracer.span(
                    "generate", metadata={**metadata, "iteration": iteration}
                ):
                    async for chunk in self._llm.stream(
                        messages=messages,
                        name="chat.generate",
                        metadata=metadata,
                        **request_kwargs,
                        **tool_kwargs,
                    ):
                        if chunk.type is StreamEventType.TEXT and chunk.text:
                            state.answer += chunk.text
                            yield _event(SSEEvent.TOKEN, {"text": chunk.text})
                        elif chunk.type is StreamEventType.THINKING and chunk.text:
                            yield _event(SSEEvent.THINKING, {"text": chunk.text})
                        elif chunk.type is StreamEventType.TOOL_USE and chunk.tool_call:
                            pending.append(dict(chunk.tool_call))
                        elif chunk.type is StreamEventType.USAGE and chunk.usage:
                            _accumulate_usage(state, chunk.usage)
                        elif chunk.type is StreamEventType.REFUSAL:
                            state.refused = True
                        elif chunk.type is StreamEventType.DONE:
                            response = chunk.response
                            state.stop_reason = chunk.stop_reason or state.stop_reason
                            if chunk.usage:
                                _accumulate_usage(state, chunk.usage)

                if not pending or iteration >= max_iterations:
                    if pending:
                        # Either the iteration budget ran out, or the model asked
                        # for a tool on a turn where none was offered. Both are
                        # dropped rather than dispatched: the second would mean
                        # executing something the route decision excluded.
                        _log.warning(
                            "tool_calls_not_dispatched",
                            reason=(
                                "budget_exhausted"
                                if max_iterations
                                else "tools_not_offered"
                            ),
                            iterations=iteration,
                            pending=len(pending),
                            session_id=state.session_id,
                        )
                    break

                if ctx.guard is not None and not ctx.guard.begin_iteration():
                    _log.warning("tool_loop_guard_stop", iterations=iteration)
                    break

                messages.append(
                    {"role": "assistant", "content": _assistant_blocks(response)}
                )
                results: list[dict[str, Any]] = []
                for call in pending:
                    call_id = str(call.get("id") or repo.new_id())
                    name = str(call.get("name") or "")
                    arguments = dict(call.get("input") or {})
                    yield _event(
                        SSEEvent.TOOL_CALL,
                        {
                            "tool_call_id": call_id,
                            "tool_name": name,
                            "kind": ctx.tool_kind(name).value,
                            "arguments": _safe_arguments(ctx, arguments),
                        },
                    )
                    outcome = await self._dispatcher.dispatch(
                        ctx, tool_call_id=call_id, tool_name=name, arguments=arguments
                    )
                    state.tool_calls.append(_tool_call_record(outcome, ctx))
                    yield _event(
                        SSEEvent.TOOL_RESULT,
                        {
                            "tool_call_id": call_id,
                            "is_error": outcome.is_error,
                            "latency_ms": outcome.latency_ms,
                            "result_summary": outcome.to_result_summary(
                                tool_config(cfg).result_summary_chars
                            ),
                            "error_message": outcome.error_message,
                            "http_status": outcome.http_status,
                            "truncated": outcome.truncated,
                        },
                    )
                    results.append(outcome.to_tool_result_block())
                messages.append({"role": "user", "content": results})

    # ------------------------------------------------------------- persistence
    async def _persist(
        self,
        session: AsyncSession | None,
        state: TurnState,
        *,
        stage: str,
        operation: Any,
    ) -> bool:
        """Run one persistence step, rolling back and degrading on failure.

        A SQLAlchemy session that has seen an error refuses every subsequent
        statement until it is rolled back, so a failed write must clean up or it
        silently takes down every later write in the same turn.

        Args:
            session: Database session, or None when the store is unavailable.
            state: The turn state, used to report a degradation.
            stage: Pipeline stage the write belongs to, for the metric label.
            operation: Zero-argument coroutine factory performing the write.

        Returns:
            True when the write committed.
        """
        if session is None:
            return False
        started = time.perf_counter()
        try:
            await operation()
            await session.commit()
        except Exception as exc:
            await _rollback(session)
            _log.warning(
                "persistence_degraded",
                stage=stage,
                error=type(exc).__name__,
                session_id=state.session_id,
                tenant_id=state.principal.tenant_id,
            )
            state.guardrail(
                GuardrailEvent(
                    stage=GuardrailStage.OUTPUT.value,
                    kind=GuardrailKind.GROUNDEDNESS.value,
                    action=GuardrailAction.WARN.value,
                    detail=(
                        f"persisting the {stage} record failed ({type(exc).__name__})"
                    ),
                )
            )
            return False
        finally:
            observe_pipeline_stage(
                stage=stage, latency_ms=(time.perf_counter() - started) * _MS
            )
        return True

    async def _ensure_session(
        self, session: AsyncSession | None, state: TurnState
    ) -> None:
        """Create or touch the tenant, user and ``chat_sessions`` rows.

        The tenant and user rows are mirrors of what Entra already asserted, so
        the first request from a new directory provisions them rather than
        failing on a missing foreign key. Nothing here grants access: the
        principal's tenant came from a validated token.

        Args:
            session: Database session, or None.
            state: The turn state.
        """

        async def _write() -> None:
            """Upsert the tenant, the user and the session row."""
            await repo.upsert_tenant(
                session,
                tenant_id=state.principal.tenant_id,
                name=state.principal.tenant_id,
                entra_tenant_id=state.principal.tenant_id,
            )
            await repo.upsert_user(session, state.principal)
            await repo.get_or_create_session(
                session,
                tenant_id=state.principal.tenant_id,
                user_id=state.principal.user_id,
                session_id=state.session_id,
                title=state.title,
            )

        await self._persist(session, state, stage="input_guard", operation=_write)

    async def _persist_user_turn(
        self, session: AsyncSession | None, state: TurnState, guard: InputDecision
    ) -> None:
        """Write the user turn.

        Only :attr:`InputDecision.redacted_text` is ever written: the prompt-safe
        form keeps the user's own words, which have not been through full
        redaction, and ``repositories.append_message`` refuses content that has
        not.

        Args:
            session: Database session, or None.
            state: The turn state.
            guard: Stage 1's decision.
        """

        async def _write() -> None:
            """Append the redacted user turn."""
            await repo.append_message(
                session,
                tenant_id=state.principal.tenant_id,
                session_id=state.session_id,
                user_id=state.principal.user_id,
                role=Role.USER,
                content=guard.redacted_text,
                pii_redacted=True,
                pii_types=guard.pii_types,
                guardrail_events=[
                    event.model_dump(mode="json") for event in guard.events
                ],
                trace_id=state.trace_id,
            )

        await self._persist(session, state, stage="input_guard", operation=_write)

    async def _persist_assistant_turn(
        self,
        session: AsyncSession | None,
        state: TurnState,
        *,
        window: SessionWindow | None,
    ) -> None:
        """Write the assistant turn, its lineage and the session window.

        Args:
            session: Database session, or None.
            state: The turn state.
            window: The live window to append the turn to, when stage 2 loaded
                one. A short-circuited turn has no window and only touches
                PostgreSQL.
        """

        async def _write() -> None:
            """Append the assistant turn and record its lineage."""
            await repo.append_message(
                session,
                tenant_id=state.principal.tenant_id,
                session_id=state.session_id,
                user_id=state.principal.user_id,
                role=Role.ASSISTANT,
                content=state.stored_answer,
                pii_redacted=True,
                message_id=state.message_id,
                citations=[c.model_dump(mode="json") for c in state.citations],
                tool_calls=[t.model_dump(mode="json") for t in state.tool_calls],
                guardrail_events=[e.model_dump(mode="json") for e in state.guardrails],
                context_stats=(
                    state.context_stats.model_dump(mode="json")
                    if state.context_stats
                    else None
                ),
                usage=dict(state.usage) if state.usage else None,
                pii_types=state.pii_types,
                model=state.model or None,
                latency_ms=state.latency_ms,
                cost_usd=state.cost_usd,
                trace_id=state.trace_id,
                stop_reason=state.stop_reason or None,
                refused=state.refused,
            )
            await self._record_lineage(session, state)

        await self._persist(session, state, stage="output_guard", operation=_write)

        if window is not None:
            with self._stage("output_guard", state, essential=False):
                await self._short_term.record_turn(
                    window,
                    role=Role.ASSISTANT,
                    content=state.stored_answer,
                    pii_redacted=True,
                    message_id=state.message_id,
                    citations=state.citations,
                    tool_calls=state.tool_calls,
                    persist=True,
                )

    async def _record_lineage(self, session: AsyncSession, state: TurnState) -> None:
        """Record the retrieval and generation lineage for this turn.

        Args:
            session: Database session.
            state: The turn state.
        """
        chunk_ids = [chunk.payload.chunk_id for chunk in state.sources]
        document_ids = sorted({chunk.payload.document_id for chunk in state.sources})
        if chunk_ids:
            await repo.record_lineage(
                session,
                tenant_id=state.principal.tenant_id,
                kind="retrieval",
                subject_id=state.message_id,
                operation="retrieve",
                actor=state.principal.user_id,
                parents=chunk_ids,
                user_id=state.principal.user_id,
                session_id=state.session_id,
                trace_id=state.trace_id,
                outputs={"documents": document_ids, "chunks": len(chunk_ids)},
                metrics={
                    "max_score": state.retrieval.max_score if state.retrieval else 0.0,
                    "cache_hit": float(
                        bool(state.retrieval and state.retrieval.cache_hit)
                    ),
                },
            )
        await repo.record_lineage(
            session,
            tenant_id=state.principal.tenant_id,
            kind="generation",
            subject_id=state.message_id,
            operation="answer",
            actor=state.model or self._settings.anthropic_model_main,
            parents=list(chunk_ids),
            user_id=state.principal.user_id,
            session_id=state.session_id,
            trace_id=state.trace_id,
            outputs={
                "citations": len(state.citations),
                "tool_calls": len(state.tool_calls),
                "refused": state.refused,
            },
            metrics={
                "latency_ms": state.latency_ms,
                "cost_usd": state.cost_usd,
                "answer_chars": float(len(state.stored_answer)),
            },
        )

    def _score(
        self,
        state: TurnState,
        citations: CitationReport,
        output: OutputDecision,
        result: RetrievalResult,
    ) -> None:
        """Emit the turn's Langfuse scores.

        Args:
            state: The turn state.
            citations: Stage 11's report.
            output: Stage 12's decision.
            result: Stage 5's result.
        """
        with suppress(Exception):
            self._tracer.score(
                name="citation_validity", value=citations.citation_validity
            )
            self._tracer.score(name="groundedness", value=output.groundedness)
            self._tracer.score(name="coverage", value=citations.coverage)
            self._tracer.score(name="retrieval_max_score", value=result.max_score)
            self._tracer.score(
                name="refused",
                value=1.0 if state.refused else 0.0,
                comment=state.stop_reason or None,
            )


# ------------------------------------------------------------------ utilities


async def _rollback(session: AsyncSession) -> None:
    """Roll a session back so later writes in the same turn can still run.

    Args:
        session: The session to clean up. A failure here is logged and swallowed:
            there is nothing further to try, and raising would turn a degraded
            write into a failed turn.
    """
    try:
        await session.rollback()
    except Exception:
        _log.warning("rollback_failed")


def _title_for(text: str, *, limit: int = 60) -> str:
    """Derive a session title from the first redacted user turn.

    Args:
        text: The redacted turn.
        limit: Maximum characters.

    Returns:
        A single-line title, truncated on a word boundary.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    clipped = flat[:limit].rsplit(" ", 1)[0]
    return f"{clipped or flat[:limit]}…"


def _markers_for(chunks: Sequence[RetrievedChunk]) -> dict[str, str]:
    """Build the chunk-id to ``[n]`` map for the given order.

    Args:
        chunks: Chunks in the order they will be numbered.

    Returns:
        Chunk id to marker.
    """
    return {
        chunk.payload.chunk_id: format_marker(index)
        for index, chunk in enumerate(chunks, start=1)
    }


def _order_changed(
    before: Sequence[RetrievedChunk], after: Sequence[RetrievedChunk]
) -> bool:
    """Whether packing changed which chunk carries which marker.

    Args:
        before: Chunks stage 7 numbered.
        after: Chunks stage 9 actually packed.

    Returns:
        True when the two orderings differ.
    """
    return [chunk.payload.chunk_id for chunk in before] != [
        chunk.payload.chunk_id for chunk in after
    ]


def _realign_notes(
    report: ContradictionReport, sources: Sequence[RetrievedChunk]
) -> str:
    """Re-render contradiction notes against the packed source order.

    A contradiction whose chunks did not both survive packing is dropped rather
    than renumbered onto a different source.

    Args:
        report: Stage 7's report.
        sources: The chunks stage 9 packed, in marker order.

    Returns:
        The re-rendered notes, empty when nothing survived.
    """
    markers = _markers_for(sources)
    kept = []
    for conflict in report.contradictions:
        current = markers.get(conflict.current_chunk_id)
        superseded = markers.get(conflict.superseded_chunk_id)
        if current is None or superseded is None:
            continue
        remarked = [
            citation.model_copy(
                update={
                    "marker": markers.get(citation.chunk_id, citation.marker),
                }
            )
            for citation in conflict.citations
            if citation.chunk_id in markers
        ]
        kept.append(conflict.model_copy(update={"citations": remarked}))
    realigned = report.model_copy(update={"contradictions": kept})
    return render_contradiction_notes(realigned)


def _highest_classification(chunks: Sequence[RetrievedChunk]) -> Classification:
    """Highest sensitivity present in the assembled context.

    This is the egress ceiling the tool layer screens against, so it must be the
    maximum rather than the median: one restricted chunk in the prompt makes the
    whole context restricted.

    Args:
        chunks: The retrieved chunks.

    Returns:
        The strictest classification, or ``PUBLIC`` when there are no chunks.
    """
    best = Classification.PUBLIC
    for chunk in chunks:
        best = max(best, chunk.payload.classification_level)
    return best


def _empty_connector() -> Any:
    """Build an empty remote-MCP connector request.

    Returns:
        A :class:`~app.rag.tools.mcp_client.ConnectorRequest` with nothing in it,
        so a turn that never planned tools still has a well-formed plan object.
    """
    from app.rag.tools.mcp_client import ConnectorRequest

    return ConnectorRequest(betas=[], mcp_servers=[], tools=[])


def _tool_request_kwargs(plan: ToolPlan) -> dict[str, Any]:
    """Project a tool plan onto ``LLMClient.stream`` keyword arguments.

    ``ToolPlan.request_kwargs`` also returns ``betas``, which the client derives
    itself from the presence of ``mcp_servers`` — passing it through would send
    the flag twice.

    Args:
        plan: The tool plan.

    Returns:
        ``tools`` and, when a remote MCP server applies, ``mcp_servers``.
    """
    kwargs = plan.request_kwargs()
    projected: dict[str, Any] = {}
    if kwargs.get("tools"):
        projected["tools"] = kwargs["tools"]
    if kwargs.get("mcp_servers"):
        projected["mcp_servers"] = kwargs["mcp_servers"]
    return projected


def _retriever_for(settings: Settings) -> Any:
    """Build the retrieval callable the built-in search tool uses.

    Injecting stage 5's retriever means a tool-issued search goes through exactly
    the same ACL filter, dedupe, rerank and MMR as the pipeline's own. The
    principal is supplied by :meth:`ToolDispatcher.dispatch` on every call rather
    than captured here, so a tool can only ever search as its actual caller.

    Args:
        settings: Active settings.

    Returns:
        An awaitable matching :class:`~app.rag.tools.builtin.RetrieveFn`.
    """

    async def _retrieve(
        *,
        query: str,
        principal: Principal,
        filters: MetadataFilter | None,
        top_n: int | None,
    ) -> RetrievalResult:
        """Run stage 5 for a tool-issued query.

        Args:
            query: The tool's query.
            principal: The caller, as passed by the dispatcher.
            filters: Facets, already narrowed against the turn's filter.
            top_n: Result count.

        Returns:
            The retrieval result.
        """
        return await retrieve(
            principal, [query], filters, top_n=top_n, settings=settings
        )

    return _retrieve


def _cache_resolver(settings: Settings) -> Any:
    """Build the stage 4 chunk resolver.

    Addendum M asks the orchestrator to inject the retriever's richer
    ``retrieve_by_ids`` so a cache hit is re-authorised with the same filter, drop
    vocabulary and score scale as a live retrieval.

    Args:
        settings: Active settings.

    Returns:
        A :class:`~app.rag.memory.semantic_cache.ChunkResolver`.
    """

    async def _resolve(
        principal: Principal,
        chunk_ids: Sequence[str],
        filters: MetadataFilter | None,
    ) -> Sequence[RetrievedChunk]:
        """Re-fetch cached chunk ids through the live ACL filter.

        Args:
            principal: The caller.
            chunk_ids: Cached ids, best first.
            filters: The turn's metadata filter.

        Returns:
            The surviving chunks.
        """
        result = await retrieve_by_ids(principal, chunk_ids, filters, settings=settings)
        return result.chunks

    return _resolve


def _assistant_blocks(response: Any) -> list[Any]:
    """Extract the assistant content blocks to replay into the next request.

    With adaptive thinking, the thinking blocks must go back verbatim alongside
    the ``tool_use`` blocks or the next call is rejected, so the whole
    ``response.content`` is echoed rather than just the tool calls.

    Args:
        response: The :class:`~ragcore.llm.LLMResponse` from the finished stream.

    Returns:
        The content blocks as plain mappings.
    """
    raw = getattr(response, "raw", None)
    content = getattr(raw, "content", None) or []
    blocks: list[Any] = []
    for block in content:
        dump = getattr(block, "model_dump", None)
        blocks.append(dump(exclude_none=True) if callable(dump) else block)
    return blocks


def _safe_arguments(ctx: ToolContext, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Redact tool arguments before they reach the stream.

    Model-authored arguments can echo the user's turn, and the ``tool_call`` event
    is emitted before anything else redacts them.

    Args:
        ctx: The turn's tool context, supplying the detector.
        arguments: Raw arguments.

    Returns:
        The redacted mapping.
    """
    from app.rag.tools.registry import redact_arguments

    try:
        return redact_arguments(arguments, detector=ctx.detector, settings=ctx.settings)
    except Exception:
        _log.warning("tool_argument_redaction_failed")
        return {"redacted": True}


def _tool_call_record(result: ToolResult, ctx: ToolContext) -> ToolCall:
    """Project a tool result onto the persisted :class:`ToolCall`.

    Args:
        result: What the dispatcher returned.
        ctx: The turn's tool context.

    Returns:
        The record. ``arguments`` is deliberately empty here: the dispatcher
        already persisted the redacted arguments to ``tool_invocations``, and
        duplicating them onto the message row would put model-authored text
        through a second redaction path.
    """
    del ctx
    return ToolCall(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        kind=result.kind.value,
        result_summary=result.to_result_summary(),
        is_error=result.is_error,
        latency_ms=result.latency_ms,
        created_at=result.created_at,
    )


def _accumulate_usage(state: TurnState, usage: Any) -> None:
    """Fold one model call's usage into the turn total.

    A tool loop makes several calls; the ``usage`` SSE event reports the turn, not
    the last call, so the buckets are summed and the cost recomputed per call
    against the model that actually served it.

    Args:
        state: The turn state.
        usage: A :class:`~ragcore.llm.LLMUsage`.
    """
    current = state.usage or {}
    state.model = usage.model or state.model
    state.usage = {
        "model": state.model,
        "input_tokens": int(current.get("input_tokens", 0)) + usage.input_tokens,
        "output_tokens": int(current.get("output_tokens", 0)) + usage.output_tokens,
        "cache_read_tokens": int(current.get("cache_read_tokens", 0))
        + usage.cache_read_tokens,
        "cache_write_tokens": int(current.get("cache_write_tokens", 0))
        + usage.cache_write_tokens,
        "cost_usd": float(current.get("cost_usd", 0.0)) + usage.cost_usd(),
    }
    if state.context_stats is not None:
        state.context_stats.cache_read_tokens = int(state.usage["cache_read_tokens"])
        state.context_stats.cache_write_tokens = int(state.usage["cache_write_tokens"])


def _empty_stats(settings: Settings) -> ContextStats:
    """Build the stats a short-circuited turn reports.

    Args:
        settings: Active settings, supplying the budget.

    Returns:
        Zeroed :class:`~ragcore.models.chat.ContextStats` with a real budget, so
        the UI's context meter renders rather than dividing by zero.
    """
    return ContextStats(
        window_tokens=0,
        budget_tokens=settings.context_prompt_budget_tokens,
        system_tokens=0,
        history_tokens=0,
        retrieved_tokens=0,
        memory_tokens=0,
        summary_tokens=0,
        messages_live=0,
        messages_suppressed=0,
        compaction_events=0,
    )


def _empty_usage(settings: Settings) -> dict[str, Any]:
    """Build the usage payload a turn without a model call reports.

    Args:
        settings: Active settings, supplying the model name.

    Returns:
        A zeroed usage mapping in the contract's shape.
    """
    return {
        "model": settings.anthropic_model_main,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
    }


_ORCHESTRATORS: dict[str, Orchestrator] = {}


def get_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Return the process-wide orchestrator.

    ``Settings`` is unhashable, so the cache key is the handful of fields that
    change which collaborators get built.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached orchestrator, so model clients, the tool registry and the
        session store are shared across requests.
    """
    cfg = settings or get_settings()
    key = "|".join(
        str(part)
        for part in (
            cfg.anthropic_model_main,
            cfg.qdrant_url,
            cfg.database_url,
            cfg.redis_url,
            cfg.tool_registry_path,
        )
    )
    orchestrator = _ORCHESTRATORS.get(key)
    if orchestrator is None:
        orchestrator = Orchestrator(settings=cfg)
        _ORCHESTRATORS[key] = orchestrator
    return orchestrator
