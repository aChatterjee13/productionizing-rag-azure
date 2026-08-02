"""Tool routing, loop control and the single dispatch chokepoint (stage 8).

Three responsibilities:

**Route.** :func:`decide_route` answers "retrieval, tools, or both?" from the
:class:`~app.rag.query_transform.TransformedQuery` hints *and* the confidence the
retrieval that already ran earned. A high-confidence retrieval does not need a tool;
a low-confidence one, or an explicit ``needs_tools``, does. The decision is a data
object so the orchestrator can trace it and the eval harness can assert on it.

**Bound.** :class:`LoopGuard` caps the loop at ``tool_max_iterations`` and breaks the
failure mode that actually happens in production: the model calling the same tool with
the same arguments over and over because the answer is genuinely not there. The second
identical call is answered from the first result with an explicit instruction to stop;
the third is refused outright.

**Dispatch.** :meth:`ToolDispatcher.dispatch` is the only path from a model
``tool_use`` block to an executed tool. It enforces, in order: tenant + role filtering,
loop detection, the per-tenant rate limit, the egress classification gate
(``RESTRICTED`` content never reaches an external API) and PII in arguments — then
executes, traces the call as a Langfuse span with **redacted** arguments, records the
Prometheus series and persists the row to ``tool_invocations``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.query_transform import TransformedQuery
from app.rag.tools.builtin import BuiltinExecutor, RetrieveFn
from app.rag.tools.mcp_client import (
    ConnectorRequest,
    LocalMcpClient,
    LocalMcpTool,
    RemoteMcpConnector,
)
from app.rag.tools.registry import (
    NEVER_FORWARD,
    RegisteredTool,
    ToolConfig,
    ToolRegistry,
    get_tool_registry,
    redact_arguments,
    tool_config,
)
from app.rag.tools.rest_tool import RestExecutor, get_rest_executor
from ragcore.db.repositories import write_tool_invocation
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.models.retrieval import MetadataFilter, RetrievalResult
from ragcore.models.tool import ToolKind, ToolResult
from ragcore.observability import Tracer, get_tracer
from ragcore.observability.metrics import observe_tool_invocation
from ragcore.pii import PIIDetector, get_pii_detector
from ragcore.settings import Settings, get_settings

__all__ = [
    "ExposedTool",
    "LoopGuard",
    "LoopVerdict",
    "RouteDecision",
    "RouteMode",
    "ToolContext",
    "ToolDispatcher",
    "ToolPlan",
    "argument_signature",
    "decide_route",
]

_log = get_logger(__name__)

_MS = 1000.0

#: Either kind of tool the loop can execute: a declarative/built-in registry tool, or
#: a tool discovered on a self-hosted MCP server.
type ExposedTool = RegisteredTool | LocalMcpTool


# ---------------------------------------------------------------------- routing


class RouteMode(StrEnum):
    """What stage 8 should do for this turn."""

    RETRIEVAL_ONLY = "retrieval_only"
    TOOLS_ONLY = "tools_only"
    BOTH = "both"
    NEITHER = "neither"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """The routing verdict, traced and asserted on by the eval harness."""

    mode: RouteMode
    reason: str
    max_iterations: int
    retrieval_confidence: float
    tool_hints: tuple[str, ...] = ()
    candidate_tools: tuple[str, ...] = ()

    @property
    def use_tools(self) -> bool:
        """Whether the tool loop should run.

        Returns:
            True for ``tools_only`` and ``both``.
        """
        return self.mode in {RouteMode.TOOLS_ONLY, RouteMode.BOTH}

    @property
    def use_retrieval(self) -> bool:
        """Whether corpus retrieval should be used for this turn.

        Returns:
            True for ``retrieval_only`` and ``both``.
        """
        return self.mode in {RouteMode.RETRIEVAL_ONLY, RouteMode.BOTH}

    def as_metadata(self) -> dict[str, Any]:
        """Render the decision for a trace or a log line.

        Returns:
            A JSON-serialisable mapping. Contains no user content.
        """
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "max_iterations": self.max_iterations,
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "tool_hints": list(self.tool_hints),
            "candidate_tools": list(self.candidate_tools),
        }


def _retrieval_confidence(retrieval: RetrievalResult | None) -> float:
    """Score how well the corpus already answered the question.

    Args:
        retrieval: The stage-5 result, or None when retrieval did not run.

    Returns:
        The best ``final_score`` in the result, or 0.0 when nothing was retrieved.
        Fusion and rerank scores are not on a common scale, which is why this is
        compared against a configurable floor rather than an absolute one.
    """
    if retrieval is None or not retrieval.chunks:
        return 0.0
    return float(retrieval.max_score)


def decide_route(
    transformed: TransformedQuery | None,
    *,
    tools: Sequence[ExposedTool] = (),
    retrieval: RetrievalResult | None = None,
    allow_tools: bool = True,
    settings: Settings | None = None,
    config: ToolConfig | None = None,
) -> RouteDecision:
    """Decide retrieval versus tools for one turn.

    Args:
        transformed: The stage-3 plan. ``None`` is treated as "retrieve".
        tools: Tools already filtered for this principal. With none exposed, the
            answer can only ever be ``retrieval_only``.
        retrieval: The stage-5 result, when retrieval has already run.
        allow_tools: The request's ``allow_tools`` flag. A user opt-out wins over
            every hint.
        settings: Platform settings; defaults to :func:`get_settings`.
        config: Resolved tool config; derived from settings when omitted.

    Returns:
        The routing decision, including the reason, so a surprising route is
        explicable after the fact.
    """
    cfg = settings or get_settings()
    resolved = config or tool_config(cfg)
    confidence = _retrieval_confidence(retrieval)
    hints = tuple(transformed.tool_hints) if transformed else ()
    names = tuple(tool.name for tool in tools)

    wants_retrieval = transformed.needs_retrieval if transformed else True
    if transformed is not None and transformed.degraded:
        # A degraded plan is the raw-query fallback; its flags are not evidence.
        wants_retrieval = True

    if not (allow_tools and resolved.enabled and names):
        reason = (
            "tools disabled by request"
            if not allow_tools
            else "no tool is exposed to this principal"
            if resolved.enabled
            else "tool_enabled=false"
        )
        mode = RouteMode.RETRIEVAL_ONLY if wants_retrieval else RouteMode.NEITHER
        return RouteDecision(
            mode=mode,
            reason=reason,
            max_iterations=0,
            retrieval_confidence=confidence,
            tool_hints=hints,
            candidate_tools=(),
        )

    asked_for_tools = bool(transformed and transformed.needs_tools) or bool(hints)
    weak_retrieval = (
        retrieval is not None and confidence < resolved.router_min_confidence
    )
    no_retrieval_wanted = transformed is not None and not wants_retrieval

    if asked_for_tools:
        reason = "the query transform asked for a tool"
    elif weak_retrieval:
        reason = (
            f"retrieval confidence {confidence:.3f} is below "
            f"tool_router_min_confidence {resolved.router_min_confidence:.3f}"
        )
    elif no_retrieval_wanted:
        reason = "the query transform said retrieval is not needed"
    else:
        return RouteDecision(
            mode=RouteMode.RETRIEVAL_ONLY,
            reason="retrieval is sufficient",
            max_iterations=0,
            retrieval_confidence=confidence,
            tool_hints=hints,
            candidate_tools=names,
        )

    mode = RouteMode.BOTH if wants_retrieval else RouteMode.TOOLS_ONLY
    return RouteDecision(
        mode=mode,
        reason=reason,
        max_iterations=resolved.max_iterations,
        retrieval_confidence=confidence,
        tool_hints=hints,
        candidate_tools=names,
    )


# ----------------------------------------------------------------- loop control


def argument_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Build the identity of one (tool, arguments) call.

    Args:
        tool_name: Name of the tool.
        arguments: Arguments the model supplied.

    Returns:
        A stable string. Key order does not matter, so two calls that differ only in
        JSON key order are recognised as the same call.
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        rendered = repr(sorted(arguments.items()))
    return f"{tool_name}:{rendered}"


class LoopVerdict(StrEnum):
    """What the loop guard decided about one proposed call."""

    ALLOW = "allow"
    REPEAT = "repeat"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"


@dataclass(slots=True)
class LoopGuard:
    """Bounds the tool loop and breaks repeat cycles.

    Attributes:
        max_iterations: Hard cap from ``tool_max_iterations``.
        repeat_limit: Identical ``(tool, arguments)`` calls tolerated before the call
            is refused, from ``tool_loop_repeat_limit``.
    """

    max_iterations: int
    repeat_limit: int = 2
    iterations: int = 0
    seen: dict[str, int] = field(default_factory=dict)
    broken_reason: str | None = None

    @classmethod
    def from_config(
        cls, config: ToolConfig, *, max_iterations: int | None = None
    ) -> LoopGuard:
        """Build a guard from the resolved tool configuration.

        Args:
            config: Resolved tool config.
            max_iterations: Override for the iteration cap, e.g. a route decision's.

        Returns:
            The guard.
        """
        return cls(
            max_iterations=(
                config.max_iterations if max_iterations is None else max_iterations
            ),
            repeat_limit=config.loop_repeat_limit,
        )

    @property
    def exhausted(self) -> bool:
        """Whether the iteration budget is spent.

        Returns:
            True once ``iterations`` reaches ``max_iterations``.
        """
        return self.iterations >= self.max_iterations

    def begin_iteration(self) -> bool:
        """Consume one loop iteration.

        Returns:
            True when the loop may run another round, False when the budget is spent.
        """
        if self.exhausted:
            self.broken_reason = self.broken_reason or "max_iterations"
            return False
        self.iterations += 1
        return True

    def check(self, tool_name: str, arguments: Mapping[str, Any]) -> LoopVerdict:
        """Classify a proposed call and record it.

        Args:
            tool_name: Name of the tool the model wants to call.
            arguments: Arguments it supplied.

        Returns:
            ``allow`` the first time, ``repeat`` on the duplicate that trips the
            limit (the caller answers it with a stop instruction rather than calling
            the tool again), and ``blocked`` for every duplicate after that.
        """
        if self.exhausted:
            self.broken_reason = "max_iterations"
            return LoopVerdict.EXHAUSTED
        signature = argument_signature(tool_name, arguments)
        count = self.seen.get(signature, 0) + 1
        self.seen[signature] = count
        if count < self.repeat_limit:
            return LoopVerdict.ALLOW
        self.broken_reason = "repeat_loop"
        _log.warning(
            "tool_loop_repeat_detected",
            tool=tool_name,
            occurrences=count,
            limit=self.repeat_limit,
        )
        return LoopVerdict.REPEAT if count == self.repeat_limit else LoopVerdict.BLOCKED

    def reset(self) -> None:
        """Clear the guard for a new turn."""
        self.iterations = 0
        self.seen.clear()
        self.broken_reason = None


# --------------------------------------------------------------------- context


@dataclass(slots=True)
class ToolContext:
    """Everything one turn's tool calls need, assembled once by the orchestrator."""

    principal: Principal
    registry: ToolRegistry
    settings: Settings
    config: ToolConfig
    session: AsyncSession | None = None
    session_id: str | None = None
    message_id: str | None = None
    base_filter: MetadataFilter | None = None
    retrieve: RetrieveFn | None = None
    context_classification: Classification = Classification.PUBLIC
    tracer: Tracer | None = None
    detector: PIIDetector | None = None
    exposed: dict[str, ExposedTool] = field(default_factory=dict)
    guard: LoopGuard | None = None

    @classmethod
    def build(
        cls,
        principal: Principal,
        *,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        **overrides: Any,
    ) -> ToolContext:
        """Assemble a context with the process-wide singletons filled in.

        Args:
            principal: The caller.
            settings: Platform settings; defaults to :func:`get_settings`.
            registry: Tool registry; defaults to the cached one.
            **overrides: Any other field of :class:`ToolContext`.

        Returns:
            The context.
        """
        cfg = settings or get_settings()
        return cls(
            principal=principal,
            registry=registry or get_tool_registry(cfg),
            settings=cfg,
            config=tool_config(cfg),
            **overrides,
        )

    def tool_kind(self, name: str) -> ToolKind:
        """Kind of an exposed tool, for metrics and persistence.

        Args:
            name: Tool name.

        Returns:
            The tool's kind, defaulting to ``rest`` for an unknown name.
        """
        tool = self.exposed.get(name)
        return tool.kind if tool is not None else ToolKind.REST


@dataclass(frozen=True, slots=True)
class ToolPlan:
    """The tool surface offered to the model for one turn."""

    tools: list[dict[str, Any]]
    connector: ConnectorRequest
    exposed: dict[str, ExposedTool]

    def request_kwargs(self) -> dict[str, Any]:
        """Render the tool-related keyword arguments for the model call.

        Returns:
            ``tools`` merged with the connector's ``mcp_toolset`` entries, plus
            ``mcp_servers`` and ``betas`` when a remote server applies. Never emits
            ``mcp_servers`` without its matching toolset entry.
        """
        connector = self.connector.as_kwargs()
        merged: dict[str, Any] = {"tools": [*self.tools, *connector.get("tools", [])]}
        if connector:
            merged["mcp_servers"] = connector["mcp_servers"]
            merged["betas"] = connector["betas"]
        return merged


# -------------------------------------------------------------------- dispatch


class ToolDispatcher:
    """The single path from a model ``tool_use`` block to an executed tool."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        rest: RestExecutor | None = None,
        builtin: BuiltinExecutor | None = None,
        local_mcp: LocalMcpClient | None = None,
        remote_mcp: RemoteMcpConnector | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialise the dispatcher.

        Args:
            settings: Platform settings; defaults to :func:`get_settings`.
            rest: REST executor; defaults to the shared one.
            builtin: Built-in executor; constructed when omitted.
            local_mcp: Self-hosted MCP client; constructed when omitted.
            remote_mcp: Remote MCP connector; constructed when omitted.
            tracer: Langfuse tracer; defaults to the cached one.
        """
        self.settings = settings or get_settings()
        self.config = tool_config(self.settings)
        self.rest = rest or get_rest_executor(self.settings)
        self.builtin = builtin or BuiltinExecutor(settings=self.settings)
        self.local_mcp = local_mcp or LocalMcpClient(
            settings=self.settings, config=self.config
        )
        self.remote_mcp = remote_mcp or RemoteMcpConnector(
            settings=self.settings, config=self.config
        )
        self.tracer = tracer or get_tracer(self.settings)

    # ------------------------------------------------------------------ plan

    async def plan(self, ctx: ToolContext) -> ToolPlan:
        """Assemble the tool surface for one turn and record it on the context.

        Args:
            ctx: The turn's tool context.

        Returns:
            The plan. Local MCP discovery failures silently reduce the surface
            instead of failing the turn.
        """
        exposed: dict[str, ExposedTool] = {}
        definitions: list[dict[str, Any]] = []
        for tool in ctx.registry.tools_for(ctx.principal):
            if tool.kind is ToolKind.MCP:
                continue  # reaches the model through the connector's toolset entry
            exposed[tool.name] = tool
            definitions.append(tool.spec.to_anthropic_tool())

        for local in await self.local_mcp.discovered_tools(ctx.registry, ctx.principal):
            if local.name in exposed:
                _log.warning("mcp_tool_name_collision", tool=local.name)
                continue
            exposed[local.name] = local
            definitions.append(local.to_anthropic_tool())

        connector = await self.remote_mcp.build(ctx.registry, ctx.principal)
        ctx.exposed = exposed
        return ToolPlan(tools=definitions, connector=connector, exposed=exposed)

    # -------------------------------------------------------------- screening

    def _detector(self, ctx: ToolContext) -> PIIDetector:
        """Resolve the PII detector for a context.

        Args:
            ctx: The turn's tool context.

        Returns:
            The detector.
        """
        if ctx.detector is None:
            ctx.detector = get_pii_detector(ctx.settings)
        return ctx.detector

    def screen(
        self, ctx: ToolContext, tool: ExposedTool, arguments: Mapping[str, Any]
    ) -> str | None:
        """Decide whether this content may be forwarded to this tool.

        Two independent gates. The classification gate compares the highest
        classification present in the turn's context against the tool's ceiling;
        ``RESTRICTED`` never reaches anything outside the platform, regardless of
        configuration. The PII gate blocks personal data in the arguments unless the
        tool's policy explicitly opts in — a lookup-by-email tool legitimately does,
        a generic weather API does not.

        Args:
            ctx: The turn's tool context.
            tool: The resolved tool.
            arguments: Arguments the model supplied.

        Returns:
            A redacted refusal reason, or None when the call may proceed.
        """
        if not tool.may_receive(ctx.context_classification):
            return (
                f"this tool may not receive {ctx.context_classification.value} content"
            )
        if tool.kind is not ToolKind.RETRIEVAL and (
            ctx.context_classification >= NEVER_FORWARD
        ):
            return "restricted content is never forwarded outside the platform"
        if tool.policy.allow_pii_in_arguments or tool.kind is ToolKind.RETRIEVAL:
            return None
        detector = self._detector(ctx)
        for value in _string_values(arguments):
            report = detector.analyze(value)
            if report.has_pii:
                _log.warning(
                    "tool_argument_pii_blocked",
                    tool=tool.name,
                    entity_types=sorted(report.entity_types),
                )
                return (
                    "the arguments contain personal data "
                    f"({', '.join(sorted(report.entity_types))}) and this tool is not "
                    "permitted to receive it; ask the user for a non-personal "
                    "identifier instead"
                )
        return None

    # --------------------------------------------------------------- dispatch

    async def dispatch(
        self,
        ctx: ToolContext,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Execute one tool call with every gate, trace and record applied.

        Args:
            ctx: The turn's tool context.
            tool_call_id: The model's ``tool_use`` block id.
            tool_name: Tool name the model used.
            arguments: Arguments the model supplied.

        Returns:
            The tool result. Every failure path — unknown tool, denied tool, loop
            break, rate limit, egress refusal, upstream error — returns
            ``is_error=True`` so the model can recover inside the loop.
        """
        started = time.perf_counter()
        args = dict(arguments or {})
        tool = ctx.exposed.get(tool_name)
        if tool is None or not tool.is_allowed_for(ctx.principal):
            _log.warning(
                "tool_call_denied",
                tool=tool_name,
                tenant_id=ctx.principal.tenant_id,
            )
            return await self._finish(
                ctx,
                ToolResult.failure(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    kind=ctx.tool_kind(tool_name),
                    message=f"tool {tool_name!r} is not available to you",
                    latency_ms=(time.perf_counter() - started) * _MS,
                ),
                arguments=args,
            )

        guard = ctx.guard
        if guard is not None:
            verdict = guard.check(tool_name, args)
            if verdict is not LoopVerdict.ALLOW:
                return await self._finish(
                    ctx,
                    ToolResult.failure(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        kind=tool.kind,
                        message=_loop_message(verdict, tool_name),
                        latency_ms=(time.perf_counter() - started) * _MS,
                    ),
                    arguments=args,
                )

        allowed = await ctx.registry.rate_limiter.acquire(
            tenant_id=ctx.principal.tenant_id,
            tool_name=tool_name,
            per_minute=tool.rate_limit_per_minute,
        )
        if not allowed:
            return await self._finish(
                ctx,
                ToolResult.failure(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    kind=tool.kind,
                    message="this tool's rate limit for your tenant is exhausted",
                    latency_ms=(time.perf_counter() - started) * _MS,
                ),
                arguments=args,
            )

        refusal = self.screen(ctx, tool, args)
        if refusal is not None:
            return await self._finish(
                ctx,
                ToolResult.failure(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    kind=tool.kind,
                    message=refusal,
                    latency_ms=(time.perf_counter() - started) * _MS,
                ),
                arguments=args,
            )

        redacted = redact_arguments(
            args, detector=self._detector(ctx), settings=ctx.settings
        )
        tracer = ctx.tracer or self.tracer
        async with tracer.span(
            f"tool.{tool_name}",
            input=redacted,
            metadata={
                "tool": tool_name,
                "kind": tool.kind.value,
                "tenant_id": ctx.principal.tenant_id,
                "server": getattr(tool, "server_name", None),
            },
        ) as span:
            result = await self._execute(ctx, tool, tool_call_id, args)
            span.update(
                output={
                    "is_error": result.is_error,
                    "latency_ms": round(result.latency_ms, 2),
                    "http_status": result.http_status,
                    "truncated": result.truncated,
                    "content_chars": len(result.content),
                }
            )
        return await self._finish(ctx, result, arguments=args, redacted=redacted)

    async def _execute(
        self,
        ctx: ToolContext,
        tool: ExposedTool,
        tool_call_id: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Route a screened call to the right executor.

        Args:
            ctx: The turn's tool context.
            tool: The resolved tool.
            tool_call_id: The model's ``tool_use`` block id.
            arguments: Arguments the model supplied.

        Returns:
            The tool result.
        """
        if isinstance(tool, LocalMcpTool):
            return await self.local_mcp.call(
                tool.server,
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                arguments=arguments,
                max_result_chars=tool.max_result_chars,
                timeout_seconds=tool.timeout_seconds,
            )
        if tool.kind is ToolKind.RETRIEVAL:
            return await self.builtin.execute(
                tool,
                tool_call_id=tool_call_id,
                arguments=arguments,
                principal=ctx.principal,
                base_filter=ctx.base_filter,
                retrieve=ctx.retrieve,
            )
        return await self.rest.execute(
            tool, tool_call_id=tool_call_id, arguments=arguments
        )

    async def _finish(
        self,
        ctx: ToolContext,
        result: ToolResult,
        *,
        arguments: Mapping[str, Any],
        redacted: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Record metrics and persist the invocation.

        Persistence never fails the turn: a database problem is logged and the result
        still reaches the model.

        Args:
            ctx: The turn's tool context.
            result: The result to record.
            arguments: Raw arguments, redacted here when not already done.
            redacted: Pre-redacted arguments, to avoid scanning twice.

        Returns:
            The unchanged result, so callers can ``return await self._finish(...)``.
        """
        observe_tool_invocation(
            tool=result.tool_name,
            kind=result.kind.value,
            latency_ms=result.latency_ms,
            is_error=result.is_error,
        )
        if ctx.session is None:
            return result
        safe = (
            dict(redacted)
            if redacted is not None
            else redact_arguments(
                arguments, detector=self._detector(ctx), settings=ctx.settings
            )
        )
        try:
            await write_tool_invocation(
                ctx.session,
                tenant_id=ctx.principal.tenant_id,
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                kind=result.kind.value,
                arguments=safe,
                session_id=ctx.session_id,
                message_id=ctx.message_id,
                user_id=ctx.principal.user_id,
                result_summary=result.to_result_summary(
                    ctx.config.result_summary_chars
                ),
                is_error=result.is_error,
                error_message=result.error_message,
                http_status=result.http_status,
                truncated=result.truncated,
                latency_ms=result.latency_ms,
                trace_id=(ctx.tracer or self.tracer).current_trace_id(),
            )
        except Exception as exc:
            _log.error(
                "tool_invocation_persist_failed",
                tool=result.tool_name,
                error=type(exc).__name__,
            )
        return result


def _loop_message(verdict: LoopVerdict, tool_name: str) -> str:
    """Build the instruction returned when the loop guard intervenes.

    Args:
        verdict: The guard's verdict.
        tool_name: Tool the model tried to call.

    Returns:
        A message telling the model to stop repeating and answer with what it has.
    """
    if verdict is LoopVerdict.EXHAUSTED:
        return (
            "the tool-call budget for this turn is spent; answer with the "
            "information you already have and say what is still missing"
        )
    return (
        f"you already called {tool_name!r} with these exact arguments and got the "
        "result above; calling it again will not change it. Answer with what you "
        "have, or call a different tool with different arguments"
    )


def _string_values(value: Any) -> list[str]:
    """Collect every string leaf in a nested argument structure.

    Args:
        value: A mapping, sequence or scalar.

    Returns:
        All string values found, depth-first.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _string_values(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _string_values(item)]
    return []
