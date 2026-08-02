"""Built-in tools: corpus search and tenant/clock context.

Two tools are always present, whatever the deployment's ``tools.yaml`` says.

``search_corpus``
    Lets the model re-query the index mid-loop with *different* filters than the
    orchestrator chose. Stage 5 runs one plan; the model frequently discovers half
    way through an answer that it needs the policy document rather than the FAQ, or
    a narrower date range. Without this tool the only recovery is a second turn.
    Every query goes through :func:`ragcore.vectorstore.filters.build_acl_filter`, so
    a tool-issued search is exactly as tenant-scoped as a pipeline search.

``current_context``
    The wall clock plus the caller's tenant, roles and clearance. "Current" and
    "last quarter" are unanswerable without a clock, and the model must not guess a
    date from its training cut-off. Returns no user content, so nothing here needs
    redaction.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.rag.tools.registry import (
    RegisteredTool,
    ToolPolicy,
    register_tool,
    tool_config,
)
from ragcore.embeddings import get_embedding_provider
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.retrieval import (
    MetadataFilter,
    RetrievalResult,
    RetrievalStage,
    RetrievedChunk,
)
from ragcore.models.tool import ToolKind, ToolResult, ToolSpec
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore import (
    build_acl_filter,
    get_client,
    hybrid_search,
    serialise_filter,
)

__all__ = [
    "CONTEXT_TOOL_NAME",
    "SEARCH_TOOL_NAME",
    "BuiltinExecutor",
    "RetrieveFn",
    "builtin_tools",
    "context_tool_spec",
    "filter_from_arguments",
    "qdrant_retrieve",
    "render_chunks",
    "search_tool_spec",
]

_log = get_logger(__name__)

_MS = 1000.0

#: Name of the corpus-search tool the model may call inside the loop.
SEARCH_TOOL_NAME = "search_corpus"

#: Name of the clock / tenant-context tool.
CONTEXT_TOOL_NAME = "current_context"


@runtime_checkable
class RetrieveFn(Protocol):
    """Callable the search tool delegates to.

    The orchestrator injects the full stage-5 retriever (rerank, dedupe, MMR and all)
    so a tool-issued search behaves identically to a pipeline search. When nothing is
    injected the tool falls back to :func:`qdrant_retrieve`, which is a real
    ACL-filtered hybrid search — degraded in ranking quality, never in tenancy.
    """

    async def __call__(
        self,
        *,
        query: str,
        principal: Principal,
        filters: MetadataFilter | None,
        top_n: int,
    ) -> RetrievalResult:
        """Run one retrieval.

        Args:
            query: The search text.
            principal: The caller, used for the ACL filter.
            filters: Metadata facets to apply on top of the ACL filter.
            top_n: Maximum chunks to return.

        Returns:
            The retrieval result.
        """
        ...


# ------------------------------------------------------------------ tool specs


def search_tool_spec(settings: Settings | None = None) -> ToolSpec:
    """Build the ``search_corpus`` tool definition.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The tool spec, with ``top_n`` bounded by ``retrieval_limit``.
    """
    cfg = settings or get_settings()
    return ToolSpec(
        name=SEARCH_TOOL_NAME,
        kind=ToolKind.RETRIEVAL,
        description=(
            "Search the indexed corpus again with different terms or filters. Call "
            "this when the sources already supplied do not answer the question, when "
            "you need a different document type or time range, or when the user asks "
            "a follow-up that the first retrieval did not cover. Results are always "
            "restricted to what the current user is allowed to see."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Standalone search question. Resolve pronouns yourself — the "
                        "tool has no conversation history."
                    ),
                    "minLength": 1,
                },
                "doc_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these doc_type values.",
                },
                "source_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "blob | sharepoint | http | sql | upload | local.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to documents carrying any of these tags.",
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ISO language codes to restrict to.",
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to specific document ids.",
                },
                "section_prefix": {
                    "type": "string",
                    "description": "Only chunks at or beneath this heading.",
                },
                "date_from": {
                    "type": "string",
                    "description": "ISO-8601 lower bound on source_modified_at.",
                },
                "date_to": {
                    "type": "string",
                    "description": "ISO-8601 upper bound on source_modified_at.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many chunks to return.",
                    "minimum": 1,
                    "maximum": cfg.retrieval_limit,
                    "default": cfg.retrieval_top_n,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        max_result_chars=cfg.tool_max_result_chars,
    )


def context_tool_spec(settings: Settings | None = None) -> ToolSpec:
    """Build the ``current_context`` tool definition.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The tool spec.
    """
    cfg = settings or get_settings()
    return ToolSpec(
        name=CONTEXT_TOOL_NAME,
        kind=ToolKind.RETRIEVAL,
        description=(
            "Return the current date and time plus the calling user's tenant, roles "
            "and clearance level. Call this before answering anything that depends "
            "on 'today', 'this quarter', 'current' or on who is asking. Never guess "
            "the date."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone for the rendered local time, e.g. "
                        f"'Europe/Berlin'. Defaults to {cfg.ingest_timezone!r}."
                    ),
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        max_result_chars=cfg.tool_max_result_chars,
    )


def builtin_tools(settings: Settings | None = None) -> list[RegisteredTool]:
    """Return the always-present tools, policy-resolved.

    Both built-ins stay inside the process, so their egress ceiling is the highest
    classification the principal can already see — nothing leaves the platform.

    Args:
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The registered built-in tools.
    """
    cfg = settings or get_settings()
    config = tool_config(cfg)
    policy = ToolPolicy(
        max_classification=Classification.RESTRICTED,
        allow_pii_in_arguments=True,
        response_pii_scan=False,
    )
    return [
        register_tool(search_tool_spec(cfg), policy=policy, config=config),
        register_tool(context_tool_spec(cfg), policy=policy, config=config),
    ]


# ------------------------------------------------------------------ retrieval


def _parse_moment(value: Any) -> datetime | None:
    """Parse an ISO-8601 date or datetime supplied by the model.

    Args:
        value: The raw argument value.

    Returns:
        A timezone-aware UTC datetime, or None when the value is absent or
        unparseable — a bad date narrows nothing rather than failing the call.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def filter_from_arguments(
    arguments: Mapping[str, Any], *, base: MetadataFilter | None = None
) -> MetadataFilter | None:
    """Build the metadata filter for a tool-issued search.

    Args:
        arguments: Validated tool arguments.
        base: The filter the caller already applied. The model's facets are
            intersected with it via :meth:`MetadataFilter.merged_with`, so a tool call
            can only ever *narrow* what the user was already allowed to search.

    Returns:
        The merged filter, or None when neither side constrains anything.
    """

    def as_list(key: str) -> list[str] | None:
        """Read a list-of-strings argument.

        Args:
            key: Argument name.

        Returns:
            The list, or None when absent or empty.
        """
        value = arguments.get(key)
        if not isinstance(value, (list, tuple)) or not value:
            return None
        return [str(item) for item in value]

    section = arguments.get("section_prefix")
    extracted = MetadataFilter(
        doc_types=as_list("doc_types"),
        source_types=as_list("source_types"),
        tags=as_list("tags"),
        languages=as_list("languages"),
        document_ids=as_list("document_ids"),
        section_prefix=str(section) if isinstance(section, str) and section else None,
        date_from=_parse_moment(arguments.get("date_from")),
        date_to=_parse_moment(arguments.get("date_to")),
    )
    if base is None:
        return extracted
    return base.merged_with(extracted)


async def qdrant_retrieve(
    *,
    query: str,
    principal: Principal,
    filters: MetadataFilter | None,
    top_n: int,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Run an ACL-filtered hybrid search directly against Qdrant.

    The fallback used when the orchestrator injects no retriever. It is a real
    dense + sparse + server-side-RRF query with the full ACL filter; it simply skips
    the rerank/MMR refinements stage 5 adds.

    Args:
        query: The search text.
        principal: The caller.
        filters: Metadata facets applied on top of the ACL filter.
        top_n: Maximum chunks to return.
        settings: Platform settings; defaults to :func:`get_settings`.

    Returns:
        The retrieval result, with ``retrieval_stage`` set to ``fusion``.
    """
    cfg = settings or get_settings()
    started = time.perf_counter()
    provider = get_embedding_provider(cfg)
    embedded = await provider.embed_query(query)
    embed_ms = (time.perf_counter() - started) * _MS

    qfilter = build_acl_filter(principal, filters)
    client = await get_client(cfg)
    fused_started = time.perf_counter()
    points = await hybrid_search(
        client,
        collection=cfg.qdrant_chunks_collection,
        query_text=query,
        dense=embedded.dense,
        sparse=embedded.sparse,
        qfilter=qfilter,
        limit=min(top_n, cfg.retrieval_limit),
        prefetch_limit=cfg.retrieval_prefetch_limit,
        fusion=cfg.retrieval_fusion,
    )
    fuse_ms = (time.perf_counter() - fused_started) * _MS

    chunks: list[RetrievedChunk] = []
    for point in points:
        if not point.payload:
            continue
        chunks.append(
            RetrievedChunk(
                payload=ChunkPayload.from_qdrant_payload(point.payload),
                fusion_score=float(point.score or 0.0),
                final_score=float(point.score or 0.0),
                retrieval_stage=RetrievalStage.FUSION.value,
            )
        )
    return RetrievalResult(
        chunks=chunks[:top_n],
        queries_used=[query],
        filter_applied=serialise_filter(qfilter),
        total_candidates=len(points),
        after_dedupe=len(chunks),
        after_rerank=len(chunks),
        latency_ms={"embed": embed_ms, "fuse": fuse_ms},
    )


def render_chunks(result: RetrievalResult, *, snippet_chars: int) -> str:
    """Render retrieved chunks as numbered sources for the model.

    Args:
        result: The retrieval result.
        snippet_chars: Characters of chunk text to include per source.

    Returns:
        A numbered block, or an explicit "no matching content" marker so the model
        does not read an empty result as a transport failure.
    """
    if not result.chunks:
        return (
            "No chunks in the indexed corpus matched that query for this user. "
            "Do not invent an answer; say what is and is not covered."
        )
    lines: list[str] = []
    for index, chunk in enumerate(result.chunks, start=1):
        payload = chunk.payload
        section = " > ".join(payload.section_path) if payload.section_path else ""
        header = f"[{index}] {payload.title or payload.document_id}"
        if section:
            header = f"{header} — {section}"
        lines.append(header)
        lines.append(f"    source: {payload.source_uri}")
        lines.append(f"    chunk_id: {payload.chunk_id}")
        text = payload.text.strip().replace("\n", " ")
        lines.append(f"    {text[:snippet_chars]}")
    return "\n".join(lines)


# ------------------------------------------------------------------- executor


class BuiltinExecutor:
    """Executes the built-in tools."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        """Initialise the executor.

        Args:
            settings: Platform settings; defaults to :func:`get_settings`.
        """
        self.settings = settings or get_settings()

    async def execute(
        self,
        tool: RegisteredTool,
        *,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        principal: Principal,
        base_filter: MetadataFilter | None = None,
        retrieve: RetrieveFn | None = None,
    ) -> ToolResult:
        """Run one built-in tool.

        Args:
            tool: The registered built-in.
            tool_call_id: The model's ``tool_use`` block id.
            arguments: Validated tool arguments.
            principal: The caller.
            base_filter: The filter the turn is already scoped to.
            retrieve: Retrieval callable injected by the orchestrator.

        Returns:
            The tool result. Failures come back as ``is_error`` results, never as
            exceptions, so the loop can continue.
        """
        if tool.name == CONTEXT_TOOL_NAME:
            return self._context(
                tool,
                tool_call_id=tool_call_id,
                arguments=arguments,
                principal=principal,
            )
        if tool.name == SEARCH_TOOL_NAME:
            return await self._search(
                tool,
                tool_call_id=tool_call_id,
                arguments=arguments,
                principal=principal,
                base_filter=base_filter,
                retrieve=retrieve,
            )
        return ToolResult.failure(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            kind=ToolKind.RETRIEVAL,
            message=f"{tool.name!r} is not a built-in tool",
        )

    async def _search(
        self,
        tool: RegisteredTool,
        *,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        principal: Principal,
        base_filter: MetadataFilter | None,
        retrieve: RetrieveFn | None,
    ) -> ToolResult:
        """Run the corpus-search tool.

        Args:
            tool: The registered tool.
            tool_call_id: The model's ``tool_use`` block id.
            arguments: Validated tool arguments.
            principal: The caller.
            base_filter: The filter the turn is already scoped to.
            retrieve: Retrieval callable injected by the orchestrator.

        Returns:
            The tool result.
        """
        started = time.perf_counter()
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                kind=ToolKind.RETRIEVAL,
                message="'query' is required and must not be empty",
                latency_ms=(time.perf_counter() - started) * _MS,
            )
        requested = arguments.get("top_n")
        top_n = (
            int(requested)
            if isinstance(requested, int) and requested > 0
            else self.settings.retrieval_top_n
        )
        top_n = min(top_n, self.settings.retrieval_limit)
        filters = filter_from_arguments(arguments, base=base_filter)

        runner = retrieve
        try:
            if runner is None:
                result = await qdrant_retrieve(
                    query=query,
                    principal=principal,
                    filters=filters,
                    top_n=top_n,
                    settings=self.settings,
                )
            else:
                result = await runner(
                    query=query,
                    principal=principal,
                    filters=filters,
                    top_n=top_n,
                )
        except Exception as exc:
            _log.warning(
                "builtin_search_failed",
                tool=tool.name,
                tenant_id=principal.tenant_id,
                error=type(exc).__name__,
            )
            return ToolResult.failure(
                tool_call_id=tool_call_id,
                tool_name=tool.name,
                kind=ToolKind.RETRIEVAL,
                message=f"corpus search failed ({type(exc).__name__})",
                latency_ms=(time.perf_counter() - started) * _MS,
            )

        content = render_chunks(
            result, snippet_chars=self.settings.retrieval_snippet_chars
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            kind=ToolKind.RETRIEVAL,
            content=content,
            structured={
                "chunk_ids": result.chunk_ids,
                "document_ids": result.document_ids,
                "max_score": result.max_score,
                "total_candidates": result.total_candidates,
            },
            latency_ms=(time.perf_counter() - started) * _MS,
        ).truncate(tool.max_result_chars)

    def _context(
        self,
        tool: RegisteredTool,
        *,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        principal: Principal,
    ) -> ToolResult:
        """Run the clock / tenant-context tool.

        Args:
            tool: The registered tool.
            tool_call_id: The model's ``tool_use`` block id.
            arguments: Validated tool arguments.
            principal: The caller.

        Returns:
            The tool result. Contains no user content, so no redaction applies.
        """
        started = time.perf_counter()
        now = datetime.now(UTC)
        requested = arguments.get("timezone")
        zone_name = (
            str(requested)
            if isinstance(requested, str) and requested
            else self.settings.ingest_timezone
        )
        try:
            zone = ZoneInfo(zone_name)
        except (ZoneInfoNotFoundError, ValueError):
            zone_name = "UTC"
            zone = UTC
        payload: dict[str, Any] = {
            "utc_now": now.isoformat(),
            "local_now": now.astimezone(zone).isoformat(),
            "timezone": zone_name,
            "iso_weekday": now.isoweekday(),
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "roles": sorted(principal.roles),
            "is_admin": principal.is_admin(),
            "max_classification": principal.max_classification.value,
            "clearance_rank": principal.clearance_rank(),
            "environment": self.settings.env,
        }
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            kind=ToolKind.RETRIEVAL,
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            structured=payload,
            latency_ms=(time.perf_counter() - started) * _MS,
        ).truncate(tool.max_result_chars)
