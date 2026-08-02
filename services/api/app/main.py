"""The ASGI application: lifespan, routing and OpenAPI metadata.

``uvicorn app.main:app`` is the entry point `services/api/Dockerfile` runs.

Startup does the work that would otherwise land on the first user's turn and make
it look broken: the Qdrant client is built and its collections ensured, the
FastEmbed dense, sparse and cross-encoder models are loaded, the Entra JWKS is
pre-fetched, and the database is probed. None of it is allowed to abort the
process — a container that refuses to start because a dependency is briefly
unreachable cannot serve ``/health``, cannot be diagnosed, and turns a degraded
dependency into an outage. Every failure is logged loudly and reported by
``/readyz``, which is what an orchestrator should gate traffic on.

Shutdown is the mirror image: the model client, the HTTP clients, the Redis
connections, the Qdrant transport and the database pool are all released, and the
tracer is flushed, so a scale-in event does not leave Azure holding abandoned
connections.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI

from app import SERVICE_VERSION, api_setting
from app.auth.entra import (
    assert_dev_mode_allowed,
    get_token_validator,
    reset_token_validator,
)
from app.deps import reset_rate_limiter
from app.middleware import install_exception_handlers, install_middleware
from app.routers import PUBLIC_ROUTERS, VERSIONED_ROUTERS
from ragcore.db import check_database, dispose_engine
from ragcore.embeddings import get_embedding_provider
from ragcore.llm import get_llm_client
from ragcore.logging import configure_logging, get_logger
from ragcore.observability import get_tracer, set_build_info
from ragcore.observability.langfuse import shutdown_tracer
from ragcore.rerank import get_reranker
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore.client import close_all_clients, get_client
from ragcore.vectorstore.collections import ensure_collections

__all__ = ["app", "create_app", "lifespan"]

_log = get_logger(__name__)

DESCRIPTION = """\
Multi-tenant retrieval-augmented generation over an ACL-aware hybrid index.

Every endpoint below `/api/v1` requires `Authorization: Bearer <Entra JWT>`. The
token's `tid` is the tenant boundary: it scopes every vector query and every SQL
statement, and no request body may name a tenant. `oid`, `roles` and `groups`
decide document visibility, and the derived clearance ceiling caps it further.

`POST /chat` streams `text/event-stream`. Events are `session`, `retrieval`,
`thinking`, `tool_call`, `tool_result`, `token`, `citations`, `context_stats`,
`guardrail`, `usage`, `done` and `error`; a client must ignore unknown names.
Errors are RFC 7807 `application/problem+json` on every route.
"""

TAGS_METADATA: list[dict[str, Any]] = [
    {"name": "chat", "description": "Ask a question; streaming or single-body."},
    {"name": "search", "description": "Retrieval without generation, and identity."},
    {"name": "sessions", "description": "Conversations, messages and compaction."},
    {"name": "documents", "description": "Corpus listing, upload and provenance."},
    {"name": "memory", "description": "Long-term memory, profile and consent."},
    {"name": "feedback", "description": "Rate an answer."},
    {"name": "eval", "description": "Golden-set evaluation runs."},
    {"name": "admin", "description": "Tenant administration. Requires `rag.admin`."},
    {"name": "health", "description": "Liveness, readiness and metrics."},
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Warm every dependency on startup and release it on shutdown.

    Args:
        application: The FastAPI application, whose ``state`` carries the
            resolved settings and the startup verdict.

    Yields:
        None, for the lifetime of the process.

    Raises:
        ConfigError: When ``entra_dev_mode`` is on in production. This one *does*
            abort startup: a process that would accept unsigned principals in
            production must not serve traffic at all.
    """
    settings: Settings = application.state.settings
    configure_logging(settings)
    assert_dev_mode_allowed(settings)
    if settings.entra_dev_mode:
        _log.warning(
            "entra_dev_mode_enabled",
            env=settings.env,
            header=settings.entra_dev_principal_header,
            detail=(
                "unsigned principals are accepted; never do this outside development"
            ),
        )

    set_build_info(
        service=settings.service_name,
        env=settings.env,
        release=settings.langfuse_release,
    )
    ready: dict[str, bool] = {}

    ready["qdrant"] = await _start_qdrant(settings)
    ready["models"] = await _warm_models(settings)
    ready["database"] = await _probe_database(settings)
    ready["identity"] = await get_token_validator(settings).warm_up()

    application.state.startup = ready
    _log.info(
        "api_started",
        env=settings.env,
        version=SERVICE_VERSION,
        prefix=settings.api_prefix,
        checks=ready,
    )
    try:
        yield
    finally:
        await _shutdown(settings)


async def _start_qdrant(settings: Settings) -> bool:
    """Build the Qdrant client and ensure the collections exist.

    Args:
        settings: Active settings.

    Returns:
        True when the collections are present. ``ensure_collections`` is
        idempotent and tolerates a concurrent creator, so several replicas may
        start at once.
    """
    try:
        client = await get_client(settings)
        if api_setting(settings, "api_ensure_collections"):
            await ensure_collections(client, settings)
    except Exception as exc:
        _log.error("qdrant_startup_failed", error=type(exc).__name__)
        return False
    return True


async def _warm_models(settings: Settings) -> bool:
    """Load the embedding and rerank models before the first request.

    FastEmbed loads ONNX weights lazily on first use, which would put a
    multi-second stall inside the first user's turn and, worse, inside the event
    loop. Loading here moves that cost to startup where a readiness probe can see
    it.

    Args:
        settings: Active settings.

    Returns:
        True when both models are resident, or when warming is disabled.
    """
    if not api_setting(settings, "api_warm_models"):
        _log.info("model_warmup_skipped")
        return True
    ok = True
    try:
        provider = get_embedding_provider(settings)
        warm = getattr(provider, "warm_up", None)
        if warm is not None:
            await warm()
    except Exception as exc:
        _log.error("embedding_warmup_failed", error=type(exc).__name__)
        ok = False
    try:
        reranker = get_reranker(settings)
        warm = getattr(reranker, "warm_up", None)
        if warm is not None:
            await warm()
    except Exception as exc:
        _log.error("rerank_warmup_failed", error=type(exc).__name__)
        ok = False
    return ok


async def _probe_database(settings: Settings) -> bool:
    """Check that the database answers.

    Args:
        settings: Active settings.

    Returns:
        True when a trivial query succeeded.
    """
    try:
        return await check_database(settings)
    except Exception as exc:
        _log.error("database_startup_probe_failed", error=type(exc).__name__)
        return False


async def _shutdown(settings: Settings) -> None:
    """Release every long-lived resource.

    Each teardown is independent: one failing must not skip the rest, or a
    restart leaks whatever came after it.

    Args:
        settings: Active settings.
    """
    from app.rag.memory.long_term import reset_long_term_memory
    from app.rag.memory.semantic_cache import reset_semantic_cache
    from app.rag.memory.short_term import get_short_term_memory
    from app.rag.tools.rest_tool import reset_rest_executor

    with suppress(Exception):
        await get_llm_client(settings).aclose()
    with suppress(Exception):
        await get_short_term_memory(settings).aclose()
    with suppress(Exception):
        await reset_rest_executor()
    with suppress(Exception):
        await reset_token_validator()
    with suppress(Exception):
        await reset_rate_limiter()
    with suppress(Exception):
        reset_semantic_cache()
    with suppress(Exception):
        reset_long_term_memory()
    with suppress(Exception):
        await close_all_clients()
    with suppress(Exception):
        await dispose_engine(settings)
    with suppress(Exception):
        get_tracer(settings).flush()
        shutdown_tracer(settings)
    _log.info("api_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Active settings. Defaults to the process settings; injectable
            so a test can build an app against an alternative configuration.

    Returns:
        The configured :class:`~fastapi.FastAPI` instance.
    """
    cfg = settings or get_settings()
    configure_logging(cfg)

    docs_enabled = bool(api_setting(cfg, "api_docs_enabled")) and not cfg.is_production
    application = FastAPI(
        title="Productionizing RAG API",
        version=SERVICE_VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        root_path=cfg.api_root_path,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        contact={"name": "Platform team"},
        license_info={"name": "Proprietary"},
        swagger_ui_parameters={"persistAuthorization": True},
    )
    application.state.settings = cfg
    application.state.startup = {}

    install_middleware(application, cfg)
    install_exception_handlers(application, cfg)

    for router in PUBLIC_ROUTERS:
        application.include_router(router)
    for router in VERSIONED_ROUTERS:
        application.include_router(router, prefix=cfg.api_prefix)
    return application


#: The application `services/api/Dockerfile` and `uvicorn app.main:app` load.
app = create_app()
