"""Qdrant connection management.

One :class:`~qdrant_client.AsyncQdrantClient` per distinct endpoint per process.
The client owns an HTTP (or gRPC) connection pool, so creating one per request would
both leak sockets and destroy keep-alive; :func:`get_client` therefore hands out a
cached singleton keyed on the connection parameters.

:class:`~ragcore.settings.Settings` is a pydantic model and unhashable, so it can
never be an ``lru_cache`` key. The cache key here is a tuple of the ``qdrant_*``
connection fields with the API key reduced to a short digest — the key material
itself never enters a dict key, a log line or a repr.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient

from ragcore.settings import Settings, get_settings

__all__ = [
    "check_qdrant",
    "close_all_clients",
    "close_client",
    "get_client",
]

# structlog's lazy proxy: no configuration side effect at import time, and it picks
# up ragcore.logging.configure_logging() as soon as the process calls it.
_log = structlog.get_logger(__name__)

#: Cached clients keyed by :func:`_client_key`.
_CLIENTS: dict[tuple[Any, ...], AsyncQdrantClient] = {}


def _client_key(settings: Settings) -> tuple[Any, ...]:
    """Build the cache key identifying one Qdrant endpoint.

    Args:
        settings: Settings carrying the ``qdrant_*`` connection fields.

    Returns:
        A hashable tuple. The API key is represented by a truncated SHA-256 digest
        so rotating the key yields a fresh client without the secret ever becoming
        part of a dict key.
    """
    digest = hashlib.sha256((settings.qdrant_api_key or "").encode()).hexdigest()[:16]
    return (
        settings.qdrant_url,
        settings.qdrant_prefer_grpc,
        settings.qdrant_grpc_port,
        settings.qdrant_https,
        int(settings.qdrant_timeout_seconds),
        digest,
    )


async def get_client(settings: Settings | None = None) -> AsyncQdrantClient:
    """Return the process-wide async Qdrant client for the configured endpoint.

    Args:
        settings: Settings to read the ``qdrant_*`` fields from. Defaults to
            :func:`ragcore.settings.get_settings`.

    Returns:
        A cached :class:`~qdrant_client.AsyncQdrantClient`. Repeated calls with
        equivalent settings return the same instance.
    """
    cfg = settings or get_settings()
    key = _client_key(cfg)
    existing = _CLIENTS.get(key)
    if existing is not None:
        return existing

    # Constructing the client performs no I/O, so there is no await between the
    # cache miss and the cache fill and therefore no need for a lock.
    client = AsyncQdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        prefer_grpc=cfg.qdrant_prefer_grpc,
        grpc_port=cfg.qdrant_grpc_port,
        https=cfg.qdrant_https,
        timeout=int(cfg.qdrant_timeout_seconds),
        # The version handshake emits a warning on every construction when the
        # server runs a different minor version; the query and payload APIs used
        # here are stable across those.
        check_compatibility=False,
    )
    _CLIENTS[key] = client
    _log.info(
        "qdrant.client.created",
        url=cfg.qdrant_url,
        prefer_grpc=cfg.qdrant_prefer_grpc,
        timeout_seconds=cfg.qdrant_timeout_seconds,
    )
    return client


async def close_client(settings: Settings | None = None) -> None:
    """Close and forget the client for one endpoint.

    Args:
        settings: Settings identifying the endpoint. Defaults to
            :func:`ragcore.settings.get_settings`.
    """
    cfg = settings or get_settings()
    client = _CLIENTS.pop(_client_key(cfg), None)
    if client is None:
        return
    await client.close()
    _log.info("qdrant.client.closed", url=cfg.qdrant_url)


async def close_all_clients() -> None:
    """Close every cached client.

    Intended for an application shutdown hook and for test teardown, so a process
    never exits with unclosed sockets.
    """
    for key, client in list(_CLIENTS.items()):
        _CLIENTS.pop(key, None)
        await client.close()
    _log.info("qdrant.client.closed_all")


async def check_qdrant(
    client: AsyncQdrantClient | None = None,
    settings: Settings | None = None,
) -> bool:
    """Readiness probe for Qdrant.

    Used by ``GET /readyz``. Never raises: a probe that throws would turn a degraded
    dependency into a 500 on the health endpoint itself.

    Args:
        client: Existing client to probe. Defaults to :func:`get_client`.
        settings: Settings used when ``client`` is omitted.

    Returns:
        True when Qdrant answered a collection listing, False otherwise.
    """
    cfg = settings or get_settings()
    try:
        probe = client or await get_client(cfg)
        await probe.get_collections()
    except Exception as exc:
        _log.warning("qdrant.probe.failed", error=type(exc).__name__)
        return False
    return True
