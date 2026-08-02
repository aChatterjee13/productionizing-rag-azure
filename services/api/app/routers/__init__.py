"""The HTTP surface from `docs/CONTRACTS.md`.

:data:`VERSIONED_ROUTERS` is mounted under ``settings.api_prefix`` and requires a
bearer token on every route. :data:`PUBLIC_ROUTERS` is mounted at the application
root and requires none — ``/health``, ``/readyz`` and ``/metrics`` must answer
without the identity provider, or a rolling deployment cannot come up while Entra
is degraded.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.routers import (
    admin,
    chat,
    documents,
    eval,
    feedback,
    health,
    memory,
    search,
    sessions,
)

__all__ = [
    "PUBLIC_ROUTERS",
    "VERSIONED_ROUTERS",
    "admin",
    "chat",
    "documents",
    "eval",
    "feedback",
    "health",
    "memory",
    "search",
    "sessions",
]

#: Routers mounted under the versioned prefix, in the contract's table order.
VERSIONED_ROUTERS: tuple[APIRouter, ...] = (
    chat.router,
    search.router,
    sessions.router,
    documents.router,
    memory.router,
    feedback.router,
    eval.router,
    admin.router,
)

#: Routers mounted at the root, outside authentication.
PUBLIC_ROUTERS: tuple[APIRouter, ...] = (health.router,)
