"""Request-scoped dependencies: identity, tenancy, storage and rate limiting.

Every tenant-scoped handler takes :data:`CurrentPrincipal`. That dependency is the
only place a :class:`~ragcore.models.acl.Principal` enters the service, and every
repository call and every Qdrant filter downstream derives its tenant from it — a
handler never reads a tenant id from a path, a query string or a body.

:func:`require_roles` is the authorisation counterpart, used as a router-level
dependency on the admin surface.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Annotated, Any

import anyio
from fastapi import Depends, Header, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.entra import get_token_validator
from ragcore.db import get_sessionmaker
from ragcore.errors import AuthError, RagError
from ragcore.logging import bind_request_context, get_logger
from ragcore.models.acl import Principal
from ragcore.settings import Settings, get_settings

__all__ = [
    "CurrentPrincipal",
    "DbSession",
    "PageLimit",
    "RateLimit",
    "RateLimiter",
    "SettingsDep",
    "TenantAdmin",
    "get_db",
    "get_principal",
    "get_rate_limiter",
    "require_roles",
    "reset_rate_limiter",
]

_log = get_logger(__name__)


class RateLimitedError(RagError):
    """The caller exceeded their per-minute request budget."""

    status_code = 429
    code = "rate_limited"


# ------------------------------------------------------------------- settings


def settings_dependency() -> Settings:
    """Provide the process settings to a handler.

    Returns:
        The cached :class:`~ragcore.settings.Settings`.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_dependency)]


# ------------------------------------------------------------------- database


async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a database session for the duration of a request.

    The session is handed out with no transaction begun, as
    :mod:`ragcore.db.repositories` expects; the handler commits.

    Yields:
        An :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
    """
    factory = get_sessionmaker(get_settings())
    session = factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


DbSession = Annotated[AsyncSession, Depends(get_db)]


# ------------------------------------------------------------------- identity


async def get_principal(
    request: Request,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the authenticated caller.

    Args:
        request: The inbound request, for the dev-principal header and audit
            context.
        settings: Active settings.
        authorization: The ``Authorization`` header.

    Returns:
        The validated :class:`~ragcore.models.acl.Principal`.

    Raises:
        AuthError: When no credential is present or validation fails.
    """
    validator = get_token_validator(settings)
    dev_header = request.headers.get(settings.entra_dev_principal_header)
    principal = await validator.principal_for_header(
        authorization, dev_principal=dev_header
    )
    # Bind tenancy onto every subsequent log line from this request. Deliberately
    # ids only: email and display name are personal data and never enter a log.
    bind_request_context(
        request_id=getattr(request.state, "request_id", None),
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        route=request.url.path,
    )
    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_roles(
    *roles: str, require_all: bool = False
) -> Callable[[Principal, Settings], Principal]:
    """Build a dependency that admits only callers holding the given app roles.

    The platform administrator role is always accepted, and its name is read from
    ``settings.entra_admin_role`` at request time — so renaming it in one place
    renames it everywhere, and no call site writes it as a literal. Calling this
    with **no** roles therefore produces an administrator-only gate.

    Args:
        *roles: App role names, any one of which admits the caller.
        require_all: Require every listed role rather than any one of them.

    Returns:
        A FastAPI dependency returning the principal when permitted.
    """
    wanted = tuple(roles)

    def _guard(principal: CurrentPrincipal, settings: SettingsDep) -> Principal:
        """Admit or refuse the caller.

        Args:
            principal: The resolved caller.
            settings: Active settings supplying the admin role name.

        Returns:
            The principal, unchanged.

        Raises:
            AuthError: 403 when the caller lacks the required role. The message
                names the requirement, not the caller's roles: telling someone
                which roles they *do* hold is an information leak on a shared
                deployment.
        """
        if principal.is_admin(settings.entra_admin_role):
            return principal
        held = set(principal.roles)
        satisfied = bool(wanted) and (
            held.issuperset(wanted) if require_all else bool(held.intersection(wanted))
        )
        if satisfied:
            return principal
        _log.warning(
            "authorisation_denied",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            required=list(wanted) or [settings.entra_admin_role],
        )
        requirement = " and ".join(wanted) if wanted else settings.entra_admin_role
        raise AuthError(
            f"this operation requires the {requirement} role",
            code="auth_missing_role",
            status_code=403,
        )

    return _guard


#: Dependency admitting only platform administrators.
TenantAdmin = Annotated[Principal, Depends(require_roles())]


# --------------------------------------------------------------- rate limiting


class RateLimiter:
    """Per-``(tenant, user)`` token bucket, Redis-backed with a local fallback.

    Redis is what makes the budget hold across replicas. When it is disabled,
    absent or unreachable the limiter degrades to an in-process bucket and logs
    once: a per-replica budget is weaker than a shared one but far better than no
    limit at all, and a Redis outage must not take the API down with it.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the limiter.

        Args:
            settings: Active settings supplying the budget, burst and key prefix.
        """
        self._settings = settings
        self._prefix = str(settings.api_rate_limit_prefix)
        self._per_minute = int(settings.api_rate_limit_per_minute)
        self._burst = max(1, int(settings.api_rate_limit_burst))
        self._local: dict[str, tuple[float, float]] = {}
        self._lock = anyio.Lock()
        self._redis: Any | None = None
        self._redis_ready: bool | None = None

    @property
    def enabled(self) -> bool:
        """Whether limiting is switched on.

        Returns:
            True when ``api_rate_limit_enabled``.
        """
        return bool(self._settings.api_rate_limit_enabled)

    def _key(self, principal: Principal) -> str:
        """Bucket key for a caller.

        Args:
            principal: The caller.

        Returns:
            A tenant-first key, so one tenant's traffic cannot evict another's
            counters in a shared Redis instance.
        """
        return f"{self._prefix}{principal.tenant_id}:{principal.user_id}"

    async def _client(self) -> Any | None:
        """Resolve the Redis client, once.

        Returns:
            An async Redis client, or None when Redis is disabled or the package
            is not installed.
        """
        if self._redis_ready is not None:
            return self._redis
        async with self._lock:
            if self._redis_ready is not None:
                return self._redis
            self._redis_ready = False
            if not self._settings.redis_enabled:
                _log.info("rate_limiter_backend", backend="memory", reason="disabled")
                return None
            try:
                import redis.asyncio as redis_asyncio
            except ImportError:
                _log.warning(
                    "rate_limiter_backend", backend="memory", reason="redis_missing"
                )
                return None
            self._redis = redis_asyncio.from_url(
                self._settings.redis_url, decode_responses=True
            )
            self._redis_ready = True
            _log.info("rate_limiter_backend", backend="redis")
            return self._redis

    async def acquire(self, principal: Principal) -> tuple[bool, float]:
        """Consume one token for a caller.

        Args:
            principal: The caller.

        Returns:
            An ``(allowed, retry_after_seconds)`` pair.
        """
        if not self.enabled or self._per_minute <= 0:
            return True, 0.0
        key = self._key(principal)
        client = await self._client()
        if client is not None:
            allowed = await self._acquire_redis(client, key)
            if allowed is not None:
                return allowed
        return await self._acquire_local(key)

    async def _acquire_redis(self, client: Any, key: str) -> tuple[bool, float] | None:
        """Consume a token from the shared bucket.

        A fixed window of one minute is used rather than a sliding log: it is one
        ``INCR`` plus one ``EXPIRE``, which is cheap enough to sit in front of
        every request, and the burst allowance covers the boundary effect.

        Args:
            client: The Redis client.
            key: Bucket key.

        Returns:
            The verdict, or None when Redis failed and the caller should fall
            back to the in-process bucket.
        """
        window = int(time.time() // 60)
        try:
            pipeline = client.pipeline()
            pipeline.incr(f"{key}:{window}")
            pipeline.expire(f"{key}:{window}", 120)
            counted, _ = await pipeline.execute()
        except Exception as exc:
            _log.warning("rate_limiter_redis_failed", error=type(exc).__name__)
            self._redis_ready = False
            self._redis = None
            return None
        limit = self._per_minute + self._burst
        if int(counted) > limit:
            return False, 60.0 - (time.time() % 60.0)
        return True, 0.0

    async def _acquire_local(self, key: str) -> tuple[bool, float]:
        """Consume a token from the in-process bucket.

        Args:
            key: Bucket key.

        Returns:
            An ``(allowed, retry_after_seconds)`` pair.
        """
        rate = self._per_minute / 60.0
        capacity = float(self._per_minute + self._burst)
        now = time.monotonic()
        async with self._lock:
            tokens, updated = self._local.get(key, (capacity, now))
            tokens = min(capacity, tokens + (now - updated) * rate)
            if tokens < 1.0:
                self._local[key] = (tokens, now)
                return False, max(0.0, (1.0 - tokens) / rate)
            self._local[key] = (tokens - 1.0, now)
            return True, 0.0

    async def aclose(self) -> None:
        """Release the Redis connection, if one was opened."""
        client, self._redis, self._redis_ready = self._redis, None, None
        if client is not None:
            with_close = getattr(client, "aclose", None) or client.close
            await with_close()


_LIMITERS: dict[str, RateLimiter] = {}


def get_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    """Return the process-wide rate limiter.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        The cached limiter, keyed on the Redis URL and the budget.
    """
    cfg = settings or get_settings()
    key = f"{cfg.redis_url}|{cfg.api_rate_limit_per_minute}"
    limiter = _LIMITERS.get(key)
    if limiter is None:
        limiter = RateLimiter(cfg)
        _LIMITERS[key] = limiter
    return limiter


async def reset_rate_limiter() -> None:
    """Close and drop every cached limiter. Shutdown hook and test helper."""
    limiters: Sequence[RateLimiter] = list(_LIMITERS.values())
    _LIMITERS.clear()
    for limiter in limiters:
        await limiter.aclose()


async def enforce_rate_limit(
    principal: CurrentPrincipal, settings: SettingsDep
) -> Principal:
    """Charge one request against the caller's budget.

    Args:
        principal: The resolved caller.
        settings: Active settings.

    Returns:
        The principal, unchanged.

    Raises:
        RateLimitedError: 429 with a ``Retry-After`` hint in ``detail`` when the
            budget is exhausted.
    """
    allowed, retry_after = await get_rate_limiter(settings).acquire(principal)
    if allowed:
        return principal
    _log.warning(
        "rate_limited",
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        per_minute=settings.api_rate_limit_per_minute,
    )
    raise RateLimitedError(
        "request budget exceeded; retry shortly",
        detail={
            "retry_after_seconds": round(retry_after, 1),
            "limit_per_minute": settings.api_rate_limit_per_minute,
        },
    )


RateLimit = Annotated[Principal, Depends(enforce_rate_limit)]


# ----------------------------------------------------------------- pagination


def page_limit(
    settings: SettingsDep,
    limit: Annotated[int | None, Query(ge=1, description="Rows to return.")] = None,
) -> int:
    """Resolve and clamp a list endpoint's page size.

    Args:
        settings: Active settings supplying the default and the ceiling.
        limit: Requested page size.

    Returns:
        The effective page size, never above ``api_max_page_size`` — an unbounded
        list is a denial-of-service primitive on a multi-tenant database.
    """
    default = int(settings.api_default_page_size)
    ceiling = int(settings.api_max_page_size)
    return min(ceiling, int(limit) if limit else default)


PageLimit = Annotated[int, Depends(page_limit)]
