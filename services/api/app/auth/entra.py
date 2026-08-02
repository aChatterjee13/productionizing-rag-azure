"""Microsoft Entra ID token validation.

The whole of the platform's authentication lives here: JWKS retrieval with
rotation handling, RS256 signature verification, ``iss``/``aud``/``exp``/``nbf``
checks, directory (``tid``) pinning, and the Microsoft Graph fallback for the
group-overage claim. :mod:`app.auth.principal` then turns the verified claims into
a :class:`~ragcore.models.acl.Principal`, which is the only thing the rest of the
service ever sees.

Three properties this module is responsible for:

1. **A token is never trusted before its signature is.** Claims are read only from
   :func:`jwt.decode` output; nothing anywhere reads an unverified payload except
   the header ``kid``, which selects a key and is verified by the signature check
   that follows.
2. **Key rotation is handled without a restart.** An unknown ``kid`` triggers one
   refetch, rate-limited by ``entra_jwks_refresh_min_seconds`` so a stream of
   tokens signed by a key that will never exist cannot turn into a denial of
   service against Microsoft's endpoint.
3. **Dev mode is refused in production.** :class:`ragcore.settings.Settings`
   already refuses to construct with ``entra_dev_mode`` in production;
   :func:`assert_dev_mode_allowed` re-checks at startup and on every request, so
   an environment mutated after construction still fails closed and loudly.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
import jwt
from jwt import PyJWK, PyJWKSet

from app.auth.principal import (
    claim_list,
    principal_from_claims,
    requires_group_lookup,
)
from ragcore.errors import AuthError, ConfigError
from ragcore.logging import get_logger
from ragcore.models.acl import Principal
from ragcore.settings import Settings, get_settings

__all__ = [
    "EntraTokenValidator",
    "JWKSCache",
    "assert_dev_mode_allowed",
    "get_token_validator",
    "reset_token_validator",
]

_log = get_logger(__name__)

#: Bearer scheme prefix, compared case-insensitively as RFC 6750 requires.
_BEARER = "bearer"


def assert_dev_mode_allowed(settings: Settings) -> None:
    """Refuse unsigned dev principals outside a non-production environment.

    Args:
        settings: Active settings.

    Raises:
        ConfigError: When ``entra_dev_mode`` is on and ``env == "production"``.
    """
    if settings.entra_dev_mode and settings.is_production:
        msg = (
            "entra_dev_mode=true is refused when env='production': unsigned "
            "principals must never be accepted in production"
        )
        _log.error("entra_dev_mode_refused", env=settings.env)
        raise ConfigError(msg, code="auth_dev_mode_in_production")


# --------------------------------------------------------------------- JWKS


@dataclass(slots=True)
class JWKSCache:
    """Cached Entra signing keys with rotation-aware refresh.

    Attributes:
        settings: Active settings supplying the JWKS URL and cache TTL.
        keys: ``kid`` to :class:`jwt.PyJWK`, as of the last successful fetch.
        fetched_at: Monotonic timestamp of the last successful fetch.
        last_attempt_at: Monotonic timestamp of the last fetch attempt, successful
            or not. Rate-limits rotation-triggered refetches.
    """

    settings: Settings
    keys: dict[str, PyJWK] = field(default_factory=dict)
    fetched_at: float = 0.0
    last_attempt_at: float = 0.0
    _lock: anyio.Lock = field(default_factory=anyio.Lock, repr=False)

    @property
    def url(self) -> str:
        """JWKS endpoint for the pinned directory.

        Returns:
            The discovery keys URL.

        Raises:
            ConfigError: When ``entra_tenant_id`` is unset, so no issuer or JWKS
                endpoint can be derived.
        """
        url = self.settings.entra_jwks_url
        if not url:
            msg = (
                "entra_tenant_id is not configured: cannot validate tokens without "
                "an issuer and a JWKS endpoint"
            )
            raise ConfigError(msg, code="auth_not_configured")
        return url

    def _expired(self, now: float) -> bool:
        """Whether the cached document has aged out.

        Args:
            now: Current monotonic time.

        Returns:
            True when the cache is empty or older than ``entra_jwks_cache_seconds``.
        """
        if not self.keys:
            return True
        return (now - self.fetched_at) >= float(self.settings.entra_jwks_cache_seconds)

    async def key_for(self, kid: str, *, client: httpx.AsyncClient) -> PyJWK:
        """Resolve a signing key by ``kid``, refreshing on a miss.

        Args:
            kid: Key id from the token header.
            client: HTTP client used for the fetch.

        Returns:
            The matching :class:`jwt.PyJWK`.

        Raises:
            AuthError: When the key is unknown even after a refresh — which is what
                a forged or long-expired token looks like.
        """
        now = time.monotonic()
        if self._expired(now):
            await self.refresh(client=client, force=True)
        key = self.keys.get(kid)
        if key is not None:
            return key

        # Unknown kid: Entra rotates keys ahead of publishing them in tokens, so
        # one immediate refetch is correct. The floor stops a forged-kid flood from
        # becoming an outbound request per request.
        if (time.monotonic() - self.last_attempt_at) >= float(
            self.settings.entra_jwks_refresh_min_seconds
        ):
            await self.refresh(client=client, force=True)
            key = self.keys.get(kid)
            if key is not None:
                _log.info("entra_jwks_rotated", kid=kid, keys=len(self.keys))
                return key
        raise AuthError("token signing key is not known", code="auth_unknown_key")

    async def refresh(self, *, client: httpx.AsyncClient, force: bool = False) -> None:
        """Fetch the JWKS document.

        Args:
            client: HTTP client used for the fetch.
            force: Fetch even when the cached document is still fresh.

        Raises:
            AuthError: When the endpoint is unreachable and no cached keys exist.
                With cached keys present, a failed refresh degrades to the cache and
                logs, because rejecting every request over a transient network blip
                is worse than serving a key that is at most one TTL stale.
        """
        async with self._lock:
            now = time.monotonic()
            if not force and not self._expired(now):
                return
            self.last_attempt_at = now
            try:
                response = await client.get(
                    self.url,
                    timeout=float(self.settings.entra_jwks_timeout_seconds),
                )
                response.raise_for_status()
                document = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                if self.keys:
                    _log.warning(
                        "entra_jwks_refresh_failed",
                        error=type(exc).__name__,
                        cached_keys=len(self.keys),
                    )
                    return
                _log.error("entra_jwks_unavailable", error=type(exc).__name__)
                raise AuthError(
                    "identity provider keys are unavailable",
                    code="auth_jwks_unavailable",
                    status_code=503,
                ) from exc

            keys: dict[str, PyJWK] = {}
            for entry in PyJWKSet.from_dict(document).keys:
                if entry.key_id:
                    keys[entry.key_id] = entry
            if not keys:
                _log.error("entra_jwks_empty")
                if not self.keys:
                    raise AuthError(
                        "identity provider published no usable signing keys",
                        code="auth_jwks_empty",
                        status_code=503,
                    )
                return
            self.keys = keys
            self.fetched_at = time.monotonic()
            _log.info("entra_jwks_loaded", keys=len(keys))


# ------------------------------------------------------------- Graph groups


@dataclass(slots=True)
class _GroupCacheEntry:
    """One cached transitive group list.

    Attributes:
        groups: Group object ids.
        expires_at: Monotonic time after which the entry is stale.
    """

    groups: list[str]
    expires_at: float


class EntraTokenValidator:
    """Validates Entra ID access tokens and resolves them to principals."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialise the validator.

        Args:
            settings: Active settings. Defaults to the process settings.
            http_client: HTTP client for JWKS and Graph calls. Injectable for
                tests; one is created lazily otherwise and closed by
                :meth:`aclose`.
        """
        self._settings = settings or get_settings()
        self._jwks = JWKSCache(settings=self._settings)
        self._client = http_client
        self._owns_client = http_client is None
        self._groups: dict[str, _GroupCacheEntry] = {}
        self._graph_token: tuple[str, float] | None = None
        self._graph_warned = False
        self._lock = anyio.Lock()

    @property
    def settings(self) -> Settings:
        """Settings this validator was built from.

        Returns:
            The bound settings.
        """
        return self._settings

    @property
    def jwks(self) -> JWKSCache:
        """The signing-key cache.

        Returns:
            The cache, exposed for warm-up and tests.
        """
        return self._jwks

    async def _http(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it on first use.

        Returns:
            An :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=float(self._settings.entra_jwks_timeout_seconds),
                        follow_redirects=False,
                    )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client when this validator owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warm_up(self) -> bool:
        """Pre-fetch the JWKS during startup.

        Returns:
            True when keys are cached afterwards. Never raises: a cold start with
            an unreachable identity provider must still produce a process that can
            answer ``/health``, and the first request will retry.
        """
        if self._settings.entra_dev_mode or not self._settings.entra_tenant_id:
            return False
        try:
            await self._jwks.refresh(client=await self._http(), force=True)
        except (AuthError, ConfigError) as exc:
            _log.warning("entra_jwks_warmup_failed", error=type(exc).__name__)
            return False
        return bool(self._jwks.keys)

    # ------------------------------------------------------------- validation
    def _expected_issuers(self) -> list[str]:
        """Issuers this API accepts.

        Returns:
            The v2.0 issuer, plus the v1.0 form when
            ``entra_accept_v1_issuer`` is on.

        Raises:
            ConfigError: When no tenant is configured.
        """
        issuer = self._settings.entra_issuer
        if not issuer:
            msg = "entra_tenant_id is not configured: no issuer to validate against"
            raise ConfigError(msg, code="auth_not_configured")
        issuers = [issuer]
        if self._settings.entra_accept_v1_issuer:
            issuers.append(f"https://sts.windows.net/{self._settings.entra_tenant_id}/")
        return issuers

    def _expected_audiences(self) -> list[str]:
        """Audiences this API accepts.

        Returns:
            ``expected_audience`` plus its ``api://`` URI form, so an app
            registration that mints either passes.

        Raises:
            ConfigError: When neither ``entra_audience`` nor ``entra_client_id`` is
                set — accepting any audience would let a token minted for a
                different API in the same directory in.
        """
        audience = self._settings.expected_audience
        if not audience:
            msg = (
                "entra_audience/entra_client_id is not configured: refusing to "
                "accept a token without checking who it was issued for"
            )
            raise ConfigError(msg, code="auth_not_configured")
        audiences = [audience]
        if not audience.startswith("api://"):
            audiences.append(f"api://{audience}")
        return audiences

    async def decode(self, token: str) -> dict[str, Any]:
        """Verify a token's signature and registered claims.

        Args:
            token: The raw compact JWS from the ``Authorization`` header.

        Returns:
            The verified claim set.

        Raises:
            AuthError: On any structural, signature, issuer, audience, expiry,
                not-before or directory failure. The message names the failure
                class and never echoes the token.
        """
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthError(
                "token header is malformed", code="auth_malformed_token"
            ) from exc

        algorithm = str(header.get("alg") or "")
        allowed = list(self._settings.entra_allowed_algorithms)
        if algorithm not in allowed:
            # Explicitly refusing "none" and every symmetric algorithm here rather
            # than relying on the library's default is the point: algorithm
            # confusion is the classic JWT break.
            raise AuthError(
                f"token algorithm {algorithm!r} is not accepted",
                code="auth_bad_algorithm",
            )
        kid = str(header.get("kid") or "")
        if not kid:
            raise AuthError(
                "token header carries no 'kid'", code="auth_malformed_token"
            )

        key = await self._jwks.key_for(kid, client=await self._http())
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key.key,
                algorithms=allowed,
                audience=self._expected_audiences(),
                issuer=self._expected_issuers(),
                leeway=float(self._settings.entra_leeway_seconds),
                options={
                    "require": ["exp", "iss", "aud"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token has expired", code="auth_token_expired") from exc
        except jwt.ImmatureSignatureError as exc:
            raise AuthError(
                "token is not yet valid", code="auth_token_immature"
            ) from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError(
                "token was not issued for this API", code="auth_bad_audience"
            ) from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError(
                "token was not issued by the expected directory",
                code="auth_bad_issuer",
            ) from exc
        except jwt.PyJWTError as exc:
            raise AuthError(
                "token signature verification failed", code="auth_bad_signature"
            ) from exc

        self._check_tenant(claims)
        self._check_scope(claims)
        return claims

    def _check_tenant(self, claims: Mapping[str, Any]) -> None:
        """Pin the token's directory to the configured one.

        Args:
            claims: Verified claims.

        Raises:
            AuthError: When ``tid`` disagrees with ``entra_tenant_id``.
        """
        if not self._settings.entra_pin_tenant:
            return
        expected = self._settings.entra_tenant_id
        actual = str(claims.get("tid") or "")
        if expected and actual != expected:
            _log.error("entra_tid_mismatch", expected=expected, actual=actual)
            raise AuthError(
                "token belongs to a different directory", code="auth_bad_tenant"
            )

    def _check_scope(self, claims: Mapping[str, Any]) -> None:
        """Enforce the optional required delegated scope.

        Args:
            claims: Verified claims.

        Raises:
            AuthError: When a required scope is configured and absent. Status 403,
                because the caller is authenticated but under-privileged.
        """
        required = self._settings.entra_required_scope
        if not required:
            return
        if str(required) not in claim_list(claims, "scp"):
            raise AuthError(
                "token does not carry the required scope",
                code="auth_missing_scope",
                status_code=403,
            )

    # ----------------------------------------------------------- group overage
    async def resolve_groups(self, claims: Mapping[str, Any]) -> list[str]:
        """Resolve the caller's transitive group ids, handling the overage claim.

        Args:
            claims: Verified claims.

        Returns:
            Group object ids. When the token carried a complete ``groups`` claim
            that list is returned unchanged. On an overage, Microsoft Graph is
            queried with the API's own application identity and the answer is
            cached for ``entra_group_cache_seconds``. Any failure returns an empty
            list, which fails closed: the caller sees only unrestricted documents.
        """
        if not requires_group_lookup(claims):
            return claim_list(claims, self._settings.entra_groups_claim)
        if not self._settings.entra_group_overage_lookup:
            _log.warning("entra_group_overage_ignored", oid=str(claims.get("oid")))
            return []

        user_id = str(claims.get("oid") or claims.get("sub") or "")
        if not user_id:
            return []
        cached = self._groups.get(user_id)
        now = time.monotonic()
        if cached is not None and cached.expires_at > now:
            return list(cached.groups)

        ttl = float(self._settings.entra_group_cache_seconds)
        groups = await self._fetch_groups(user_id)
        self._groups[user_id] = _GroupCacheEntry(groups=groups, expires_at=now + ttl)
        return list(groups)

    async def _fetch_groups(self, user_id: str) -> list[str]:
        """Page Microsoft Graph for a user's transitive group memberships.

        Args:
            user_id: The caller's ``oid``.

        Returns:
            Group object ids, or an empty list when Graph is unavailable or the
            API has no application identity to call it with.
        """
        token = await self._graph_access_token()
        if not token:
            return []
        base = self._settings.azure_graph_base_url.rstrip("/")
        path = str(self._settings.entra_group_lookup_path)
        # The configured path is user-relative ("/me/..."); the application-identity
        # call addresses the user explicitly instead.
        suffix = path.replace("/me/", f"/users/{user_id}/", 1)
        page = int(self._settings.entra_group_page_size)
        url: str | None = f"{base}{suffix}?$select=id&$top={page}"
        timeout = float(self._settings.entra_group_lookup_timeout_seconds)
        client = await self._http()
        groups: list[str] = []
        try:
            while url:
                response = await client.get(
                    url,
                    headers={"authorization": f"Bearer {token}"},
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                for item in body.get("value") or []:
                    identifier = item.get("id")
                    if identifier:
                        groups.append(str(identifier))
                url = body.get("@odata.nextLink")
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning(
                "entra_group_lookup_failed",
                error=type(exc).__name__,
                resolved=len(groups),
            )
            return []
        _log.info("entra_group_overage_resolved", groups=len(groups))
        return groups

    async def _graph_access_token(self) -> str | None:
        """Acquire an application token for Microsoft Graph.

        Two sources are tried, in order: the Azure instance metadata service
        (managed identity, the deployed configuration), then a client-credentials
        grant using ``entra_client_secret``. Both are cached until
        ``entra_graph_token_skew_seconds`` before expiry.

        Returns:
            The bearer token, or None when neither source is configured or
            reachable. The absence is logged once rather than per request.
        """
        cached = self._graph_token
        now = time.monotonic()
        if cached is not None and cached[1] > now:
            return cached[0]

        scope = self._settings.azure_graph_scope
        client = await self._http()
        token: tuple[str, float] | None = None
        if self._settings.azure_use_managed_identity:
            token = await self._imds_token(client, scope)
        if token is None:
            token = await self._client_credentials_token(client, scope)
        if token is None:
            if not self._graph_warned:
                _log.warning("entra_graph_identity_unavailable")
                self._graph_warned = True
            return None
        self._graph_token = token
        return token[0]

    async def _imds_token(
        self, client: httpx.AsyncClient, scope: str
    ) -> tuple[str, float] | None:
        """Fetch a Graph token from the instance metadata service.

        Args:
            client: HTTP client to use.
            scope: Requested scope, converted to a resource URI for IMDS.

        Returns:
            A ``(token, monotonic_expiry)`` pair, or None when IMDS is absent.
        """
        endpoint = str(self._settings.entra_managed_identity_endpoint)
        params = {
            "api-version": str(self._settings.entra_managed_identity_api_version),
            "resource": scope.removesuffix("/.default"),
        }
        if self._settings.azure_client_id:
            params["client_id"] = self._settings.azure_client_id
        try:
            response = await client.get(
                endpoint,
                params=params,
                headers={"metadata": "true"},
                timeout=float(self._settings.entra_group_lookup_timeout_seconds),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        return self._token_pair(body)

    async def _client_credentials_token(
        self, client: httpx.AsyncClient, scope: str
    ) -> tuple[str, float] | None:
        """Perform a client-credentials grant for Graph.

        Args:
            client: HTTP client to use.
            scope: Requested scope.

        Returns:
            A ``(token, monotonic_expiry)`` pair, or None when no client secret is
            configured or the endpoint refuses.
        """
        secret = self._settings.entra_client_secret
        client_id = self._settings.entra_client_id
        tenant = self._settings.entra_tenant_id
        if not (secret and client_id and tenant):
            return None
        url = (
            f"{self._settings.entra_authority_host.rstrip('/')}"
            f"/{tenant}/oauth2/v2.0/token"
        )
        try:
            response = await client.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": str(secret),
                    "scope": scope,
                },
                timeout=float(self._settings.entra_group_lookup_timeout_seconds),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning("entra_graph_token_failed", error=type(exc).__name__)
            return None
        return self._token_pair(body)

    def _token_pair(self, body: Mapping[str, Any]) -> tuple[str, float] | None:
        """Convert a token endpoint response into a cache entry.

        Args:
            body: Parsed JSON body.

        Returns:
            A ``(token, monotonic_expiry)`` pair, or None when the body carries no
            access token.
        """
        token = body.get("access_token")
        if not token:
            return None
        try:
            expires_in = float(body.get("expires_in", 3600))
        except (TypeError, ValueError):
            expires_in = 3600.0
        skew = float(self._settings.entra_graph_token_skew_seconds)
        return str(token), time.monotonic() + max(30.0, expires_in - skew)

    # ------------------------------------------------------------- principals
    async def principal_for_token(self, token: str) -> Principal:
        """Validate a bearer token and resolve it to a principal.

        Args:
            token: Raw compact JWS.

        Returns:
            The resolved :class:`~ragcore.models.acl.Principal`.

        Raises:
            AuthError: On any validation failure.
        """
        claims = await self.decode(token)
        groups = await self.resolve_groups(claims)
        return principal_from_claims(claims, settings=self._settings, groups=groups)

    async def principal_for_header(
        self, authorization: str | None, *, dev_principal: str | None = None
    ) -> Principal:
        """Resolve a principal from request headers.

        Args:
            authorization: The ``Authorization`` header value, if any.
            dev_principal: The dev-principal header value, honoured only when
                ``entra_dev_mode`` is on.

        Returns:
            The resolved principal.

        Raises:
            AuthError: When no usable credential is present or validation fails.
            ConfigError: When dev mode is on in production.
        """
        if self._settings.entra_dev_mode:
            assert_dev_mode_allowed(self._settings)
            if dev_principal:
                from app.auth.principal import parse_dev_principal

                _log.warning(
                    "entra_dev_principal_accepted",
                    env=self._settings.env,
                    header=self._settings.entra_dev_principal_header,
                )
                return parse_dev_principal(dev_principal, settings=self._settings)
        token = extract_bearer(authorization)
        if not token:
            raise AuthError(
                "missing bearer token", code="auth_missing_token", status_code=401
            )
        return await self.principal_for_token(token)


def extract_bearer(authorization: str | None) -> str | None:
    """Pull the credential out of an ``Authorization`` header.

    Args:
        authorization: Raw header value.

    Returns:
        The token, or None when the header is absent, empty or not a Bearer
        challenge.
    """
    if not authorization:
        return None
    scheme, _, credential = authorization.partition(" ")
    if scheme.strip().lower() != _BEARER:
        return None
    token = credential.strip()
    return token or None


_VALIDATORS: dict[str, EntraTokenValidator] = {}


def _validator_key(settings: Settings) -> str:
    """Cache key for a validator.

    ``Settings`` is a pydantic model and therefore unhashable, so the key is the
    tuple of fields that change what a validator accepts.

    Args:
        settings: Active settings.

    Returns:
        A stable string key.
    """
    return "|".join(
        str(part)
        for part in (
            settings.entra_tenant_id,
            settings.expected_audience,
            settings.entra_authority_host,
            settings.entra_dev_mode,
            settings.azure_graph_base_url,
        )
    )


def get_token_validator(settings: Settings | None = None) -> EntraTokenValidator:
    """Return the process-wide token validator.

    Args:
        settings: Active settings. Defaults to the process settings.

    Returns:
        A cached :class:`EntraTokenValidator`, so the JWKS and group caches are
        shared by every request.
    """
    cfg = settings or get_settings()
    key = _validator_key(cfg)
    validator = _VALIDATORS.get(key)
    if validator is None:
        validator = EntraTokenValidator(cfg)
        _VALIDATORS[key] = validator
    return validator


async def reset_token_validator() -> None:
    """Close and drop every cached validator. Shutdown hook and test helper."""
    validators: Sequence[EntraTokenValidator] = list(_VALIDATORS.values())
    _VALIDATORS.clear()
    for validator in validators:
        await validator.aclose()
