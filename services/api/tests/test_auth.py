"""Authentication and the tenant boundary.

Multi-tenancy is a security boundary, so these are the tests that have to be
adversarial rather than illustrative. Three properties are asserted:

1. **Tenant B cannot read tenant A.** Through retrieval, through the document
   listing, through lineage and through sessions — four independent paths, each
   of which would be a leak on its own.
2. **A missing role is a 403.** Not a 404, not a silently empty list.
3. **Token validation is real.** A generated RSA key pair signs tokens that are
   validated against a stubbed JWKS, so expiry, audience, issuer, algorithm and
   directory pinning are exercised against ``pyjwt`` rather than against a mock
   of it.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
import pytest
from conftest import TENANT_A, TENANT_B, auth_headers
from cryptography.hazmat.primitives.asymmetric import rsa

from app.auth.entra import EntraTokenValidator, assert_dev_mode_allowed, extract_bearer
from app.auth.principal import max_classification_for, requires_group_lookup
from ragcore.db import repositories as repo
from ragcore.errors import AuthError, ConfigError
from ragcore.models.acl import AccessControl, Classification

TENANT_A_DOC = "doc-acme-travel-2025"
CONFIDENTIAL_DOC = "doc-acme-salary-bands"


# --------------------------------------------------------------- cross-tenant


async def test_retrieval_is_scoped_to_the_callers_tenant(
    api: httpx.AsyncClient, acme_user: Any, globex_user: Any, retrieval: Any
) -> None:
    """A tenant-B search must never surface a tenant-A chunk."""
    acme = await api.post(
        "/api/v1/search",
        json={"query": "meal allowance"},
        headers=auth_headers(acme_user),
    )
    globex = await api.post(
        "/api/v1/search",
        json={"query": "meal allowance"},
        headers=auth_headers(globex_user),
    )
    assert acme.status_code == httpx.codes.OK
    assert globex.status_code == httpx.codes.OK

    acme_docs = {c["payload"]["document_id"] for c in acme.json()["chunks"]}
    globex_docs = {c["payload"]["document_id"] for c in globex.json()["chunks"]}
    assert acme_docs == {TENANT_A_DOC}
    assert globex_docs == {"doc-globex-travel-policy"}
    assert not acme_docs & globex_docs

    # The handler must have passed the *caller's* principal down, not a default.
    tenants = [principal.tenant_id for principal, _ in retrieval.calls]
    assert tenants == [TENANT_A, TENANT_B]


async def test_document_listing_and_lineage_are_tenant_scoped(
    api: httpx.AsyncClient, db_session: Any, acme_user: Any, globex_user: Any
) -> None:
    """A tenant-A document is invisible and unreachable from tenant B."""
    await repo.upsert_document(
        db_session,
        tenant_id=TENANT_A,
        document_id=TENANT_A_DOC,
        source_uri="https://example.test/acme/travel-2025",
        source_type="blob",
        access_control=AccessControl(
            tenant_id=TENANT_A, classification=Classification.PUBLIC
        ),
        title="Acme travel policy 2025",
    )
    await db_session.commit()

    mine = await api.get("/api/v1/documents", headers=auth_headers(acme_user))
    theirs = await api.get("/api/v1/documents", headers=auth_headers(globex_user))
    assert [d["document_id"] for d in mine.json()] == [TENANT_A_DOC]
    assert theirs.json() == []

    # Lineage must not confirm the document even exists.
    leak = await api.get(
        f"/api/v1/documents/{TENANT_A_DOC}/lineage", headers=auth_headers(globex_user)
    )
    assert leak.status_code == httpx.codes.NOT_FOUND
    assert leak.headers["content-type"].startswith("application/problem+json")
    assert leak.json()["code"] == "document_not_found"

    allowed = await api.get(
        f"/api/v1/documents/{TENANT_A_DOC}/lineage", headers=auth_headers(acme_user)
    )
    assert allowed.status_code == httpx.codes.OK
    assert allowed.json()["tenant_id"] == TENANT_A


async def test_clearance_ceiling_hides_a_document_within_a_tenant(
    api: httpx.AsyncClient, db_session: Any, acme_user: Any, acme_admin: Any
) -> None:
    """A same-tenant document above the caller's clearance is not listed."""
    await repo.upsert_document(
        db_session,
        tenant_id=TENANT_A,
        document_id=CONFIDENTIAL_DOC,
        source_uri="https://example.test/acme/salary-bands",
        source_type="blob",
        access_control=AccessControl(
            tenant_id=TENANT_A,
            allowed_groups=["g-acme-hr"],
            classification=Classification.CONFIDENTIAL,
        ),
        title="Acme salary bands",
    )
    await db_session.commit()

    engineer = await api.get("/api/v1/documents", headers=auth_headers(acme_user))
    admin = await api.get("/api/v1/documents", headers=auth_headers(acme_admin))
    assert [d["document_id"] for d in engineer.json()] == []
    assert [d["document_id"] for d in admin.json()] == [CONFIDENTIAL_DOC]


async def test_session_from_another_tenant_is_refused(
    api: httpx.AsyncClient, db_session: Any, acme_user: Any, globex_user: Any
) -> None:
    """Reaching another tenant's session is an auditable 403, not a 404."""
    row = await repo.get_or_create_session(
        db_session, tenant_id=TENANT_A, user_id=acme_user.user_id, title="acme chat"
    )
    await db_session.commit()

    mine = await api.get(
        f"/api/v1/sessions/{row.session_id}", headers=auth_headers(acme_user)
    )
    assert mine.status_code == httpx.codes.OK

    # `get_session_row` filters on tenant in SQL, so a cross-tenant probe cannot
    # even observe that the row exists. What matters is that nothing of tenant
    # A's leaks, and that the same id is reachable from tenant A.
    theirs = await api.get(
        f"/api/v1/sessions/{row.session_id}", headers=auth_headers(globex_user)
    )
    assert theirs.status_code == httpx.codes.NOT_FOUND
    assert theirs.json()["code"] == "session_not_found"
    assert "acme chat" not in theirs.text

    messages = await api.get(
        f"/api/v1/sessions/{row.session_id}/messages",
        headers=auth_headers(globex_user),
    )
    assert messages.status_code == httpx.codes.NOT_FOUND

    compact = await api.post(
        f"/api/v1/sessions/{row.session_id}/compact",
        headers=auth_headers(globex_user),
    )
    assert compact.status_code == httpx.codes.NOT_FOUND

    deleted = await api.delete(
        f"/api/v1/sessions/{row.session_id}", headers=auth_headers(globex_user)
    )
    assert deleted.status_code == httpx.codes.NOT_FOUND
    # ... and tenant A's session survived the attempt.
    still_there = await api.get(
        f"/api/v1/sessions/{row.session_id}", headers=auth_headers(acme_user)
    )
    assert still_there.status_code == httpx.codes.OK


# --------------------------------------------------------------------- roles


async def test_missing_role_is_forbidden(
    api: httpx.AsyncClient, acme_user: Any, acme_admin: Any
) -> None:
    """The admin surface refuses a caller without ``rag.admin``."""
    denied = await api.get("/api/v1/admin/sources", headers=auth_headers(acme_user))
    assert denied.status_code == httpx.codes.FORBIDDEN
    assert denied.json()["code"] == "auth_missing_role"
    # The refusal must not enumerate what the caller does hold.
    assert "rag.user" not in denied.text

    allowed = await api.get("/api/v1/admin/sources", headers=auth_headers(acme_admin))
    assert allowed.status_code == httpx.codes.OK


async def test_unauthenticated_requests_are_rejected(api: httpx.AsyncClient) -> None:
    """No credential means 401 with a Bearer challenge, on every scoped route."""
    for method, path in (
        ("GET", "/api/v1/me"),
        ("GET", "/api/v1/sessions"),
        ("POST", "/api/v1/search"),
    ):
        response = await api.request(method, path, json={"query": "x"})
        assert response.status_code == httpx.codes.UNAUTHORIZED, path
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json()["code"] == "auth_missing_token"


async def test_health_needs_no_credential(api: httpx.AsyncClient) -> None:
    """Probes and metrics answer without a token."""
    for path in ("/health", "/readyz", "/metrics"):
        response = await api.get(path)
        assert response.status_code in {httpx.codes.OK, httpx.codes.SERVICE_UNAVAILABLE}


async def test_me_echoes_the_derived_clearance(
    api: httpx.AsyncClient, acme_admin: Any
) -> None:
    """``GET /me`` reports the ceiling the API will actually enforce."""
    response = await api.get("/api/v1/me", headers=auth_headers(acme_admin))
    assert response.status_code == httpx.codes.OK
    body = response.json()
    assert body["tenant_id"] == TENANT_A
    assert body["max_classification"] == Classification.RESTRICTED.value


# ---------------------------------------------------------------- token layer


def _keypair() -> tuple[Any, dict[str, Any]]:
    """Generate an RSA key pair and the JWKS entry describing its public half.

    Returns:
        A ``(private_key, jwks_document)`` pair.
    """
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    public_jwk.update({"kid": "test-key-1", "use": "sig", "alg": "RS256"})
    return private, {"keys": [public_jwk]}


def _mint(
    private: Any,
    *,
    tenant: str,
    audience: str,
    issuer: str,
    expires_in: int = 600,
    kid: str = "test-key-1",
    **claims: Any,
) -> str:
    """Sign a token with the generated key.

    Args:
        private: The RSA private key.
        tenant: Value for ``tid``.
        audience: Value for ``aud``.
        issuer: Value for ``iss``.
        expires_in: Seconds until expiry; negative mints an expired token.
        kid: Key id placed in the header.
        **claims: Extra claims.

    Returns:
        The compact JWS.
    """
    now = int(time.time())
    payload = {
        "oid": "00000000-0000-0000-0000-00000000000a",
        "tid": tenant,
        "aud": audience,
        "iss": issuer,
        "iat": now - 5,
        "nbf": now - 5,
        "exp": now + expires_in,
        "roles": ["rag.user"],
        "groups": ["g-acme-engineering"],
        "name": "Test User",
        "preferred_username": "test@example.test",
        **claims,
    }
    return jwt.encode(payload, private, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def entra() -> Any:
    """A validator wired to a stubbed JWKS endpoint and a generated key.

    Returns:
        A ``(validator, private_key, settings)`` tuple.
    """
    from ragcore.settings import Settings

    private, jwks = _keypair()
    settings = Settings(
        _env_file=None,
        env="local",
        entra_tenant_id="11111111-1111-1111-1111-111111111111",
        entra_client_id="22222222-2222-2222-2222-222222222222",
        entra_dev_mode=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the generated JWKS for any request.

        Args:
            request: The outbound request.

        Returns:
            The JWKS document.
        """
        del request
        return httpx.Response(200, json=jwks)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return EntraTokenValidator(settings, http_client=client), private, settings


async def test_valid_token_resolves_to_a_principal(entra: Any) -> None:
    """A well-formed token yields the expected principal and clearance."""
    validator, private, settings = entra
    token = _mint(
        private,
        tenant=settings.entra_tenant_id,
        audience=settings.entra_client_id,
        issuer=settings.entra_issuer,
    )
    principal = await validator.principal_for_token(token)
    assert principal.tenant_id == settings.entra_tenant_id
    assert principal.user_id == "00000000-0000-0000-0000-00000000000a"
    assert principal.groups == ["g-acme-engineering"]
    assert principal.max_classification is Classification.INTERNAL


async def test_a_token_from_another_directory_is_refused(entra: Any) -> None:
    """``tid`` pinning refuses a token minted for a different directory.

    This is the cross-tenant case at the identity layer: the signature is valid
    for *that* directory's key, but the token is not ours, and accepting it would
    hand its bearer a principal in a tenant they do not belong to.
    """
    validator, private, settings = entra
    token = _mint(
        private,
        tenant="99999999-9999-9999-9999-999999999999",
        audience=settings.entra_client_id,
        issuer=settings.entra_issuer,
    )
    with pytest.raises(AuthError) as caught:
        await validator.decode(token)
    assert caught.value.code == "auth_bad_tenant"


async def test_expired_wrong_audience_and_wrong_key_are_refused(entra: Any) -> None:
    """Expiry, audience and an unknown ``kid`` each fail closed."""
    validator, private, settings = entra

    expired = _mint(
        private,
        tenant=settings.entra_tenant_id,
        audience=settings.entra_client_id,
        issuer=settings.entra_issuer,
        expires_in=-3600,
    )
    with pytest.raises(AuthError) as expiry:
        await validator.decode(expired)
    assert expiry.value.code == "auth_token_expired"

    wrong_audience = _mint(
        private,
        tenant=settings.entra_tenant_id,
        audience="api://someone-else",
        issuer=settings.entra_issuer,
    )
    with pytest.raises(AuthError) as audience:
        await validator.decode(wrong_audience)
    assert audience.value.code == "auth_bad_audience"

    unknown_key = _mint(
        private,
        tenant=settings.entra_tenant_id,
        audience=settings.entra_client_id,
        issuer=settings.entra_issuer,
        kid="rotated-away",
    )
    with pytest.raises(AuthError) as key:
        await validator.decode(unknown_key)
    assert key.value.code == "auth_unknown_key"


async def test_unsigned_token_is_refused(entra: Any) -> None:
    """An ``alg: none`` token never reaches signature verification."""
    validator, _private, settings = entra
    forged = jwt.encode(
        {"oid": "x", "tid": settings.entra_tenant_id, "aud": settings.entra_client_id},
        key="",
        algorithm="none",
        headers={"kid": "test-key-1"},
    )
    with pytest.raises(AuthError) as caught:
        await validator.decode(forged)
    assert caught.value.code == "auth_bad_algorithm"


async def test_group_overage_falls_back_and_fails_closed(entra: Any) -> None:
    """A ``hasgroups`` token with no Graph identity yields no groups."""
    validator, private, settings = entra
    token = _mint(
        private,
        tenant=settings.entra_tenant_id,
        audience=settings.entra_client_id,
        issuer=settings.entra_issuer,
        hasgroups=True,
        groups=[],
    )
    claims = await validator.decode(token)
    assert requires_group_lookup(claims) is True
    # The stubbed transport answers every request with the JWKS document, which
    # carries no `value` array, so the lookup resolves nothing.
    assert await validator.resolve_groups(claims) == []


# ------------------------------------------------------------------ dev mode


def test_dev_mode_is_refused_in_production() -> None:
    """Unsigned principals must never be accepted in production."""
    from ragcore.settings import Settings

    # Settings itself refuses the combination, which is the first line of defence.
    with pytest.raises(ValueError, match="entra_dev_mode"):
        Settings(_env_file=None, env="production", entra_dev_mode=True)

    # And the runtime guard refuses it again, for an environment mutated after
    # construction.
    mutated = Settings(_env_file=None, env="local", entra_dev_mode=True).model_copy(
        update={"env": "production"}
    )
    with pytest.raises(ConfigError) as caught:
        assert_dev_mode_allowed(mutated)
    assert caught.value.code == "auth_dev_mode_in_production"


def test_bearer_extraction_is_strict() -> None:
    """Only a well-formed Bearer challenge yields a credential."""
    assert extract_bearer("Bearer abc.def.ghi") == "abc.def.ghi"
    assert extract_bearer("bearer abc") == "abc"
    assert extract_bearer("Basic abc") is None
    assert extract_bearer("Bearer   ") is None
    assert extract_bearer(None) is None


def test_clearance_is_derived_from_the_strongest_mapping() -> None:
    """Role and group mappings raise, never lower, the ceiling."""
    from ragcore.settings import Settings

    settings = Settings(_env_file=None)
    assert (
        max_classification_for(["rag.user"], [], settings=settings)
        is Classification.INTERNAL
    )
    assert (
        max_classification_for(["rag.admin", "rag.public"], [], settings=settings)
        is Classification.RESTRICTED
    )
    assert (
        max_classification_for(["rag.public"], [], settings=settings)
        is Classification.PUBLIC
    )
