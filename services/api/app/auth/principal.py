"""Claim-to-:class:`~ragcore.models.acl.Principal` mapping and its tunables.

Kept separate from :mod:`app.auth.entra` so that the mapping rules — which decide
what a caller is *allowed to see* — can be unit-tested without a JWKS, a network
or a token, and so the dev-mode path and the validated-token path provably produce
the same shape of principal.

**Tunables.** Read through :func:`auth_setting`, the same indirection
:func:`app.api_setting` uses: the real ``ragcore.settings`` field wins whenever it
exists, otherwise the documented default in :data:`AUTH_SETTING_DEFAULTS` applies.
No clearance level, claim name, cache TTL or Graph URL is written at a call site.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragcore.errors import AuthError
from ragcore.logging import get_logger
from ragcore.models.acl import Classification, Principal
from ragcore.settings import Settings

__all__ = [
    "AUTH_SETTING_DEFAULTS",
    "GROUP_OVERAGE_CLAIMS",
    "auth_setting",
    "claim_list",
    "max_classification_for",
    "parse_dev_principal",
    "principal_from_claims",
    "requires_group_lookup",
]

_log = get_logger(__name__)

#: Claims Entra sets instead of ``groups`` when the token would overflow. Either
#: one means "the group list is incomplete, ask Microsoft Graph".
GROUP_OVERAGE_CLAIMS: tuple[str, ...] = ("hasgroups", "_claim_names", "_claim_sources")


#: Identity tunables not yet declared on :class:`ragcore.settings.Settings`.
AUTH_SETTING_DEFAULTS: dict[str, Any] = {
    # ---------------------------------------------------------- clearance map
    # App role -> clearance ceiling. The strongest *matching* rule across a
    # caller's roles and groups wins, so holding `rag.admin` alongside
    # `rag.public` still yields `restricted`. Keys are role values exactly as
    # they appear in the `roles` claim; a role that is not a key contributes
    # nothing, and a caller matching no rule at all gets
    # `entra_default_classification`.
    "entra_clearance_roles": {
        "rag.admin": "restricted",
        "rag.restricted": "restricted",
        "rag.confidential": "confidential",
        "rag.internal": "internal",
        "rag.public": "public",
    },
    # Entra group object id -> clearance ceiling. Empty by default: group ids are
    # deployment-specific, so an operator maps them per tenant.
    "entra_clearance_groups": {},
    # Clearance for a caller matching no role or group rule. `internal` mirrors
    # `AccessControl.classification`'s default: a caller sees ordinary internal
    # material and nothing sensitive until an operator says so.
    "entra_default_classification": "internal",
    # ------------------------------------------------------------- validation
    # Accept the v1.0 issuer (`https://sts.windows.net/{tid}/`) alongside the v2.0
    # one. MSAL SPAs issue v2.0 tokens; some app registrations still mint v1.0.
    "entra_accept_v1_issuer": True,
    # Optional scope (`scp`) every caller must present. None disables the check;
    # app-role-only tokens (client credentials) carry no `scp` at all.
    "entra_required_scope": None,
    # Reject a token whose `tid` is not the pinned directory. Turning this off is
    # only correct for a genuinely multi-directory registration.
    "entra_pin_tenant": True,
    # ---------------------------------------------------------------- caching
    # Seconds an unsuccessful JWKS fetch is remembered before retrying, and the
    # floor between two refetches triggered by an unknown `kid`. Without a floor,
    # a stream of tokens signed by a key we will never have becomes a DoS on the
    # Microsoft endpoint.
    "entra_jwks_refresh_min_seconds": 60.0,
    "entra_jwks_timeout_seconds": 10.0,
    # ------------------------------------------------- group overage fallback
    # Resolve the group-overage claim through Microsoft Graph. With this off, a
    # caller whose token overflowed simply has no groups, which fails closed.
    "entra_group_overage_lookup": True,
    "entra_group_cache_seconds": 900.0,
    "entra_group_page_size": 999,
    "entra_group_lookup_timeout_seconds": 10.0,
    # Graph endpoint used for the transitive group lookup, relative to
    # ``settings.azure_graph_base_url``. Written in the user-relative "/me/" form;
    # the application-identity call substitutes "/users/{oid}/".
    "entra_group_lookup_path": "/me/transitiveMemberOf/microsoft.graph.group",
    # ------------------------------------- application identity used for Graph
    # Client secret for the client-credentials grant, used only when managed
    # identity is unavailable. Source it from Key Vault; never commit one.
    "entra_client_secret": None,
    # Azure instance metadata service, the managed-identity token source.
    "entra_managed_identity_endpoint": (
        "http://169.254.169.254/metadata/identity/oauth2/token"
    ),
    "entra_managed_identity_api_version": "2018-02-01",
    # Refresh a Graph token this many seconds before it actually expires.
    "entra_graph_token_skew_seconds": 60.0,
}


def auth_setting(settings: Settings, name: str) -> Any:
    """Read an identity tunable, preferring the settings field over the default.

    Args:
        settings: Active settings object.
        name: A key of :data:`AUTH_SETTING_DEFAULTS`.

    Returns:
        ``getattr(settings, name)`` when the field exists and is not None,
        otherwise the documented default.

    Raises:
        KeyError: If ``name`` is not a declared tunable.
    """
    if name not in AUTH_SETTING_DEFAULTS:
        msg = f"unknown auth tunable {name!r}"
        raise KeyError(msg)
    value = getattr(settings, name, None)
    return AUTH_SETTING_DEFAULTS[name] if value is None else value


def claim_list(claims: Mapping[str, Any], key: str) -> list[str]:
    """Read a claim that may be a list, a single string or absent.

    Entra emits ``roles`` as an array but ``scp`` as a space-delimited string, and
    a single-valued array is sometimes flattened by intermediate proxies.

    Args:
        claims: Decoded token claims.
        key: Claim name.

    Returns:
        The claim's values as a list of non-empty strings, order preserved.
    """
    raw = claims.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        parts: Iterable[str] = raw.replace(",", " ").split()
    elif isinstance(raw, Sequence):
        parts = [str(item) for item in raw]
    else:  # pragma: no cover - defensive against a non-conforming issuer
        return []
    return [value.strip() for value in parts if str(value).strip()]


def requires_group_lookup(claims: Mapping[str, Any]) -> bool:
    """Whether the token's group list is truncated and needs a Graph lookup.

    Args:
        claims: Decoded token claims.

    Returns:
        True when Entra signalled a group overage. ``hasgroups`` is the v2.0
        signal; ``_claim_names``/``_claim_sources`` is the v1.0 distributed-claim
        form. Either means the ``groups`` claim, if present at all, is incomplete.
    """
    if claims.get("hasgroups"):
        return True
    names = claims.get("_claim_names")
    return bool(isinstance(names, Mapping) and "groups" in names)


def max_classification_for(
    roles: Sequence[str],
    groups: Sequence[str],
    *,
    settings: Settings,
) -> Classification:
    """Derive a caller's clearance ceiling from their roles and groups.

    The strongest **matching** rule wins. When nothing matches, the configured
    default applies — so a caller whose roles the operator has said nothing about
    gets the deployment's baseline, and a role deliberately mapped to ``public``
    genuinely caps that caller below it.

    Args:
        roles: App roles from the token.
        groups: Entra group object ids, after any overage lookup.
        settings: Active settings supplying ``entra_clearance_roles``,
            ``entra_clearance_groups`` and ``entra_default_classification``.

    Returns:
        The clearance ceiling. An unparseable configured value is ignored with a
        warning rather than crashing the request or silently granting more.
    """
    role_map: Mapping[str, Any] = auth_setting(settings, "entra_clearance_roles") or {}
    group_map: Mapping[str, Any] = (
        auth_setting(settings, "entra_clearance_groups") or {}
    )
    matched: list[Classification] = []
    for source, values in ((role_map, roles), (group_map, groups)):
        for value in values:
            if value not in source:
                continue
            candidate = _classification_or_none(source[value])
            if candidate is None:
                _log.warning(
                    "clearance_mapping_invalid", key=value, value=str(source[value])
                )
                continue
            matched.append(candidate)
    if matched:
        return max(matched)

    fallback = _classification_or_none(
        auth_setting(settings, "entra_default_classification")
    )
    if fallback is None:  # pragma: no cover - defensive against bad configuration
        _log.warning("clearance_default_invalid")
        return Classification.INTERNAL
    return fallback


def _classification_or_none(value: Any) -> Classification | None:
    """Coerce a configured value to a :class:`Classification`.

    Args:
        value: A classification name, rank or enum member.

    Returns:
        The classification, or None when the value is not recognisable.
    """
    if isinstance(value, Classification):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Classification.from_rank(value)
    if isinstance(value, str):
        try:
            return Classification(value.strip().lower())
        except ValueError:
            return None
    return None


def principal_from_claims(
    claims: Mapping[str, Any],
    *,
    settings: Settings,
    groups: Sequence[str] | None = None,
) -> Principal:
    """Build a :class:`Principal` from validated Entra ID token claims.

    Args:
        claims: Decoded, signature-verified claims.
        settings: Active settings supplying the claim names and clearance map.
        groups: Group object ids resolved out of band (the overage lookup). When
            given they replace the token's ``groups`` claim entirely, because an
            overage token's partial list is not a subset worth merging.

    Returns:
        The resolved principal.

    Raises:
        AuthError: If the token carries no ``oid``/``sub`` or no ``tid``. Both are
            structural: without them there is no user and no tenant boundary.
    """
    user_id = str(claims.get("oid") or claims.get("sub") or "").strip()
    if not user_id:
        raise AuthError("token carries no 'oid' claim", code="auth_missing_subject")
    tenant_id = str(claims.get("tid") or "").strip()
    if not tenant_id:
        raise AuthError("token carries no 'tid' claim", code="auth_missing_tenant")

    roles = claim_list(claims, settings.entra_roles_claim)
    resolved_groups = (
        list(groups)
        if groups is not None
        else claim_list(claims, settings.entra_groups_claim)
    )
    email = claims.get("preferred_username") or claims.get("email") or claims.get("upn")
    return Principal(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles,
        groups=resolved_groups,
        email=str(email) if email else None,
        display_name=str(claims["name"]) if claims.get("name") else None,
        max_classification=max_classification_for(
            roles, resolved_groups, settings=settings
        ),
    )


def parse_dev_principal(raw: str, *, settings: Settings) -> Principal:
    """Parse the unsigned dev-mode principal header.

    Used only when ``entra_dev_mode`` is on, which :class:`ragcore.settings.Settings`
    refuses outright in production. The header carries
    ``Principal.model_dump_json()``; ``max_classification`` may be omitted, in which
    case it is derived from the supplied roles and groups exactly as a real token's
    would be.

    Args:
        raw: Header value: a JSON object.
        settings: Active settings supplying the clearance map.

    Returns:
        The parsed principal.

    Raises:
        AuthError: If the header is not a JSON object or does not describe a valid
            principal.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AuthError(
            "dev principal header is not valid JSON", code="auth_dev_principal_invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise AuthError(
            "dev principal header must be a JSON object",
            code="auth_dev_principal_invalid",
        )
    if not payload.get("max_classification"):
        payload.pop("max_classification", None)
        payload["max_classification"] = max_classification_for(
            [str(role) for role in payload.get("roles") or []],
            [str(group) for group in payload.get("groups") or []],
            settings=settings,
        ).value
    try:
        return Principal.model_validate(payload)
    except ValueError as exc:
        # The message can quote the offending field name but never a token value:
        # a dev principal carries no secret, and the field names are the contract's.
        raise AuthError(
            "dev principal header does not describe a valid Principal",
            code="auth_dev_principal_invalid",
        ) from exc
