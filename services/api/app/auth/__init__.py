"""Authentication: Entra ID token validation and principal resolution.

    from app.auth import get_token_validator, principal_from_claims

Importing this package pulls ``httpx`` and ``pyjwt`` but nothing heavier, so it is
safe from the dependency layer.
"""

from __future__ import annotations

from app.auth.entra import (
    EntraTokenValidator,
    JWKSCache,
    assert_dev_mode_allowed,
    extract_bearer,
    get_token_validator,
    reset_token_validator,
)
from app.auth.principal import (
    AUTH_SETTING_DEFAULTS,
    GROUP_OVERAGE_CLAIMS,
    auth_setting,
    claim_list,
    max_classification_for,
    parse_dev_principal,
    principal_from_claims,
    requires_group_lookup,
)

__all__ = [
    "AUTH_SETTING_DEFAULTS",
    "GROUP_OVERAGE_CLAIMS",
    "EntraTokenValidator",
    "JWKSCache",
    "assert_dev_mode_allowed",
    "auth_setting",
    "claim_list",
    "extract_bearer",
    "get_token_validator",
    "max_classification_for",
    "parse_dev_principal",
    "principal_from_claims",
    "requires_group_lookup",
    "reset_token_validator",
]
