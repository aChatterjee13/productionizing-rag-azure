"""Access-control primitives: classification levels, document ACLs and principals.

Multi-tenancy is a security boundary. A chunk is visible to a principal iff **all**
of the following hold:

1. ``tenant_id`` matches exactly, and
2. the principal is not in ``denied_users`` (explicit deny always wins), and
3. ``allowed_roles`` / ``allowed_groups`` / ``allowed_users`` are all empty (the
   document is unrestricted within the tenant) **or** the principal matches at
   least one of them, and
4. ``chunk.classification.rank <= principal.clearance_rank()``.

The only component permitted to translate these rules into a Qdrant filter is
:func:`ragcore.vectorstore.filters.build_acl_filter`. :meth:`AccessControl.permits`
here is the in-process mirror of the same rule, used for defence-in-depth checks
after retrieval and before output.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ADMIN_ROLE",
    "AccessControl",
    "Classification",
    "Principal",
]

#: App role that grants access to the admin surface. Mirrored by the
#: ``entra_admin_role`` setting for call sites that need it configurable.
ADMIN_ROLE = "rag.admin"


class Classification(StrEnum):
    """Ordered data-sensitivity label.

    Members compare by :attr:`rank`, not lexicographically, so
    ``Classification.PUBLIC < Classification.RESTRICTED`` is True and sorting a
    list of classifications yields least-to-most sensitive order.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        """Numeric sensitivity, ascending.

        Returns:
            0 for ``PUBLIC`` through 3 for ``RESTRICTED``. Denormalised into
            ``ChunkPayload.classification_rank`` so Qdrant can range-filter on it.
        """
        return _CLASSIFICATION_RANKS[self]

    @classmethod
    def from_rank(cls, rank: int) -> Classification:
        """Invert :attr:`rank`.

        Args:
            rank: Numeric sensitivity, clamped into the valid range.

        Returns:
            The matching classification.
        """
        ordered = _CLASSIFICATION_ORDER
        index = min(max(rank, 0), len(ordered) - 1)
        return ordered[index]

    @classmethod
    def max_rank(cls) -> int:
        """Highest valid rank.

        Returns:
            The rank of the most sensitive classification.
        """
        return len(_CLASSIFICATION_ORDER) - 1

    def __lt__(self, other: object) -> bool:
        """Compare by rank rather than by string value.

        Args:
            other: Another classification.

        Returns:
            True when this classification is strictly less sensitive.
        """
        if isinstance(other, Classification):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Compare by rank rather than by string value.

        Args:
            other: Another classification.

        Returns:
            True when this classification is no more sensitive than ``other``.
        """
        if isinstance(other, Classification):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Compare by rank rather than by string value.

        Args:
            other: Another classification.

        Returns:
            True when this classification is strictly more sensitive.
        """
        if isinstance(other, Classification):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Compare by rank rather than by string value.

        Args:
            other: Another classification.

        Returns:
            True when this classification is at least as sensitive as ``other``.
        """
        if isinstance(other, Classification):
            return self.rank >= other.rank
        return NotImplemented


_CLASSIFICATION_ORDER: tuple[Classification, ...] = (
    Classification.PUBLIC,
    Classification.INTERNAL,
    Classification.CONFIDENTIAL,
    Classification.RESTRICTED,
)
_CLASSIFICATION_RANKS: dict[Classification, int] = {
    level: index for index, level in enumerate(_CLASSIFICATION_ORDER)
}


def _dedupe(values: list[str]) -> list[str]:
    """Normalise an ACL list: strip blanks, de-duplicate, preserve order.

    Args:
        values: Raw list of identifiers.

    Returns:
        A cleaned list with stable ordering.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


class AccessControl(BaseModel):
    """Who may see a document or chunk, within one tenant.

    Attributes:
        tenant_id: Owning tenant. Never optional — it is the security boundary.
        allowed_roles: App roles permitted. Empty means no role restriction.
        allowed_groups: Entra group object IDs permitted. Empty means no restriction.
        allowed_users: Entra user object IDs (``oid``) permitted. Empty means no
            restriction.
        denied_users: Entra user object IDs explicitly denied. Always wins over
            every allow rule.
        classification: Sensitivity label gating which principals may read it.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Owning tenant id; the security boundary.")
    allowed_roles: list[str] = Field(
        default_factory=list,
        description="Permitted app roles; empty means unrestricted.",
    )
    allowed_groups: list[str] = Field(
        default_factory=list,
        description="Permitted Entra group object IDs; empty means unrestricted.",
    )
    allowed_users: list[str] = Field(
        default_factory=list,
        description="Permitted Entra user object IDs; empty means unrestricted.",
    )
    denied_users: list[str] = Field(
        default_factory=list,
        description="Explicitly denied Entra user object IDs; deny always wins.",
    )
    classification: Classification = Field(
        default=Classification.INTERNAL,
        description="Sensitivity label compared against the principal's clearance.",
    )

    @field_validator("allowed_roles", "allowed_groups", "allowed_users", "denied_users")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        """Normalise ACL lists so equality and payload hashing are stable.

        Args:
            value: Raw identifier list.

        Returns:
            Cleaned, de-duplicated list.
        """
        return _dedupe(value)

    @property
    def is_unrestricted(self) -> bool:
        """Whether every principal in the tenant may read this, clearance aside.

        Returns:
            True when all three allow lists are empty.
        """
        return not (self.allowed_roles or self.allowed_groups or self.allowed_users)

    @property
    def classification_rank(self) -> int:
        """Denormalised numeric sensitivity for range filtering.

        Returns:
            The rank of :attr:`classification`.
        """
        return self.classification.rank

    def permits(self, principal: Principal) -> bool:
        """Evaluate the visibility rule in process.

        This mirrors :func:`ragcore.vectorstore.filters.build_acl_filter` exactly and
        exists for defence in depth (stage 12 output guard, ACL negative tests). It
        is not a substitute for filtering in Qdrant.

        Args:
            principal: The caller.

        Returns:
            True when the principal may read the protected content.
        """
        if principal.tenant_id != self.tenant_id:
            return False
        if principal.user_id in self.denied_users:
            return False
        if self.classification.rank > principal.clearance_rank():
            return False
        if self.is_unrestricted:
            return True
        if principal.user_id in self.allowed_users:
            return True
        if self.allowed_groups and not set(self.allowed_groups).isdisjoint(
            principal.groups
        ):
            return True
        return bool(
            self.allowed_roles
            and not set(self.allowed_roles).isdisjoint(principal.roles)
        )

    def merged_with(self, other: AccessControl) -> AccessControl:
        """Combine two ACLs so the result never grants more than ``self`` does.

        ``self`` is the authoritative ACL — the one resolved from the item itself —
        and ``other`` supplies defaults:

        * **Allow lists fill in, they never add.** When ``self`` restricts at all its
          three allow lists are kept verbatim and ``other``'s are ignored; only when
          ``self`` is unrestricted do ``other``'s apply. :meth:`permits` reads the
          three lists as a *disjunction*, so unioning them grants access to every
          principal ``other`` names — a source default of
          ``allowed_groups=["g-staff"]`` would hand the whole of ``g-staff`` read on
          an item SharePoint had restricted to a single user.
        * **Deny lists union.** Deny always wins, so more of it is always stricter.
        * **Classification takes the maximum**, the more sensitive label.

        For every principal ``merged.permits(p)`` implies ``self.permits(p)``, which
        is the property the ingestion path depends on when it layers source defaults
        onto permissions read from the source system.

        Args:
            other: The ACL to merge in (typically a source default).

        Returns:
            A new :class:`AccessControl`, no more permissive than ``self``.

        Raises:
            ValueError: If the two ACLs belong to different tenants.
        """
        if self.tenant_id != other.tenant_id:
            msg = (
                "cannot merge access control across tenants: "
                f"{self.tenant_id!r} vs {other.tenant_id!r}"
            )
            raise ValueError(msg)

        # Pivot on the whole ACL rather than per list: filling one empty list from
        # `other` widens the disjunction just as surely as unioning all three would.
        grants = other if self.is_unrestricted else self
        return AccessControl(
            tenant_id=self.tenant_id,
            allowed_roles=list(grants.allowed_roles),
            allowed_groups=list(grants.allowed_groups),
            allowed_users=list(grants.allowed_users),
            denied_users=_dedupe([*self.denied_users, *other.denied_users]),
            classification=max(self.classification, other.classification),
        )

    def to_flat(self) -> dict[str, Any]:
        """Flatten to the payload fields Qdrant can index.

        Nested objects cannot be payload-indexed, so ``ChunkPayload`` stores these
        six fields at the top level. Use
        :meth:`ragcore.models.chunk.ChunkPayload.from_access_control` rather than
        calling this by hand.

        Returns:
            A mapping of flat ACL payload fields.
        """
        return {
            "tenant_id": self.tenant_id,
            "allowed_roles": list(self.allowed_roles),
            "allowed_groups": list(self.allowed_groups),
            "allowed_users": list(self.allowed_users),
            "denied_users": list(self.denied_users),
            "classification": self.classification.value,
            "classification_rank": self.classification.rank,
        }

    @classmethod
    def from_flat(cls, payload: dict[str, Any]) -> Self:
        """Rebuild an ACL from flat payload fields.

        Args:
            payload: Mapping containing the flat ACL keys produced by
                :meth:`to_flat` (extra keys are ignored).

        Returns:
            The reconstructed :class:`AccessControl`.
        """
        return cls(
            tenant_id=str(payload["tenant_id"]),
            allowed_roles=list(payload.get("allowed_roles") or []),
            allowed_groups=list(payload.get("allowed_groups") or []),
            allowed_users=list(payload.get("allowed_users") or []),
            denied_users=list(payload.get("denied_users") or []),
            classification=Classification(
                payload.get("classification") or Classification.INTERNAL.value
            ),
        )


class Principal(BaseModel):
    """An authenticated caller, resolved from Entra ID token claims.

    Attributes:
        user_id: Entra ``oid`` claim; the stable per-user object id.
        tenant_id: Entra ``tid`` claim; the tenant this caller is scoped to.
        roles: App roles from the ``roles`` claim.
        groups: Entra group object IDs from the ``groups`` claim.
        email: Preferred username or email, when present in the token.
        display_name: Human-readable name, when present in the token.
        max_classification: Highest sensitivity this caller may read.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="Entra `oid` claim: stable user object id.")
    tenant_id: str = Field(description="Entra `tid` claim: the caller's tenant.")
    roles: list[str] = Field(
        default_factory=list, description="App roles from the token's `roles` claim."
    )
    groups: list[str] = Field(
        default_factory=list,
        description="Entra group object IDs from the token's `groups` claim.",
    )
    email: str | None = Field(
        default=None, description="Email or preferred_username, when present."
    )
    display_name: str | None = Field(
        default=None, description="Display name, when present."
    )
    max_classification: Classification = Field(
        default=Classification.INTERNAL,
        description="Clearance ceiling: content above this is never returned.",
    )

    @field_validator("roles", "groups")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        """Normalise claim lists.

        Args:
            value: Raw claim values.

        Returns:
            Cleaned, de-duplicated list.
        """
        return _dedupe(value)

    def clearance_rank(self) -> int:
        """Numeric clearance ceiling for range filtering.

        Returns:
            The rank of :attr:`max_classification`.
        """
        return self.max_classification.rank

    def is_admin(self, admin_role: str = ADMIN_ROLE) -> bool:
        """Whether this caller holds the admin app role.

        Args:
            admin_role: Role name to check. Defaults to :data:`ADMIN_ROLE`; pass
                ``settings.entra_admin_role`` when it has been overridden.

        Returns:
            True when the role is present.
        """
        return admin_role in self.roles

    def can_read(self, acl: AccessControl) -> bool:
        """Whether this caller may read content protected by ``acl``.

        Args:
            acl: The access control to evaluate.

        Returns:
            True when :meth:`AccessControl.permits` allows this principal.
        """
        return acl.permits(self)

    def audit_identity(self) -> dict[str, Any]:
        """Low-cardinality identity fields safe to log and persist.

        Returns:
            A mapping with the tenant, user id, role/group counts and clearance.
            Deliberately excludes email and display name so audit rows carry no
            unredacted personal data.
        """
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role_count": len(self.roles),
            "group_count": len(self.groups),
            "max_classification": self.max_classification.value,
            "is_admin": self.is_admin(),
        }
