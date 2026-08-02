"""Per-document ACL resolution for every connector.

Multi-tenancy is a security boundary, so the ACL stamped on a chunk is never
guessed at the write site. Every connector resolves an
:class:`~ragcore.models.acl.AccessControl` here, from one of four sources:

* a ``<name>.acl.json`` sidecar next to the document (blob and local filesystem),
* blob metadata / index tags (``acl_allowed_groups``, ``classification``, ...),
* Microsoft Graph ``/permissions`` on the SharePoint item, or
* the owning :class:`~ragcore.models.document.SourceConfig` defaults.

Whatever the origin, the result is merged with the source defaults through
:meth:`AccessControl.merged_with`, which keeps the stricter classification and never
lets an unrestricted parent widen a restricted child.

:func:`acl_fingerprint` is the stable hash recorded in
:class:`~ragcore.models.document.IngestManifestEntry`; a change in the fingerprint
alone triggers the cheap ACL-only reindex path instead of a full re-embed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl, Classification
from ragcore.models.document import SourceConfig

__all__ = [
    "ACL_METADATA_PREFIX",
    "SIDECAR_SUFFIX",
    "access_control_from_graph_permissions",
    "access_control_from_mapping",
    "access_control_from_metadata",
    "access_control_from_sidecar",
    "acl_fingerprint",
    "merge_with_source_defaults",
    "parse_identifier_list",
    "sidecar_name_for",
]

_log = get_logger(__name__)

#: Suffix of the ACL sidecar file that accompanies a blob or local document.
SIDECAR_SUFFIX = ".acl.json"

#: Prefix accepted on blob metadata keys and index tags, e.g. ``acl_allowed_groups``.
ACL_METADATA_PREFIX = "acl_"

#: Accepted aliases for each ACL list, checked with and without the prefix.
_LIST_ALIASES: dict[str, tuple[str, ...]] = {
    "allowed_roles": ("allowed_roles", "roles", "allowedroles"),
    "allowed_groups": ("allowed_groups", "groups", "allowedgroups"),
    "allowed_users": ("allowed_users", "users", "allowedusers"),
    "denied_users": ("denied_users", "denied", "deny", "denyusers", "deniedusers"),
}

#: Accepted aliases for the classification label.
_CLASSIFICATION_ALIASES: tuple[str, ...] = (
    "classification",
    "sensitivity",
    "confidentiality",
)

#: Characters treated as separators inside a single metadata string value.
_SEPARATORS = ",;|"


def sidecar_name_for(name: str) -> str:
    """Build the sidecar path that carries the ACL for a document.

    Args:
        name: Blob name or file name of the document, e.g. ``"hr/policy.pdf"``.

    Returns:
        The sidecar name, e.g. ``"hr/policy.pdf.acl.json"``.
    """
    return f"{name}{SIDECAR_SUFFIX}"


def parse_identifier_list(value: Any) -> list[str]:
    """Coerce a metadata value into a clean list of identifiers.

    Blob metadata and index tags can only hold strings, so a list arrives either as a
    JSON array or as a separator-delimited string. Both forms are accepted, as is an
    already-parsed sequence.

    Args:
        value: Raw value from a sidecar, metadata entry, tag or SQL column.

    Returns:
        A list of non-empty, de-duplicated identifiers in first-seen order.
    """
    if value is None:
        return []
    raw: Iterable[Any]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                raw = decoded
            else:
                raw = _split_delimited(text)
        else:
            raw = _split_delimited(text)
    elif isinstance(value, Sequence):
        raw = value
    else:
        raw = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _split_delimited(text: str) -> list[str]:
    """Split a delimited metadata string on any supported separator.

    Args:
        text: The raw string value.

    Returns:
        The split tokens, still unstripped.
    """
    normalised = text
    for separator in _SEPARATORS:
        normalised = normalised.replace(separator, "\n")
    return normalised.split("\n")


def _lookup(data: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    """Find the first present alias in a case-insensitive mapping.

    Args:
        data: Mapping whose keys may be prefixed and arbitrarily cased.
        aliases: Candidate key names, without the ``acl_`` prefix.

    Returns:
        The matching value, or None when no alias is present.
    """
    lowered = {str(key).strip().lower(): value for key, value in data.items()}
    for alias in aliases:
        for candidate in (alias, f"{ACL_METADATA_PREFIX}{alias}"):
            if candidate in lowered:
                return lowered[candidate]
    return None


def _parse_classification(value: Any, fallback: Classification) -> Classification:
    """Parse a classification label, tolerating case and unknown values.

    Args:
        value: Raw label such as ``"Confidential"``.
        fallback: Label used when the value is missing or unrecognised.

    Returns:
        The parsed :class:`Classification`, or ``fallback``.
    """
    if value is None:
        return fallback
    token = str(value).strip().lower()
    if not token:
        return fallback
    try:
        return Classification(token)
    except ValueError:
        _log.warning("acl.unknown_classification", label=token)
        return fallback


def access_control_from_mapping(
    data: Mapping[str, Any],
    source: SourceConfig,
) -> AccessControl | None:
    """Build an item ACL from a flat mapping of metadata-style keys.

    Recognised keys (with or without the ``acl_`` prefix, case-insensitive):
    ``allowed_roles``/``roles``, ``allowed_groups``/``groups``,
    ``allowed_users``/``users``, ``denied_users``/``deny`` and ``classification``.

    Args:
        data: The metadata mapping.
        source: Owning source config, used for the tenant and the default label.

    Returns:
        The item ACL, or None when the mapping carries no ACL information at all
        (so the caller can fall back to the source defaults without merging).

    Raises:
        ValueError: If the mapping declares a ``tenant_id`` other than the source's.
            Cross-tenant metadata is a security bug, never a silent override.
    """
    if not data:
        return None

    declared_tenant = _lookup(data, ("tenant_id", "tenant"))
    if declared_tenant is not None and str(declared_tenant).strip() != source.tenant_id:
        msg = (
            "document metadata declares a different tenant than its source: "
            f"source tenant {source.tenant_id!r}"
        )
        raise ValueError(msg)

    lists = {
        field: parse_identifier_list(_lookup(data, aliases))
        for field, aliases in _LIST_ALIASES.items()
    }
    raw_classification = _lookup(data, _CLASSIFICATION_ALIASES)
    if raw_classification is None and not any(lists.values()):
        return None

    return AccessControl(
        tenant_id=source.tenant_id,
        classification=_parse_classification(
            raw_classification, source.default_classification
        ),
        **lists,
    )


def access_control_from_sidecar(
    raw: bytes | str,
    source: SourceConfig,
) -> AccessControl | None:
    """Parse a ``<name>.acl.json`` sidecar.

    The sidecar is a JSON object using the same key names as
    :class:`~ragcore.models.acl.AccessControl`; unknown keys are ignored so a sidecar
    may also carry human notes.

    Args:
        raw: Sidecar contents as bytes or text.
        source: Owning source config.

    Returns:
        The item ACL, or None when the sidecar is empty, malformed or carries no ACL
        keys. A malformed sidecar is logged and treated as absent rather than being
        allowed to widen access.

    Raises:
        ValueError: If the sidecar declares a different tenant than its source.
    """
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        _log.warning("acl.sidecar_unparseable", source_id=source.source_id)
        return None
    if not isinstance(payload, Mapping):
        _log.warning("acl.sidecar_not_object", source_id=source.source_id)
        return None
    return access_control_from_mapping(payload, source)


def access_control_from_metadata(
    metadata: Mapping[str, Any] | None,
    tags: Mapping[str, Any] | None,
    source: SourceConfig,
) -> AccessControl | None:
    """Build an item ACL from blob metadata and blob index tags.

    Index tags win over metadata for the same key, because tags are what Azure can
    filter on server-side and are therefore the authoritative copy.

    Args:
        metadata: Blob metadata dictionary, or None.
        tags: Blob index tags dictionary, or None.
        source: Owning source config.

    Returns:
        The item ACL, or None when neither carries ACL information.

    Raises:
        ValueError: If either mapping declares a different tenant than its source.
    """
    merged: dict[str, Any] = {}
    merged.update(metadata or {})
    merged.update(tags or {})
    return access_control_from_mapping(merged, source)


def access_control_from_graph_permissions(
    permissions: Iterable[Mapping[str, Any]],
    source: SourceConfig,
) -> AccessControl | None:
    """Map Microsoft Graph ``/permissions`` onto an item ACL.

    Graph returns one permission object per grant. Modern responses carry
    ``grantedToV2`` / ``grantedToIdentitiesV2`` with ``user``, ``group`` or
    ``siteGroup`` identities; the legacy ``grantedTo`` / ``grantedToIdentities``
    shapes are also read. Only Entra object IDs are used — display names are
    deliberately ignored, since they are neither stable nor unique.

    A grant whose ``link.scope`` is ``"anonymous"`` or ``"organization"`` is treated
    as tenant-wide: it contributes no allow entries, leaving the document
    unrestricted within the tenant (still subject to classification).

    Args:
        permissions: The ``value`` array from ``/drives/{id}/items/{id}/permissions``.
        source: Owning source config.

    Returns:
        The item ACL, or None when no object IDs could be resolved.
    """
    groups: list[str] = []
    users: list[str] = []
    tenant_wide = False

    for permission in permissions:
        link = permission.get("link")
        if isinstance(link, Mapping):
            scope = str(link.get("scope", "")).lower()
            if scope in {"anonymous", "organization"}:
                tenant_wide = True

        identities: list[Mapping[str, Any]] = []
        for key in ("grantedToV2", "grantedTo"):
            value = permission.get(key)
            if isinstance(value, Mapping):
                identities.append(value)
        for key in ("grantedToIdentitiesV2", "grantedToIdentities"):
            value = permission.get(key)
            if isinstance(value, list):
                identities.extend(item for item in value if isinstance(item, Mapping))

        for identity in identities:
            for kind in ("group", "siteGroup"):
                object_id = _identity_object_id(identity.get(kind))
                if object_id:
                    groups.append(object_id)
            for kind in ("user", "siteUser", "application"):
                object_id = _identity_object_id(identity.get(kind))
                if object_id:
                    users.append(object_id)

    if not groups and not users:
        if tenant_wide:
            return AccessControl(
                tenant_id=source.tenant_id,
                classification=source.default_classification,
            )
        return None

    return AccessControl(
        tenant_id=source.tenant_id,
        allowed_groups=parse_identifier_list(groups),
        allowed_users=parse_identifier_list(users),
        classification=source.default_classification,
    )


def _identity_object_id(identity: Any) -> str | None:
    """Extract the Entra object id from a Graph identity object.

    Args:
        identity: A Graph ``identity`` object, or anything else.

    Returns:
        The object id, or None when the identity is absent or carries only a
        SharePoint-local id (``loginName``/display name), which is not an Entra
        object id and must not be used as an ACL entry.
    """
    if not isinstance(identity, Mapping):
        return None
    for key in ("id", "objectId"):
        value = identity.get(key)
        if value:
            token = str(value).strip()
            # SharePoint-local principal ids are short integers; Entra object ids are
            # GUIDs. Refuse the former so a numeric site-group id never masquerades
            # as a directory object id.
            if token and not token.isdigit():
                return token
    return None


def merge_with_source_defaults(
    item: AccessControl | None,
    source: SourceConfig,
) -> AccessControl:
    """Combine a resolved item ACL with the source defaults.

    Args:
        item: ACL resolved from the item itself, or None when the source provided
            none.
        source: Owning source config.

    Returns:
        The effective ACL. When ``source.inherit_source_permissions`` is False the
        item ACL is discarded entirely and only the source defaults apply, so an
        operator can pin a whole source to one ACL regardless of item permissions.
    """
    defaults = source.default_access_control()
    if item is None or not source.inherit_source_permissions:
        return defaults
    return item.merged_with(defaults)


def acl_fingerprint(ac: AccessControl) -> str:
    """Hash an ACL so an ACL-only change can be detected without a re-embed.

    The digest covers the flattened form with every list sorted, so reordering the
    same group ids does not look like a change.

    Args:
        ac: The access control to fingerprint.

    Returns:
        A 64-character lowercase hex SHA-256 digest.
    """
    flat = ac.to_flat()
    canonical = {
        key: sorted(value) if isinstance(value, list) else value
        for key, value in sorted(flat.items())
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
