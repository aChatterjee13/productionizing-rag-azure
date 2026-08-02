"""The single security chokepoint: every Qdrant filter in the platform is built here.

Nothing else in this repository may construct a :class:`qdrant_client.models.Filter`
that serves data to a principal. Centralising it means the tenant boundary, the deny
list and the classification ceiling are one reviewable function instead of a rule
repeated at a dozen call sites — and it means the unit tests in
``packages/ragcore/tests/test_filters.py`` are a proof about the whole system.

:func:`build_acl_filter` composition, mirroring
:meth:`ragcore.models.acl.AccessControl.permits` clause for clause:

``must``
    ``tenant_id == principal.tenant_id``, ``is_deleted == False``,
    ``classification_rank <= clearance``, plus every :class:`MetadataFilter` clause.
``must_not``
    ``denied_users`` contains ``principal.user_id`` — explicit deny always wins,
    because ``must_not`` is evaluated independently of the permissive branch.
``should`` / ``min_should=1``
    the permissive branch: ``allowed_users`` matches, **or** ``allowed_groups``
    intersects, **or** ``allowed_roles`` intersects, **or** the document is
    unrestricted (all three lists empty, expressed with ``IsEmpty``).

Two properties are worth stating explicitly because they are what the tests assert:

* A principal from tenant A can never match a tenant-B point. The tenant clause is in
  ``must`` and there is no code path that omits it — ``build_acl_filter`` takes the
  tenant from the :class:`Principal`, never from caller-supplied input.
* ``denied_users`` beats a matching group. ``must_not`` and ``should`` are ANDed by
  Qdrant, so satisfying the permissive branch cannot rescue a denied user.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from qdrant_client import models as qm

from ragcore.models.acl import Classification, Principal
from ragcore.models.memory import MemoryKind
from ragcore.models.retrieval import MetadataFilter
from ragcore.vectorstore.collections import point_id_for_chunk

__all__ = [
    "FINGERPRINT_VERSION",
    "MIN_SHOULD",
    "build_acl_filter",
    "build_acl_filter_for_chunk_ids",
    "build_cache_filter",
    "build_memory_filter",
    "build_tenant_filter",
    "classification_ceiling",
    "filter_fingerprint",
    "serialise_filter",
]

#: Number of permissive-branch conditions a point must satisfy. One: the branch is a
#: disjunction (user match OR group match OR role match OR unrestricted).
MIN_SHOULD = 1

#: Bumped whenever the fingerprint inputs change, so old semantic-cache entries can
#: never be reused under new filter semantics.
FINGERPRINT_VERSION = "v1"


def _match_any(key: str, values: Sequence[str]) -> qm.FieldCondition:
    """Build an "any of" condition on a keyword field.

    Args:
        key: Payload field name.
        values: Accepted values; matches when the field (or any element of it, for
            array fields) equals one of them.

    Returns:
        The field condition.
    """
    return qm.FieldCondition(key=key, match=qm.MatchAny(any=list(values)))


def _match_value(key: str, value: str) -> qm.FieldCondition:
    """Build an equality condition on a keyword field.

    Args:
        key: Payload field name.
        value: Value to match. For an array field this matches when any element
            equals ``value``.

    Returns:
        The field condition.
    """
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


def _is_empty(key: str) -> qm.IsEmptyCondition:
    """Build an "field is absent, null or an empty array" condition.

    Args:
        key: Payload field name.

    Returns:
        The IsEmpty condition.
    """
    return qm.IsEmptyCondition(is_empty=qm.PayloadField(key=key))


def _unrestricted_branch() -> qm.Filter:
    """Condition matching a document with no allow-list restrictions at all.

    A document is unrestricted when ``allowed_roles``, ``allowed_groups`` and
    ``allowed_users`` are *all* empty; it is then readable by any principal in the
    tenant, subject to classification and the deny list. Expressed as a nested
    ``Filter`` so the three ``IsEmpty`` conditions are ANDed inside the single
    ``should`` slot they occupy.

    Returns:
        A nested filter requiring all three allow lists to be empty.
    """
    return qm.Filter(
        must=[
            _is_empty("allowed_roles"),
            _is_empty("allowed_groups"),
            _is_empty("allowed_users"),
        ]
    )


def _effective_clearance(principal: Principal, extra: MetadataFilter | None) -> int:
    """Resolve the classification ceiling for this request.

    A :class:`MetadataFilter` may only *narrow* the ceiling. Taking the minimum of
    the two means a caller cannot widen their clearance by sending
    ``max_classification=restricted``.

    Args:
        principal: The caller.
        extra: Optional metadata filter that may carry ``max_classification``.

    Returns:
        The effective maximum ``classification_rank`` the caller may see.
    """
    ceiling = principal.clearance_rank()
    if extra is not None and extra.max_classification is not None:
        ceiling = min(ceiling, extra.max_classification.rank)
    return ceiling


def _metadata_conditions(extra: MetadataFilter) -> list[qm.Condition]:
    """Translate a :class:`MetadataFilter` into ``must`` conditions.

    Every clause narrows the result set; nothing here can widen it, which is why
    these are safe to append to ``must`` alongside the ACL clauses.

    Args:
        extra: The facet filter.

    Returns:
        Conditions to AND into the ``must`` list. Empty when nothing is set.
    """
    conditions: list[qm.Condition] = []
    if extra.doc_types:
        conditions.append(_match_any("doc_type", extra.doc_types))
    if extra.source_types:
        conditions.append(_match_any("source_type", extra.source_types))
    if extra.tags:
        conditions.append(_match_any("tags", extra.tags))
    if extra.authors:
        conditions.append(_match_any("author", extra.authors))
    if extra.languages:
        conditions.append(_match_any("language", extra.languages))
    if extra.document_ids:
        conditions.append(_match_any("document_id", extra.document_ids))
    if extra.section_prefix:
        # section_path is a keyword array holding the breadcrumb root-to-leaf, so a
        # heading appearing anywhere in it means this chunk lives at or beneath that
        # heading. That is the section-subtree reading of "prefix", and it is the one
        # a keyword index can answer without a payload scan.
        conditions.append(_match_value("section_path", extra.section_prefix))
    if extra.date_from is not None or extra.date_to is not None:
        conditions.append(
            qm.FieldCondition(
                key="source_modified_at",
                range=qm.DatetimeRange(
                    gte=extra.date_from,
                    lte=extra.date_to,
                ),
            )
        )
    if extra.exclude_pii:
        conditions.append(_is_empty("pii_types"))
    return conditions


def build_acl_filter(
    principal: Principal,
    extra: MetadataFilter | None = None,
    *,
    include_deleted: bool = False,
) -> qm.Filter:
    """Build the mandatory access-control filter for a chunk search.

    This is the only supported way to produce a filter for ``rag_chunks``. It is
    intentionally not parameterised by tenant: the tenant comes from ``principal``,
    so no caller can substitute one.

    Args:
        principal: The authenticated caller. Supplies the tenant, the user id
            checked against ``denied_users``, the groups and roles for the
            permissive branch, and the clearance ceiling.
        extra: Optional facet filter. Every clause narrows the result set;
            ``max_classification`` can only lower the ceiling, never raise it.
        include_deleted: Include soft-deleted chunks. Forensic and admin tooling
            only — leaving this True in a retrieval path leaks tombstoned content.

    Returns:
        A :class:`qdrant_client.models.Filter` with ``must``, ``must_not``,
        ``should`` and ``min_should`` populated as documented in this module.

    Example:
        >>> from ragcore.models.acl import Principal
        >>> f = build_acl_filter(Principal(user_id="u1", tenant_id="t1"))
        >>> f.min_should.min_count
        1
    """
    must: list[qm.Condition] = [
        _match_value("tenant_id", principal.tenant_id),
    ]
    if not include_deleted:
        must.append(
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False))
        )
    must.append(
        qm.FieldCondition(
            key="classification_rank",
            range=qm.Range(lte=_effective_clearance(principal, extra)),
        )
    )
    if extra is not None:
        must.extend(_metadata_conditions(extra))

    # Explicit deny. Evaluated independently of `should`, so a denied user cannot be
    # rescued by also matching an allowed group.
    must_not: list[qm.Condition] = [_match_value("denied_users", principal.user_id)]

    should: list[qm.Condition] = [
        _match_value("allowed_users", principal.user_id),
        _unrestricted_branch(),
    ]
    if principal.groups:
        should.append(_match_any("allowed_groups", principal.groups))
    if principal.roles:
        should.append(_match_any("allowed_roles", principal.roles))

    return qm.Filter(
        must=must,
        must_not=must_not,
        should=should,
        # Redundant with `should` (Qdrant already requires at least one `should` to
        # match) but stated explicitly: the permissive branch is a disjunction whose
        # threshold is exactly one, and an accidental future edit that turned
        # `should` into an AND would fail the test that asserts min_count == 1.
        min_should=qm.MinShould(conditions=list(should), min_count=MIN_SHOULD),
    )


def build_acl_filter_for_chunk_ids(
    principal: Principal,
    chunk_ids: Sequence[str],
    extra: MetadataFilter | None = None,
    *,
    include_deleted: bool = False,
) -> qm.Filter:
    """ACL filter narrowed to a specific set of chunk ids.

    This is the semantic-cache re-fetch path (pipeline stage 4). Cached ``chunk_ids``
    are *never* trusted: they are re-fetched through the full ACL filter, so a
    principal who has since lost access simply gets fewer chunks instead of a stale
    leak. The id restriction is expressed with ``HasIdCondition`` against the
    deterministic point ids from
    :func:`ragcore.vectorstore.collections.point_id_for_chunk`, which is indexed by
    construction and needs no payload scan.

    Args:
        principal: The authenticated caller.
        chunk_ids: Logical chunk ids to restrict to. An empty sequence yields a
            filter that matches nothing, which is the safe interpretation.
        extra: Optional facet filter, applied as in :func:`build_acl_filter`.
        include_deleted: See :func:`build_acl_filter`.

    Returns:
        The composed filter.
    """
    base = build_acl_filter(principal, extra, include_deleted=include_deleted)
    must = list(base.must or [])
    must.append(
        qm.HasIdCondition(has_id=[point_id_for_chunk(cid) for cid in chunk_ids])
    )
    return qm.Filter(
        must=must,
        must_not=base.must_not,
        should=base.should,
        min_should=base.min_should,
    )


def build_tenant_filter(
    tenant_id: str,
    *,
    document_ids: Sequence[str] | None = None,
    exclude_document_ids: Sequence[str] | None = None,
    source_id: str | None = None,
    chunk_ids: Sequence[str] | None = None,
    include_deleted: bool = True,
    only_deleted: bool = False,
) -> qm.Filter:
    """Tenant-scoped filter for ingestion and administrative **writes**.

    This is **not** an ACL filter and must never be used to serve a read to a
    principal — it has no deny list, no permissive branch and no classification
    ceiling, because ingestion is not acting on behalf of a user. It lives here
    anyway so that no module outside this one ever hand-rolls a
    :class:`qdrant_client.models.Filter`.

    Args:
        tenant_id: Tenant the operation is scoped to. Always applied.
        document_ids: Restrict to these documents.
        exclude_document_ids: Exclude these documents (used to tombstone everything
            a manifest no longer lists).
        source_id: Restrict to one source config.
        chunk_ids: Restrict to specific chunks, by deterministic point id.
        include_deleted: When False, exclude already soft-deleted chunks so a
            tombstone pass does not rewrite rows it has already tombstoned.
        only_deleted: When True, restrict to soft-deleted chunks (purge tooling).

    Returns:
        The composed filter.

    Raises:
        ValueError: If both ``include_deleted=False`` and ``only_deleted=True`` are
            requested, which can never match anything.
    """
    if only_deleted and not include_deleted:
        msg = "only_deleted=True contradicts include_deleted=False"
        raise ValueError(msg)

    must: list[qm.Condition] = [_match_value("tenant_id", tenant_id)]
    if source_id is not None:
        must.append(_match_value("source_id", source_id))
    if document_ids:
        must.append(_match_any("document_id", document_ids))
    if exclude_document_ids:
        must.append(
            qm.FieldCondition(
                key="document_id",
                match=qm.MatchExcept(**{"except": list(exclude_document_ids)}),
            )
        )
    if chunk_ids is not None:
        must.append(
            qm.HasIdCondition(has_id=[point_id_for_chunk(cid) for cid in chunk_ids])
        )
    if only_deleted:
        must.append(
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=True))
        )
    elif not include_deleted:
        must.append(
            qm.FieldCondition(key="is_deleted", match=qm.MatchValue(value=False))
        )
    return qm.Filter(must=must)


def build_memory_filter(
    principal: Principal,
    kinds: Iterable[MemoryKind | str] | None = None,
    *,
    include_expired: bool = False,
    now: datetime | None = None,
) -> qm.Filter:
    """Build the filter for a long-term memory search in ``rag_memories``.

    Memories are private to one user inside one tenant: both clauses are in ``must``,
    so a memory can never surface for a different user even if the dense vector is a
    perfect match.

    Args:
        principal: The caller whose memories are being read.
        kinds: Restrict to these :class:`~ragcore.models.memory.MemoryKind` values.
            None or empty means every kind.
        include_expired: Include memories past their ``expires_at``. Retrieval never
            should; cleanup tooling does.
        now: Reference time for the expiry check. Defaults to now (UTC).

    Returns:
        The composed filter.
    """
    must: list[qm.Condition] = [
        _match_value("tenant_id", principal.tenant_id),
        _match_value("user_id", principal.user_id),
    ]
    kind_values = [str(kind) for kind in (kinds or [])]
    if kind_values:
        must.append(_match_any("kind", kind_values))

    if include_expired:
        return qm.Filter(must=must)

    # "Not expired" is a disjunction: either no expiry was set, or it is in the
    # future. Both arms live in `should` with a threshold of one.
    reference = now or datetime.now(UTC)
    should: list[qm.Condition] = [
        _is_empty("expires_at"),
        qm.FieldCondition(
            key="expires_at",
            range=qm.DatetimeRange(gt=reference),
        ),
    ]
    return qm.Filter(
        must=must,
        should=should,
        min_should=qm.MinShould(conditions=list(should), min_count=MIN_SHOULD),
    )


def build_cache_filter(principal: Principal, fingerprint: str) -> qm.Filter:
    """Build the filter for a semantic-cache probe in ``rag_semantic_cache``.

    The fingerprint match is exact and mandatory: a cache hit must have been produced
    under the same metadata filter and the same clearance, otherwise the cached
    retrieval plan does not describe this request. Similarity alone is never enough
    (see :meth:`ragcore.models.memory.SemanticCacheEntry.matches`).

    Args:
        principal: The caller probing the cache.
        fingerprint: Output of :func:`filter_fingerprint` for this request.

    Returns:
        The composed filter. Matches the caller's own entries and tenant-wide entries
        (``user_id`` unset), never another user's.
    """
    must: list[qm.Condition] = [
        _match_value("tenant_id", principal.tenant_id),
        _match_value("filter_fingerprint", fingerprint),
    ]
    should: list[qm.Condition] = [
        _match_value("user_id", principal.user_id),
        _is_empty("user_id"),
    ]
    return qm.Filter(
        must=must,
        should=should,
        min_should=qm.MinShould(conditions=list(should), min_count=MIN_SHOULD),
    )


def filter_fingerprint(principal: Principal, extra: MetadataFilter | None) -> str:
    """Hash the request's filter identity for semantic-cache keying.

    Stable and order-independent: the payload is canonical JSON with sorted keys, and
    :meth:`ragcore.models.retrieval.MetadataFilter.fingerprint_payload` sorts every
    list, so two filters differing only in ordering hash identically.

    The tenant is included even though
    :meth:`ragcore.models.memory.SemanticCacheEntry.make_cache_id` also keys on it —
    defence in depth means a fingerprint collision across tenants is impossible by
    construction, not merely unlikely.

    Args:
        principal: The caller. Only the tenant and the clearance rank contribute;
            group and role membership deliberately do not, because cached chunk ids
            are re-filtered through the live ACL filter on reuse.
        extra: The metadata filter for this request, or None.

    Returns:
        A 32-character hex digest.
    """
    payload = {
        "version": FINGERPRINT_VERSION,
        "tenant_id": principal.tenant_id,
        "clearance_rank": principal.clearance_rank(),
        "filter": extra.fingerprint_payload() if extra is not None else {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def serialise_filter(qfilter: qm.Filter) -> dict[str, object]:
    """Render a filter for tracing and for ``RetrievalResult.filter_applied``.

    Args:
        qfilter: The filter that was applied.

    Returns:
        A JSON-safe mapping. Filters carry ids, tenant ids and facet values but never
        document text, so this is safe to log and to return to the client.
    """
    return qfilter.model_dump(mode="json", exclude_none=True)


def classification_ceiling(
    principal: Principal, extra: MetadataFilter | None = None
) -> Classification:
    """Typed view of the effective classification ceiling.

    Stage 12's output guard re-checks emitted content against this, so the
    defence-in-depth check uses exactly the ceiling the filter used.

    Args:
        principal: The caller.
        extra: Optional metadata filter that may narrow the ceiling.

    Returns:
        The most sensitive classification this request may surface.
    """
    return Classification.from_rank(_effective_clearance(principal, extra))
