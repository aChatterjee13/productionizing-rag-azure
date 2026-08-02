"""Tests for the ACL filter — the security chokepoint for requirement #1.

Two complementary levels:

* **Structure.** Assert the filter Qdrant receives has the tenant clause in ``must``,
  the deny clause in ``must_not``, and a four-arm permissive branch with
  ``min_should=1``. A regression that quietly moved the tenant clause into ``should``
  would still return plausible results, so shape is worth pinning.
* **Semantics.** A self-contained evaluator (``evaluate``) interprets the generated
  filter against synthetic payloads exactly as Qdrant does, and the visibility
  assertions run through it. That is what actually proves cross-tenant isolation and
  deny-beats-group: the filter is not just shaped right, it *decides* right.

The evaluator is deliberately independent of ``qdrant_client.local`` — testing the
filter against the same library that built it would prove far less.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from qdrant_client import models as qm

from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.memory import MemoryKind
from ragcore.models.retrieval import MetadataFilter
from ragcore.vectorstore.collections import point_id_for_chunk
from ragcore.vectorstore.filters import (
    build_acl_filter,
    build_acl_filter_for_chunk_ids,
    build_cache_filter,
    build_memory_filter,
    build_tenant_filter,
    classification_ceiling,
    filter_fingerprint,
    serialise_filter,
)

# ---------------------------------------------------------------------------
# A minimal, independent Qdrant filter evaluator.
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    """Qdrant treats an array field as a set of values and a scalar as a singleton."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _check_range(condition: qm.Range, values: list[Any]) -> bool:
    numbers = [v for v in values if isinstance(v, (int, float))]
    for number in numbers:
        if condition.lt is not None and not number < condition.lt:
            continue
        if condition.lte is not None and not number <= condition.lte:
            continue
        if condition.gt is not None and not number > condition.gt:
            continue
        if condition.gte is not None and not number >= condition.gte:
            continue
        return True
    return False


def _check_datetime_range(condition: qm.DatetimeRange, values: list[Any]) -> bool:
    for raw in values:
        moment = _to_datetime(raw)
        if moment is None:
            continue
        bounds = (
            (condition.lt, lambda a, b: a < b),
            (condition.lte, lambda a, b: a <= b),
            (condition.gt, lambda a, b: a > b),
            (condition.gte, lambda a, b: a >= b),
        )
        if all(
            bound is None or compare(moment, _to_datetime(bound))
            for bound, compare in bounds
        ):
            return True
    return False


def _check_field(condition: qm.FieldCondition, payload: dict[str, Any]) -> bool:
    values = _as_list(payload.get(condition.key))
    if condition.match is not None:
        match = condition.match
        if isinstance(match, qm.MatchValue):
            return match.value in values
        if isinstance(match, qm.MatchAny):
            return bool(set(match.any) & set(values))
        if isinstance(match, qm.MatchExcept):
            return any(value not in set(match.except_) for value in values)
        raise AssertionError(f"evaluator does not handle match {match!r}")
    if condition.range is not None:
        if isinstance(condition.range, qm.DatetimeRange):
            return _check_datetime_range(condition.range, values)
        return _check_range(condition.range, values)
    raise AssertionError(f"evaluator does not handle condition {condition!r}")


def _check_condition(
    condition: Any, payload: dict[str, Any], point_id: str | None
) -> bool:
    if isinstance(condition, qm.Filter):
        return evaluate(condition, payload, point_id=point_id)
    if isinstance(condition, qm.IsEmptyCondition):
        value = payload.get(condition.is_empty.key)
        return value is None or (isinstance(value, list) and len(value) == 0)
    if isinstance(condition, qm.HasIdCondition):
        return point_id is not None and point_id in {
            str(candidate) for candidate in condition.has_id
        }
    if isinstance(condition, qm.FieldCondition):
        return _check_field(condition, payload)
    raise AssertionError(f"evaluator does not handle {condition!r}")


def evaluate(
    qfilter: qm.Filter, payload: dict[str, Any], *, point_id: str | None = None
) -> bool:
    """Decide whether a payload satisfies a filter, the way Qdrant would.

    ``must``, ``must_not``, ``should`` and ``min_should`` are four independent gates,
    all of which must pass. That independence is the mechanical reason ``must_not``
    beats ``should``.
    """
    for condition in qfilter.must or []:
        if not _check_condition(condition, payload, point_id):
            return False
    for condition in qfilter.must_not or []:
        if _check_condition(condition, payload, point_id):
            return False
    should = qfilter.should or []
    if should and not any(
        _check_condition(condition, payload, point_id) for condition in should
    ):
        return False
    if qfilter.min_should is not None:
        matched = sum(
            1
            for condition in qfilter.min_should.conditions
            if _check_condition(condition, payload, point_id)
        )
        if matched < qfilter.min_should.min_count:
            return False
    return True


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


def make_principal(
    *,
    tenant_id: str = TENANT_A,
    user_id: str = "user-1",
    roles: list[str] | None = None,
    groups: list[str] | None = None,
    clearance: Classification = Classification.INTERNAL,
) -> Principal:
    return Principal(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles or [],
        groups=groups or [],
        max_classification=clearance,
    )


def make_payload(
    *,
    tenant_id: str = TENANT_A,
    allowed_roles: list[str] | None = None,
    allowed_groups: list[str] | None = None,
    allowed_users: list[str] | None = None,
    denied_users: list[str] | None = None,
    classification: Classification = Classification.INTERNAL,
    is_deleted: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build the flat ACL payload Qdrant stores, via AccessControl.to_flat()."""
    acl = AccessControl(
        tenant_id=tenant_id,
        allowed_roles=allowed_roles or [],
        allowed_groups=allowed_groups or [],
        allowed_users=allowed_users or [],
        denied_users=denied_users or [],
        classification=classification,
    )
    payload: dict[str, Any] = {**acl.to_flat(), "is_deleted": is_deleted}
    payload.update(extra)
    return payload


def visible(principal: Principal, payload: dict[str, Any], **kwargs: Any) -> bool:
    return evaluate(build_acl_filter(principal, **kwargs), payload)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_acl_filter_structure_has_every_mandated_clause():
    principal = make_principal(roles=["reader"], groups=["g-eng"])
    qfilter = build_acl_filter(principal)

    must = list(qfilter.must or [])
    tenant_clauses = [
        c for c in must if isinstance(c, qm.FieldCondition) and c.key == "tenant_id"
    ]
    assert len(tenant_clauses) == 1
    assert tenant_clauses[0].match.value == TENANT_A

    deleted = [
        c for c in must if isinstance(c, qm.FieldCondition) and c.key == "is_deleted"
    ]
    assert len(deleted) == 1
    assert deleted[0].match.value is False

    rank = [
        c
        for c in must
        if isinstance(c, qm.FieldCondition) and c.key == "classification_rank"
    ]
    assert len(rank) == 1
    assert rank[0].range.lte == Classification.INTERNAL.rank

    must_not = list(qfilter.must_not or [])
    assert len(must_not) == 1
    assert must_not[0].key == "denied_users"
    assert must_not[0].match.value == principal.user_id

    # Permissive branch: user, unrestricted, groups, roles.
    should = list(qfilter.should or [])
    assert len(should) == 4
    keys = {c.key for c in should if isinstance(c, qm.FieldCondition)}
    assert keys == {"allowed_users", "allowed_groups", "allowed_roles"}
    nested = [c for c in should if isinstance(c, qm.Filter)]
    assert len(nested) == 1
    unrestricted_keys = {
        c.is_empty.key for c in nested[0].must if isinstance(c, qm.IsEmptyCondition)
    }
    assert unrestricted_keys == {"allowed_roles", "allowed_groups", "allowed_users"}

    assert qfilter.min_should is not None
    assert qfilter.min_should.min_count == 1


def test_acl_filter_omits_empty_group_and_role_arms():
    """A principal with no groups/roles gets a two-arm branch, not empty MatchAny."""
    qfilter = build_acl_filter(make_principal())
    should = list(qfilter.should or [])
    assert len(should) == 2
    assert qfilter.min_should.min_count == 1


def test_include_deleted_drops_the_tombstone_clause():
    qfilter = build_acl_filter(make_principal(), include_deleted=True)
    keys = [c.key for c in qfilter.must if isinstance(c, qm.FieldCondition)]
    assert "is_deleted" not in keys


def test_metadata_filter_clauses_land_in_must():
    extra = MetadataFilter(
        doc_types=["policy"],
        source_types=["sharepoint"],
        tags=["hr"],
        authors=["ada"],
        languages=["en"],
        document_ids=["doc-1"],
        section_prefix="Benefits",
        date_from=datetime(2024, 1, 1, tzinfo=UTC),
        date_to=datetime(2024, 12, 31, tzinfo=UTC),
        exclude_pii=True,
    )
    qfilter = build_acl_filter(make_principal(), extra)
    field_keys = [c.key for c in qfilter.must if isinstance(c, qm.FieldCondition)]
    for key in (
        "doc_type",
        "source_type",
        "tags",
        "author",
        "language",
        "document_id",
        "section_path",
        "source_modified_at",
    ):
        assert key in field_keys
    empties = [
        c.is_empty.key for c in qfilter.must if isinstance(c, qm.IsEmptyCondition)
    ]
    assert "pii_types" in empties


def test_metadata_filter_can_only_narrow_the_clearance():
    principal = make_principal(clearance=Classification.CONFIDENTIAL)

    narrowed = build_acl_filter(
        principal, MetadataFilter(max_classification=Classification.PUBLIC)
    )
    rank = next(
        c for c in narrowed.must if getattr(c, "key", None) == "classification_rank"
    )
    assert rank.range.lte == Classification.PUBLIC.rank

    # A caller asking for RESTRICTED must not be granted it.
    widened = build_acl_filter(
        principal, MetadataFilter(max_classification=Classification.RESTRICTED)
    )
    rank = next(
        c for c in widened.must if getattr(c, "key", None) == "classification_rank"
    )
    assert rank.range.lte == Classification.CONFIDENTIAL.rank
    assert (
        classification_ceiling(
            principal, MetadataFilter(max_classification=Classification.RESTRICTED)
        )
        is Classification.CONFIDENTIAL
    )


# ---------------------------------------------------------------------------
# Semantics: cross-tenant isolation
# ---------------------------------------------------------------------------


def test_cross_tenant_point_is_never_visible():
    """The headline guarantee: tenant A can never match a tenant-B point."""
    principal = make_principal(tenant_id=TENANT_A)
    own = make_payload(tenant_id=TENANT_A)
    foreign = make_payload(tenant_id=TENANT_B)

    assert visible(principal, own) is True
    assert visible(principal, foreign) is False


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        {},
        {"allowed_users": ["user-1"]},
        {"allowed_groups": ["g-eng"]},
        {"allowed_roles": ["reader"]},
        {"classification": Classification.PUBLIC},
    ],
    ids=["unrestricted", "user-match", "group-match", "role-match", "public"],
)
def test_no_permissive_branch_can_cross_the_tenant_boundary(payload_kwargs):
    """Even a point that satisfies every allow rule stays invisible across tenants."""
    principal = make_principal(
        tenant_id=TENANT_A, user_id="user-1", roles=["reader"], groups=["g-eng"]
    )
    foreign = make_payload(tenant_id=TENANT_B, **payload_kwargs)
    assert visible(principal, foreign) is False
    # ...and the identical point inside the caller's own tenant *is* visible, so the
    # test is failing on tenancy and not on some unrelated clause.
    own = make_payload(tenant_id=TENANT_A, **payload_kwargs)
    assert visible(principal, own) is True


def test_admin_role_does_not_cross_tenants():
    admin = make_principal(
        tenant_id=TENANT_A, roles=["rag.admin"], clearance=Classification.RESTRICTED
    )
    assert admin.is_admin() is True
    assert visible(admin, make_payload(tenant_id=TENANT_B)) is False


# ---------------------------------------------------------------------------
# Semantics: deny beats allow
# ---------------------------------------------------------------------------


def test_denied_user_beats_a_matching_group():
    """Explicit deny wins over an otherwise-matching group."""
    principal = make_principal(user_id="mallory", groups=["g-eng"])
    payload = make_payload(allowed_groups=["g-eng"], denied_users=["mallory"])

    # The permissive branch really does match, so the deny is what excludes it.
    permissive = build_acl_filter(principal).should
    assert any(_check_condition(c, payload, None) for c in permissive)
    assert visible(principal, payload) is False

    # A colleague in the same group, not on the deny list, still sees it.
    colleague = make_principal(user_id="alice", groups=["g-eng"])
    assert visible(colleague, payload) is True


@pytest.mark.parametrize(
    "payload_kwargs",
    [
        {"allowed_groups": ["g-eng"]},
        {"allowed_roles": ["reader"]},
        {"allowed_users": ["mallory"]},
        {},
    ],
    ids=["group", "role", "explicit-user", "unrestricted"],
)
def test_deny_beats_every_allow_form(payload_kwargs):
    principal = make_principal(user_id="mallory", roles=["reader"], groups=["g-eng"])
    payload = make_payload(denied_users=["mallory"], **payload_kwargs)
    assert visible(principal, payload) is False


# ---------------------------------------------------------------------------
# Semantics: the permissive branch and the classification ceiling
# ---------------------------------------------------------------------------


def test_unrestricted_document_is_visible_to_any_tenant_member():
    stranger = make_principal(user_id="nobody-in-particular")
    assert visible(stranger, make_payload()) is True


def test_restricted_document_needs_a_matching_allow_entry():
    principal = make_principal(user_id="bob", groups=["g-sales"], roles=["reader"])
    assert visible(principal, make_payload(allowed_groups=["g-eng"])) is False
    assert visible(principal, make_payload(allowed_groups=["g-sales"])) is True
    assert visible(principal, make_payload(allowed_roles=["reader"])) is True
    assert visible(principal, make_payload(allowed_users=["bob"])) is True
    assert visible(principal, make_payload(allowed_users=["carol"])) is False


def test_partial_allow_lists_are_still_a_restriction():
    """A doc with only allowed_users set is not 'unrestricted' for everybody else."""
    outsider = make_principal(user_id="dave", groups=["g-eng"], roles=["reader"])
    assert visible(outsider, make_payload(allowed_users=["erin"])) is False


def test_classification_ceiling_is_enforced():
    payload = make_payload(classification=Classification.CONFIDENTIAL)
    assert visible(make_principal(clearance=Classification.INTERNAL), payload) is False
    assert (
        visible(make_principal(clearance=Classification.CONFIDENTIAL), payload) is True
    )
    assert visible(make_principal(clearance=Classification.RESTRICTED), payload) is True


def test_soft_deleted_chunk_is_hidden_unless_explicitly_included():
    principal = make_principal()
    payload = make_payload(is_deleted=True)
    assert visible(principal, payload) is False
    assert visible(principal, payload, include_deleted=True) is True


# ---------------------------------------------------------------------------
# The filter must agree with AccessControl.permits (the stage-12 mirror)
# ---------------------------------------------------------------------------

_MATRIX_PRINCIPALS = [
    make_principal(user_id="u1"),
    make_principal(user_id="u2", groups=["g-eng"]),
    make_principal(user_id="u3", roles=["reader"]),
    make_principal(user_id="u4", clearance=Classification.RESTRICTED),
    make_principal(user_id="u5", tenant_id=TENANT_B, groups=["g-eng"]),
    make_principal(
        user_id="u6",
        groups=["g-eng"],
        roles=["reader"],
        clearance=Classification.CONFIDENTIAL,
    ),
]

_MATRIX_ACLS = [
    AccessControl(tenant_id=TENANT_A),
    AccessControl(tenant_id=TENANT_A, allowed_groups=["g-eng"]),
    AccessControl(tenant_id=TENANT_A, allowed_roles=["reader"]),
    AccessControl(tenant_id=TENANT_A, allowed_users=["u1"]),
    AccessControl(tenant_id=TENANT_A, denied_users=["u2"], allowed_groups=["g-eng"]),
    AccessControl(tenant_id=TENANT_A, classification=Classification.CONFIDENTIAL),
    AccessControl(tenant_id=TENANT_A, classification=Classification.RESTRICTED),
    AccessControl(tenant_id=TENANT_B, allowed_groups=["g-eng"]),
    AccessControl(
        tenant_id=TENANT_B,
        allowed_users=["u1", "u2", "u3", "u4", "u5", "u6"],
        classification=Classification.PUBLIC,
    ),
]


def _acl_id(acl: AccessControl) -> str:
    parts = [acl.tenant_id, acl.classification.value]
    for label, values in (
        ("roles", acl.allowed_roles),
        ("groups", acl.allowed_groups),
        ("users", acl.allowed_users),
        ("deny", acl.denied_users),
    ):
        if values:
            parts.append(f"{label}={'|'.join(values)}")
    return "-".join(parts)


@pytest.mark.parametrize("principal", _MATRIX_PRINCIPALS, ids=lambda p: p.user_id)
@pytest.mark.parametrize("acl", _MATRIX_ACLS, ids=_acl_id)
def test_filter_matches_in_process_acl_check(principal, acl):
    """build_acl_filter and AccessControl.permits must never disagree.

    They are the same rule expressed twice — once for Qdrant, once for the
    defence-in-depth output guard. A divergence means either the filter leaks or the
    guard rejects content the user is entitled to.
    """
    payload = {**acl.to_flat(), "is_deleted": False}
    assert visible(principal, payload) == acl.permits(principal)


# ---------------------------------------------------------------------------
# Cached chunk-id refetch
# ---------------------------------------------------------------------------


def test_chunk_id_refetch_keeps_the_full_acl_filter():
    principal = make_principal(user_id="u1")
    qfilter = build_acl_filter_for_chunk_ids(principal, ["chunk-a", "chunk-b"])

    assert qfilter.must_not is not None
    assert qfilter.min_should.min_count == 1
    has_id = [c for c in qfilter.must if isinstance(c, qm.HasIdCondition)]
    assert len(has_id) == 1
    assert set(has_id[0].has_id) == {
        point_id_for_chunk("chunk-a"),
        point_id_for_chunk("chunk-b"),
    }

    payload = make_payload()
    assert evaluate(qfilter, payload, point_id=point_id_for_chunk("chunk-a")) is True
    assert evaluate(qfilter, payload, point_id=point_id_for_chunk("chunk-z")) is False


def test_chunk_id_refetch_still_denies_a_revoked_principal():
    """A cache hit must not resurrect access the principal has since lost."""
    principal = make_principal(user_id="mallory", groups=["g-eng"])
    qfilter = build_acl_filter_for_chunk_ids(principal, ["chunk-a"])
    revoked = make_payload(allowed_groups=["g-eng"], denied_users=["mallory"])
    assert evaluate(qfilter, revoked, point_id=point_id_for_chunk("chunk-a")) is False


def test_empty_chunk_id_list_matches_nothing():
    qfilter = build_acl_filter_for_chunk_ids(make_principal(), [])
    assert (
        evaluate(qfilter, make_payload(), point_id=point_id_for_chunk("chunk-a"))
        is False
    )


# ---------------------------------------------------------------------------
# Memory, cache and tenant-write filters
# ---------------------------------------------------------------------------


def test_memory_filter_is_scoped_to_tenant_and_user():
    principal = make_principal(user_id="u1")
    qfilter = build_memory_filter(principal)
    keys = {c.key: c for c in qfilter.must if isinstance(c, qm.FieldCondition)}
    assert keys["tenant_id"].match.value == TENANT_A
    assert keys["user_id"].match.value == "u1"

    mine = {"tenant_id": TENANT_A, "user_id": "u1", "kind": "fact", "expires_at": None}
    assert evaluate(qfilter, mine) is True
    assert evaluate(qfilter, {**mine, "user_id": "u2"}) is False
    assert evaluate(qfilter, {**mine, "tenant_id": TENANT_B}) is False


def test_memory_filter_restricts_kinds_and_hides_expired():
    principal = make_principal(user_id="u1")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    qfilter = build_memory_filter(
        principal, [MemoryKind.PREFERENCE, MemoryKind.FACT], now=now
    )

    base = {"tenant_id": TENANT_A, "user_id": "u1"}
    assert evaluate(qfilter, {**base, "kind": "fact", "expires_at": None}) is True
    assert evaluate(qfilter, {**base, "kind": "episode", "expires_at": None}) is False

    fresh = (now + timedelta(days=1)).isoformat()
    stale = (now - timedelta(days=1)).isoformat()
    assert evaluate(qfilter, {**base, "kind": "fact", "expires_at": fresh}) is True
    assert evaluate(qfilter, {**base, "kind": "fact", "expires_at": stale}) is False

    permissive = build_memory_filter(principal, include_expired=True, now=now)
    assert evaluate(permissive, {**base, "kind": "fact", "expires_at": stale}) is True


def test_cache_filter_requires_an_exact_fingerprint_and_owns_the_entry():
    principal = make_principal(user_id="u1")
    qfilter = build_cache_filter(principal, "fp-123")

    mine = {"tenant_id": TENANT_A, "user_id": "u1", "filter_fingerprint": "fp-123"}
    assert evaluate(qfilter, mine) is True
    # Tenant-wide entry (user_id unset) is reusable.
    assert evaluate(qfilter, {**mine, "user_id": None}) is True
    # Another user's entry is not.
    assert evaluate(qfilter, {**mine, "user_id": "u2"}) is False
    # A different fingerprint is not.
    assert evaluate(qfilter, {**mine, "filter_fingerprint": "fp-999"}) is False
    # Another tenant's entry is not, even with a matching fingerprint.
    assert evaluate(qfilter, {**mine, "tenant_id": TENANT_B}) is False


def test_tenant_write_filter_scopes_and_excludes():
    qfilter = build_tenant_filter(
        TENANT_A,
        source_id="src-1",
        exclude_document_ids=["doc-live"],
        include_deleted=False,
    )
    live = {
        "tenant_id": TENANT_A,
        "source_id": "src-1",
        "document_id": "doc-live",
        "is_deleted": False,
    }
    gone = {**live, "document_id": "doc-gone"}
    assert evaluate(qfilter, live) is False
    assert evaluate(qfilter, gone) is True
    assert evaluate(qfilter, {**gone, "tenant_id": TENANT_B}) is False
    assert evaluate(qfilter, {**gone, "source_id": "src-2"}) is False
    assert evaluate(qfilter, {**gone, "is_deleted": True}) is False


def test_tenant_write_filter_rejects_a_contradiction():
    with pytest.raises(ValueError, match="contradicts"):
        build_tenant_filter(TENANT_A, include_deleted=False, only_deleted=True)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_and_order_independent():
    principal = make_principal()
    a = MetadataFilter(doc_types=["policy", "faq"], tags=["hr", "eu"])
    b = MetadataFilter(doc_types=["faq", "policy"], tags=["eu", "hr"])
    assert filter_fingerprint(principal, a) == filter_fingerprint(principal, b)
    assert filter_fingerprint(principal, a) == filter_fingerprint(principal, a)
    assert len(filter_fingerprint(principal, None)) == 32


def test_fingerprint_separates_tenants_and_clearances():
    base = MetadataFilter(doc_types=["policy"])
    a = filter_fingerprint(make_principal(tenant_id=TENANT_A), base)
    b = filter_fingerprint(make_principal(tenant_id=TENANT_B), base)
    assert a != b

    low = filter_fingerprint(make_principal(clearance=Classification.PUBLIC), base)
    high = filter_fingerprint(make_principal(clearance=Classification.RESTRICTED), base)
    assert low != high


def test_fingerprint_changes_with_the_filter():
    principal = make_principal()
    assert filter_fingerprint(principal, None) != filter_fingerprint(
        principal, MetadataFilter(doc_types=["policy"])
    )
    assert filter_fingerprint(
        principal, MetadataFilter(exclude_pii=True)
    ) != filter_fingerprint(principal, MetadataFilter(exclude_pii=False))


def test_serialise_filter_is_json_safe_and_carries_no_text():
    qfilter = build_acl_filter(
        make_principal(groups=["g-eng"]), MetadataFilter(doc_types=["policy"])
    )
    data = serialise_filter(qfilter)
    assert set(data) <= {"must", "must_not", "should", "min_should"}
    assert "must" in data
    assert isinstance(data["must"], list)
