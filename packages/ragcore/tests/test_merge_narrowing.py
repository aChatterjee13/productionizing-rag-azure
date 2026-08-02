"""Merging must never widen — the property both merge helpers exist to preserve.

Two independent merges layer defaults onto constraints that are already in force:

* :meth:`AccessControl.merged_with` folds a source's default ACL onto the ACL an
  ingestion connector read from the source system itself. The item's own permissions
  are authoritative, so the merge may only ever *remove* grants.
* :meth:`MetadataFilter.merged_with` folds facets a tool or the query transformer
  extracted onto the filter the caller supplied, so a tool call can only ever narrow
  what the user was already searching.

Both widened before these tests existed, and neither had any coverage:

* ``AccessControl`` unioned the three allow lists, which :meth:`permits` reads as a
  *disjunction* — so a source default of ``allowed_groups=["g-staff"]`` granted the
  whole of ``g-staff`` read on an item SharePoint had restricted to one user.
* ``MetadataFilter`` intersected correctly but returned ``[]``, which the model
  validator collapses to None — "no constraint" — dropping the clause entirely.

So these assert the invariant directly, over a population of principals and over the
generated Qdrant clause, rather than pinning the shape of the output. A future
rewrite that preserves the property passes unchanged.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.retrieval import MetadataFilter
from ragcore.vectorstore.filters import build_acl_filter, serialise_filter

TENANT = "t1"

#: A population broad enough that any widening shows up as a principal the merged
#: ACL admits and the item ACL did not: the named user, a group member, a role
#: holder, a bystander with nothing, and one of each who is also explicitly denied.
POPULATION: tuple[Principal, ...] = (
    Principal(user_id="u1", tenant_id=TENANT),
    Principal(user_id="u2", tenant_id=TENANT),
    Principal(user_id="u3", tenant_id=TENANT, groups=["g-staff"]),
    Principal(user_id="u4", tenant_id=TENANT, groups=["g-other"]),
    Principal(user_id="u5", tenant_id=TENANT, roles=["r-reader"]),
    Principal(user_id="u6", tenant_id=TENANT, roles=["r-other"]),
    Principal(user_id="u7", tenant_id=TENANT, groups=["g-staff"], roles=["r-reader"]),
    Principal(user_id="u1", tenant_id=TENANT, groups=["g-staff"]),
    Principal(
        user_id="u8",
        tenant_id=TENANT,
        groups=["g-staff"],
        max_classification=Classification.RESTRICTED,
    ),
)

#: ACLs spanning every combination of "restricts nothing" and "restricts via each of
#: the three dimensions", so the matrix below covers item/default pairs that restrict
#: on the same dimension and on different ones.
ACLS: tuple[AccessControl, ...] = (
    AccessControl(tenant_id=TENANT),
    AccessControl(tenant_id=TENANT, allowed_users=["u1"]),
    AccessControl(tenant_id=TENANT, allowed_users=["u2"]),
    AccessControl(tenant_id=TENANT, allowed_groups=["g-staff"]),
    AccessControl(tenant_id=TENANT, allowed_roles=["r-reader"]),
    AccessControl(tenant_id=TENANT, allowed_users=["u1"], allowed_groups=["g-staff"]),
    AccessControl(
        tenant_id=TENANT,
        allowed_users=["u2"],
        classification=Classification.CONFIDENTIAL,
    ),
    AccessControl(tenant_id=TENANT, denied_users=["u1"]),
)


def _strings(node: object) -> Iterator[str]:
    """Yield every string anywhere in a decoded JSON structure.

    Args:
        node: A value from :func:`json.loads`.

    Yields:
        Each string leaf, so a test can assert over all of them at once.
    """
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _acl_id(acl: AccessControl) -> str:
    """Render an ACL as a short test id.

    Args:
        acl: The ACL to label.

    Returns:
        A compact description such as ``users=u1+deny=u2`` or ``open``.
    """
    parts = [
        f"{label}={'|'.join(values)}"
        for label, values in (
            ("users", acl.allowed_users),
            ("groups", acl.allowed_groups),
            ("roles", acl.allowed_roles),
            ("deny", acl.denied_users),
        )
        if values
    ]
    if acl.classification is not Classification.INTERNAL:
        parts.append(acl.classification.value)
    return "+".join(parts) or "open"


class TestAccessControlMerge:
    """``AccessControl.merged_with`` never grants what the item ACL withheld."""

    @pytest.mark.parametrize(
        ("item", "default"),
        list(itertools.product(ACLS, ACLS)),
        ids=lambda acl: _acl_id(acl),
    )
    def test_merge_never_grants_more_than_the_item_acl(
        self, item: AccessControl, default: AccessControl
    ) -> None:
        """For every principal, the merged ACL implies the item ACL.

        This is the whole contract. ``item`` is what the source system said; layering
        an operator's defaults on top may narrow it but must never admit a principal
        the source system excluded.

        Args:
            item: ACL resolved from the item itself.
            default: The source's default ACL.
        """
        merged = item.merged_with(default)
        for principal in POPULATION:
            if merged.permits(principal):
                assert item.permits(principal), (
                    f"{principal.user_id} (groups={principal.groups}, "
                    f"roles={principal.roles}) gained access: "
                    f"item={item!r} default={default!r} merged={merged!r}"
                )

    def test_group_default_does_not_unlock_a_user_restricted_item(self) -> None:
        """The SharePoint regression, stated explicitly.

        An item restricted to one user, merged with a source default naming a broad
        group, must not become readable by that group.
        """
        item = AccessControl(tenant_id=TENANT, allowed_users=["u1"])
        default = AccessControl(tenant_id=TENANT, allowed_groups=["g-staff"])
        merged = item.merged_with(default)

        bystander = Principal(user_id="u3", tenant_id=TENANT, groups=["g-staff"])
        assert not merged.permits(bystander)
        assert merged.permits(Principal(user_id="u1", tenant_id=TENANT))

    def test_user_default_does_not_add_a_second_user(self) -> None:
        """Same dimension: a default naming another user must not grant them read."""
        merged = AccessControl(tenant_id=TENANT, allowed_users=["u1"]).merged_with(
            AccessControl(tenant_id=TENANT, allowed_users=["u2"])
        )
        assert not merged.permits(Principal(user_id="u2", tenant_id=TENANT))

    def test_unrestricted_item_inherits_the_defaults(self) -> None:
        """Defaults still apply — and narrow — when the item restricts nothing."""
        merged = AccessControl(tenant_id=TENANT).merged_with(
            AccessControl(tenant_id=TENANT, allowed_groups=["g-staff"])
        )
        assert merged.allowed_groups == ["g-staff"]
        assert merged.permits(
            Principal(user_id="u3", tenant_id=TENANT, groups=["g-staff"])
        )
        assert not merged.permits(Principal(user_id="u9", tenant_id=TENANT))

    def test_both_unrestricted_stays_unrestricted(self) -> None:
        """Two unrestricted ACLs merge to an unrestricted one."""
        merged = AccessControl(tenant_id=TENANT).merged_with(
            AccessControl(tenant_id=TENANT)
        )
        assert merged.is_unrestricted

    def test_deny_lists_union(self) -> None:
        """Deny always wins, so both sides' denials survive the merge."""
        merged = AccessControl(tenant_id=TENANT, denied_users=["a"]).merged_with(
            AccessControl(tenant_id=TENANT, denied_users=["b"])
        )
        assert sorted(merged.denied_users) == ["a", "b"]

    def test_deny_beats_an_inherited_grant(self) -> None:
        """A denied principal stays denied even when the defaults would admit them."""
        merged = AccessControl(tenant_id=TENANT, denied_users=["u3"]).merged_with(
            AccessControl(tenant_id=TENANT, allowed_groups=["g-staff"])
        )
        assert not merged.permits(
            Principal(user_id="u3", tenant_id=TENANT, groups=["g-staff"])
        )

    def test_classification_takes_the_more_sensitive_label(self) -> None:
        """The stricter of the two classifications wins, in both argument orders."""
        low = AccessControl(tenant_id=TENANT, classification=Classification.PUBLIC)
        high = AccessControl(tenant_id=TENANT, classification=Classification.RESTRICTED)
        assert low.merged_with(high).classification is Classification.RESTRICTED
        assert high.merged_with(low).classification is Classification.RESTRICTED

    def test_cross_tenant_merge_is_refused(self) -> None:
        """The tenant boundary is never crossed silently."""
        with pytest.raises(ValueError, match="across tenants"):
            AccessControl(tenant_id=TENANT).merged_with(AccessControl(tenant_id="t2"))


class TestMetadataFilterMerge:
    """``MetadataFilter.merged_with`` only ever narrows.

    An empty intersection must match nothing rather than everything.
    """

    LIST_FACETS = (
        "doc_types",
        "source_types",
        "tags",
        "authors",
        "languages",
        "document_ids",
    )

    @pytest.mark.parametrize("facet", LIST_FACETS)
    def test_disjoint_facets_become_unsatisfiable(self, facet: str) -> None:
        """Sharing no value must not read as "no constraint".

        Args:
            facet: Name of the list facet under test.
        """
        merged = MetadataFilter(**{facet: ["a"]}).merged_with(
            MetadataFilter(**{facet: ["b"]})
        )
        assert merged.unsatisfiable is True
        # The most constraining filter there is, so callers that skip an empty
        # filter must not skip this one.
        assert not merged.is_empty

    @pytest.mark.parametrize("facet", LIST_FACETS)
    def test_overlapping_facets_intersect(self, facet: str) -> None:
        """A shared value survives and the unshared ones do not.

        Args:
            facet: Name of the list facet under test.
        """
        merged = MetadataFilter(**{facet: ["a", "b"]}).merged_with(
            MetadataFilter(**{facet: ["b", "c"]})
        )
        assert getattr(merged, facet) == ["b"]

    @pytest.mark.parametrize("facet", LIST_FACETS)
    def test_one_sided_facet_is_carried_through(self, facet: str) -> None:
        """A facet only one side constrains is kept as-is.

        Args:
            facet: Name of the list facet under test.
        """
        merged = MetadataFilter(**{facet: ["a"]}).merged_with(MetadataFilter())
        assert getattr(merged, facet) == ["a"]

    def test_an_unsatisfiable_filter_matches_nothing_in_qdrant(self) -> None:
        """The regression, asserted where it actually mattered.

        A dropped facet is invisible in the model but decisive in the filter: without
        a clause Qdrant returns every document the ACL allows instead of none. The
        filter must carry a condition no chunk can satisfy.
        """
        merged = MetadataFilter(document_ids=["docA"]).merged_with(
            MetadataFilter(document_ids=["docB"])
        )
        qfilter = build_acl_filter(Principal(user_id="u1", tenant_id=TENANT), merged)
        ranks = [
            condition
            for condition in (qfilter.must or [])
            if getattr(condition, "key", None) == "classification_rank"
            and getattr(condition.range, "lt", None) is not None
        ]
        assert ranks, "unsatisfiable filter carried no impossible clause"
        # classification_rank is written onto every chunk and is never negative.
        assert ranks[0].range.lt <= 0

    def test_the_serialised_filter_carries_no_control_characters(self) -> None:
        """Nothing unprintable may reach the client or a log line.

        ``filter_applied`` is returned in the API response, and a NUL byte would be
        rejected outright by Postgres ``jsonb``/``text`` if the filter were ever
        persisted. Flagging the model keeps the impossibility out of the payload.
        """
        merged = MetadataFilter(document_ids=["docA"]).merged_with(
            MetadataFilter(document_ids=["docB"])
        )
        qfilter = build_acl_filter(Principal(user_id="u1", tenant_id=TENANT), merged)
        encoded = json.dumps(serialise_filter(qfilter))
        offenders = [
            text
            for text in _strings(json.loads(encoded))
            if any(character < " " for character in text)
        ]
        assert not offenders, f"control characters reached the client: {offenders}"

    def test_disjoint_section_prefixes_are_unsatisfiable(self) -> None:
        """Two different headings name disjoint subtrees, so neither may win."""
        merged = MetadataFilter(section_prefix="A").merged_with(
            MetadataFilter(section_prefix="B")
        )
        assert merged.unsatisfiable is True

    def test_crossed_date_bounds_are_unsatisfiable_not_an_error(self) -> None:
        """An empty date range means no results, not a 500.

        The constructor refuses an inverted range, so ``merged_with`` has to notice
        the crossing itself rather than hand the pair to the validator.
        """
        merged = MetadataFilter(date_from=datetime(2024, 6, 1, tzinfo=UTC)).merged_with(
            MetadataFilter(date_to=datetime(2023, 1, 1, tzinfo=UTC))
        )
        assert merged.unsatisfiable is True
        assert (
            merged.date_from is None
            or merged.date_to is None
            or (merged.date_from <= merged.date_to)
        )

    def test_unsatisfiable_is_contagious(self) -> None:
        """Merging an unsatisfiable filter with anything stays unsatisfiable."""
        impossible = MetadataFilter(doc_types=["a"]).merged_with(
            MetadataFilter(doc_types=["b"])
        )
        assert impossible.merged_with(MetadataFilter(tags=["x"])).unsatisfiable
        assert MetadataFilter(tags=["x"]).merged_with(impossible).unsatisfiable

    def test_identical_section_prefixes_survive(self) -> None:
        """Agreement is not a conflict."""
        merged = MetadataFilter(section_prefix="A").merged_with(
            MetadataFilter(section_prefix="A")
        )
        assert merged.section_prefix == "A"

    def test_classification_ceiling_takes_the_minimum(self) -> None:
        """The ceiling may only ever come down."""
        merged = MetadataFilter(
            max_classification=Classification.RESTRICTED
        ).merged_with(MetadataFilter(max_classification=Classification.PUBLIC))
        assert merged.max_classification is Classification.PUBLIC

    def test_exclude_pii_is_sticky(self) -> None:
        """Either side asking to exclude PII is enough."""
        empty = MetadataFilter()
        assert MetadataFilter(exclude_pii=True).merged_with(empty).exclude_pii
        assert empty.merged_with(MetadataFilter(exclude_pii=True)).exclude_pii

    def test_date_bounds_intersect(self) -> None:
        """The merged window is the overlap: latest start, earliest end."""
        merged = MetadataFilter(
            date_from=datetime(2024, 1, 1, tzinfo=UTC),
            date_to=datetime(2024, 12, 31, tzinfo=UTC),
        ).merged_with(
            MetadataFilter(
                date_from=datetime(2024, 6, 1, tzinfo=UTC),
                date_to=datetime(2025, 6, 1, tzinfo=UTC),
            )
        )
        assert merged.date_from == datetime(2024, 6, 1, tzinfo=UTC)
        assert merged.date_to == datetime(2024, 12, 31, tzinfo=UTC)

    def test_merging_none_is_a_copy(self) -> None:
        """Merging with nothing changes nothing."""
        original = MetadataFilter(tags=["x"], exclude_pii=True)
        merged = original.merged_with(None)
        assert merged == original
        assert merged is not original
