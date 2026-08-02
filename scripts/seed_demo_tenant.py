#!/usr/bin/env python3
"""Seed two demo tenants, four personas and a deliberately awkward document set.

This module is the **single source of truth** for the demo fixture. The evaluation
golden set (`services/eval/golden/golden_set.yaml`), the ACL negative tests and
`scripts/smoke_test.py` all bind to the identifiers exported here — import them,
never retype them:

    from seed_demo_tenant import PERSONAS, DEMO_DOCUMENTS, CANARIES

What the corpus is built to prove:

* **Tenant isolation.** ``tenant-acme`` and ``tenant-globex`` both carry a travel
  policy whose wording is nearly identical but whose meal allowance differs
  (EUR 60 vs EUR 30). A cross-tenant leak therefore shows up as a wrong *number*,
  not just a wrong document.
* **Clearance.** Documents span every :class:`~ragcore.models.acl.Classification`, so
  the ``PUBLIC``-clearance intern persona must not see the ``CONFIDENTIAL`` salary
  bands or the ``RESTRICTED`` incident report.
* **Group and deny semantics.** ``doc-acme-vpn-runbook`` is limited to the engineering
  group; ``doc-acme-contractor-nda`` is limited to the same group *and* explicitly
  denies the engineer, which is the only case that proves deny beats allow.
* **Overlap and near-duplication.** The onboarding handbook repeats a paragraph from
  the travel policy verbatim, so dedupe has something real to collapse.
* **Contradiction.** ``doc-acme-travel-2023`` and ``doc-acme-travel-2025`` disagree
  about the meal allowance; the newer ``effective_from`` must win while both are cited.
* **PII.** ``doc-acme-hr-contact`` contains a name, an email address and a phone
  number so the PII guardrail has a real target.

Every restricted document carries a unique canary token (``CANARIES``). A test asserts
a leak by searching for the canary: if it appears in an answer for a principal who may
not read that document, the ACL boundary has failed.

Usage:

    uv run python scripts/seed_demo_tenant.py
    uv run python scripts/seed_demo_tenant.py --purge      # replace, don't merge
    uv run python scripts/seed_demo_tenant.py --dry-run    # show the plan only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ragcore.db import repositories as repo
from ragcore.db import session_scope
from ragcore.db.models import SourceConfigRow
from ragcore.dedupe import content_sha256, simhash_hex
from ragcore.embeddings import get_embedding_provider
from ragcore.logging import configure_logging, get_logger
from ragcore.models.acl import AccessControl, Classification, Principal
from ragcore.models.chunk import ChunkPayload
from ragcore.models.document import SourceType
from ragcore.settings import Settings, get_settings
from ragcore.vectorstore.collections import ensure_collections, get_client
from ragcore.vectorstore.filters import build_tenant_filter
from ragcore.vectorstore.writer import (
    ChunkPoint,
    count_chunks,
    hard_delete_by_filter,
    upsert_chunks,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger("scripts.seed_demo_tenant")

EXIT_OK = 0
EXIT_FAILED = 1

# --------------------------------------------------------------------------- tenants
#: Tenant ids double as the Entra ``tid`` claim for the dev-mode personas, so a payload
#: written here matches a principal resolved from the dev header without translation.
TENANT_ACME = "tenant-acme"
TENANT_GLOBEX = "tenant-globex"

#: Entra group object ids used by the demo ACLs.
GROUP_ACME_ENGINEERING = "g-acme-engineering"
GROUP_ACME_HR = "g-acme-hr"
GROUP_ACME_INTERNS = "g-acme-interns"
GROUP_GLOBEX_OPS = "g-globex-operations"

#: App roles. ``rag.admin`` must stay in step with ``ragcore.models.acl.ADMIN_ROLE``.
ROLE_ADMIN = "rag.admin"
ROLE_USER = "rag.user"

#: The ingest run id recorded on every seeded chunk, so lineage can tell seeded data
#: from data a real ingestion run produced.
SEED_RUN_ID = "seed-demo-tenant"


@dataclass(frozen=True, slots=True)
class DemoTenant:
    """A tenant in the demo fixture.

    Attributes:
        tenant_id: Tenant id, also used as the Entra ``tid`` for dev-mode personas.
        name: Human-readable tenant name.
        entra_tenant_id: The real directory GUID this tenant would map to.
        default_classification: Default classification for new sources.
    """

    tenant_id: str
    name: str
    entra_tenant_id: str
    default_classification: Classification = Classification.INTERNAL


@dataclass(frozen=True, slots=True)
class DemoSection:
    """One heading plus its body; becomes exactly one chunk.

    Attributes:
        heading: Section heading, used for ``section_path`` and the contextual header.
        text: Verbatim body text. Citation spans must match this exactly.
    """

    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class DemoDocument:
    """A document in the demo fixture, with its ACL and its sections.

    Attributes:
        document_id: Stable document id referenced by the golden set.
        tenant_id: Owning tenant.
        title: Document title shown in citations.
        doc_type: Document class; also the contradiction-authority key.
        source_id: Source config that "ingested" it.
        source_type: Connector that would have produced it.
        source_uri: Canonical URI.
        classification: Sensitivity label.
        allowed_roles: Permitted app roles; empty means unrestricted.
        allowed_groups: Permitted Entra group ids; empty means unrestricted.
        allowed_users: Permitted user ids; empty means unrestricted.
        denied_users: Explicitly denied user ids; deny always wins.
        sections: Ordered sections, one chunk each.
        tags: Free-form tags for metadata filtering.
        author: Document author.
        language: ISO 639-1 language code.
        source_modified_at: Last modification time at source.
        effective_from: Start of the validity window.
        effective_to: End of the validity window.
        canary: Unique token embedded in the text, used by leak assertions.
    """

    document_id: str
    tenant_id: str
    title: str
    doc_type: str
    source_id: str
    source_type: SourceType
    source_uri: str
    classification: Classification
    sections: tuple[DemoSection, ...]
    allowed_roles: tuple[str, ...] = ()
    allowed_groups: tuple[str, ...] = ()
    allowed_users: tuple[str, ...] = ()
    denied_users: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    author: str | None = None
    language: str = "en"
    source_modified_at: datetime | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None

    @property
    def canary(self) -> str | None:
        """Look up the canary token embedded in this document's text.

        Returns:
            The token from :data:`CANARIES`, or None when this document carries none.
        """
        return CANARIES.get(self.document_id)

    def access_control(self) -> AccessControl:
        """Build the nested ACL for this document.

        Returns:
            The :class:`~ragcore.models.acl.AccessControl` every chunk inherits.
        """
        return AccessControl(
            tenant_id=self.tenant_id,
            allowed_roles=list(self.allowed_roles),
            allowed_groups=list(self.allowed_groups),
            allowed_users=list(self.allowed_users),
            denied_users=list(self.denied_users),
            classification=self.classification,
        )

    @property
    def full_text(self) -> str:
        """Concatenate every section, headings included.

        Returns:
            The whole document as plain text.
        """
        return "\n\n".join(f"{s.heading}\n{s.text}" for s in self.sections)


@dataclass(slots=True)
class SeedReport:
    """What a seed run changed.

    Attributes:
        tenants: Tenant ids written.
        users: Principal ids written.
        sources: Source config ids written.
        documents: Document ids written.
        chunks: Number of chunk points upserted.
        purged: Number of stale points removed by ``--purge``.
        dry_run: Whether anything was actually written.
    """

    tenants: list[str] = field(default_factory=list)
    users: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    chunks: int = 0
    purged: int = 0
    dry_run: bool = False


def _utc(year: int, month: int, day: int) -> datetime:
    """Build a timezone-aware midnight UTC datetime.

    Args:
        year: Calendar year.
        month: Calendar month.
        day: Calendar day.

    Returns:
        The moment at 00:00 UTC.
    """
    return datetime(year, month, day, tzinfo=UTC)


DEMO_TENANTS: tuple[DemoTenant, ...] = (
    DemoTenant(
        tenant_id=TENANT_ACME,
        name="Acme Industrial GmbH",
        entra_tenant_id="11111111-1111-4111-8111-111111111111",
        default_classification=Classification.INTERNAL,
    ),
    DemoTenant(
        tenant_id=TENANT_GLOBEX,
        name="Globex Logistics Ltd",
        entra_tenant_id="22222222-2222-4222-8222-222222222222",
        default_classification=Classification.INTERNAL,
    ),
)

#: Persona key -> principal. The keys are what ``GoldenItem.as_user`` carries.
PERSONAS: dict[str, Principal] = {
    "acme_admin": Principal(
        user_id="11111111-1111-4111-8111-000000000001",
        tenant_id=TENANT_ACME,
        roles=[ROLE_ADMIN, ROLE_USER],
        groups=[GROUP_ACME_ENGINEERING, GROUP_ACME_HR],
        email="dana.admin@acme.example",
        display_name="Dana Admin",
        max_classification=Classification.RESTRICTED,
    ),
    "acme_engineer": Principal(
        user_id="11111111-1111-4111-8111-000000000002",
        tenant_id=TENANT_ACME,
        roles=[ROLE_USER],
        groups=[GROUP_ACME_ENGINEERING],
        email="erik.engineer@acme.example",
        display_name="Erik Engineer",
        max_classification=Classification.CONFIDENTIAL,
    ),
    "acme_intern": Principal(
        user_id="11111111-1111-4111-8111-000000000003",
        tenant_id=TENANT_ACME,
        roles=[ROLE_USER],
        groups=[GROUP_ACME_INTERNS],
        email="ines.intern@acme.example",
        display_name="Ines Intern",
        max_classification=Classification.PUBLIC,
    ),
    "globex_analyst": Principal(
        user_id="22222222-2222-4222-8222-000000000001",
        tenant_id=TENANT_GLOBEX,
        roles=[ROLE_USER],
        groups=[GROUP_GLOBEX_OPS],
        email="gita.analyst@globex.example",
        display_name="Gita Analyst",
        max_classification=Classification.CONFIDENTIAL,
    ),
}

#: Canary tokens embedded in restricted or tenant-specific text. Assert a leak by
#: searching for the token: it must never reach a principal who may not read the
#: owning document.
CANARIES: dict[str, str] = {
    "doc-acme-salary-bands": "CANARY-ACME-SALARY-7F3A",
    "doc-acme-security-incident": "CANARY-ACME-INCIDENT-9B21",
    "doc-acme-contractor-nda": "CANARY-ACME-NDA-4C88",
    "doc-acme-vpn-runbook": "CANARY-ACME-VPN-5A17",
    "doc-globex-travel-policy": "CANARY-GLOBEX-TRAVEL-1E44",
    "doc-globex-warehouse-safety": "CANARY-GLOBEX-SAFETY-2D55",
}

#: The contradictory pair: same claim, different value, different effective window.
CONTRADICTION_PAIR: tuple[str, str] = (
    "doc-acme-travel-2023",
    "doc-acme-travel-2025",
)

#: Paragraph duplicated verbatim between the travel policy and the onboarding handbook,
#: so simhash/exact-hash dedupe has a real duplicate to collapse.
_SHARED_EXPENSE_PARAGRAPH = (
    "Expense claims are submitted in the Acme Expense portal within 30 days of the "
    "trip. Each claim needs an itemised receipt; card statements are not accepted as "
    "evidence. Claims older than 90 days are rejected without exception."
)

DEMO_DOCUMENTS: tuple[DemoDocument, ...] = (
    # ------------------------------------------------------------------ acme, public
    DemoDocument(
        document_id="doc-acme-travel-2023",
        tenant_id=TENANT_ACME,
        title="Acme Travel and Expense Policy (2023 edition)",
        doc_type="policy",
        source_id="src-acme-policies",
        source_type=SourceType.BLOB,
        source_uri="https://acme.example/policies/travel-2023.md",
        classification=Classification.PUBLIC,
        tags=("travel", "expenses", "superseded"),
        author="Acme People Operations",
        source_modified_at=_utc(2023, 1, 9),
        effective_from=_utc(2023, 1, 1),
        effective_to=_utc(2025, 3, 31),
        sections=(
            DemoSection(
                heading="Meal allowance",
                text=(
                    "Employees travelling on Acme business claim a flat daily meal "
                    "allowance of EUR 45 per full day away from their usual place of "
                    "work. Partial days are reimbursed at half the daily rate. No "
                    "receipts are required for the meal allowance itself."
                ),
            ),
            DemoSection(
                heading="Flights and rail",
                text=(
                    "Journeys shorter than six hours are booked in economy class. "
                    "Rail travel is preferred where the total door-to-door time is "
                    "within one hour of flying."
                ),
            ),
            DemoSection(heading="Expense claims", text=_SHARED_EXPENSE_PARAGRAPH),
        ),
    ),
    DemoDocument(
        document_id="doc-acme-travel-2025",
        tenant_id=TENANT_ACME,
        title="Acme Travel and Expense Policy (2025 edition)",
        doc_type="policy",
        source_id="src-acme-policies",
        source_type=SourceType.BLOB,
        source_uri="https://acme.example/policies/travel-2025.md",
        classification=Classification.PUBLIC,
        tags=("travel", "expenses", "current"),
        author="Acme People Operations",
        source_modified_at=_utc(2025, 3, 18),
        effective_from=_utc(2025, 4, 1),
        sections=(
            DemoSection(
                heading="Meal allowance",
                text=(
                    "Employees travelling on Acme business claim a flat daily meal "
                    "allowance of EUR 60 per full day away from their usual place of "
                    "work. This supersedes the EUR 45 rate that applied until 31 "
                    "March 2025. Partial days are reimbursed at half the daily rate."
                ),
            ),
            DemoSection(
                heading="Flights and rail",
                text=(
                    "Journeys shorter than six hours are booked in economy class. "
                    "Rail travel is preferred where the total door-to-door time is "
                    "within one hour of flying."
                ),
            ),
            DemoSection(heading="Expense claims", text=_SHARED_EXPENSE_PARAGRAPH),
        ),
    ),
    DemoDocument(
        document_id="doc-acme-onboarding",
        tenant_id=TENANT_ACME,
        title="Acme New Joiner Handbook",
        doc_type="handbook",
        source_id="src-acme-intranet",
        source_type=SourceType.SHAREPOINT,
        source_uri="https://acme.sharepoint.example/sites/people/onboarding.docx",
        classification=Classification.PUBLIC,
        tags=("onboarding", "handbook"),
        author="Acme People Operations",
        source_modified_at=_utc(2025, 5, 6),
        sections=(
            DemoSection(
                heading="Your first week",
                text=(
                    "Your manager schedules a 30-minute check-in on day one, day "
                    "three and at the end of week one. Mandatory security awareness "
                    "training must be completed before the end of week two."
                ),
            ),
            # Deliberately identical to the travel policy paragraph: dedupe must
            # collapse these two chunks and report the drop rather than hide it.
            DemoSection(heading="Claiming expenses", text=_SHARED_EXPENSE_PARAGRAPH),
            DemoSection(
                heading="Equipment",
                text=(
                    "New joiners receive a laptop, a headset and a monitor. The "
                    "one-off home-office stipend is EUR 400 and is claimed through "
                    "the same expense portal."
                ),
            ),
        ),
    ),
    DemoDocument(
        document_id="doc-acme-hr-contact",
        tenant_id=TENANT_ACME,
        title="Acme HR Duty Roster",
        doc_type="note",
        source_id="src-acme-intranet",
        source_type=SourceType.SHAREPOINT,
        source_uri="https://acme.sharepoint.example/sites/people/hr-duty.docx",
        classification=Classification.INTERNAL,
        tags=("hr", "contact", "pii"),
        author="Acme People Operations",
        source_modified_at=_utc(2025, 6, 2),
        sections=(
            DemoSection(
                heading="Who to contact",
                text=(
                    "The HR duty officer this quarter is Priya Raman. Reach her at "
                    "priya.raman@acme.example or on +49 151 2345 6789. For payroll "
                    "questions, write to payroll@acme.example instead."
                ),
            ),
        ),
    ),
    # ---------------------------------------------------------------- acme, internal
    DemoDocument(
        document_id="doc-acme-remote-work",
        tenant_id=TENANT_ACME,
        title="Acme Remote Work Standard",
        doc_type="standard",
        source_id="src-acme-policies",
        source_type=SourceType.BLOB,
        source_uri="https://acme.example/policies/remote-work.md",
        classification=Classification.INTERNAL,
        tags=("remote", "workplace"),
        author="Acme Workplace Services",
        source_modified_at=_utc(2025, 2, 14),
        effective_from=_utc(2025, 3, 1),
        sections=(
            DemoSection(
                heading="Eligibility",
                text=(
                    "Employees may work remotely up to three days a week with their "
                    "manager's agreement. Roles that require physical presence on the "
                    "production floor are excluded."
                ),
            ),
            DemoSection(
                heading="Home office equipment",
                text=(
                    "The one-off home-office stipend is EUR 400 and covers a desk, a "
                    "chair or a lamp. It is claimed once per employee and overlaps "
                    "with the new joiner equipment allocation."
                ),
            ),
        ),
    ),
    DemoDocument(
        document_id="doc-acme-vpn-runbook",
        tenant_id=TENANT_ACME,
        title="Acme VPN Recovery Runbook",
        doc_type="runbook",
        source_id="src-acme-engineering",
        source_type=SourceType.HTTP,
        source_uri="https://wiki.acme.example/runbooks/vpn-recovery",
        classification=Classification.INTERNAL,
        allowed_groups=(GROUP_ACME_ENGINEERING,),
        tags=("network", "runbook", "engineering"),
        author="Acme Network Engineering",
        source_modified_at=_utc(2025, 4, 22),
        sections=(
            DemoSection(
                heading="Symptom: tunnel flaps every 30 seconds",
                text=(
                    "Rekey mismatch between the two concentrators. Reference "
                    "CANARY-ACME-VPN-5A17. Drain node vpn-b, force a rekey with a "
                    "600-second lifetime on both peers, then return the node to the "
                    "pool once three consecutive keepalives succeed."
                ),
            ),
        ),
    ),
    # ------------------------------------------------------------ acme, confidential
    DemoDocument(
        document_id="doc-acme-salary-bands",
        tenant_id=TENANT_ACME,
        title="Acme Salary Bands FY25 (HR only)",
        doc_type="standard",
        source_id="src-acme-hr",
        source_type=SourceType.SHAREPOINT,
        source_uri="https://acme.sharepoint.example/sites/hr/salary-bands-fy25.xlsx",
        classification=Classification.CONFIDENTIAL,
        allowed_groups=(GROUP_ACME_HR,),
        tags=("compensation", "hr", "restricted"),
        author="Acme Reward",
        source_modified_at=_utc(2025, 1, 31),
        sections=(
            DemoSection(
                heading="Engineering bands",
                text=(
                    "Reference CANARY-ACME-SALARY-7F3A. Band E3 runs from EUR 62,000 "
                    "to EUR 78,000 base. Band E4 runs from EUR 79,000 to EUR 96,000 "
                    "base. Bonus opportunity is 10 per cent of base at both bands."
                ),
            ),
        ),
    ),
    DemoDocument(
        document_id="doc-acme-contractor-nda",
        tenant_id=TENANT_ACME,
        title="Acme Contractor NDA Addendum",
        doc_type="contract",
        source_id="src-acme-legal",
        source_type=SourceType.BLOB,
        source_uri="https://acme.example/legal/contractor-nda-addendum.pdf",
        classification=Classification.CONFIDENTIAL,
        # Allowed for engineering *except* one named engineer: the only shape that
        # proves an explicit deny overrides an otherwise-matching group.
        allowed_groups=(GROUP_ACME_ENGINEERING,),
        denied_users=(PERSONAS["acme_engineer"].user_id,),
        tags=("legal", "contract", "restricted"),
        author="Acme Legal",
        source_modified_at=_utc(2024, 11, 12),
        sections=(
            DemoSection(
                heading="Clause 7: residual knowledge",
                text=(
                    "Reference CANARY-ACME-NDA-4C88. Contractors may retain general "
                    "skills but must not reuse Acme process parameters, including the "
                    "sintering temperature profile, for 24 months after the "
                    "engagement ends."
                ),
            ),
        ),
    ),
    # -------------------------------------------------------------- acme, restricted
    DemoDocument(
        document_id="doc-acme-security-incident",
        tenant_id=TENANT_ACME,
        title="Acme Security Incident 2025-0142 (admins only)",
        doc_type="runbook",
        source_id="src-acme-security",
        source_type=SourceType.SHAREPOINT,
        source_uri="https://acme.sharepoint.example/sites/security/inc-2025-0142.docx",
        classification=Classification.RESTRICTED,
        allowed_roles=(ROLE_ADMIN,),
        tags=("security", "incident", "restricted"),
        author="Acme Security Operations",
        source_modified_at=_utc(2025, 6, 27),
        sections=(
            DemoSection(
                heading="Timeline and containment",
                text=(
                    "Reference CANARY-ACME-INCIDENT-9B21. A build agent token leaked "
                    "through a public pipeline log at 02:14 UTC. The token was "
                    "revoked at 02:41 UTC and the agent pool was rebuilt from a "
                    "clean image. No customer data was accessed."
                ),
            ),
        ),
    ),
    # ------------------------------------------------------------------------ globex
    DemoDocument(
        document_id="doc-globex-travel-policy",
        tenant_id=TENANT_GLOBEX,
        title="Globex Travel and Expense Policy",
        doc_type="policy",
        source_id="src-globex-policies",
        source_type=SourceType.BLOB,
        source_uri="https://globex.example/policies/travel.md",
        classification=Classification.PUBLIC,
        tags=("travel", "expenses"),
        author="Globex Finance",
        source_modified_at=_utc(2025, 2, 3),
        effective_from=_utc(2025, 2, 1),
        sections=(
            DemoSection(
                heading="Meal allowance",
                # Deliberately near-identical wording to Acme's policy with a
                # different number: a cross-tenant leak shows up as a wrong amount.
                text=(
                    "Employees travelling on Globex business claim a flat daily meal "
                    "allowance of EUR 30 per full day away from their usual place of "
                    "work. Reference CANARY-GLOBEX-TRAVEL-1E44. Partial days are "
                    "reimbursed at half the daily rate."
                ),
            ),
            DemoSection(
                heading="Approvals",
                text=(
                    "Any single trip above EUR 1,500 needs written approval from a "
                    "director before booking."
                ),
            ),
        ),
    ),
    DemoDocument(
        document_id="doc-globex-warehouse-safety",
        tenant_id=TENANT_GLOBEX,
        title="Globex Warehouse Safety Standard",
        doc_type="standard",
        source_id="src-globex-policies",
        source_type=SourceType.BLOB,
        source_uri="https://globex.example/policies/warehouse-safety.md",
        classification=Classification.INTERNAL,
        tags=("safety", "warehouse"),
        author="Globex Health and Safety",
        source_modified_at=_utc(2025, 5, 19),
        sections=(
            DemoSection(
                heading="Forklift operation",
                text=(
                    "Reference CANARY-GLOBEX-SAFETY-2D55. Forklift operators "
                    "revalidate their licence every 24 months. The aisle speed limit "
                    "is 8 km/h and reversing without a spotter is prohibited in aisle "
                    "blocks C and D."
                ),
            ),
        ),
    ),
)

#: Source configs the ingestion service can iterate over after seeding.
_SOURCE_DEFINITIONS: tuple[tuple[str, str, str, SourceType, dict[str, str]], ...] = (
    (TENANT_ACME, "src-acme-policies", "Acme policy blobs", SourceType.BLOB, {}),
    (TENANT_ACME, "src-acme-intranet", "Acme intranet", SourceType.SHAREPOINT, {}),
    (TENANT_ACME, "src-acme-engineering", "Acme wiki", SourceType.HTTP, {}),
    (TENANT_ACME, "src-acme-hr", "Acme HR library", SourceType.SHAREPOINT, {}),
    (TENANT_ACME, "src-acme-legal", "Acme legal blobs", SourceType.BLOB, {}),
    (
        TENANT_ACME,
        "src-acme-security",
        "Acme security library",
        SourceType.SHAREPOINT,
        {},
    ),
    (TENANT_GLOBEX, "src-globex-policies", "Globex policy blobs", SourceType.BLOB, {}),
)


def documents_for_tenant(tenant_id: str) -> tuple[DemoDocument, ...]:
    """Select the demo documents belonging to one tenant.

    Args:
        tenant_id: Tenant to filter by.

    Returns:
        The matching documents, in declaration order.
    """
    return tuple(doc for doc in DEMO_DOCUMENTS if doc.tenant_id == tenant_id)


def documents_visible_to(principal: Principal) -> tuple[DemoDocument, ...]:
    """Compute which demo documents a principal may read.

    Uses :meth:`ragcore.models.acl.AccessControl.permits`, the in-process mirror of
    ``build_acl_filter``, so tests can state an expectation without querying Qdrant.

    Args:
        principal: The caller.

    Returns:
        Every demo document the principal is allowed to see.
    """
    return tuple(
        doc for doc in DEMO_DOCUMENTS if doc.access_control().permits(principal)
    )


def forbidden_canaries_for(principal: Principal) -> tuple[str, ...]:
    """List canary tokens that must never reach a principal.

    Args:
        principal: The caller.

    Returns:
        Canary tokens belonging to documents this principal may not read.
    """
    visible = {doc.document_id for doc in documents_visible_to(principal)}
    return tuple(
        token for document_id, token in CANARIES.items() if document_id not in visible
    )


def chunk_id_for(document: DemoDocument, index: int) -> str:
    """Derive the deterministic chunk id for one section.

    The id is an opaque, readable string so a golden item can name it. Qdrant only
    accepts UUID or integer point ids, but that translation belongs to
    ``ragcore.vectorstore.collections.point_id_for_chunk`` (applied by
    :class:`~ragcore.vectorstore.writer.ChunkPoint`), not here.

    Args:
        document: Owning document.
        index: Zero-based section index.

    Returns:
        A stable id, so a re-seed upserts in place instead of duplicating.
    """
    return f"{document.document_id}::{index:04d}"


def build_chunks(document: DemoDocument, settings: Settings) -> list[ChunkPayload]:
    """Turn one demo document into its chunk payloads.

    One section becomes one chunk. The contextual header carries the title and the
    section breadcrumb and is stored separately from ``text`` so a citation span stays
    verbatim.

    Args:
        document: The document to chunk.
        settings: Resolved settings; supplies the dedupe shingle width and whether a
            contextual header is generated at all.

    Returns:
        Chunk payloads in section order.
    """
    access = document.access_control()
    chunks: list[ChunkPayload] = []
    for index, section in enumerate(document.sections):
        header = (
            f"{document.title} > {section.heading}"
            if settings.chunk_contextual_header_enabled
            else ""
        )
        payload = ChunkPayload.from_access_control(
            access,
            chunk_id=chunk_id_for(document, index),
            document_id=document.document_id,
            chunk_index=index,
            source_type=document.source_type.value,
            source_id=document.source_id,
            source_uri=document.source_uri,
            title=document.title,
            section_path=[section.heading],
            page=None,
            text=section.text,
            contextual_header=header,
            keywords=list(document.tags),
            doc_type=document.doc_type,
            tags=list(document.tags),
            author=document.author,
            language=document.language,
            content_sha256=content_sha256(section.text),
            simhash=simhash_hex(section.text),
            # Approximated on purpose: seeding must not need an Anthropic API key just
            # to count tokens. The ingestion pipeline records exact counts.
            token_count=max(1, len(section.text) // 4),
            source_modified_at=document.source_modified_at,
            effective_from=document.effective_from,
            effective_to=document.effective_to,
            version=1,
            is_deleted=False,
            ingest_run_id=SEED_RUN_ID,
        )
        chunks.append(payload)
    return chunks


def annotate_pii(chunks: Sequence[ChunkPayload], settings: Settings) -> None:
    """Record which PII entity types each seeded chunk contains.

    The seeded corpus is authored fixture text, not user content, so it is stored
    verbatim — but ``pii_types`` is populated honestly so the guardrail and the eval
    ``pii`` category have something truthful to assert against. No chunk text is logged.

    Args:
        chunks: Chunks to annotate in place.
        settings: Resolved settings; skipped entirely when ``pii_enabled`` is false.
    """
    if not settings.pii_enabled:
        return
    # Imported lazily so a seed run with PII detection switched off never pays the
    # Presidio import cost (and never needs the optional `pii` extra installed).
    from ragcore.pii import get_pii_detector

    detector = get_pii_detector(settings)
    for chunk in chunks:
        report = detector.analyze(chunk.text, language=chunk.language)
        if report.has_pii:
            chunk.pii_types = list(report.entity_types)


async def upsert_relational(session: AsyncSession, report: SeedReport) -> None:
    """Write tenants, users, source configs and documents to PostgreSQL.

    Args:
        session: Active database session. The caller commits.
        report: Mutated in place with what was written.
    """
    for tenant in DEMO_TENANTS:
        await repo.upsert_tenant(
            session,
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            entra_tenant_id=tenant.entra_tenant_id,
            default_classification=tenant.default_classification,
            settings_blob={"seeded_by": SEED_RUN_ID},
        )
        report.tenants.append(tenant.tenant_id)

    for key, principal in PERSONAS.items():
        await repo.upsert_user(session, principal)
        report.users.append(f"{key}={principal.user_id}")

    for tenant_id, source_id, name, source_type, options in _SOURCE_DEFINITIONS:
        row = await session.get(SourceConfigRow, source_id)
        if row is None:
            row = SourceConfigRow(source_id=source_id, tenant_id=tenant_id)
            session.add(row)
        elif row.tenant_id != tenant_id:
            msg = (
                f"source config {source_id!r} already exists under tenant "
                f"{row.tenant_id!r}, refusing to move it to {tenant_id!r}"
            )
            raise ValueError(msg)
        row.source_type = source_type.value
        row.name = name
        row.enabled = True
        row.options = dict(options)
        row.doc_type = "document"
        row.language = "en"
        await session.flush()
        report.sources.append(source_id)

    for document in DEMO_DOCUMENTS:
        chunk_count = len(document.sections)
        await repo.upsert_document(
            session,
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            source_uri=document.source_uri,
            source_type=document.source_type.value,
            access_control=document.access_control(),
            source_id=document.source_id,
            title=document.title,
            doc_type=document.doc_type,
            language=document.language,
            author=document.author,
            content_sha256=content_sha256(document.full_text),
            simhash=simhash_hex(document.full_text),
            tags=list(document.tags),
            chunk_count=chunk_count,
            token_count=max(1, len(document.full_text) // 4),
            size_bytes=len(document.full_text.encode("utf-8")),
            source_modified_at=document.source_modified_at,
            effective_from=document.effective_from,
            effective_to=document.effective_to,
            ingest_run_id=SEED_RUN_ID,
        )
        report.documents.append(document.document_id)


async def purge_seeded_points(
    client: AsyncQdrantClient,
    settings: Settings,
    report: SeedReport,
) -> None:
    """Delete previously seeded chunk points so a re-seed replaces rather than merges.

    Administrative write path, not a retrieval path: the selector comes from
    :func:`ragcore.vectorstore.filters.build_tenant_filter`, which is tenant-scoped and
    additionally narrowed to the demo document ids. Reads must always go through
    ``build_acl_filter`` instead.

    Args:
        client: Connected Qdrant client.
        settings: Resolved settings; supplies the collection name.
        report: Mutated in place with the number of points removed.
    """
    removed = 0
    for tenant in DEMO_TENANTS:
        document_ids = [
            doc.document_id for doc in documents_for_tenant(tenant.tenant_id)
        ]
        if not document_ids:
            continue
        qfilter = build_tenant_filter(tenant.tenant_id, document_ids=document_ids)
        removed += await count_chunks(
            client,
            collection=settings.qdrant_chunks_collection,
            qfilter=qfilter,
        )
        await hard_delete_by_filter(
            client,
            collection=settings.qdrant_chunks_collection,
            qfilter=qfilter,
        )
    report.purged = removed


async def upsert_vectors(
    client: AsyncQdrantClient,
    settings: Settings,
    chunks: Sequence[ChunkPayload],
    report: SeedReport,
) -> None:
    """Embed and upsert every seeded chunk into the chunks collection.

    Points are grouped by tenant before writing: ``upsert_chunks`` raises
    ``TenantMismatchError`` on a batch that spans tenants, because a batch is the retry
    unit and a partially-applied retry is how cross-tenant rows appear.

    Args:
        client: Connected Qdrant client.
        settings: Resolved settings; supplies the collection name and batch size.
        chunks: Chunks to index.
        report: Mutated in place with the number of points upserted.

    Raises:
        RuntimeError: If the embedder returns a different number of vectors than chunks.
    """
    provider = get_embedding_provider(settings)
    embedded = await provider.embed_documents([chunk.embed_text for chunk in chunks])
    if len(embedded) != len(chunks):
        msg = f"embedder returned {len(embedded)} vectors for {len(chunks)} chunks"
        raise RuntimeError(msg)

    by_tenant: dict[str, list[ChunkPoint]] = {}
    for chunk, vector in zip(chunks, embedded, strict=True):
        point = ChunkPoint(
            payload=chunk,
            dense=vector.dense,
            sparse=vector.sparse,
        )
        by_tenant.setdefault(chunk.tenant_id, []).append(point)

    total = 0
    for tenant_id, points in sorted(by_tenant.items()):
        written = await upsert_chunks(
            client,
            collection=settings.qdrant_chunks_collection,
            chunks=points,
            settings=settings,
        )
        logger.info("seeded_chunks", tenant_id=tenant_id, chunks=written)
        total += written
    report.chunks = total


async def seed(
    settings: Settings,
    *,
    purge: bool = False,
    dry_run: bool = False,
) -> SeedReport:
    """Seed the demo tenants, personas and corpus.

    Args:
        settings: Resolved platform settings.
        purge: Delete previously seeded chunk points before writing.
        dry_run: Build everything and report, but write nothing.

    Returns:
        A :class:`SeedReport` describing what was written.
    """
    report = SeedReport(dry_run=dry_run)

    all_chunks: list[ChunkPayload] = []
    for document in DEMO_DOCUMENTS:
        all_chunks.extend(build_chunks(document, settings))
    annotate_pii(all_chunks, settings)

    if dry_run:
        report.tenants = [tenant.tenant_id for tenant in DEMO_TENANTS]
        report.users = [
            f"{key}={principal.user_id}" for key, principal in PERSONAS.items()
        ]
        report.sources = [definition[1] for definition in _SOURCE_DEFINITIONS]
        report.documents = [doc.document_id for doc in DEMO_DOCUMENTS]
        report.chunks = len(all_chunks)
        return report

    async with session_scope(settings) as session:
        await upsert_relational(session, report)
        await session.commit()

    client = await get_client(settings)
    try:
        await ensure_collections(client, settings)
        if purge:
            await purge_seeded_points(client, settings, report)
        await upsert_vectors(client, settings, all_chunks, report)
    finally:
        await client.close()

    return report


def print_report(report: SeedReport) -> None:
    """Print a human-readable summary of the seed run.

    Args:
        report: The report to print.
    """
    prefix = "would seed" if report.dry_run else "seeded"
    print(f"{prefix}:")
    print(f"  tenants   {len(report.tenants)}: {', '.join(report.tenants)}")
    print(f"  users     {len(report.users)}: {', '.join(report.users)}")
    print(f"  sources   {len(report.sources)}")
    print(f"  documents {len(report.documents)}")
    print(f"  chunks    {report.chunks}")
    if report.purged:
        print(f"  purged    {report.purged} stale points")
    print()
    print("visibility matrix (documents each persona may read):")
    for key, principal in PERSONAS.items():
        visible = [doc.document_id for doc in documents_visible_to(principal)]
        print(
            f"  {key:<15} clearance={principal.max_classification.value:<12}"
            f" {len(visible)} docs"
        )
        for document_id in visible:
            print(f"      {document_id}")
    print()
    print("canaries that must never leak:")
    for key, principal in PERSONAS.items():
        forbidden = forbidden_canaries_for(principal)
        print(f"  {key:<15} {', '.join(forbidden) if forbidden else '(none)'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="seed_demo_tenant.py",
        description=(
            "Seed two demo tenants, four personas with different roles, groups and "
            "clearances, and a corpus with overlapping, restricted and contradictory "
            "content. The eval golden set and the ACL negative tests depend on it."
        ),
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete previously seeded chunk points first (replace instead of merge)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the fixture and print the plan without writing anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    try:
        report = asyncio.run(seed(settings, purge=args.purge, dry_run=args.dry_run))
    except Exception as exc:
        logger.exception("seed_failed")
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print_report(report)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
