"""The ingestion pipeline. All of it. ``function_app.py`` is only an adapter.

Every entry point — the nightly timer, the admin HTTP trigger, the blob trigger, the
retry queue, the Durable orchestrator, and a plain CLI call — lands in this module, so
what runs in Azure is exactly what runs on a laptop.

The run is deliberately split into three phases, because that split is what makes the
serverless fan-out both cheap and resumable:

1. :func:`plan_source` — enumerate descriptors, resolve ACLs, and classify on the
   *cheap* signals (ETag, modification time, ACL fingerprint). Documents that are
   provably unchanged never get downloaded. What survives becomes a
   :class:`DocumentTask`.
2. :func:`process_document` — one document, start to finish: fetch, hash, re-classify
   on the content hash, parse, enrich, chunk, embed, dedupe, upsert, record lineage.
   Stateless and idempotent, so a Durable activity can run it on any worker and a
   retry cannot duplicate anything.
3. :func:`finalize_run` — fold the outcomes into the manifest, tombstone what
   disappeared, advance the source cursor, and write the
   :class:`~ragcore.models.document.IngestRunSummary`.

Two-tier delta, stated plainly: phase 1 skips on metadata, phase 2 skips on content.
A file that was touched but not edited passes phase 1's ETag check as *changed*, gets
downloaded once in phase 2, hashes identically, and is skipped — no parse, no
embedding, no write. A re-run over an untouched corpus does no writes at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from ingestion.acl import acl_fingerprint
from ingestion.chunk import chunk_document
from ingestion.connectors import get_connector, make_document_id
from ingestion.connectors.base import guess_media_type
from ingestion.delta import (
    ChangeSet,
    DeltaDecision,
    ManifestStore,
    classify_document,
    detect_deletions,
    get_manifest_store,
    manifest_entry_for,
    mirror_items_to_postgres,
    tombstone_documents,
)
from ingestion.enrich import enrich_document, scan_and_redact
from ingestion.parse import parse_document
from ingestion.upsert import NullChunkWriter, RunUpserter, get_chunk_writer
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl, Classification
from ragcore.models.chunk import SourceType
from ragcore.models.document import (
    IngestAction,
    IngestManifest,
    IngestManifestEntry,
    IngestRunSummary,
    IngestStatus,
    IngestTrigger,
    SourceConfig,
    SourceDocument,
)
from ragcore.settings import Settings, get_settings

__all__ = [
    "DocumentOutcome",
    "DocumentTask",
    "SourcePlan",
    "changeset_from_plan",
    "dry_run_source",
    "finalize_run",
    "guard_scheduled_run",
    "ingest_single_document",
    "ingest_uploaded_document",
    "list_active_tenants",
    "load_source_configs",
    "plan_source",
    "process_document",
    "process_documents",
    "resolve_sources",
    "run_ingest",
    "run_ingestion",
    "start_run",
]

_log = get_logger(__name__)


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def _new_run_id() -> str:
    """Generate a run id.

    Returns:
        A 32-character hex id from ``ragcore.db.repositories.new_id``.
    """
    from ragcore.db.repositories import new_id

    return new_id()


def _redact_error(exc: BaseException) -> str:
    """Render an exception as a message safe to persist.

    Ingested documents can contain personal data, and an exception message can quote
    it. Only the exception type and — for the platform's own errors, whose messages are
    contractually content-free — the message itself are kept.

    Args:
        exc: The exception to describe.

    Returns:
        A short, content-free description.
    """
    if isinstance(exc, RagError):
        return f"{type(exc).__name__}: {exc.message}"
    return type(exc).__name__


# --------------------------------------------------------------------- the guard
def guard_scheduled_run(
    settings: Settings, now: datetime | None = None, *, forced: bool = False
) -> tuple[bool, str]:
    """Decide whether a scheduled run may start.

    Requirement #1 asks for a delta refresh *outside working hours*, configurably. The
    authority is :meth:`ragcore.settings.Settings.may_start_scheduled_ingest`; this
    wrapper only adds the per-invocation ``forced`` override so an admin can trigger a
    mid-afternoon run without changing configuration.

    Args:
        settings: Process settings.
        now: Moment the run would start. Defaults to now in ``ingest_timezone``.
        forced: Per-invocation override of the working-hours guard.

    Returns:
        An ``(allowed, reason)`` pair where reason is ``"ok"``, ``"forced"``,
        ``"disabled"`` or ``"working_hours"``.
    """
    allowed, reason = settings.may_start_scheduled_ingest(now)
    if allowed or reason == "disabled":
        return allowed, reason
    if forced:
        return True, "forced"
    return False, reason


# ------------------------------------------------------------------ source state
def _source_config_from_row(row: Any) -> SourceConfig:
    """Convert a ``source_configs`` row into its pydantic model.

    Args:
        row: A :class:`ragcore.db.models.SourceConfigRow`.

    Returns:
        The equivalent :class:`SourceConfig`.
    """
    return SourceConfig(
        source_id=row.source_id,
        tenant_id=row.tenant_id,
        source_type=SourceType(row.source_type),
        name=row.name,
        enabled=row.enabled,
        options=dict(row.options or {}),
        default_classification=Classification(row.default_classification),
        default_allowed_roles=list(row.default_allowed_roles or []),
        default_allowed_groups=list(row.default_allowed_groups or []),
        default_allowed_users=list(row.default_allowed_users or []),
        default_denied_users=list(row.default_denied_users or []),
        inherit_source_permissions=row.inherit_source_permissions,
        doc_type=row.doc_type,
        tags=list(row.tags or []),
        language=row.language,
        cron_override=row.cron_override,
        timezone_override=row.timezone_override,
        cursor=row.cursor,
        cursor_updated_at=row.cursor_updated_at,
        last_run_id=row.last_run_id,
        last_run_at=row.last_run_at,
        last_status=IngestStatus(row.last_status) if row.last_status else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def list_active_tenants(settings: Settings | None = None) -> list[str]:
    """List tenants eligible for a scheduled run.

    Args:
        settings: Process settings.

    Returns:
        Active tenant ids, sorted.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.models import Tenant

    active = settings or get_settings()
    async with session_scope(active) as session:
        result = await session.execute(
            select(Tenant.tenant_id).where(Tenant.is_active.is_(True))
        )
        return sorted(str(value) for value in result.scalars())


async def load_source_configs(
    *,
    settings: Settings | None = None,
    tenant_id: str | None = None,
    source_ids: Sequence[str] | None = None,
) -> list[SourceConfig]:
    """Load the source configurations a run should process.

    Args:
        settings: Process settings.
        tenant_id: Restrict to one tenant. When omitted every active tenant is
            processed, which is what the nightly timer does.
        source_ids: Restrict to specific sources within the tenant scope.

    Returns:
        Enabled source configs, tenant by tenant.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.repositories import list_source_configs

    active = settings or get_settings()
    tenants = [tenant_id] if tenant_id else await list_active_tenants(active)
    wanted = set(source_ids or ())
    out: list[SourceConfig] = []
    async with session_scope(active) as session:
        for tenant in tenants:
            rows = await list_source_configs(
                session, tenant_id=tenant, enabled_only=True
            )
            for row in rows:
                if wanted and row.source_id not in wanted:
                    continue
                out.append(_source_config_from_row(row))
    return out


async def _persist_source_state(
    *,
    settings: Settings,
    source: SourceConfig,
    cursor: str | None,
    run_id: str,
    status: IngestStatus,
    started_at: datetime,
) -> None:
    """Write a source's delta cursor and last-run state back to Postgres.

    The manifest is the authoritative cursor store (it lives next to the entries it
    describes); this mirrors it into ``source_configs`` so the admin API can show
    "last run" without touching Blob Storage.

    Args:
        settings: Process settings.
        source: The source that ran.
        cursor: Cursor to persist, or None to leave the stored one alone.
        run_id: Run id.
        status: Terminal status of the run.
        started_at: When the run started.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.models import SourceConfigRow

    values: dict[str, Any] = {
        "last_run_id": run_id,
        "last_run_at": started_at,
        "last_status": status.value,
    }
    if cursor is not None:
        values["cursor"] = cursor
        values["cursor_updated_at"] = _utcnow()
    async with session_scope(settings) as session:
        await session.execute(
            update(SourceConfigRow)
            .where(
                SourceConfigRow.tenant_id == source.tenant_id,
                SourceConfigRow.source_id == source.source_id,
            )
            .values(**values)
        )
        await session.commit()


# ------------------------------------------------------------------ the payloads
class DocumentTask(BaseModel):
    """One document's worth of work, serialisable for a Durable activity.

    The task carries everything the activity needs: the source config (so it can build
    a connector on any worker), the descriptor, the planned action, and the manifest
    entry the plan compared against (so phase 2 can re-classify on the content hash
    without re-reading the manifest).
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Parent run id.")
    tenant_id: str = Field(description="Owning tenant id.")
    source: SourceConfig = Field(description="Source configuration to read.")
    descriptor: SourceDocument = Field(description="Content-free document descriptor.")
    planned_action: IngestAction = Field(
        description="Action the plan phase decided on the cheap signals."
    )
    reason: str = Field(default="", description="Why the plan chose that action.")
    previous_entry: IngestManifestEntry | None = Field(
        default=None, description="Manifest entry from the last successful run."
    )
    forced: bool = Field(
        default=False, description="Reindex regardless of the recorded state."
    )
    trigger: IngestTrigger = Field(
        default=IngestTrigger.TIMER, description="What started the run."
    )


class DocumentOutcome(BaseModel):
    """What actually happened to one document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Document id.")
    source_uri: str = Field(description="Canonical source URI.")
    action: IngestAction = Field(description="Action finally taken.")
    status: IngestStatus = Field(
        default=IngestStatus.SUCCEEDED, description="Per-document status."
    )
    reason: str = Field(default="", description="Machine-readable justification.")
    chunk_count: int = Field(default=0, ge=0, description="Chunks written.")
    chunks_deleted: int = Field(default=0, ge=0, description="Stale chunks pruned.")
    token_count: int = Field(default=0, ge=0, description="Tokens embedded.")
    duplicates_dropped: int = Field(default=0, ge=0, description="Chunks deduped.")
    drop_reasons: dict[str, int] = Field(
        default_factory=dict, description="Drop reason -> count."
    )
    content_sha256: str = Field(default="", description="Hash indexed.")
    acl_fingerprint: str = Field(default="", description="ACL fingerprint indexed.")
    etag: str | None = Field(default=None, description="ETag observed.")
    source_modified_at: datetime | None = Field(
        default=None, description="Modification time observed."
    )
    version: int = Field(default=1, ge=1, description="Document version indexed.")
    pii_detected: bool = Field(
        default=False, description="Whether PII was found in this document."
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Processing time.")
    error_message: str | None = Field(
        default=None, description="Redacted failure description."
    )

    def manifest_entry(
        self, descriptor: SourceDocument, run_id: str
    ) -> IngestManifestEntry:
        """Build the manifest entry this outcome implies.

        Args:
            descriptor: The document descriptor, for URI and modification time.
            run_id: Run id to stamp.

        Returns:
            The entry to upsert into the source manifest.
        """
        updated = descriptor.model_copy(
            update={
                "etag": self.etag or descriptor.etag,
                "source_modified_at": self.source_modified_at
                or descriptor.source_modified_at,
                "content_sha256": self.content_sha256 or descriptor.content_sha256,
            }
        )
        return manifest_entry_for(
            updated,
            run_id=run_id,
            acl_fingerprint=self.acl_fingerprint,
            version=self.version,
            chunk_count=self.chunk_count,
            token_count=self.token_count,
        )


class SourcePlan(BaseModel):
    """The plan for one source: work to do, work already known unnecessary."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Run id.")
    tenant_id: str = Field(description="Owning tenant id.")
    source_id: str = Field(description="Source config id.")
    tasks: list[DocumentTask] = Field(
        default_factory=list, description="Documents that need fetching."
    )
    skipped: list[DeltaDecision] = Field(
        default_factory=list,
        description="Documents skipped on metadata alone, never downloaded.",
    )
    deletions: list[str] = Field(
        default_factory=list, description="Document ids to tombstone."
    )
    documents_seen: int = Field(default=0, ge=0, description="Descriptors enumerated.")
    full_scan: bool = Field(
        default=True, description="Whether the enumeration covered the whole source."
    )
    cursor: str | None = Field(
        default=None, description="Delta cursor to persist after a successful run."
    )
    trigger: IngestTrigger = Field(
        default=IngestTrigger.TIMER, description="What started the run."
    )
    forced: bool = Field(default=False, description="Working-hours guard overridden.")
    error_message: str | None = Field(
        default=None, description="Redacted enumeration failure, when planning failed."
    )


# ---------------------------------------------------------------------- phase 1
async def plan_source(
    source: SourceConfig,
    *,
    run_id: str,
    settings: Settings | None = None,
    trigger: IngestTrigger = IngestTrigger.TIMER,
    forced: bool = False,
    since: datetime | None = None,
    store: ManifestStore | None = None,
) -> SourcePlan:
    """Enumerate a source and decide what needs fetching.

    Classification here uses only what a listing gives away — ETag, modification time
    and the resolved ACL — so an unchanged corpus costs one listing call and no
    downloads.

    Args:
        source: Source configuration to read.
        run_id: Run id to stamp on tasks.
        settings: Process settings.
        trigger: What started the run.
        forced: Reindex everything regardless of recorded state.
        since: Lower bound on modification time, for connectors that filter on it.
        store: Manifest store; resolved from settings when omitted.

    Returns:
        A :class:`SourcePlan`. Enumeration failures are captured in
        ``error_message`` rather than raised, so one broken source cannot abort a
        multi-source nightly run.
    """
    active = settings or get_settings()
    manifest_store = store or get_manifest_store(active)
    plan = SourcePlan(
        run_id=run_id,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        trigger=trigger,
        forced=forced,
    )

    manifest = await manifest_store.load(source.tenant_id, source.source_id)
    connector = get_connector(source, active)
    connector.prime_delta_state(manifest.entries)
    seen: set[str] = set()

    try:
        async for descriptor in connector.list_documents(since):
            seen.add(descriptor.document_id)
            plan.documents_seen += 1
            try:
                access_control = await connector.resolve_acl(descriptor)
            except Exception as exc:
                _log.warning(
                    "plan.acl_resolution_failed",
                    tenant_id=source.tenant_id,
                    source_id=source.source_id,
                    error=_redact_error(exc),
                )
                access_control = source.default_access_control()
            _assert_tenant(access_control, source)
            fingerprint = acl_fingerprint(access_control)
            resolved = descriptor.model_copy(update={"access_control": access_control})

            decision = classify_document(
                manifest, resolved, acl_fingerprint=fingerprint, force=forced
            )
            if decision.action is IngestAction.SKIP:
                plan.skipped.append(decision)
                continue
            plan.tasks.append(
                DocumentTask(
                    run_id=run_id,
                    tenant_id=source.tenant_id,
                    source=source,
                    descriptor=resolved,
                    planned_action=decision.action,
                    reason=decision.reason,
                    previous_entry=manifest.get(resolved.document_id),
                    forced=forced,
                    trigger=trigger,
                )
            )
        plan.full_scan = connector.performed_full_scan
        plan.cursor = connector.cursor
        plan.deletions = detect_deletions(
            manifest,
            seen,
            full_scan=plan.full_scan,
            enabled=active.ingest_delete_missing,
        )
    except Exception as exc:
        plan.error_message = _redact_error(exc)
        _log.error(
            "plan.enumeration_failed",
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            error=plan.error_message,
        )
    finally:
        await connector.close()

    _log.info(
        "plan.complete",
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        run_id=run_id,
        seen=plan.documents_seen,
        tasks=len(plan.tasks),
        skipped=len(plan.skipped),
        deletions=len(plan.deletions),
        full_scan=plan.full_scan,
    )
    return plan


def _assert_tenant(access_control: AccessControl, source: SourceConfig) -> None:
    """Refuse an ACL that does not belong to the source's tenant.

    Args:
        access_control: The resolved ACL.
        source: The owning source config.

    Raises:
        ValueError: If the tenants disagree. Multi-tenancy is a security boundary, so
            this is a hard failure rather than a corrected value.
    """
    if access_control.tenant_id != source.tenant_id:
        msg = (
            "resolved access control belongs to a different tenant than its source: "
            f"expected {source.tenant_id!r}"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------- phase 2
async def process_document(
    task: DocumentTask,
    *,
    settings: Settings | None = None,
    upserter: RunUpserter | None = None,
    connector: Any | None = None,
) -> DocumentOutcome:
    """Fetch, re-classify, parse, enrich, chunk, embed and write one document.

    This is the unit a Durable activity runs. It is idempotent: chunk point ids are
    deterministic, the ``documents`` row is an upsert, and a document whose content
    hash matches the manifest performs no writes at all.

    Args:
        task: The unit of work.
        settings: Process settings.
        upserter: Shared upserter. When provided, its run-level dedupe state spans
            every document in the batch, which is how cross-document duplicates are
            caught in one process.
        connector: Shared connector for the task's source. Built and closed here when
            omitted.

    Returns:
        The outcome. Failures are returned as ``status=FAILED`` with a redacted
        message, never raised, so one bad document cannot fail a run.
    """
    active = settings or get_settings()
    started = time.perf_counter()
    descriptor = task.descriptor
    outcome = DocumentOutcome(
        document_id=descriptor.document_id,
        source_uri=descriptor.source_uri,
        action=task.planned_action,
        reason=task.reason,
        etag=descriptor.etag,
        source_modified_at=descriptor.source_modified_at,
        version=max(1, task.previous_entry.version if task.previous_entry else 1),
    )

    owned_connector = connector is None
    owned_upserter = upserter is None
    active_connector = connector or get_connector(task.source, active)
    if task.previous_entry is not None:
        active_connector.prime_delta_state(
            {task.previous_entry.document_id: task.previous_entry}
        )
    writer_owner: RunUpserter | None = None

    try:
        if upserter is None:
            writer = await get_chunk_writer(active)
            writer_owner = RunUpserter(
                settings=active, writer=writer, run_id=task.run_id
            )
            active_upserter = writer_owner
        else:
            active_upserter = upserter

        return await _process(
            task, active, active_connector, active_upserter, outcome, started
        )
    except Exception as exc:
        outcome.status = IngestStatus.FAILED
        outcome.error_message = _redact_error(exc)
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        _log.error(
            "process.failed",
            tenant_id=task.tenant_id,
            document_id=descriptor.document_id,
            error=outcome.error_message,
        )
        return outcome
    finally:
        if owned_connector:
            await active_connector.close()
        if owned_upserter and writer_owner is not None:
            await writer_owner.close()


async def _process(
    task: DocumentTask,
    settings: Settings,
    connector: Any,
    upserter: RunUpserter,
    outcome: DocumentOutcome,
    started: float,
) -> DocumentOutcome:
    """Run the per-document stages once the collaborators are resolved.

    Args:
        task: The unit of work.
        settings: Process settings.
        connector: Connector for the task's source.
        upserter: Upserter carrying run-level dedupe state.
        outcome: Outcome being filled in.
        started: ``time.perf_counter`` reading at the start of the document.

    Returns:
        The completed outcome.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.repositories import mark_documents_deleted, upsert_document

    descriptor = task.descriptor
    source = task.source

    if task.planned_action is IngestAction.DELETE:
        outcome.chunks_deleted = await upserter.tombstone(
            tenant_id=task.tenant_id, document_id=descriptor.document_id
        )
        async with session_scope(settings) as session:
            await mark_documents_deleted(
                session,
                tenant_id=task.tenant_id,
                document_ids=[descriptor.document_id],
            )
            await session.commit()
        outcome.action = IngestAction.DELETE
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    fingerprint = acl_fingerprint(descriptor.access_control)
    outcome.acl_fingerprint = fingerprint

    if task.planned_action is IngestAction.ACL_ONLY:
        await upserter.acl_only_reindex(
            tenant_id=task.tenant_id,
            document_id=descriptor.document_id,
            access_control=descriptor.access_control,
        )
        async with session_scope(settings) as session:
            await upsert_document(
                session,
                tenant_id=task.tenant_id,
                document_id=descriptor.document_id,
                source_uri=descriptor.source_uri,
                source_type=descriptor.source_type.value,
                access_control=descriptor.access_control,
                source_id=source.source_id,
                title=descriptor.title,
                doc_type=descriptor.doc_type,
                language=descriptor.language or source.language,
                author=descriptor.author,
                content_sha256=task.previous_entry.content_sha256
                if task.previous_entry
                else "",
                etag=descriptor.etag,
                acl_fingerprint=fingerprint,
                tags=descriptor.tags,
                chunk_count=task.previous_entry.chunk_count
                if task.previous_entry
                else 0,
                token_count=task.previous_entry.token_count
                if task.previous_entry
                else 0,
                source_modified_at=descriptor.source_modified_at,
                ingest_run_id=task.run_id,
            )
            await session.commit()
        outcome.action = IngestAction.ACL_ONLY
        outcome.content_sha256 = (
            task.previous_entry.content_sha256 if task.previous_entry else ""
        )
        outcome.chunk_count = (
            task.previous_entry.chunk_count if task.previous_entry else 0
        )
        outcome.token_count = (
            task.previous_entry.token_count if task.previous_entry else 0
        )
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    fetched = await connector.fetch(descriptor)
    if fetched.not_modified:
        outcome.action = IngestAction.SKIP
        outcome.status = IngestStatus.SKIPPED
        outcome.reason = "not_modified"
        outcome.content_sha256 = (
            task.previous_entry.content_sha256 if task.previous_entry else ""
        )
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    document = connector.apply_fetched(descriptor, fetched)
    if document.size_bytes > settings.ingest_max_document_bytes:
        outcome.action = IngestAction.SKIP
        outcome.status = IngestStatus.SKIPPED
        outcome.reason = "too_large"
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    outcome.etag = document.etag
    outcome.source_modified_at = document.source_modified_at
    outcome.content_sha256 = document.content_sha256

    # Phase 2 delta: the bytes are in hand, so the content hash — not the ETag —
    # decides whether this document has really changed.
    single = IngestManifest(tenant_id=task.tenant_id, source_id=source.source_id)
    if task.previous_entry is not None:
        single.upsert(task.previous_entry)
    decision = classify_document(
        single, document, acl_fingerprint=fingerprint, force=task.forced
    )
    outcome.reason = decision.reason
    outcome.version = decision.next_version

    if decision.action is IngestAction.SKIP:
        outcome.action = IngestAction.SKIP
        outcome.status = IngestStatus.SKIPPED
        outcome.chunk_count = (
            task.previous_entry.chunk_count if task.previous_entry else 0
        )
        outcome.token_count = (
            task.previous_entry.token_count if task.previous_entry else 0
        )
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    if decision.action is IngestAction.ACL_ONLY:
        rewritten = await upserter.acl_only_reindex(
            tenant_id=task.tenant_id,
            document_id=document.document_id,
            access_control=document.access_control,
        )
        outcome.action = IngestAction.ACL_ONLY
        outcome.chunk_count = rewritten
        outcome.duration_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    parsed = parse_document(document, settings)
    enriched = await enrich_document(parsed, settings=settings)
    outcome.pii_detected = bool(enriched.metadata.get("pii_types"))
    drafts = chunk_document(enriched, settings)

    async with session_scope(settings) as session:
        written = await upserter.upsert_document(
            enriched, drafts, version=decision.next_version, session=session
        )
        await upsert_document(
            session,
            tenant_id=task.tenant_id,
            document_id=enriched.document_id,
            source_uri=enriched.source_uri,
            source_type=enriched.source_type.value,
            access_control=enriched.access_control,
            source_id=source.source_id,
            title=enriched.title,
            doc_type=enriched.doc_type,
            language=enriched.language,
            author=enriched.author,
            content_sha256=enriched.content_sha256,
            etag=document.etag,
            acl_fingerprint=fingerprint,
            tags=enriched.tags,
            keywords=enriched.keywords,
            summary=enriched.summary,
            page_count=enriched.page_count,
            chunk_count=written.chunks_written,
            token_count=written.tokens_embedded,
            size_bytes=document.size_bytes,
            source_modified_at=enriched.source_modified_at,
            effective_from=enriched.effective_from,
            effective_to=enriched.effective_to,
            pii_types=list(enriched.metadata.get("pii_types") or []),
            pii_redacted=bool(enriched.metadata.get("pii_redacted")),
            ingest_run_id=task.run_id,
            blob_url=document.blob_url,
            extra={"media_type": document.media_type},
        )
        await session.commit()

    outcome.action = decision.action
    outcome.chunk_count = written.chunks_written
    outcome.chunks_deleted = written.chunks_deleted
    outcome.token_count = written.tokens_embedded
    outcome.duplicates_dropped = written.duplicates_dropped
    outcome.drop_reasons = dict(written.drop_reasons)
    outcome.duration_ms = (time.perf_counter() - started) * 1000.0
    _log.info(
        "process.complete",
        tenant_id=task.tenant_id,
        document_id=enriched.document_id,
        action=outcome.action.value,
        chunks=outcome.chunk_count,
        duplicates_dropped=outcome.duplicates_dropped,
        pii_detected=outcome.pii_detected,
    )
    return outcome


async def process_documents(
    tasks: Sequence[DocumentTask],
    *,
    settings: Settings | None = None,
    share_dedupe_state: bool = True,
) -> list[DocumentOutcome]:
    """Process a batch of documents with bounded concurrency.

    Args:
        tasks: Units of work, all from the same run.
        settings: Process settings; ``ingest_max_parallel_docs`` bounds concurrency.
        share_dedupe_state: Share one :class:`RunUpserter` across the batch so
            duplicate chunks are detected across documents, not just within one.

    Returns:
        Outcomes in input order.
    """
    active = settings or get_settings()
    if not tasks:
        return []

    shared: RunUpserter | None = None
    if share_dedupe_state:
        writer = await get_chunk_writer(active)
        shared = RunUpserter(settings=active, writer=writer, run_id=tasks[0].run_id)

    connectors: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(active.ingest_max_parallel_docs)

    async def one(task: DocumentTask) -> DocumentOutcome:
        async with semaphore:
            connector = connectors.get(task.source.source_id)
            if connector is None:
                connector = get_connector(task.source, active)
                connectors[task.source.source_id] = connector
            return await process_document(
                task, settings=active, upserter=shared, connector=connector
            )

    try:
        return list(await asyncio.gather(*(one(task) for task in tasks)))
    finally:
        for connector in connectors.values():
            await connector.close()
        if shared is not None:
            await shared.close()


# ---------------------------------------------------------------------- phase 3
async def finalize_run(
    plan: SourcePlan,
    outcomes: Sequence[DocumentOutcome],
    *,
    source: SourceConfig,
    settings: Settings | None = None,
    store: ManifestStore | None = None,
    started_at: datetime | None = None,
    within_working_hours: bool = False,
) -> IngestRunSummary:
    """Fold outcomes into the manifest, tombstone deletions and write the summary.

    Finalisation is idempotent, which is what makes a failed run resumable: re-running
    it re-applies the same manifest entries and re-writes the same run row.

    Args:
        plan: The plan this run executed.
        outcomes: Per-document outcomes.
        source: The source configuration.
        settings: Process settings.
        store: Manifest store; resolved from settings when omitted.
        started_at: Run start time, for the summary.
        within_working_hours: Whether the run started inside working hours.

    Returns:
        The completed :class:`IngestRunSummary`.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.repositories import record_ingest_run

    active = settings or get_settings()
    manifest_store = store or get_manifest_store(active)
    summary = IngestRunSummary(
        run_id=plan.run_id,
        tenant_id=plan.tenant_id,
        source_id=plan.source_id,
        trigger=plan.trigger,
        started_at=started_at or _utcnow(),
        forced=plan.forced,
        within_working_hours=within_working_hours,
        documents_seen=plan.documents_seen,
        documents_skipped=len(plan.skipped),
    )
    if plan.error_message:
        summary.errors.append(plan.error_message)
        summary.error_message = plan.error_message

    manifest = await manifest_store.load(plan.tenant_id, plan.source_id)
    descriptors = {task.descriptor.document_id: task.descriptor for task in plan.tasks}
    decisions: list[DeltaDecision] = list(plan.skipped)
    statuses: dict[str, str] = {}
    metrics: dict[str, dict[str, float]] = {}

    writer = await get_chunk_writer(active)
    upserter = RunUpserter(settings=active, writer=writer, run_id=plan.run_id)

    try:
        for outcome in outcomes:
            _accumulate(summary, outcome)
            decisions.append(
                DeltaDecision(
                    document_id=outcome.document_id,
                    source_uri=outcome.source_uri,
                    action=outcome.action,
                    reason=outcome.reason or outcome.action.value,
                    content_sha256=outcome.content_sha256,
                    acl_fingerprint=outcome.acl_fingerprint,
                    previous_version=max(0, outcome.version - 1),
                )
            )
            statuses[outcome.document_id] = outcome.status.value
            metrics[outcome.document_id] = {
                "chunk_count": float(outcome.chunk_count),
                "token_count": float(outcome.token_count),
                "duplicates_dropped": float(outcome.duplicates_dropped),
                "duration_ms": outcome.duration_ms,
            }
            if outcome.error_message:
                summary.errors.append(outcome.error_message)
            if outcome.status is IngestStatus.FAILED:
                continue
            if outcome.action is IngestAction.DELETE:
                manifest.mark_deleted(outcome.document_id, plan.run_id)
                continue
            descriptor = descriptors.get(outcome.document_id)
            if descriptor is not None:
                manifest.upsert(outcome.manifest_entry(descriptor, plan.run_id))

        # A metadata-only skip still refreshes last_seen_at so the entry does not look
        # abandoned, and records the ETag so the next run short-circuits earlier.
        for decision in plan.skipped:
            entry = manifest.get(decision.document_id)
            if entry is not None:
                entry.last_run_id = plan.run_id
                entry.last_seen_at = _utcnow()

        if plan.deletions:
            summary.chunks_deleted += await tombstone_documents(
                tenant_id=plan.tenant_id,
                document_ids=plan.deletions,
                manifest=manifest,
                run_id=plan.run_id,
                upserter=upserter,
            )
            summary.documents_deleted += len(plan.deletions)
            decisions.extend(
                DeltaDecision(
                    document_id=document_id,
                    source_uri=(
                        manifest.entries[document_id].source_uri
                        if document_id in manifest.entries
                        else ""
                    ),
                    action=IngestAction.DELETE,
                    reason="missing_from_full_scan",
                )
                for document_id in plan.deletions
            )
    finally:
        await upserter.close()

    manifest.cursor = plan.cursor if plan.cursor is not None else manifest.cursor
    manifest.last_run_id = plan.run_id
    if plan.full_scan:
        manifest.last_full_scan_at = _utcnow()
    await manifest_store.save(manifest)

    summary.finish(IngestStatus.FAILED if plan.error_message and not outcomes else None)
    summary.metrics.update(
        {
            "documents_per_second": (
                summary.documents_seen / summary.duration_seconds
                if summary.duration_seconds
                else 0.0
            ),
            "tracked_documents": float(len(manifest.entries)),
            "live_documents": float(manifest.live_count),
        }
    )

    async with session_scope(active) as session:
        await record_ingest_run(session, summary)
        await mirror_items_to_postgres(
            session,
            tenant_id=plan.tenant_id,
            run_id=plan.run_id,
            decisions=decisions,
            statuses=statuses,
            metrics=metrics,
        )
        await session.commit()

    await _persist_source_state(
        settings=active,
        source=source,
        cursor=plan.cursor,
        run_id=plan.run_id,
        status=summary.status,
        started_at=summary.started_at,
    )
    if store is None:
        await manifest_store.close()

    _log.info(
        "run.finished",
        tenant_id=plan.tenant_id,
        source_id=plan.source_id,
        run_id=plan.run_id,
        status=summary.status.value,
        created=summary.documents_created,
        updated=summary.documents_updated,
        skipped=summary.documents_skipped,
        deleted=summary.documents_deleted,
        failed=summary.documents_failed,
        chunks=summary.chunks_upserted,
        duplicates_dropped=summary.duplicates_dropped,
    )
    return summary


def _accumulate(summary: IngestRunSummary, outcome: DocumentOutcome) -> None:
    """Fold one document's outcome into the run summary.

    Args:
        summary: The summary being accumulated.
        outcome: The document outcome.
    """
    if outcome.status is IngestStatus.FAILED:
        summary.documents_failed += 1
        return
    if outcome.action is IngestAction.CREATE:
        summary.documents_created += 1
    elif outcome.action is IngestAction.UPDATE:
        summary.documents_updated += 1
    elif outcome.action is IngestAction.DELETE:
        summary.documents_deleted += 1
    elif outcome.action is IngestAction.ACL_ONLY:
        summary.documents_updated += 1
    else:
        summary.documents_skipped += 1
    summary.chunks_upserted += outcome.chunk_count
    summary.chunks_deleted += outcome.chunks_deleted
    summary.tokens_embedded += outcome.token_count
    summary.duplicates_dropped += outcome.duplicates_dropped
    if outcome.pii_detected:
        summary.pii_documents += 1


# ------------------------------------------------------------------ orchestration
async def run_ingestion(
    *,
    tenant_id: str | None = None,
    source_ids: Sequence[str] | None = None,
    trigger: IngestTrigger = IngestTrigger.MANUAL,
    forced: bool = False,
    settings: Settings | None = None,
    now: datetime | None = None,
    enforce_schedule: bool = False,
    since: datetime | None = None,
) -> list[IngestRunSummary]:
    """Run ingestion in this process, one summary per source.

    This is the CLI and "wait for the result" HTTP path, and the reference behaviour
    the Durable orchestrator reproduces across activities.

    Args:
        tenant_id: Restrict to one tenant; every active tenant otherwise.
        source_ids: Restrict to specific sources.
        trigger: What started the run.
        forced: Override the working-hours guard.
        settings: Process settings.
        now: Moment used for the schedule guard.
        enforce_schedule: Apply the working-hours guard. True for timer runs, False
            for an explicit admin or CLI run.
        since: Lower bound on modification time handed to connectors.

    Returns:
        One :class:`IngestRunSummary` per source. When the schedule guard refuses, a
        single skipped summary is returned carrying ``skip_reason``.
    """
    active = settings or get_settings()
    within_hours = active.is_within_working_hours(now)

    if enforce_schedule:
        allowed, reason = guard_scheduled_run(active, now, forced=forced)
        if not allowed:
            return [
                _skipped_summary(
                    tenant_id=tenant_id or "*",
                    trigger=trigger,
                    reason=reason,
                    within_working_hours=within_hours,
                    forced=forced,
                )
            ]
        forced = forced or reason == "forced"

    sources = await load_source_configs(
        settings=active, tenant_id=tenant_id, source_ids=source_ids
    )
    if not sources:
        _log.warning("run.no_sources", tenant_id=tenant_id or "*")
        return []

    summaries: list[IngestRunSummary] = []
    store = get_manifest_store(active)
    try:
        for source in sources:
            summaries.append(
                await _run_source(
                    source,
                    settings=active,
                    store=store,
                    trigger=trigger,
                    forced=forced,
                    since=since,
                    within_working_hours=within_hours,
                )
            )
    finally:
        await store.close()
    return summaries


async def _run_source(
    source: SourceConfig,
    *,
    settings: Settings,
    store: ManifestStore,
    trigger: IngestTrigger,
    forced: bool,
    since: datetime | None,
    within_working_hours: bool,
) -> IngestRunSummary:
    """Plan, process and finalise one source in this process.

    Args:
        source: The source configuration to read.
        settings: Process settings.
        store: Manifest store shared across the sources of one invocation.
        trigger: What started the run.
        forced: Reindex regardless of the recorded state.
        since: Lower bound on modification time handed to connectors.
        within_working_hours: Whether the run started inside working hours.

    Returns:
        The completed :class:`IngestRunSummary` for that source.
    """
    run_id, started_at = await start_run(
        source, trigger=trigger, forced=forced, settings=settings
    )
    plan = await plan_source(
        source,
        run_id=run_id,
        settings=settings,
        trigger=trigger,
        forced=forced,
        since=since,
        store=store,
    )
    outcomes = await process_documents(plan.tasks, settings=settings)
    return await finalize_run(
        plan,
        outcomes,
        source=source,
        settings=settings,
        store=store,
        started_at=started_at,
        within_working_hours=within_working_hours,
    )


def _skipped_summary(
    *,
    tenant_id: str,
    trigger: IngestTrigger,
    reason: str,
    within_working_hours: bool,
    forced: bool,
) -> IngestRunSummary:
    """Build the summary for a run the guard refused.

    Args:
        tenant_id: Tenant scope of the refused run.
        trigger: What tried to start it.
        reason: ``"disabled"`` or ``"working_hours"``.
        within_working_hours: Whether the moment was inside working hours.
        forced: Whether forcing was requested.

    Returns:
        A finished summary with status ``SKIPPED``.
    """
    summary = IngestRunSummary(
        run_id=_new_run_id(),
        tenant_id=tenant_id,
        trigger=trigger,
        skip_reason=reason,
        within_working_hours=within_working_hours,
        forced=forced,
    )
    _log.warning("run.refused", tenant_id=tenant_id, reason=reason)
    return summary.finish(IngestStatus.SKIPPED)


async def resolve_sources(
    *,
    tenant_id: str,
    source_id: str | None = None,
    source_type: SourceType | None = None,
    sources: Sequence[SourceConfig] | None = None,
    settings: Settings | None = None,
) -> list[SourceConfig]:
    """Work out which sources a run should process.

    Args:
        tenant_id: Owning tenant id. Always applied.
        source_id: Run exactly this registered source.
        source_type: Run every registered source of this type.
        sources: Use these configurations verbatim, without requiring a
            ``source_configs`` row. This is what makes a local run self-sufficient.
        settings: Process settings.

    Returns:
        The source configurations to process, in a stable order.

    Raises:
        ValueError: If an explicitly supplied config belongs to another tenant.
            Multi-tenancy is a security boundary, so this is refused rather than
            corrected.
    """
    if sources is not None:
        for candidate in sources:
            if candidate.tenant_id != tenant_id:
                msg = (
                    "source config belongs to a different tenant than the run: "
                    f"expected {tenant_id!r}"
                )
                raise ValueError(msg)
        selected = list(sources)
    else:
        selected = await load_source_configs(
            settings=settings,
            tenant_id=tenant_id,
            source_ids=[source_id] if source_id else None,
        )
    if source_type is not None:
        selected = [item for item in selected if item.source_type is source_type]
    return selected


async def run_ingest(
    *,
    tenant_id: str,
    source_id: str | None = None,
    source_type: SourceType | None = None,
    sources: Sequence[SourceConfig] | None = None,
    trigger: IngestTrigger = IngestTrigger.MANUAL,
    force: bool = False,
    dry_run: bool = False,
    full_scan: bool = False,
    settings: Settings | None = None,
) -> list[IngestRunSummary]:
    """Run ingestion for one tenant — the operational entry point.

    This is the coroutine ``scripts/run_ingest_local.py``, the admin trigger and the
    CLI all call, and it is the same code the Azure Functions timer runs.

    The working-hours guard is always evaluated. When it refuses, a single skipped
    summary carrying ``skip_reason`` is returned rather than running silently, so an
    operator can always tell the difference between "nothing changed" and "the run
    never started".

    Args:
        tenant_id: Owning tenant id. Always applied to every query.
        source_id: Run exactly this registered source.
        source_type: Run every registered source of this type.
        sources: Use these configurations verbatim, without a ``source_configs`` row.
        trigger: What started the run.
        force: Override the working-hours guard and reindex regardless of the
            recorded content hash.
        dry_run: Enumerate, fetch, parse, chunk and dedupe, but write nothing to
            Qdrant, PostgreSQL or the manifest.
        full_scan: Ignore each source's stored delta cursor so the enumeration covers
            the whole source. Deletion detection needs a full scan, so this is how an
            operator repairs a missed deletion.
        settings: Process settings.

    Returns:
        One :class:`~ragcore.models.document.IngestRunSummary` per source processed,
        or a single skipped summary when the guard refused.

    Raises:
        ValueError: If a config in ``sources`` belongs to another tenant.
    """
    active = settings or get_settings()
    within_hours = active.is_within_working_hours()
    allowed, reason = guard_scheduled_run(active, forced=force)
    if not allowed:
        return [
            _skipped_summary(
                tenant_id=tenant_id,
                trigger=trigger,
                reason=reason,
                within_working_hours=within_hours,
                forced=force,
            )
        ]

    selected = await resolve_sources(
        tenant_id=tenant_id,
        source_id=source_id,
        source_type=source_type,
        sources=sources,
        settings=active,
    )
    if not selected:
        _log.warning(
            "run.no_sources",
            tenant_id=tenant_id,
            source_id=source_id or "*",
            source_type=source_type.value if source_type else "*",
        )
        return []
    if full_scan:
        selected = [item.model_copy(update={"cursor": None}) for item in selected]

    summaries: list[IngestRunSummary] = []
    store = get_manifest_store(active)
    try:
        for source in selected:
            if dry_run:
                summaries.append(
                    await dry_run_source(
                        source,
                        settings=active,
                        trigger=trigger,
                        forced=force,
                        store=store,
                        within_working_hours=within_hours,
                    )
                )
                continue
            summaries.append(
                await _run_source(
                    source,
                    settings=active,
                    store=store,
                    trigger=trigger,
                    forced=force,
                    since=None,
                    within_working_hours=within_hours,
                )
            )
    finally:
        await store.close()
    return summaries


async def dry_run_source(
    source: SourceConfig,
    *,
    settings: Settings | None = None,
    trigger: IngestTrigger = IngestTrigger.MANUAL,
    forced: bool = False,
    since: datetime | None = None,
    store: ManifestStore | None = None,
    within_working_hours: bool = False,
) -> IngestRunSummary:
    """Project what a run would do, without writing anything.

    Everything up to the vector store happens for real — enumeration, ACL resolution,
    delta classification, fetch, parse, PII scan, chunking and dedupe — so the
    projection reflects the actual corpus rather than a guess. Nothing is embedded and
    nothing is persisted: no Qdrant client is opened, no ``documents`` or
    ``ingest_runs`` row is written, and the manifest is read but never saved.

    Summary semantics: ``documents_*`` counters are the *projection* (what the run
    would do), ``chunks_upserted`` and ``tokens_embedded`` stay at zero because
    nothing was written, and the would-be totals are reported in ``metrics`` as
    ``chunks_planned`` / ``tokens_planned``. ``metrics["dry_run"]`` is 1.0.

    Args:
        source: The source configuration to read.
        settings: Process settings.
        trigger: What started the run.
        forced: Reindex regardless of the recorded state.
        since: Lower bound on modification time handed to the connector.
        store: Manifest store; resolved from settings when omitted.
        within_working_hours: Whether the run started inside working hours.

    Returns:
        The projected :class:`IngestRunSummary`.
    """
    active = settings or get_settings()
    manifest_store = store or get_manifest_store(active)
    run_id = _new_run_id()
    started_at = _utcnow()
    summary = IngestRunSummary(
        run_id=run_id,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        trigger=trigger,
        started_at=started_at,
        forced=forced,
        within_working_hours=within_working_hours,
    )

    plan = await plan_source(
        source,
        run_id=run_id,
        settings=active,
        trigger=trigger,
        forced=forced,
        since=since,
        store=manifest_store,
    )
    summary.documents_seen = plan.documents_seen
    summary.documents_skipped = len(plan.skipped)
    summary.documents_deleted = len(plan.deletions)
    if plan.error_message:
        summary.errors.append(plan.error_message)
        summary.error_message = plan.error_message

    upserter = RunUpserter(
        settings=active,
        writer=NullChunkWriter(),
        run_id=run_id,
        dry_run=True,
    )
    connector = get_connector(source, active)
    semaphore = asyncio.Semaphore(active.ingest_max_parallel_docs)
    chunks_planned = 0
    tokens_planned = 0

    async def project(task: DocumentTask) -> tuple[int, int, str]:
        """Fetch, parse and chunk one document without writing anything.

        Args:
            task: The unit of work.

        Returns:
            A ``(chunks, tokens, status)`` triple where status is one of
            ``"created"``, ``"updated"``, ``"skipped"``, ``"pii"`` or ``"failed"``.
        """
        async with semaphore:
            try:
                fetched = await connector.fetch(task.descriptor)
                if fetched.not_modified:
                    return 0, 0, "skipped"
                document = connector.apply_fetched(task.descriptor, fetched)
                if document.size_bytes > active.ingest_max_document_bytes:
                    return 0, 0, "skipped"
                parsed = parse_document(document, active)
                redacted, pii_types, was_redacted = scan_and_redact(parsed, active)
                redacted = redacted.model_copy(
                    update={
                        "metadata": {
                            **redacted.metadata,
                            "pii_types": pii_types,
                            "pii_redacted": was_redacted,
                        }
                    }
                )
                drafts = chunk_document(redacted, active)
                written = await upserter.upsert_document(redacted, drafts)
                status = (
                    "created"
                    if task.planned_action is IngestAction.CREATE
                    else ("updated")
                )
                if pii_types:
                    summary.pii_documents += 1
                summary.duplicates_dropped += written.duplicates_dropped
                return len(written.chunk_ids), written.tokens_embedded, status
            except Exception as exc:
                message = _redact_error(exc)
                summary.errors.append(message)
                _log.warning(
                    "dry_run.document_failed",
                    tenant_id=source.tenant_id,
                    document_id=task.descriptor.document_id,
                    error=message,
                )
                return 0, 0, "failed"

    try:
        results = await asyncio.gather(*(project(task) for task in plan.tasks))
    finally:
        await connector.close()
        await upserter.close()
        if store is None:
            await manifest_store.close()

    for chunks, tokens, status in results:
        chunks_planned += chunks
        tokens_planned += tokens
        if status == "created":
            summary.documents_created += 1
        elif status == "updated":
            summary.documents_updated += 1
        elif status == "failed":
            summary.documents_failed += 1
        else:
            summary.documents_skipped += 1

    summary.metrics.update(
        {
            "dry_run": 1.0,
            "chunks_planned": float(chunks_planned),
            "tokens_planned": float(tokens_planned),
            "deletions_planned": float(len(plan.deletions)),
        }
    )
    _log.info(
        "run.dry_run",
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        run_id=run_id,
        seen=summary.documents_seen,
        chunks_planned=chunks_planned,
    )
    return summary.finish()


async def start_run(
    source: SourceConfig,
    *,
    trigger: IngestTrigger,
    forced: bool = False,
    settings: Settings | None = None,
) -> tuple[str, datetime]:
    """Open a new run for one source and persist its ``running`` row.

    Called by the Durable plan activity and by :func:`run_ingestion` so both paths
    create the same auditable record before any document is touched.

    Args:
        source: Source about to be processed.
        trigger: What started the run.
        forced: Whether the working-hours guard was overridden.
        settings: Process settings.

    Returns:
        A ``(run_id, started_at)`` pair.
    """
    active = settings or get_settings()
    run_id = _new_run_id()
    started_at = _utcnow()
    await _record_running(active, run_id, source, trigger, started_at, forced)
    return run_id, started_at


async def _record_running(
    settings: Settings,
    run_id: str,
    source: SourceConfig,
    trigger: IngestTrigger,
    started_at: datetime,
    forced: bool,
) -> None:
    """Write the ``running`` run row before any work starts.

    An interrupted serverless invocation therefore still leaves an auditable record
    rather than vanishing.

    Args:
        settings: Process settings.
        run_id: Run id.
        source: Source being processed.
        trigger: What started the run.
        started_at: Run start time.
        forced: Whether the guard was overridden.
    """
    from ragcore.db.base import session_scope
    from ragcore.db.repositories import record_ingest_run

    summary = IngestRunSummary(
        run_id=run_id,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        trigger=trigger,
        status=IngestStatus.RUNNING,
        started_at=started_at,
        forced=forced,
    )
    async with session_scope(settings) as session:
        await record_ingest_run(session, summary)
        await session.commit()


# ------------------------------------------------------------- single-document
async def ingest_single_document(
    *,
    tenant_id: str,
    source_id: str,
    source_uri: str,
    trigger: IngestTrigger = IngestTrigger.UPLOAD,
    settings: Settings | None = None,
    forced: bool = False,
) -> IngestRunSummary:
    """Ingest exactly one document, immediately.

    This is the blob-trigger and reindex path: a document lands in the container and
    is indexed within seconds instead of waiting for the nightly pass.

    Args:
        tenant_id: Owning tenant id.
        source_id: Source config the document belongs to.
        source_uri: Canonical URI of the document.
        trigger: What started the run.
        settings: Process settings.
        forced: Reindex even when the content hash is unchanged.

    Returns:
        A one-document :class:`IngestRunSummary`.

    Raises:
        ValueError: If the source does not exist for that tenant, or the document is
            not present in it.
    """
    active = settings or get_settings()
    sources = await load_source_configs(
        settings=active, tenant_id=tenant_id, source_ids=[source_id]
    )
    if not sources:
        msg = f"source {source_id!r} not found for tenant {tenant_id!r}"
        raise ValueError(msg)
    source = sources[0]

    run_id, started_at = await start_run(
        source, trigger=trigger, forced=forced, settings=active
    )
    store = get_manifest_store(active)
    document_id = make_document_id(tenant_id, source_uri)
    try:
        plan = await plan_source(
            source,
            run_id=run_id,
            settings=active,
            trigger=trigger,
            forced=forced,
            store=store,
        )
        plan.deletions = []
        plan.full_scan = False
        plan.tasks = [
            task for task in plan.tasks if task.descriptor.document_id == document_id
        ]
        plan.skipped = [
            decision for decision in plan.skipped if decision.document_id == document_id
        ]
        if not plan.tasks and not plan.skipped:
            msg = "document was not found in its source"
            raise ValueError(msg)
        outcomes = await process_documents(plan.tasks, settings=active)
        return await finalize_run(
            plan,
            outcomes,
            source=source,
            settings=active,
            store=store,
            started_at=started_at,
            within_working_hours=active.is_within_working_hours(),
        )
    finally:
        await store.close()


async def ingest_uploaded_document(
    *,
    tenant_id: str,
    source_id: str,
    filename: str,
    payload: bytes,
    access_control: AccessControl,
    media_type: str | None = None,
    title: str = "",
    doc_type: str = "document",
    tags: Sequence[str] | None = None,
    author: str | None = None,
    language: str | None = None,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> DocumentOutcome:
    """Ingest bytes handed straight to the platform, with no connector involved.

    ``POST /documents`` uses this: the ACL comes from the uploading principal rather
    than from a source, and the document is indexed in the same pipeline as everything
    else so its chunks are indistinguishable from crawled ones.

    Args:
        tenant_id: Owning tenant id.
        source_id: Logical source to attribute the upload to.
        filename: Original file name; selects the parser.
        payload: Raw bytes.
        access_control: ACL to stamp on every chunk. Must belong to ``tenant_id``.
        media_type: MIME type; guessed from the file name when omitted.
        title: Document title; the file stem is used when omitted.
        doc_type: Document class.
        tags: Tags to stamp.
        author: Author, when known.
        language: Declared language.
        settings: Process settings.
        run_id: Run id to attribute the write to; generated when omitted.

    Returns:
        The document outcome.

    Raises:
        ValueError: If the ACL belongs to a different tenant.
    """
    active = settings or get_settings()
    if access_control.tenant_id != tenant_id:
        msg = "access control tenant does not match the upload tenant"
        raise ValueError(msg)

    source_uri = f"upload://{tenant_id}/{source_id}/{filename}"
    document = SourceDocument(
        document_id=make_document_id(tenant_id, source_uri),
        tenant_id=tenant_id,
        source_id=source_id,
        source_type=SourceType.UPLOAD,
        source_uri=source_uri,
        title=title or filename.rsplit(".", 1)[0].replace("_", " "),
        content_bytes=payload,
        media_type=media_type or guess_media_type(filename),
        filename=filename,
        size_bytes=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        source_modified_at=_utcnow(),
        access_control=access_control,
        doc_type=doc_type,
        tags=list(tags or []),
        author=author,
        language=language,
    )

    parsed = parse_document(document, active)
    enriched = await enrich_document(parsed, settings=active)
    drafts = chunk_document(enriched, active)

    from ragcore.db.base import session_scope
    from ragcore.db.repositories import upsert_document

    effective_run_id = run_id or _new_run_id()
    writer = await get_chunk_writer(active)
    upserter = RunUpserter(settings=active, writer=writer, run_id=effective_run_id)
    started = time.perf_counter()
    try:
        async with session_scope(active) as session:
            written = await upserter.upsert_document(
                enriched, drafts, version=1, session=session
            )
            await upsert_document(
                session,
                tenant_id=tenant_id,
                document_id=enriched.document_id,
                source_uri=source_uri,
                source_type=SourceType.UPLOAD.value,
                access_control=access_control,
                source_id=source_id,
                title=enriched.title,
                doc_type=enriched.doc_type,
                language=enriched.language,
                author=enriched.author,
                content_sha256=enriched.content_sha256,
                acl_fingerprint=acl_fingerprint(access_control),
                tags=enriched.tags,
                keywords=enriched.keywords,
                summary=enriched.summary,
                page_count=enriched.page_count,
                chunk_count=written.chunks_written,
                token_count=written.tokens_embedded,
                size_bytes=len(payload),
                source_modified_at=enriched.source_modified_at,
                effective_from=enriched.effective_from,
                effective_to=enriched.effective_to,
                pii_types=list(enriched.metadata.get("pii_types") or []),
                pii_redacted=bool(enriched.metadata.get("pii_redacted")),
                ingest_run_id=effective_run_id,
            )
            await session.commit()
    finally:
        await upserter.close()

    return DocumentOutcome(
        document_id=enriched.document_id,
        source_uri=source_uri,
        action=IngestAction.CREATE,
        reason="upload",
        chunk_count=written.chunks_written,
        chunks_deleted=written.chunks_deleted,
        token_count=written.tokens_embedded,
        duplicates_dropped=written.duplicates_dropped,
        drop_reasons=dict(written.drop_reasons),
        content_sha256=enriched.content_sha256,
        acl_fingerprint=acl_fingerprint(access_control),
        pii_detected=bool(enriched.metadata.get("pii_types")),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def changeset_from_plan(plan: SourcePlan) -> ChangeSet:
    """Render a plan as a :class:`~ingestion.delta.ChangeSet`.

    Args:
        plan: The plan to describe.

    Returns:
        The equivalent change set, for logging and admin display.
    """
    return ChangeSet(
        tenant_id=plan.tenant_id,
        source_id=plan.source_id,
        decisions=[
            *plan.skipped,
            *(
                DeltaDecision(
                    document_id=task.descriptor.document_id,
                    source_uri=task.descriptor.source_uri,
                    action=task.planned_action,
                    reason=task.reason,
                )
                for task in plan.tasks
            ),
        ],
        deletions=list(plan.deletions),
        full_scan=plan.full_scan,
    )
