"""Delta state: the manifest, change classification and tombstoning.

A nightly refresh must be cheap, and "cheap" means: enumerate, compare, and touch
only what actually changed. The comparison state is an
:class:`~ragcore.models.document.IngestManifest` persisted per
``(tenant_id, source_id)`` in Blob Storage — durable across serverless invocations,
and mirrored into the ``ingest_items`` table so operators can see per-document
outcomes in SQL rather than by reading JSON out of a container.

Classification rules, in order (:func:`classify_document`):

1. the connector reported the item deleted -> ``DELETE``
2. no manifest entry -> ``CREATE``
3. content hash equals the recorded hash -> ``SKIP``, or ``ACL_ONLY`` when just the
   ACL fingerprint moved
4. content hash differs -> ``UPDATE``
5. no content hash available (the connector short-circuited on an unchanged ETag) ->
   defer to :meth:`IngestManifestEntry.decide`

Step 3 is the point of the whole module: **content hash beats ETag.** A file that was
copied, re-uploaded or had its metadata edited gets a new ETag and a new
``last_modified`` while its bytes are identical, and re-embedding it would be pure
waste. ``IngestManifestEntry.decide`` treats a moved ETag as an update, which is the
right default for a caller that has no hash; this module always has one, so it
prefers it and then refreshes the stored ETag so the *next* run can short-circuit at
the connector without downloading anything at all.

Deletions are detected by set difference, but only after a **full scan**. A delta pass
(a SharePoint ``deltaLink``, a SQL watermark) deliberately does not mention unchanged
documents, and treating silence as deletion would empty the index.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio
from pydantic import BaseModel, ConfigDict, Field

from ragcore.logging import get_logger
from ragcore.models.document import (
    IngestAction,
    IngestManifest,
    IngestManifestEntry,
    IngestStatus,
    SourceDocument,
)
from ragcore.settings import Settings

__all__ = [
    "BlobManifestStore",
    "ChangeSet",
    "DeltaDecision",
    "LocalManifestStore",
    "ManifestStore",
    "classify_document",
    "detect_deletions",
    "get_manifest_store",
    "manifest_entry_for",
    "manifest_summary",
    "mirror_items_to_postgres",
    "tombstone_documents",
]

_log = get_logger(__name__)

#: Reasons attached to a :class:`DeltaDecision`, surfaced in ``ingest_items.reason``.
REASON_NEW = "new"
REASON_CONTENT_CHANGED = "content_changed"
REASON_UNCHANGED = "unchanged"
REASON_UNCHANGED_TOUCHED = "unchanged_touched"
REASON_ACL_CHANGED = "acl_changed"
REASON_DELETED_AT_SOURCE = "deleted_at_source"
REASON_FORCED = "forced"
REASON_NOT_MODIFIED = "not_modified"
REASON_REAPPEARED = "reappeared"

#: Attempts made to write a manifest under optimistic concurrency.
_MANIFEST_WRITE_ATTEMPTS = 3


def _utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


class DeltaDecision(BaseModel):
    """What a run decided to do with one enumerated document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Deterministic document id.")
    source_uri: str = Field(description="Canonical source URI.")
    action: IngestAction = Field(
        description="create | update | delete | skip | acl_only"
    )
    reason: str = Field(description="Machine-readable justification for the action.")
    content_sha256: str = Field(default="", description="Hash observed this run.")
    acl_fingerprint: str = Field(default="", description="ACL hash observed this run.")
    previous_version: int = Field(
        default=0, ge=0, description="Version recorded by the last successful run."
    )

    @property
    def needs_content(self) -> bool:
        """Whether the action requires parsing and embedding.

        Returns:
            True for ``CREATE`` and ``UPDATE`` only. ``ACL_ONLY`` rewrites payloads,
            ``SKIP`` does nothing and ``DELETE`` tombstones.
        """
        return self.action in {IngestAction.CREATE, IngestAction.UPDATE}

    @property
    def next_version(self) -> int:
        """Document version this decision will produce.

        Returns:
            The previous version plus one for an update, 1 for a create, and the
            previous version unchanged for everything else.
        """
        if self.action is IngestAction.CREATE:
            return max(1, self.previous_version)
        if self.action is IngestAction.UPDATE:
            return max(1, self.previous_version) + 1
        return max(1, self.previous_version)


class ChangeSet(BaseModel):
    """The full plan for one source: what to do, and what disappeared."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(description="Owning tenant id.")
    source_id: str = Field(description="Source config id.")
    decisions: list[DeltaDecision] = Field(
        default_factory=list, description="One decision per enumerated document."
    )
    deletions: list[str] = Field(
        default_factory=list,
        description="Document ids present in the manifest but absent from a full scan.",
    )
    full_scan: bool = Field(
        default=True,
        description="Whether the enumeration saw every live document. Deletion "
        "detection is only valid when this is True.",
    )

    @property
    def actionable(self) -> list[DeltaDecision]:
        """Decisions that require any work at all.

        Returns:
            Every decision except ``SKIP``.
        """
        return [d for d in self.decisions if d.action is not IngestAction.SKIP]

    def counts(self) -> dict[str, int]:
        """Summarise decisions by action.

        Returns:
            A mapping of action value to count, plus ``"deletions"``.
        """
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.action.value] = counts.get(decision.action.value, 0) + 1
        counts["deletions"] = len(self.deletions)
        return counts


# ------------------------------------------------------------------- the stores
@runtime_checkable
class ManifestStore(Protocol):
    """Durable storage for per-source delta manifests."""

    async def load(self, tenant_id: str, source_id: str) -> IngestManifest:
        """Load a manifest, returning an empty one when none exists.

        Args:
            tenant_id: Owning tenant id.
            source_id: Source config id.

        Returns:
            The stored manifest, or a fresh empty manifest.
        """
        ...

    async def save(self, manifest: IngestManifest) -> None:
        """Persist a manifest.

        Args:
            manifest: The manifest to write.
        """
        ...

    async def close(self) -> None:
        """Release any transport the store owns."""
        ...


def _manifest_path(tenant_id: str, source_id: str) -> str:
    """Blob/file path for a manifest.

    Args:
        tenant_id: Owning tenant id.
        source_id: Source config id.

    Returns:
        A ``<tenant_id>/<source_id>.json`` relative path. Tenant-first so a container
        policy or a local directory listing is already tenant-partitioned.
    """
    return f"{tenant_id}/{source_id}.json"


def _parse_manifest(raw: bytes, tenant_id: str, source_id: str) -> IngestManifest:
    """Parse manifest JSON, tolerating corruption.

    Args:
        raw: Stored bytes.
        tenant_id: Expected tenant id.
        source_id: Expected source id.

    Returns:
        The parsed manifest, or an empty one when the payload is unusable. A corrupt
        manifest degrades a run to a full re-scan; it never fails the run.
    """
    try:
        manifest = IngestManifest.model_validate_json(raw)
    except (ValueError, UnicodeDecodeError):
        _log.warning(
            "delta.manifest_unparseable", tenant_id=tenant_id, source_id=source_id
        )
        return IngestManifest(tenant_id=tenant_id, source_id=source_id)
    if manifest.tenant_id != tenant_id or manifest.source_id != source_id:
        _log.error(
            "delta.manifest_identity_mismatch",
            tenant_id=tenant_id,
            source_id=source_id,
            stored_tenant_id=manifest.tenant_id,
        )
        return IngestManifest(tenant_id=tenant_id, source_id=source_id)
    return manifest


class LocalManifestStore:
    """Filesystem manifest store, used for local runs and tests."""

    def __init__(self, root: Path) -> None:
        """Create the store.

        Args:
            root: Directory under which ``<tenant>/<source>.json`` files live.
        """
        self.root = Path(root)

    async def load(self, tenant_id: str, source_id: str) -> IngestManifest:
        """Read a manifest from disk.

        Args:
            tenant_id: Owning tenant id.
            source_id: Source config id.

        Returns:
            The stored manifest, or an empty one.
        """
        path = self.root / _manifest_path(tenant_id, source_id)
        if not path.is_file():
            return IngestManifest(tenant_id=tenant_id, source_id=source_id)
        raw = await anyio.to_thread.run_sync(path.read_bytes)
        return _parse_manifest(raw, tenant_id, source_id)

    async def save(self, manifest: IngestManifest) -> None:
        """Write a manifest to disk atomically.

        Args:
            manifest: The manifest to write.
        """
        path = self.root / _manifest_path(manifest.tenant_id, manifest.source_id)
        manifest.updated_at = _utcnow()
        payload = manifest.model_dump_json().encode("utf-8")

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)

        await anyio.to_thread.run_sync(_write)

    async def close(self) -> None:
        """No transport to release."""
        return None


class BlobManifestStore:
    """Azure Blob manifest store with optimistic concurrency.

    Writes carry the ETag observed at load time, so two concurrent runs over the same
    source cannot silently clobber each other's manifest: the loser reloads, merges
    the newer entries and retries.
    """

    def __init__(self, settings: Settings) -> None:
        """Create the store.

        Args:
            settings: Process settings supplying the container and endpoint.
        """
        self.settings = settings
        self.container_name = settings.ingest_manifest_container
        self._service: Any = None
        self._credential: Any = None
        self._etags: dict[str, str] = {}

    async def _container(self) -> Any:
        """Build (once) the container client, creating the container if needed.

        Returns:
            An ``azure.storage.blob.aio.ContainerClient``.

        Raises:
            RuntimeError: If the Blob SDK is unavailable or no endpoint is configured.
        """
        if self._service is None:
            try:
                from azure.storage.blob.aio import BlobServiceClient
            except ImportError as exc:  # pragma: no cover - optional install
                msg = "azure-storage-blob is required for the blob manifest store"
                raise RuntimeError(msg) from exc
            if self.settings.azure_blob_connection_string:
                self._service = BlobServiceClient.from_connection_string(
                    self.settings.azure_blob_connection_string
                )
            elif self.settings.azure_blob_account_url:
                from ingestion.connectors.base import azure_credential

                self._credential = azure_credential(self.settings)
                self._service = BlobServiceClient(
                    account_url=self.settings.azure_blob_account_url,
                    credential=self._credential,
                )
            else:
                msg = "manifest store needs RAG_AZURE_BLOB_ACCOUNT_URL"
                raise RuntimeError(msg)
        container = self._service.get_container_client(self.container_name)
        try:
            await container.create_container()
        except Exception as exc:
            # Already exists, or this identity may only write blobs. Either way the
            # blob-level calls below are the real test of access.
            _log.debug("delta.manifest_container_exists", error=str(exc))
        return container

    async def load(self, tenant_id: str, source_id: str) -> IngestManifest:
        """Download a manifest blob.

        Args:
            tenant_id: Owning tenant id.
            source_id: Source config id.

        Returns:
            The stored manifest, or an empty one when the blob does not exist.
        """
        name = _manifest_path(tenant_id, source_id)
        container = await self._container()
        try:
            downloader = await container.download_blob(name)
            raw = await downloader.readall()
        except Exception:
            self._etags.pop(name, None)
            return IngestManifest(tenant_id=tenant_id, source_id=source_id)
        etag = getattr(getattr(downloader, "properties", None), "etag", None)
        if etag:
            self._etags[name] = str(etag)
        return _parse_manifest(bytes(raw), tenant_id, source_id)

    async def save(self, manifest: IngestManifest) -> None:
        """Upload a manifest, retrying on a concurrent write.

        Args:
            manifest: The manifest to write.

        Raises:
            RuntimeError: If the manifest could not be written after the configured
                number of attempts.
        """
        name = _manifest_path(manifest.tenant_id, manifest.source_id)
        container = await self._container()
        current = manifest
        for attempt in range(_MANIFEST_WRITE_ATTEMPTS):
            current.updated_at = _utcnow()
            payload = current.model_dump_json().encode("utf-8")
            etag = self._etags.get(name)
            try:
                blob = container.get_blob_client(name)
                if etag:
                    from azure.core import MatchConditions

                    result = await blob.upload_blob(
                        payload,
                        overwrite=True,
                        etag=etag,
                        match_condition=MatchConditions.IfNotModified,
                    )
                else:
                    result = await blob.upload_blob(payload, overwrite=True)
            except Exception as exc:
                _log.warning(
                    "delta.manifest_write_conflict",
                    tenant_id=manifest.tenant_id,
                    source_id=manifest.source_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                latest = await self.load(manifest.tenant_id, manifest.source_id)
                current = _merge_manifests(latest, manifest)
                continue
            new_etag = (result or {}).get("etag") if isinstance(result, dict) else None
            if new_etag:
                self._etags[name] = str(new_etag)
            return
        msg = "could not persist ingest manifest"
        raise RuntimeError(msg)

    async def close(self) -> None:
        """Close the service client and credential."""
        if self._service is not None:
            await self._service.close()
            self._service = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None


def _merge_manifests(latest: IngestManifest, mine: IngestManifest) -> IngestManifest:
    """Merge our entries onto a newer stored manifest.

    Args:
        latest: The manifest just re-read from storage.
        mine: The manifest this run wants to write.

    Returns:
        ``latest`` with our entries applied, keeping the most recently seen entry per
        document and our own cursor (which is the one we actually consumed).
    """
    merged = latest.model_copy(deep=True)
    for document_id, entry in mine.entries.items():
        existing = merged.entries.get(document_id)
        if existing is None or entry.last_seen_at >= existing.last_seen_at:
            merged.entries[document_id] = entry
    merged.cursor = mine.cursor or merged.cursor
    merged.last_run_id = mine.last_run_id or merged.last_run_id
    merged.last_full_scan_at = mine.last_full_scan_at or merged.last_full_scan_at
    return merged


def get_manifest_store(settings: Settings) -> ManifestStore:
    """Choose a manifest store for this environment.

    Args:
        settings: Process settings.

    Returns:
        A :class:`BlobManifestStore` when Blob Storage is configured, otherwise a
        :class:`LocalManifestStore` rooted next to ``ingest_local_root`` and named
        after ``ingest_manifest_container`` — so the local layout mirrors the cloud
        one.
    """
    if settings.azure_blob_account_url or settings.azure_blob_connection_string:
        return BlobManifestStore(settings)
    root = Path(settings.ingest_local_root).expanduser().parent
    return LocalManifestStore(root / settings.ingest_manifest_container)


# --------------------------------------------------------------- classification
def classify_document(
    manifest: IngestManifest,
    doc: SourceDocument,
    *,
    acl_fingerprint: str,
    not_modified: bool = False,
    force: bool = False,
) -> DeltaDecision:
    """Decide what to do with one enumerated document.

    Args:
        manifest: The source's manifest from the last successful run.
        doc: The enumerated document. ``content_sha256`` is set once the payload has
            been fetched; it may be empty when the connector short-circuited.
        acl_fingerprint: Fingerprint of the freshly resolved ACL.
        not_modified: True when a conditional request proved the content unchanged.
        force: Reindex regardless of the recorded state.

    Returns:
        The decision, carrying the reason and the previous version.
    """
    entry = manifest.get(doc.document_id)
    previous_version = entry.version if entry else 0

    if doc.deleted:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.DELETE,
            reason=REASON_DELETED_AT_SOURCE,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )

    if entry is None:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.CREATE,
            reason=REASON_NEW,
            content_sha256=doc.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=0,
        )

    if force:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.UPDATE,
            reason=REASON_FORCED,
            content_sha256=doc.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )

    if entry.is_deleted:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.CREATE,
            reason=REASON_REAPPEARED,
            content_sha256=doc.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )

    acl_changed = bool(acl_fingerprint) and acl_fingerprint != entry.acl_fingerprint

    if not_modified or not doc.content_sha256:
        # No hash available this run: the ETag/Last-Modified comparison recorded in
        # the manifest entry is the only signal, so defer to the contract's own rule.
        action = entry.decide(
            content_sha256=doc.content_sha256,
            etag=doc.etag,
            acl_fingerprint=acl_fingerprint,
        )
        reason = {
            IngestAction.UPDATE: REASON_CONTENT_CHANGED,
            IngestAction.ACL_ONLY: REASON_ACL_CHANGED,
            IngestAction.SKIP: REASON_NOT_MODIFIED,
        }.get(action, REASON_NOT_MODIFIED)
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=action,
            reason=reason,
            content_sha256=doc.content_sha256 or entry.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )

    if doc.content_sha256 != entry.content_sha256:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.UPDATE,
            reason=REASON_CONTENT_CHANGED,
            content_sha256=doc.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )

    # Bytes are identical. Anything else that moved (ETag, modification time) is
    # metadata churn and must not cost an embedding.
    if acl_changed:
        return DeltaDecision(
            document_id=doc.document_id,
            source_uri=doc.source_uri,
            action=IngestAction.ACL_ONLY,
            reason=REASON_ACL_CHANGED,
            content_sha256=doc.content_sha256,
            acl_fingerprint=acl_fingerprint,
            previous_version=previous_version,
        )
    touched = bool(doc.etag) and bool(entry.etag) and doc.etag != entry.etag
    return DeltaDecision(
        document_id=doc.document_id,
        source_uri=doc.source_uri,
        action=IngestAction.SKIP,
        reason=REASON_UNCHANGED_TOUCHED if touched else REASON_UNCHANGED,
        content_sha256=doc.content_sha256,
        acl_fingerprint=acl_fingerprint,
        previous_version=previous_version,
    )


def detect_deletions(
    manifest: IngestManifest,
    seen: Collection[str],
    *,
    full_scan: bool,
    enabled: bool = True,
) -> list[str]:
    """Find documents that disappeared from their source.

    Args:
        manifest: The source's manifest.
        seen: Document ids enumerated by this run.
        full_scan: Whether the enumeration covered the whole source. Deletion by set
            difference is unsound otherwise.
        enabled: ``settings.ingest_delete_missing``.

    Returns:
        Document ids to tombstone; empty when detection is disabled or the run was
        incremental.
    """
    if not enabled or not full_scan:
        return []
    return manifest.missing_document_ids(set(seen))


def manifest_entry_for(
    doc: SourceDocument,
    *,
    run_id: str,
    acl_fingerprint: str,
    version: int,
    chunk_count: int = 0,
    token_count: int = 0,
    content_sha256: str | None = None,
) -> IngestManifestEntry:
    """Build the manifest entry recording what this run indexed.

    Args:
        doc: The document just processed.
        run_id: Current run id.
        acl_fingerprint: ACL fingerprint stored for future ACL-only detection.
        version: Document version now indexed.
        chunk_count: Chunks written.
        token_count: Tokens embedded.
        content_sha256: Hash to record; defaults to the document's own.

    Returns:
        The entry to upsert into the manifest.
    """
    digest = content_sha256 if content_sha256 is not None else doc.content_sha256
    return IngestManifestEntry(
        document_id=doc.document_id,
        source_uri=doc.source_uri,
        content_sha256=digest,
        etag=doc.etag,
        source_modified_at=doc.source_modified_at,
        acl_fingerprint=acl_fingerprint,
        version=max(1, version),
        chunk_count=chunk_count,
        token_count=token_count,
        last_run_id=run_id,
        last_seen_at=_utcnow(),
        is_deleted=False,
    )


# ------------------------------------------------------------------- mirroring
async def mirror_items_to_postgres(
    session: Any,
    *,
    tenant_id: str,
    run_id: str,
    decisions: Sequence[DeltaDecision],
    statuses: Mapping[str, str] | None = None,
    metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> int:
    """Mirror per-document decisions into the ``ingest_items`` table.

    The manifest is the machine-readable delta state; ``ingest_items`` is the operator
    view. Writing both means "why was this document skipped last night?" is answerable
    with SQL.

    Args:
        session: Database session; the caller commits.
        tenant_id: Owning tenant id.
        run_id: Parent run id.
        decisions: Decisions to mirror.
        statuses: Optional per-document status override
            (``succeeded``/``failed``/``skipped``).
        metrics: Optional per-document counters, keys ``chunk_count``,
            ``token_count``, ``duplicates_dropped`` and ``duration_ms``.

    Returns:
        The number of rows written.
    """
    from ragcore.db.repositories import record_ingest_item

    default_status = {
        IngestAction.SKIP: IngestStatus.SKIPPED.value,
    }
    written = 0
    for decision in decisions:
        status = (statuses or {}).get(
            decision.document_id,
            default_status.get(decision.action, IngestStatus.SUCCEEDED.value),
        )
        counters = (metrics or {}).get(decision.document_id, {})
        await record_ingest_item(
            session,
            tenant_id=tenant_id,
            run_id=run_id,
            source_uri=decision.source_uri,
            action=decision.action.value,
            status=status,
            document_id=decision.document_id,
            reason=decision.reason,
            chunk_count=int(counters.get("chunk_count", 0)),
            token_count=int(counters.get("token_count", 0)),
            duplicates_dropped=int(counters.get("duplicates_dropped", 0)),
            duration_ms=float(counters.get("duration_ms", 0.0)),
        )
        written += 1
    return written


async def tombstone_documents(
    *,
    tenant_id: str,
    document_ids: Sequence[str],
    manifest: IngestManifest,
    run_id: str,
    upserter: Any,
    session: Any | None = None,
) -> int:
    """Soft-delete documents that vanished from their source.

    Tombstoning is three coordinated writes: the Qdrant points get
    ``is_deleted=True`` (so ``build_acl_filter`` stops returning them), the
    ``documents`` row is marked deleted, and the manifest entry is flagged so a later
    reappearance is classified as a fresh create.

    Args:
        tenant_id: Owning tenant id.
        document_ids: Documents to tombstone.
        manifest: The source manifest, mutated in place.
        run_id: Run performing the deletion.
        upserter: A :class:`ingestion.upsert.RunUpserter`.
        session: Optional database session for the relational soft-delete.

    Returns:
        The number of chunk points tombstoned.
    """
    if not document_ids:
        return 0
    points = 0
    for document_id in document_ids:
        points += await upserter.tombstone(tenant_id=tenant_id, document_id=document_id)
        manifest.mark_deleted(document_id, run_id)
    if session is not None:
        from ragcore.db.repositories import mark_documents_deleted

        await mark_documents_deleted(
            session, tenant_id=tenant_id, document_ids=list(document_ids)
        )
    _log.info(
        "delta.tombstoned",
        tenant_id=tenant_id,
        documents=len(document_ids),
        points=points,
    )
    return points


def manifest_summary(manifest: IngestManifest) -> str:
    """Render a compact, PII-free manifest description for logs.

    Args:
        manifest: The manifest to describe.

    Returns:
        A JSON string with counts only — never document URIs, which can contain
        personal data such as a customer name in a file path.
    """
    return json.dumps(
        {
            "tenant_id": manifest.tenant_id,
            "source_id": manifest.source_id,
            "tracked": len(manifest.entries),
            "live": manifest.live_count,
            "has_cursor": bool(manifest.cursor),
        },
        sort_keys=True,
    )
