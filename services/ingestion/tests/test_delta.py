"""Delta classification, deletion detection, tombstoning and manifest persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ingestion.acl import acl_fingerprint
from ingestion.delta import (
    REASON_ACL_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_DELETED_AT_SOURCE,
    REASON_FORCED,
    REASON_NEW,
    REASON_REAPPEARED,
    REASON_UNCHANGED,
    REASON_UNCHANGED_TOUCHED,
    LocalManifestStore,
    classify_document,
    detect_deletions,
    manifest_entry_for,
    manifest_summary,
)
from ingestion.upsert import QdrantChunkWriter, chunk_id_for
from ragcore.models.acl import AccessControl, Classification
from ragcore.models.chunk import SourceType
from ragcore.models.document import (
    IngestAction,
    IngestManifest,
    IngestManifestEntry,
    IngestTrigger,
    SourceConfig,
    SourceDocument,
)
from ragcore.settings import Settings
from ragcore.vectorstore.collections import point_id_for_chunk

TENANT = "tenant-a"
SOURCE = "src-1"
DOC_ID = "doc-0001"
URI = "file:///corpus/hr/leave-policy.pdf"
HASH_A = "a" * 64
HASH_B = "b" * 64
NOW = datetime(2026, 3, 1, 2, 30, tzinfo=UTC)


def make_acl(**overrides: object) -> AccessControl:
    fields: dict[str, object] = {"tenant_id": TENANT}
    fields.update(overrides)
    return AccessControl(**fields)  # type: ignore[arg-type]


def make_doc(
    *,
    content_sha256: str = HASH_A,
    etag: str | None = "etag-1",
    deleted: bool = False,
    acl: AccessControl | None = None,
    modified: datetime | None = NOW,
) -> SourceDocument:
    return SourceDocument(
        document_id=DOC_ID,
        tenant_id=TENANT,
        source_id=SOURCE,
        source_type=SourceType.LOCAL,
        source_uri=URI,
        title="Leave policy",
        content_sha256=content_sha256,
        etag=etag,
        source_modified_at=modified,
        access_control=acl or make_acl(),
        deleted=deleted,
    )


def make_entry(
    *,
    content_sha256: str = HASH_A,
    etag: str | None = "etag-1",
    fingerprint: str = "",
    version: int = 3,
    is_deleted: bool = False,
) -> IngestManifestEntry:
    return IngestManifestEntry(
        document_id=DOC_ID,
        source_uri=URI,
        content_sha256=content_sha256,
        etag=etag,
        source_modified_at=NOW,
        acl_fingerprint=fingerprint or acl_fingerprint(make_acl()),
        version=version,
        chunk_count=7,
        token_count=3200,
        last_run_id="run-old",
        is_deleted=is_deleted,
    )


def manifest_with(entry: IngestManifestEntry | None = None) -> IngestManifest:
    manifest = IngestManifest(tenant_id=TENANT, source_id=SOURCE)
    if entry is not None:
        manifest.upsert(entry)
    return manifest


# --------------------------------------------------------------- classification
def test_unknown_document_is_created():
    decision = classify_document(
        manifest_with(), make_doc(), acl_fingerprint=acl_fingerprint(make_acl())
    )
    assert decision.action is IngestAction.CREATE
    assert decision.reason == REASON_NEW
    assert decision.previous_version == 0
    assert decision.next_version == 1
    assert decision.needs_content is True


def test_identical_content_and_etag_is_skipped():
    fingerprint = acl_fingerprint(make_acl())
    decision = classify_document(
        manifest_with(make_entry(fingerprint=fingerprint)),
        make_doc(),
        acl_fingerprint=fingerprint,
    )
    assert decision.action is IngestAction.SKIP
    assert decision.reason == REASON_UNCHANGED
    assert decision.needs_content is False


def test_touched_but_unchanged_file_is_skipped_not_reindexed():
    """The requirement: a new ETag with identical bytes must not cost an embedding."""
    fingerprint = acl_fingerprint(make_acl())
    entry = make_entry(etag="etag-1", fingerprint=fingerprint)
    # Same bytes, but the file was copied so size-mtime (and therefore the ETag) moved.
    touched = make_doc(etag="etag-2", modified=NOW + timedelta(days=1))

    decision = classify_document(
        manifest_with(entry), touched, acl_fingerprint=fingerprint
    )

    assert decision.action is IngestAction.SKIP
    assert decision.reason == REASON_UNCHANGED_TOUCHED
    # And IngestManifestEntry.decide alone would have said UPDATE — which is exactly
    # why classify_document prefers the content hash.
    assert (
        entry.decide(content_sha256=HASH_A, etag="etag-2", acl_fingerprint=fingerprint)
        is IngestAction.UPDATE
    )


def test_changed_content_is_updated_and_bumps_version():
    fingerprint = acl_fingerprint(make_acl())
    decision = classify_document(
        manifest_with(make_entry(version=3, fingerprint=fingerprint)),
        make_doc(content_sha256=HASH_B, etag="etag-2"),
        acl_fingerprint=fingerprint,
    )
    assert decision.action is IngestAction.UPDATE
    assert decision.reason == REASON_CONTENT_CHANGED
    assert decision.previous_version == 3
    assert decision.next_version == 4


def test_acl_only_change_avoids_reembedding():
    old = acl_fingerprint(make_acl(allowed_groups=["g-hr"]))
    new_acl = make_acl(allowed_groups=["g-hr", "g-legal"])
    decision = classify_document(
        manifest_with(make_entry(fingerprint=old)),
        make_doc(acl=new_acl),
        acl_fingerprint=acl_fingerprint(new_acl),
    )
    assert decision.action is IngestAction.ACL_ONLY
    assert decision.reason == REASON_ACL_CHANGED
    assert decision.needs_content is False


def test_classification_change_is_an_acl_change():
    old_acl = make_acl(classification=Classification.INTERNAL)
    new_acl = make_acl(classification=Classification.CONFIDENTIAL)
    decision = classify_document(
        manifest_with(make_entry(fingerprint=acl_fingerprint(old_acl))),
        make_doc(acl=new_acl),
        acl_fingerprint=acl_fingerprint(new_acl),
    )
    assert decision.action is IngestAction.ACL_ONLY


def test_deleted_at_source_is_a_delete():
    decision = classify_document(
        manifest_with(make_entry()),
        make_doc(deleted=True, content_sha256=""),
        acl_fingerprint="",
    )
    assert decision.action is IngestAction.DELETE
    assert decision.reason == REASON_DELETED_AT_SOURCE


def test_reappearing_document_is_recreated():
    fingerprint = acl_fingerprint(make_acl())
    decision = classify_document(
        manifest_with(make_entry(is_deleted=True, fingerprint=fingerprint)),
        make_doc(),
        acl_fingerprint=fingerprint,
    )
    assert decision.action is IngestAction.CREATE
    assert decision.reason == REASON_REAPPEARED


def test_force_reindexes_unchanged_content():
    fingerprint = acl_fingerprint(make_acl())
    decision = classify_document(
        manifest_with(make_entry(fingerprint=fingerprint)),
        make_doc(),
        acl_fingerprint=fingerprint,
        force=True,
    )
    assert decision.action is IngestAction.UPDATE
    assert decision.reason == REASON_FORCED


def test_not_modified_without_hash_defers_to_the_entry():
    fingerprint = acl_fingerprint(make_acl())
    decision = classify_document(
        manifest_with(make_entry(fingerprint=fingerprint)),
        make_doc(content_sha256=""),
        acl_fingerprint=fingerprint,
        not_modified=True,
    )
    assert decision.action is IngestAction.SKIP
    # The recorded hash is carried forward so the manifest entry is not blanked.
    assert decision.content_sha256 == HASH_A


def test_not_modified_still_detects_an_acl_change():
    new_acl = make_acl(allowed_users=["u-1"])
    decision = classify_document(
        manifest_with(make_entry(fingerprint=acl_fingerprint(make_acl()))),
        make_doc(content_sha256=""),
        acl_fingerprint=acl_fingerprint(new_acl),
        not_modified=True,
    )
    assert decision.action is IngestAction.ACL_ONLY


# ------------------------------------------------------------------- deletions
def test_full_scan_detects_missing_documents():
    manifest = IngestManifest(tenant_id=TENANT, source_id=SOURCE)
    for index in range(3):
        manifest.upsert(
            IngestManifestEntry(
                document_id=f"doc-{index}",
                source_uri=f"file:///corpus/{index}.txt",
                content_sha256=HASH_A,
            )
        )
    missing = detect_deletions(manifest, {"doc-0", "doc-2"}, full_scan=True)
    assert missing == ["doc-1"]


def test_incremental_pass_never_deletes():
    manifest = IngestManifest(tenant_id=TENANT, source_id=SOURCE)
    manifest.upsert(
        IngestManifestEntry(document_id="doc-0", source_uri=URI, content_sha256=HASH_A)
    )
    assert detect_deletions(manifest, set(), full_scan=False) == []


def test_deletion_detection_can_be_disabled():
    manifest = IngestManifest(tenant_id=TENANT, source_id=SOURCE)
    manifest.upsert(
        IngestManifestEntry(document_id="doc-0", source_uri=URI, content_sha256=HASH_A)
    )
    assert detect_deletions(manifest, set(), full_scan=True, enabled=False) == []


def test_already_deleted_documents_are_not_re_deleted():
    manifest = IngestManifest(tenant_id=TENANT, source_id=SOURCE)
    manifest.upsert(make_entry(is_deleted=True))
    assert detect_deletions(manifest, set(), full_scan=True) == []


# -------------------------------------------------------------------- entries
def test_manifest_entry_records_what_was_indexed():
    fingerprint = acl_fingerprint(make_acl(allowed_groups=["g-hr"]))
    entry = manifest_entry_for(
        make_doc(content_sha256=HASH_B, etag="etag-9"),
        run_id="run-42",
        acl_fingerprint=fingerprint,
        version=4,
        chunk_count=11,
        token_count=5000,
    )
    assert entry.content_sha256 == HASH_B
    assert entry.etag == "etag-9"
    assert entry.acl_fingerprint == fingerprint
    assert entry.version == 4
    assert entry.chunk_count == 11
    assert entry.token_count == 5000
    assert entry.last_run_id == "run-42"
    assert entry.is_deleted is False


def test_acl_fingerprint_is_order_insensitive_but_content_sensitive():
    left = make_acl(allowed_groups=["g-b", "g-a"], allowed_roles=["r1"])
    right = make_acl(allowed_groups=["g-a", "g-b"], allowed_roles=["r1"])
    assert acl_fingerprint(left) == acl_fingerprint(right)
    assert acl_fingerprint(left) != acl_fingerprint(
        make_acl(allowed_groups=["g-a"], allowed_roles=["r1"])
    )
    assert acl_fingerprint(left) != acl_fingerprint(
        make_acl(
            allowed_groups=["g-b", "g-a"],
            allowed_roles=["r1"],
            classification=Classification.RESTRICTED,
        )
    )


def test_acl_fingerprint_separates_tenants():
    assert acl_fingerprint(AccessControl(tenant_id="t1")) != acl_fingerprint(
        AccessControl(tenant_id="t2")
    )


def test_manifest_summary_leaks_no_uris():
    manifest = manifest_with(make_entry())
    manifest.cursor = "delta-token"
    rendered = manifest_summary(manifest)
    assert URI not in rendered
    assert '"tracked": 1' in rendered
    assert '"has_cursor": true' in rendered


# ---------------------------------------------------------------------- store
async def test_local_manifest_store_round_trip(tmp_path):
    store = LocalManifestStore(tmp_path / "manifests")

    empty = await store.load(TENANT, SOURCE)
    assert empty.entries == {}
    assert empty.tenant_id == TENANT

    empty.upsert(make_entry())
    empty.cursor = "watermark-2026-03-01"
    await store.save(empty)

    reloaded = await store.load(TENANT, SOURCE)
    assert set(reloaded.entries) == {DOC_ID}
    assert reloaded.entries[DOC_ID].content_sha256 == HASH_A
    assert reloaded.cursor == "watermark-2026-03-01"
    assert reloaded.live_count == 1


async def test_local_manifest_store_is_tenant_partitioned(tmp_path):
    store = LocalManifestStore(tmp_path / "manifests")
    mine = IngestManifest(tenant_id="t1", source_id=SOURCE)
    mine.upsert(make_entry())
    await store.save(mine)

    other = await store.load("t2", SOURCE)
    assert other.entries == {}


async def test_corrupt_manifest_degrades_to_a_full_rescan(tmp_path):
    root = tmp_path / "manifests"
    path = root / TENANT / f"{SOURCE}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    manifest = await LocalManifestStore(root).load(TENANT, SOURCE)
    assert manifest.entries == {}
    assert manifest.tenant_id == TENANT


async def test_manifest_from_another_tenant_is_rejected(tmp_path):
    root = tmp_path / "manifests"
    path = root / TENANT / f"{SOURCE}.json"
    path.parent.mkdir(parents=True)
    foreign = IngestManifest(tenant_id="someone-else", source_id=SOURCE)
    foreign.upsert(make_entry())
    path.write_text(foreign.model_dump_json(), encoding="utf-8")

    manifest = await LocalManifestStore(root).load(TENANT, SOURCE)
    assert manifest.entries == {}


# ------------------------------------------------------------------ tombstoning
@dataclass
class Record:
    """A minimal stand-in for a Qdrant scroll record."""

    payload: dict[str, Any]


@dataclass
class Count:
    """A minimal stand-in for a Qdrant count result."""

    count: int


@dataclass
class FakeQdrant:
    """An async Qdrant double that records what the writer asked it to do."""

    records: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[Any] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    upserted: list[Any] = field(default_factory=list)
    scroll_filters: list[Any] = field(default_factory=list)

    async def scroll(self, **kwargs: Any) -> tuple[list[Record], None]:
        self.scroll_filters.append(kwargs["scroll_filter"])
        return [Record(item) for item in self.records], None

    async def count(self, **kwargs: Any) -> Count:
        return Count(len(self.records))

    async def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs["points_selector"].filter)

    async def set_payload(self, **kwargs: Any) -> None:
        self.payloads.append(kwargs["payload"])

    async def upsert(self, **kwargs: Any) -> None:
        self.upserted.extend(kwargs["points"])


def writer_for(client: FakeQdrant) -> QdrantChunkWriter:
    settings = Settings(langfuse_enabled=False)
    return QdrantChunkWriter(
        client, settings, collection=settings.qdrant_chunks_collection
    )


def has_id_values(qfilter: Any) -> list[Any]:
    return [
        value
        for condition in (qfilter.must or [])
        for value in getattr(condition, "has_id", []) or []
    ]


def test_chunk_ids_are_readable_and_stable():
    assert chunk_id_for("doc-acme-travel-2025", 0) == "doc-acme-travel-2025::0000"
    assert chunk_id_for("doc-acme-travel-2025", 42) == "doc-acme-travel-2025::0042"
    # The Qdrant point id is derived from the logical id, never minted separately.
    assert point_id_for_chunk(chunk_id_for("d", 1)) == point_id_for_chunk("d::0001")


async def test_prune_removes_only_the_positions_the_document_lost():
    client = FakeQdrant([{"chunk_id": chunk_id_for(DOC_ID, i)} for i in range(5)])
    keep = [chunk_id_for(DOC_ID, i) for i in range(3)]

    pruned = await writer_for(client).prune_document(
        tenant_id=TENANT, document_id=DOC_ID, keep_chunk_ids=keep
    )

    assert pruned == 2
    assert len(client.deleted) == 1
    expected = {
        point_id_for_chunk(chunk_id_for(DOC_ID, 3)),
        point_id_for_chunk(chunk_id_for(DOC_ID, 4)),
    }
    assert set(has_id_values(client.deleted[0])) == expected


async def test_prune_is_a_noop_when_the_chunk_set_is_unchanged():
    client = FakeQdrant([{"chunk_id": chunk_id_for(DOC_ID, i)} for i in range(3)])
    keep = [chunk_id_for(DOC_ID, i) for i in range(3)]

    pruned = await writer_for(client).prune_document(
        tenant_id=TENANT, document_id=DOC_ID, keep_chunk_ids=keep
    )

    assert pruned == 0
    assert client.deleted == []


async def test_document_deletion_tombstones_instead_of_purging():
    client = FakeQdrant([{"chunk_id": chunk_id_for(DOC_ID, 0)}])

    affected = await writer_for(client).soft_delete_document(
        tenant_id=TENANT, document_id=DOC_ID, run_id="run-7"
    )

    assert affected == 1
    assert client.deleted == []
    assert client.payloads[0]["is_deleted"] is True
    assert client.payloads[0]["ingest_run_id"] == "run-7"


async def test_content_hash_probe_is_tenant_scoped():
    client = FakeQdrant([{"content_sha256": HASH_A, "document_id": "other-doc"}])

    found = await writer_for(client).find_by_content_hash(
        tenant_id=TENANT, hashes=[HASH_A]
    )

    assert found == {HASH_A: "other-doc"}
    keys = {
        getattr(condition, "key", None)
        for condition in (client.scroll_filters[0].must or [])
    }
    assert "tenant_id" in keys
    assert "content_sha256" in keys


# ----------------------------------------------------------------- the guard
def guard_settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "langfuse_enabled": False,
        "ingest_working_hours_start": 0,
        "ingest_working_hours_end": 24,
        "ingest_working_days": [0, 1, 2, 3, 4, 5, 6],
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


async def test_run_ingest_refuses_inside_working_hours():
    from ingestion.pipeline import run_ingest

    summaries = await run_ingest(
        tenant_id=TENANT,
        sources=[],
        trigger=IngestTrigger.MANUAL,
        settings=guard_settings(),
    )

    assert len(summaries) == 1
    assert summaries[0].skip_reason == "working_hours"
    assert summaries[0].within_working_hours is True
    assert summaries[0].tenant_id == TENANT


async def test_force_overrides_the_working_hours_guard():
    from ingestion.pipeline import run_ingest

    summaries = await run_ingest(
        tenant_id=TENANT, sources=[], force=True, settings=guard_settings()
    )

    # The guard passed; there was simply nothing configured to ingest.
    assert summaries == []


async def test_disabled_ingestion_cannot_be_forced():
    from ingestion.pipeline import run_ingest

    summaries = await run_ingest(
        tenant_id=TENANT,
        sources=[],
        force=True,
        settings=guard_settings(ingest_enabled=False),
    )

    assert [summary.skip_reason for summary in summaries] == ["disabled"]


async def test_run_ingest_refuses_a_source_from_another_tenant():
    from ingestion.pipeline import run_ingest

    foreign = SourceConfig(
        source_id=SOURCE,
        tenant_id="tenant-b",
        source_type=SourceType.LOCAL,
        name="someone else's corpus",
        options={"root": "/tmp"},  # noqa: S108 - never read; the tenant check fires
    )

    with pytest.raises(ValueError, match="different tenant"):
        await run_ingest(
            tenant_id=TENANT,
            sources=[foreign],
            force=True,
            settings=guard_settings(),
        )
