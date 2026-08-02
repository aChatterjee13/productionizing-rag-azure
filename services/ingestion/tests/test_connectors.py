"""Connector delta logic, driven with mocked clients. No network, no cloud."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ingestion.acl import (
    access_control_from_graph_permissions,
    access_control_from_metadata,
    access_control_from_sidecar,
    parse_identifier_list,
)
from ingestion.connectors import (
    AzureBlobConnector,
    HttpCrawlerConnector,
    LocalFilesystemConnector,
    SharePointConnector,
    SqlSourceConnector,
    get_connector,
    make_document_id,
)
from ingestion.connectors.http_crawler import extract_links
from ragcore.errors import ConfigError
from ragcore.models.acl import Classification, Principal
from ragcore.models.chunk import SourceType
from ragcore.models.document import IngestManifestEntry, SourceConfig
from ragcore.settings import Settings

TENANT = "tenant-a"


def make_settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {"env": "local"}
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def make_source(
    source_type: SourceType, options: dict[str, Any], **overrides: object
) -> SourceConfig:
    fields: dict[str, object] = {
        "source_id": "src-1",
        "tenant_id": TENANT,
        "source_type": source_type,
        "name": "test-source",
        "options": options,
    }
    fields.update(overrides)
    return SourceConfig(**fields)  # type: ignore[arg-type]


async def collect(connector: Any, since: datetime | None = None) -> list[Any]:
    return [doc async for doc in connector.list_documents(since)]


# ============================================================== identity + registry
def test_document_id_is_deterministic_and_tenant_scoped():
    left = make_document_id(TENANT, "file:///corpus/a.pdf")
    assert left == make_document_id(TENANT, "file:///corpus/a.pdf")
    assert left != make_document_id("tenant-b", "file:///corpus/a.pdf")
    assert left != make_document_id(TENANT, "file:///corpus/b.pdf")


def test_registry_maps_every_source_type():
    settings = make_settings()
    pairs = [
        (SourceType.LOCAL, {"root": "corpus"}, LocalFilesystemConnector),
        (SourceType.BLOB, {"container": "c"}, AzureBlobConnector),
        (
            SourceType.SHAREPOINT,
            {"site_id": "s", "drive_id": "d"},
            SharePointConnector,
        ),
        (
            SourceType.HTTP,
            {"start_urls": ["https://example.test/"]},
            HttpCrawlerConnector,
        ),
        (
            SourceType.SQL,
            {
                "dsn_secret_ref": "DSN",
                "query": "SELECT 1",
                "watermark_column": "updated_at",
                "id_column": "id",
            },
            SqlSourceConnector,
        ),
    ]
    for source_type, options, expected in pairs:
        connector = get_connector(make_source(source_type, options), settings)
        assert isinstance(connector, expected)
        assert connector.supports_delta is True


def test_registry_refuses_an_unsupported_source_type():
    with pytest.raises(ConfigError):
        get_connector(make_source(SourceType.UPLOAD, {}), make_settings())


# ================================================================ local filesystem
def write_corpus(root) -> None:
    (root / "hr").mkdir(parents=True)
    (root / "hr" / "leave.md").write_text("# Leave\n\nTwenty-five days.\n")
    (root / "hr" / "leave.md.acl.json").write_text(
        json.dumps(
            {
                "allowed_groups": ["g-hr", "g-legal"],
                "denied_users": ["u-contractor"],
                "classification": "confidential",
            }
        )
    )
    (root / "notes.txt").write_text("plain note")
    (root / "skip.tmp").write_text("temporary")


async def test_local_connector_lists_documents_and_hides_sidecars(tmp_path):
    write_corpus(tmp_path)
    source = make_source(SourceType.LOCAL, {"root": str(tmp_path)})
    connector = LocalFilesystemConnector(source, make_settings())

    docs = await collect(connector)
    names = sorted(doc.filename for doc in docs)

    assert names == ["leave.md", "notes.txt", "skip.tmp"]
    assert all(not doc.filename.endswith(".acl.json") for doc in docs)
    assert all(doc.tenant_id == TENANT for doc in docs)
    assert all(doc.source_type is SourceType.LOCAL for doc in docs)
    assert all(doc.etag for doc in docs)
    assert connector.performed_full_scan is True


async def test_local_connector_applies_glob_filters(tmp_path):
    write_corpus(tmp_path)
    source = make_source(
        SourceType.LOCAL,
        {
            "root": str(tmp_path),
            "include_globs": ["**/*.md", "*.txt"],
            "exclude_globs": ["**/*.tmp"],
        },
    )
    docs = await collect(LocalFilesystemConnector(source, make_settings()))
    assert sorted(doc.filename for doc in docs) == ["leave.md", "notes.txt"]


async def test_local_connector_resolves_the_sidecar_acl(tmp_path):
    write_corpus(tmp_path)
    source = make_source(SourceType.LOCAL, {"root": str(tmp_path)})
    connector = LocalFilesystemConnector(source, make_settings())

    docs = await collect(connector)
    leave = next(doc for doc in docs if doc.filename == "leave.md")
    acl = await connector.resolve_acl(leave)

    assert acl.tenant_id == TENANT
    assert set(acl.allowed_groups) == {"g-hr", "g-legal"}
    assert acl.denied_users == ["u-contractor"]
    assert acl.classification is Classification.CONFIDENTIAL


async def test_local_connector_falls_back_to_source_defaults(tmp_path):
    write_corpus(tmp_path)
    source = make_source(
        SourceType.LOCAL,
        {"root": str(tmp_path)},
        default_allowed_groups=["g-everyone"],
        default_classification=Classification.INTERNAL,
    )
    connector = LocalFilesystemConnector(source, make_settings())

    docs = await collect(connector)
    note = next(doc for doc in docs if doc.filename == "notes.txt")
    acl = await connector.resolve_acl(note)

    assert acl.allowed_groups == ["g-everyone"]
    assert acl.classification is Classification.INTERNAL


async def test_local_connector_skips_the_read_when_the_etag_has_not_moved(tmp_path):
    write_corpus(tmp_path)
    source = make_source(SourceType.LOCAL, {"root": str(tmp_path)})
    connector = LocalFilesystemConnector(source, make_settings())

    first = await collect(connector)
    note = next(doc for doc in first if doc.filename == "notes.txt")
    fetched = await connector.fetch(note)
    assert fetched.not_modified is False
    assert fetched.content_text == "plain note"
    assert fetched.sha256

    # Second run: prime the connector with what the last run recorded.
    connector.prime_delta_state(
        {
            note.document_id: IngestManifestEntry(
                document_id=note.document_id,
                source_uri=note.source_uri,
                content_sha256=fetched.sha256,
                etag=note.etag,
            )
        }
    )
    again = await connector.fetch(note)
    assert again.not_modified is True
    assert again.content_bytes is None
    assert again.content_text is None


async def test_local_connector_since_filter_excludes_older_files(tmp_path):
    write_corpus(tmp_path)
    source = make_source(SourceType.LOCAL, {"root": str(tmp_path)})
    connector = LocalFilesystemConnector(source, make_settings())
    future = datetime(2999, 1, 1, tzinfo=UTC)

    assert await collect(connector, future) == []
    # An incremental pass must not claim to have seen the whole source.
    assert connector.performed_full_scan is False


# =========================================================================== blob
def test_blob_connector_builds_stable_uris_without_touching_azure():
    source = make_source(SourceType.BLOB, {"container": "docs", "prefix": "hr/"})
    settings = make_settings(
        azure_blob_account_url="https://acct.blob.core.windows.net"
    )
    connector = AzureBlobConnector(source, settings)

    assert (
        connector._uri("hr/leave.pdf")
        == "https://acct.blob.core.windows.net/docs/hr/leave.pdf"
    )
    assert connector.container == "docs"
    assert connector.prefix == "hr/"


def test_blob_metadata_and_tags_resolve_an_acl_with_tags_winning():
    source = make_source(SourceType.BLOB, {"container": "docs"})
    acl = access_control_from_metadata(
        {"acl_allowed_groups": "g-a,g-b", "classification": "internal"},
        {"classification": "restricted"},
        source,
    )
    assert acl is not None
    assert set(acl.allowed_groups) == {"g-a", "g-b"}
    assert acl.classification is Classification.RESTRICTED


def test_blob_metadata_from_another_tenant_is_refused():
    source = make_source(SourceType.BLOB, {"container": "docs"})
    with pytest.raises(ValueError, match="different tenant"):
        access_control_from_metadata({"tenant_id": "tenant-b"}, None, source)


def test_sidecar_variants_parse():
    source = make_source(SourceType.LOCAL, {"root": "corpus"})
    assert access_control_from_sidecar(b"not json", source) is None
    assert access_control_from_sidecar(b"[1,2]", source) is None
    acl = access_control_from_sidecar(
        json.dumps({"groups": ["g-a"], "note": "ignored"}), source
    )
    assert acl is not None
    assert acl.allowed_groups == ["g-a"]


def test_identifier_lists_accept_json_and_delimited_forms():
    assert parse_identifier_list('["a","b"]') == ["a", "b"]
    assert parse_identifier_list("a, b; c|d") == ["a", "b", "c", "d"]
    assert parse_identifier_list(["a", "a", " b "]) == ["a", "b"]
    assert parse_identifier_list(None) == []
    assert parse_identifier_list("") == []


# ==================================================================== http crawler
SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.test/a.html</loc><lastmod>2026-02-01</lastmod></url>
  <url><loc>https://example.test/private/b.html</loc></url>
</urlset>
"""

ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /\n"
ROBOTS_BLOCK_PRIVATE = "User-agent: *\nDisallow: /private/\n"


def crawler_for(
    handler: Any, source: SourceConfig, settings: Settings | None = None
) -> HttpCrawlerConnector:
    connector = HttpCrawlerConnector(source, settings or make_settings())
    connector.set_client(
        httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    )
    return connector


async def test_sitemap_listing_reads_lastmod_and_never_downloads_pages():
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=SITEMAP)
        return httpx.Response(200, text="<html><body><p>hi</p></body></html>")

    source = make_source(
        SourceType.HTTP, {"sitemap_url": "https://example.test/sitemap.xml"}
    )
    connector = crawler_for(handler, source)
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    assert [doc.source_uri for doc in docs] == [
        "https://example.test/a.html",
        "https://example.test/private/b.html",
    ]
    assert docs[0].source_modified_at == datetime(2026, 2, 1, tzinfo=UTC)
    # Only robots.txt and the sitemap were fetched: enumeration downloads no pages.
    assert set(requested) == {"/robots.txt", "/sitemap.xml"}


async def test_robots_disallow_is_respected():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_BLOCK_PRIVATE)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=SITEMAP)
        return httpx.Response(200, text="<html></html>")

    source = make_source(
        SourceType.HTTP, {"sitemap_url": "https://example.test/sitemap.xml"}
    )
    connector = crawler_for(handler, source)
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    assert [doc.source_uri for doc in docs] == ["https://example.test/a.html"]


async def test_off_domain_urls_are_never_crawled():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        if request.url.host == "example.test":
            return httpx.Response(
                200,
                text=(
                    '<html><body><a href="https://elsewhere.test/x.html">x</a>'
                    '<a href="/inside.html">in</a></body></html>'
                ),
                headers={"content-type": "text/html"},
            )
        raise AssertionError("crawler left the allowed domain")

    source = make_source(
        SourceType.HTTP,
        {"start_urls": ["https://example.test/index.html"], "max_depth": 1},
    )
    connector = crawler_for(handler, source)
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    hosts = {httpx.URL(doc.source_uri).host for doc in docs}
    assert hosts == {"example.test"}


async def test_crawl_caches_pages_so_fetch_makes_no_second_request():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        return httpx.Response(
            200,
            text='<html><body><a href="/a.html">a</a>body text</body></html>',
            headers={"content-type": "text/html", "etag": '"v1"'},
        )

    source = make_source(
        SourceType.HTTP,
        {"start_urls": ["https://example.test/index.html"], "max_depth": 1},
    )
    connector = crawler_for(handler, source)
    try:
        docs = await collect(connector)
        page_calls = len([call for call in calls if "robots" not in call])
        fetched = [await connector.fetch(doc) for doc in docs]
    finally:
        await connector.close()

    assert len(docs) == 2
    assert all(item.content_text for item in fetched)
    assert all(item.etag == "v1" for item in fetched)
    # No page was requested twice.
    assert len([call for call in calls if "robots" not in call]) == page_calls


async def test_conditional_get_turns_a_304_into_a_skip():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, text=SITEMAP)
        if request.headers.get("if-none-match") == "v1":
            return httpx.Response(304)
        return httpx.Response(
            200, text="<html>fresh</html>", headers={"content-type": "text/html"}
        )

    source = make_source(
        SourceType.HTTP, {"sitemap_url": "https://example.test/sitemap.xml"}
    )
    connector = crawler_for(handler, source)
    url = "https://example.test/a.html"
    document_id = make_document_id(TENANT, url)
    connector.prime_delta_state(
        {
            document_id: IngestManifestEntry(
                document_id=document_id,
                source_uri=url,
                content_sha256="x" * 64,
                etag="v1",
                source_modified_at=datetime(2026, 2, 1, tzinfo=UTC),
            )
        }
    )
    try:
        docs = await collect(connector)
        target = next(doc for doc in docs if doc.source_uri == url)
        result = await connector.fetch(target)
    finally:
        await connector.close()

    assert result.not_modified is True
    assert result.etag == "v1"


async def test_page_cap_bounds_a_sitemap_crawl():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        return httpx.Response(200, text=SITEMAP)

    source = make_source(
        SourceType.HTTP, {"sitemap_url": "https://example.test/sitemap.xml"}
    )
    connector = crawler_for(handler, source, make_settings(ingest_http_max_pages=1))
    try:
        docs = await collect(connector)
    finally:
        await connector.close()
    assert len(docs) == 1


def test_extract_links_resolves_relative_and_drops_non_http():
    html = (
        '<a href="/a.html">a</a><a href="b.html#frag">b</a>'
        '<a href="mailto:x@y.test">m</a><a href="javascript:void(0)">j</a>'
        '<a href="https://other.test/c">c</a>'
    )
    links = extract_links(html, "https://example.test/dir/index.html")
    assert links == [
        "https://example.test/a.html",
        "https://example.test/dir/b.html",
        "https://other.test/c",
    ]


def test_crawler_requires_seeds():
    with pytest.raises(ValueError, match="start_urls"):
        HttpCrawlerConnector(make_source(SourceType.HTTP, {}), make_settings())


# ====================================================================== sharepoint
class FakeGraphConnector(SharePointConnector):
    """SharePoint connector with the Graph transport replaced by canned pages."""

    def __init__(
        self,
        source: SourceConfig,
        settings: Settings,
        pages: list[dict[str, Any]],
        permissions: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Build a connector whose Graph transport is replaced by canned pages."""
        super().__init__(source, settings)
        self.pages = list(pages)
        self.permissions = permissions or {}
        self.urls: list[str] = []

    async def graph_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.urls.append(url)
        if "/permissions" in url:
            item_id = url.split("/items/")[1].split("/")[0]
            return {"value": self.permissions.get(item_id, [])}
        return self.pages.pop(0)


def sharepoint_source(**overrides: object) -> SourceConfig:
    return make_source(
        SourceType.SHAREPOINT, {"site_id": "site-1", "drive_id": "drive-1"}, **overrides
    )


DELTA_PAGE_ONE = {
    "value": [
        {
            "id": "item-1",
            "name": "leave-policy.docx",
            "size": 2048,
            "cTag": "ctag-1",
            "webUrl": "https://contoso.sharepoint.com/sites/hr/leave-policy.docx",
            "file": {
                "mimeType": (
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                )
            },
            "lastModifiedDateTime": "2026-02-10T09:12:44Z",
            "lastModifiedBy": {"user": {"displayName": "Dana Ops"}},
        },
        {"id": "folder-1", "name": "archive", "folder": {"childCount": 3}},
    ],
    "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
}

DELTA_PAGE_TWO = {
    "value": [{"id": "item-2", "name": "old.docx", "deleted": {"state": "deleted"}}],
    "@odata.deltaLink": "https://graph.microsoft.com/v1.0/delta?token=abc",
}


async def test_graph_delta_pages_are_followed_and_the_delta_link_persisted():
    connector = FakeGraphConnector(
        sharepoint_source(), make_settings(), [DELTA_PAGE_ONE, DELTA_PAGE_TWO]
    )
    docs = await collect(connector)

    assert [doc.filename for doc in docs] == ["leave-policy.docx", "old.docx"]
    # Folders are not documents.
    assert all(doc.filename != "archive" for doc in docs)
    assert docs[0].etag == "ctag-1"
    assert docs[0].author == "Dana Ops"
    assert docs[0].source_modified_at == datetime(2026, 2, 10, 9, 12, 44, tzinfo=UTC)
    assert docs[1].deleted is True
    assert connector.cursor == "https://graph.microsoft.com/v1.0/delta?token=abc"
    assert connector.performed_full_scan is True
    assert connector.urls[1] == "https://graph.microsoft.com/v1.0/next-page"


async def test_a_resumed_delta_pass_is_not_a_full_scan():
    source = sharepoint_source(cursor="https://graph.microsoft.com/v1.0/delta?token=x")
    connector = FakeGraphConnector(source, make_settings(), [DELTA_PAGE_TWO])
    await collect(connector)
    assert connector.performed_full_scan is False
    # A resumed run starts from the stored deltaLink, not from /root/delta.
    assert connector.urls[0] == "https://graph.microsoft.com/v1.0/delta?token=x"


async def test_document_identity_survives_a_rename():
    renamed = {
        "value": [
            {
                **DELTA_PAGE_ONE["value"][0],
                "name": "leave-policy-2026.docx",
                "webUrl": (
                    "https://contoso.sharepoint.com/sites/hr/leave-policy-2026.docx"
                ),
            }
        ],
        "@odata.deltaLink": "https://graph.microsoft.com/v1.0/delta?token=abc",
    }
    first = {
        "value": [DELTA_PAGE_ONE["value"][0]],
        "@odata.deltaLink": "https://graph.microsoft.com/v1.0/delta?token=abc",
    }
    original = await collect(
        FakeGraphConnector(sharepoint_source(), make_settings(), [first])
    )
    after = await collect(
        FakeGraphConnector(sharepoint_source(), make_settings(), [renamed])
    )

    assert original[0].document_id == after[0].document_id
    assert original[0].source_uri != after[0].source_uri


async def test_a_tombstone_maps_to_the_same_document_id_as_the_live_item():
    live = {
        "value": [{**DELTA_PAGE_ONE["value"][0], "id": "item-9"}],
        "@odata.deltaLink": "x",
    }
    dead = {
        "value": [{"id": "item-9", "deleted": {"state": "deleted"}}],
        "@odata.deltaLink": "y",
    }
    before = await collect(
        FakeGraphConnector(sharepoint_source(), make_settings(), [live])
    )
    after = await collect(
        FakeGraphConnector(sharepoint_source(), make_settings(), [dead])
    )
    assert before[0].document_id == after[0].document_id
    assert after[0].deleted is True


async def test_graph_permissions_become_group_and_user_object_ids():
    permissions = {
        "item-1": [
            {"grantedToV2": {"group": {"id": "11111111-1111-1111-1111-111111111111"}}},
            {
                "grantedToIdentitiesV2": [
                    {"user": {"id": "22222222-2222-2222-2222-222222222222"}},
                    {"siteGroup": {"id": "7"}},
                ]
            },
        ]
    }
    connector = FakeGraphConnector(
        sharepoint_source(),
        make_settings(),
        [{"value": DELTA_PAGE_ONE["value"], "@odata.deltaLink": "x"}],
        permissions,
    )
    docs = await collect(connector)
    acl = await connector.resolve_acl(docs[0])

    assert acl.tenant_id == TENANT
    assert acl.allowed_groups == ["11111111-1111-1111-1111-111111111111"]
    assert acl.allowed_users == ["22222222-2222-2222-2222-222222222222"]
    # A numeric SharePoint-local principal id is not an Entra object id.
    assert "7" not in acl.allowed_groups


async def test_pinned_source_permissions_skip_the_graph_call():
    connector = FakeGraphConnector(
        sharepoint_source(
            inherit_source_permissions=False, default_allowed_groups=["g-pinned"]
        ),
        make_settings(),
        [{"value": DELTA_PAGE_ONE["value"], "@odata.deltaLink": "x"}],
    )
    docs = await collect(connector)
    before = len(connector.urls)
    acl = await connector.resolve_acl(docs[0])

    assert acl.allowed_groups == ["g-pinned"]
    assert len(connector.urls) == before


async def test_source_defaults_never_widen_an_item_s_own_permissions():
    """A source default must not grant read on an item SharePoint restricted.

    This is the reachable end of ``AccessControl.merged_with``: with
    ``inherit_source_permissions`` left at its default, every item ACL is merged with
    the source defaults. Because ``permits`` reads the three allow lists as a
    disjunction, unioning them handed everyone in ``default_allowed_groups`` read on
    documents the source system had restricted to a single user.
    """
    owner = "22222222-2222-2222-2222-222222222222"
    permissions = {"item-1": [{"grantedToV2": {"user": {"id": owner}}}]}
    connector = FakeGraphConnector(
        sharepoint_source(default_allowed_groups=["g-staff"]),
        make_settings(),
        [{"value": DELTA_PAGE_ONE["value"], "@odata.deltaLink": "x"}],
        permissions,
    )
    docs = await collect(connector)
    acl = await connector.resolve_acl(docs[0])

    assert acl.permits(Principal(user_id=owner, tenant_id=TENANT))
    assert not acl.permits(
        Principal(user_id="someone-else", tenant_id=TENANT, groups=["g-staff"])
    )


def test_organization_wide_link_leaves_the_document_unrestricted():
    source = sharepoint_source()
    acl = access_control_from_graph_permissions(
        [{"link": {"scope": "organization"}}], source
    )
    assert acl is not None
    assert acl.is_unrestricted is True
    assert acl.tenant_id == TENANT


def test_no_resolvable_permissions_returns_none():
    source = sharepoint_source()
    assert access_control_from_graph_permissions([{"roles": ["read"]}], source) is None


def test_sharepoint_requires_drive_coordinates():
    with pytest.raises(ValueError, match="drive_id"):
        SharePointConnector(
            make_source(SourceType.SHAREPOINT, {"site_id": "s"}), make_settings()
        )


# ============================================================================= sql
SQL_OPTIONS: dict[str, Any] = {
    "dsn_secret_ref": "TEST_DSN",
    "query": "SELECT * FROM tickets",
    "watermark_column": "updated_at",
    "id_column": "id",
    "tenant_column": "tenant",
    "acl_groups_column": "acl_groups",
    "classification_column": "sensitivity",
    "title_column": "subject",
    "text_columns": ["subject", "body"],
    "deleted_column": "is_deleted",
    "uri_template": "crm://tickets/{id}",
}

ROWS = [
    (
        1,
        TENANT,
        "Broken laptop",
        "Needs a new battery",
        "g-it",
        "internal",
        "2026-01-05T10:00:00+00:00",
        0,
    ),
    (
        2,
        TENANT,
        "VPN access",
        "Cannot reach the intranet",
        "g-it,g-sec",
        "confidential",
        "2026-01-06T11:00:00+00:00",
        0,
    ),
    (
        3,
        "tenant-b",
        "Other tenant",
        "Must never be ingested",
        "g-x",
        "internal",
        "2026-01-07T12:00:00+00:00",
        0,
    ),
    (
        4,
        TENANT,
        "Closed ticket",
        "",
        "g-it",
        "internal",
        "2026-01-08T13:00:00+00:00",
        1,
    ),
]


async def sql_engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/tickets.db")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE tickets (id INTEGER PRIMARY KEY, tenant TEXT, "
                "subject TEXT, body TEXT, acl_groups TEXT, sensitivity TEXT, "
                "updated_at TEXT, is_deleted INTEGER)"
            )
        )
        for row in ROWS:
            await connection.execute(
                text(
                    "INSERT INTO tickets VALUES (:id, :tenant, :subject, :body, "
                    ":acl_groups, :sensitivity, :updated_at, :is_deleted)"
                ),
                dict(
                    zip(
                        (
                            "id",
                            "tenant",
                            "subject",
                            "body",
                            "acl_groups",
                            "sensitivity",
                            "updated_at",
                            "is_deleted",
                        ),
                        row,
                        strict=True,
                    )
                ),
            )
    return engine


def sql_connector(engine, **overrides: object) -> SqlSourceConnector:
    source = make_source(SourceType.SQL, dict(SQL_OPTIONS), **overrides)
    connector = SqlSourceConnector(source, make_settings())
    connector.set_engine(engine)
    return connector


def test_watermark_is_a_bind_parameter_not_string_interpolation():
    connector = sql_connector(None)
    sql, params = connector.build_statement("2026-01-05T10:00:00+00:00")
    assert ":watermark" in sql
    assert "2026-01-05" not in sql
    assert params == {"watermark": "2026-01-05T10:00:00+00:00"}

    first_sql, first_params = connector.build_statement(None)
    assert "WHERE" not in first_sql
    assert first_params == {}


def test_a_hostile_watermark_column_is_refused():
    options = dict(SQL_OPTIONS)
    options["watermark_column"] = "updated_at; DROP TABLE tickets"
    with pytest.raises(ValueError, match="plain SQL identifier"):
        SqlSourceConnector(make_source(SourceType.SQL, options), make_settings())


async def test_sql_full_load_drops_foreign_tenant_rows_and_sets_the_cursor(tmp_path):
    engine = await sql_engine(tmp_path)
    connector = sql_connector(engine)
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    assert [doc.source_uri for doc in docs] == [
        "crm://tickets/1",
        "crm://tickets/2",
        "crm://tickets/4",
    ]
    assert all(doc.tenant_id == TENANT for doc in docs)
    assert connector.performed_full_scan is True
    assert connector.cursor == "2026-01-08T13:00:00+00:00"
    # The closed ticket arrives as a tombstone.
    assert docs[2].deleted is True


async def test_sql_incremental_pass_returns_only_newer_rows(tmp_path):
    engine = await sql_engine(tmp_path)
    connector = sql_connector(engine, cursor="2026-01-06T11:00:00+00:00")
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    assert [doc.source_uri for doc in docs] == ["crm://tickets/4"]
    assert connector.performed_full_scan is False


async def test_sql_rows_carry_their_own_acl_and_rendered_body(tmp_path):
    engine = await sql_engine(tmp_path)
    connector = sql_connector(engine)
    try:
        docs = await collect(connector)
        vpn = next(doc for doc in docs if doc.source_uri == "crm://tickets/2")
        acl = await connector.resolve_acl(vpn)
        fetched = await connector.fetch(vpn)
    finally:
        await connector.close()

    assert vpn.title == "VPN access"
    assert set(acl.allowed_groups) == {"g-it", "g-sec"}
    assert acl.classification is Classification.CONFIDENTIAL
    assert acl.tenant_id == TENANT
    assert "Cannot reach the intranet" in (fetched.content_text or "")
    assert fetched.media_type == "text/markdown"
    assert fetched.sha256


async def test_sql_template_renders_the_document_body(tmp_path):
    engine = await sql_engine(tmp_path)
    options = dict(SQL_OPTIONS)
    options["template"] = "# {subject}\n\n{body}\n\nOwner group: {acl_groups}"
    source = make_source(SourceType.SQL, options)
    connector = SqlSourceConnector(source, make_settings())
    connector.set_engine(engine)
    try:
        docs = await collect(connector)
    finally:
        await connector.close()

    body = docs[0].content_text or ""
    assert body.startswith("# Broken laptop")
    assert "Owner group: g-it" in body


async def test_sql_refuses_a_synchronous_driver(tmp_path):
    from ingestion.connectors.base import ConnectorError

    monkeypatched = sql_connector(None)
    monkeypatched._engine = None

    async def fake_secret(_ref: str, _settings: Settings) -> str:
        return f"sqlite:///{tmp_path}/plain.db"

    import ingestion.connectors.sql_source as module

    original = module.resolve_secret
    module.resolve_secret = fake_secret  # type: ignore[assignment]
    try:
        with pytest.raises(ConnectorError, match="async SQLAlchemy driver"):
            await monkeypatched._get_engine()
    finally:
        module.resolve_secret = original  # type: ignore[assignment]
