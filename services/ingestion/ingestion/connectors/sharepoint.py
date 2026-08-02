"""SharePoint / OneDrive connector built on the Microsoft Graph delta query.

Incremental sync uses Graph's own change feed rather than re-listing the drive:

1. the first run calls ``GET /drives/{drive_id}/root/delta`` (or
   ``/root:/{folder}:/delta`` when ``folder_path`` is set),
2. pages are followed via ``@odata.nextLink``,
3. the terminal ``@odata.deltaLink`` is persisted as the source cursor, and
4. the next run resumes from that link and receives only what changed — including
   tombstones, as items carrying a ``deleted`` facet.

Because a delta pass deliberately does *not* enumerate everything, the connector
reports ``performed_full_scan == False`` on a resumed run, which stops the pipeline
from mistaking "not mentioned" for "deleted".

Identity is stable across renames: ``document_id`` is derived from
``sharepoint://{drive_id}/items/{item_id}`` while ``source_uri`` carries the item's
``webUrl`` for citations. A rename therefore updates a document rather than orphaning
one, and a tombstone (which carries only the item id) still addresses the right one.

Authentication is MSAL client credentials when a client secret reference is
configured, falling back to ``DefaultAzureCredential`` (federated / managed identity)
so a deployment can avoid holding a secret at all.

Permissions are real: ``GET /drives/{drive_id}/items/{item_id}/permissions`` is mapped
onto Entra group and user object IDs by
:func:`ingestion.acl.access_control_from_graph_permissions`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import anyio
import httpx

from ingestion.acl import (
    access_control_from_graph_permissions,
    merge_with_source_defaults,
)
from ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    FetchedContent,
    azure_credential,
    guess_media_type,
    make_document_id,
    resolve_secret,
    utcnow,
)
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import SourceConfig, SourceDocument
from ragcore.settings import Settings

__all__ = ["SharePointConnector"]

_log = get_logger(__name__)

#: Fields requested from the delta endpoint. Narrowing the projection keeps the
#: change feed small and avoids paging on metadata the pipeline never reads.
_DELTA_SELECT = (
    "id,name,eTag,cTag,size,webUrl,file,folder,deleted,createdBy,lastModifiedBy,"
    "lastModifiedDateTime,createdDateTime,parentReference"
)

#: Seconds of head-room applied to a cached access token's expiry.
_TOKEN_SKEW_SECONDS = 120


class SharePointConnector(BaseConnector):
    """Ingest a SharePoint document library or OneDrive via Microsoft Graph.

    Options: ``site_id`` and ``drive_id`` (required), ``folder_path`` (optional),
    plus the authentication references ``client_secret_ref`` (Key Vault secret name
    or environment variable name), ``client_id`` and ``tenant_id``, which default to
    ``settings.entra_client_id`` / ``settings.entra_tenant_id``.
    """

    supports_delta = True

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Resolve drive coordinates and authentication references.

        Args:
            source: Source configuration.
            settings: Process settings supplying the Graph endpoint, scope and
                timeout.

        Raises:
            ValueError: If ``site_id`` or ``drive_id`` is missing.
        """
        super().__init__(source, settings)
        self.source_type = SourceType.SHAREPOINT
        self.site_id = str(source.require_option("site_id"))
        self.drive_id = str(source.require_option("drive_id"))
        self.folder_path = str(source.option("folder_path") or "").strip("/")
        self.client_id = str(
            source.option("client_id") or settings.entra_client_id or ""
        )
        self.tenant_id_directory = str(
            source.option("tenant_id") or settings.entra_tenant_id or ""
        )
        self.client_secret_ref = source.option("client_secret_ref")
        self._base = settings.azure_graph_base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._credential: Any = None
        self._token: str | None = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=UTC)

    # --------------------------------------------------------------------- http
    def _http(self) -> httpx.AsyncClient:
        """Build (once) the HTTP client used for Graph calls.

        Returns:
            A shared :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.ingest_http_timeout_seconds,
                headers={"User-Agent": self.settings.ingest_http_user_agent},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and any credential owned by this connector."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None

    # --------------------------------------------------------------------- auth
    async def _access_token(self) -> str:
        """Acquire (and cache) an app-only Graph access token.

        MSAL client credentials are used when a secret reference resolves; otherwise
        ``DefaultAzureCredential`` provides a federated or managed-identity token.

        Returns:
            A bearer token valid for ``settings.azure_graph_scope``.

        Raises:
            ConnectorError: If no credential path is available or the token request
                fails.
        """
        if self._token and utcnow() < self._token_expires_at:
            return self._token

        secret = await self._resolve_client_secret()
        if secret and self.client_id and self.tenant_id_directory:
            token, expires_in = await self._token_via_msal(secret)
        elif self.settings.azure_use_managed_identity:
            token, expires_in = await self._token_via_credential()
        else:
            msg = (
                "sharepoint source needs options['client_secret_ref'] with "
                "client_id/tenant_id, or RAG_AZURE_USE_MANAGED_IDENTITY=true"
            )
            raise ConnectorError(msg, detail={"source_id": self.source.source_id})

        self._token = token
        self._token_expires_at = utcnow() + timedelta(
            seconds=max(60, expires_in - _TOKEN_SKEW_SECONDS)
        )
        return token

    async def _resolve_client_secret(self) -> str | None:
        """Resolve the Graph client secret from its reference.

        Returns:
            The secret value, or None when no reference is configured.
        """
        if not self.client_secret_ref:
            return None
        return await resolve_secret(str(self.client_secret_ref), self.settings)

    async def _token_via_msal(self, secret: str) -> tuple[str, int]:
        """Run the MSAL client-credentials flow.

        MSAL is synchronous, so the call is offloaded to a worker thread rather than
        blocking the event loop.

        Args:
            secret: The application client secret.

        Returns:
            A ``(token, expires_in_seconds)`` pair.

        Raises:
            ConnectorError: If MSAL is unavailable or the directory refuses the
                request.
        """
        try:
            import msal
        except ImportError as exc:  # pragma: no cover - optional install
            msg = "msal is required for the SharePoint connector"
            raise ConnectorError(msg) from exc

        authority = (
            f"{self.settings.entra_authority_host.rstrip('/')}/"
            f"{self.tenant_id_directory}"
        )

        def _acquire() -> dict[str, Any]:
            app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                authority=authority,
                client_credential=secret,
            )
            return app.acquire_token_for_client(
                scopes=[self.settings.azure_graph_scope]
            )

        result = await anyio.to_thread.run_sync(_acquire)
        token = result.get("access_token")
        if not token:
            msg = "MSAL client-credentials flow returned no access token"
            raise ConnectorError(
                msg,
                detail={
                    "source_id": self.source.source_id,
                    "error": str(result.get("error") or "unknown"),
                },
            )
        return str(token), int(result.get("expires_in") or 3600)

    async def _token_via_credential(self) -> tuple[str, int]:
        """Fetch a Graph token through ``DefaultAzureCredential``.

        Returns:
            A ``(token, expires_in_seconds)`` pair.

        Raises:
            ConnectorError: If the credential cannot issue a token.
        """
        if self._credential is None:
            self._credential = azure_credential(self.settings)
        try:
            token = await self._credential.get_token(self.settings.azure_graph_scope)
        except Exception as exc:
            msg = "managed-identity token request for Microsoft Graph failed"
            raise ConnectorError(
                msg, detail={"source_id": self.source.source_id, "error": str(exc)}
            ) from exc
        expires_in = int(token.expires_on - utcnow().timestamp())
        return str(token.token), max(60, expires_in)

    # -------------------------------------------------------------------- graph
    async def graph_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Issue an authenticated Graph GET with throttling-aware retries.

        Args:
            url: Absolute URL, or a path relative to the Graph base URL.
            params: Optional query parameters.

        Returns:
            The decoded JSON body.

        Raises:
            ConnectorError: If the request keeps failing after
                ``settings.ingest_retry_attempts`` retries, or returns a
                non-retryable error status.
        """
        target = url if url.startswith("http") else f"{self._base}/{url.lstrip('/')}"
        attempts = self.settings.ingest_retry_attempts + 1
        last_error = ""
        for attempt in range(attempts):
            token = await self._access_token()
            response = await self._http().get(
                target,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == httpx.codes.OK:
                return dict(response.json())
            if response.status_code in {
                httpx.codes.TOO_MANY_REQUESTS,
                httpx.codes.INTERNAL_SERVER_ERROR,
                httpx.codes.BAD_GATEWAY,
                httpx.codes.SERVICE_UNAVAILABLE,
                httpx.codes.GATEWAY_TIMEOUT,
            }:
                delay = _retry_after(response, attempt)
                last_error = f"HTTP {response.status_code}"
                _log.warning(
                    "graph.retrying",
                    status=response.status_code,
                    attempt=attempt + 1,
                    delay_seconds=delay,
                )
                await anyio.sleep(delay)
                continue
            if response.status_code == httpx.codes.UNAUTHORIZED:
                # Force a fresh token once, then fail rather than looping.
                self._token = None
                last_error = "HTTP 401"
                if attempt + 1 < attempts:
                    continue
            msg = f"Microsoft Graph request failed with HTTP {response.status_code}"
            raise ConnectorError(
                msg,
                detail={"source_id": self.source.source_id, "url": target},
            )
        msg = f"Microsoft Graph request gave up after {attempts} attempts"
        raise ConnectorError(
            msg, detail={"source_id": self.source.source_id, "error": last_error}
        )

    def _delta_url(self) -> str:
        """Build the delta URL for the first page of an enumeration.

        Returns:
            The persisted ``deltaLink`` when one exists, otherwise the drive-root
            (or folder) delta endpoint with a narrowed projection.
        """
        if self._cursor:
            return self._cursor
        if self.folder_path:
            root = f"drives/{self.drive_id}/root:/{self.folder_path}:/delta"
        else:
            root = f"drives/{self.drive_id}/root/delta"
        return f"{self._base}/{root}?$select={_DELTA_SELECT}"

    def _stable_uri(self, item_id: str) -> str:
        """Rename-proof identity URI for a drive item.

        Args:
            item_id: Graph drive-item id.

        Returns:
            A ``sharepoint://`` URI used only to derive the document id.
        """
        return f"sharepoint://{self.drive_id}/items/{item_id}"

    async def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Walk the Graph delta feed, yielding changed and deleted items.

        Args:
            since: Ignored — Graph's ``deltaLink`` is a stronger cursor than a
                timestamp, and mixing the two would silently drop changes whose
                ``lastModifiedDateTime`` is older than the boundary.

        Yields:
            Content-free descriptors. Tombstones are yielded with ``deleted=True``.
        """
        resumed = bool(self._cursor)
        self._full_scan = not resumed
        url: str | None = self._delta_url()
        pages = 0

        while url:
            payload = await self.graph_get(url)
            pages += 1
            for item in payload.get("value") or []:
                descriptor = self._descriptor_for(item)
                if descriptor is not None:
                    yield descriptor
            next_link = payload.get("@odata.nextLink")
            delta_link = payload.get("@odata.deltaLink")
            if delta_link:
                self._cursor = str(delta_link)
            url = str(next_link) if next_link else None

        _log.info(
            "sharepoint.delta_complete",
            source_id=self.source.source_id,
            pages=pages,
            resumed=resumed,
        )

    def _descriptor_for(self, item: dict[str, Any]) -> SourceDocument | None:
        """Convert one Graph drive item into a descriptor.

        Args:
            item: A member of the delta feed's ``value`` array.

        Returns:
            The descriptor, or None for folders and for items too large to ingest.
        """
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            return None
        document_id = make_document_id(self.source.tenant_id, self._stable_uri(item_id))

        if item.get("deleted") is not None:
            return self.descriptor(
                source_uri=self._stable_uri(item_id),
                document_id=document_id,
                filename=str(item.get("name") or item_id),
                deleted=True,
                metadata={"item_id": item_id},
            )
        if item.get("folder") is not None:
            return None

        name = str(item.get("name") or item_id)
        size = int(item.get("size") or 0)
        if not self.within_size_limit(size):
            _log.warning(
                "sharepoint.document_too_large",
                source_id=self.source.source_id,
                size_bytes=size,
            )
            return None

        return self.descriptor(
            source_uri=str(item.get("webUrl") or self._stable_uri(item_id)),
            document_id=document_id,
            title=name.rsplit(".", 1)[0].replace("_", " ").strip(),
            filename=name,
            media_type=_item_media_type(item, name),
            etag=str(item.get("cTag") or item.get("eTag") or "") or None,
            source_modified_at=_parse_graph_datetime(item.get("lastModifiedDateTime")),
            size_bytes=size,
            author=_identity_display_name(item.get("lastModifiedBy"))
            or _identity_display_name(item.get("createdBy")),
            metadata={
                "item_id": item_id,
                "site_id": self.site_id,
                "drive_id": self.drive_id,
                "web_url": item.get("webUrl"),
                "created_at_source": item.get("createdDateTime"),
            },
        )

    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Download one drive item's content.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The payload, or a ``not_modified`` result when the cTag has not moved.

        Raises:
            ConnectorError: If the download fails.
        """
        known = self.known_etag(doc.document_id)
        if known and doc.etag and known == doc.etag:
            return FetchedContent(
                media_type=doc.media_type, etag=doc.etag, not_modified=True
            )
        item_id = str(doc.metadata.get("item_id") or "")
        if not item_id:
            msg = "sharepoint descriptor is missing its item id"
            raise ConnectorError(msg, detail={"document_id": doc.document_id})

        token = await self._access_token()
        url = f"{self._base}/drives/{self.drive_id}/items/{item_id}/content"
        response = await self._http().get(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code != httpx.codes.OK:
            msg = f"SharePoint content download failed with {response.status_code}"
            raise ConnectorError(
                msg,
                detail={
                    "document_id": doc.document_id,
                    "source_id": self.source.source_id,
                },
            )
        payload = response.content
        media_type = (
            response.headers.get("content-type", "").split(";")[0].strip()
            or doc.media_type
        )
        return FetchedContent(
            content_bytes=payload,
            media_type=media_type,
            etag=doc.etag,
            source_modified_at=doc.source_modified_at,
            size_bytes=len(payload),
        )

    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Resolve an item's ACL from Graph ``/permissions``.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The item ACL merged with the source defaults. When
            ``inherit_source_permissions`` is False no Graph call is made and the
            source defaults are used verbatim.
        """
        if not self.source.inherit_source_permissions:
            return self.source.default_access_control()
        item_id = str(doc.metadata.get("item_id") or "")
        if not item_id:
            return self.source.default_access_control()
        try:
            payload = await self.graph_get(
                f"drives/{self.drive_id}/items/{item_id}/permissions"
            )
        except ConnectorError as exc:
            # Fail closed: without a permission list we fall back to the source
            # defaults rather than indexing the item as unrestricted.
            _log.warning(
                "sharepoint.permissions_unavailable",
                source_id=self.source.source_id,
                error=exc.message,
            )
            return self.source.default_access_control()
        item = access_control_from_graph_permissions(
            [p for p in (payload.get("value") or []) if isinstance(p, dict)],
            self.source,
        )
        return merge_with_source_defaults(item, self.source)


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """Compute the backoff delay for a throttled Graph response.

    Args:
        response: The throttled response.
        attempt: Zero-based attempt index.

    Returns:
        Seconds to wait: the ``Retry-After`` header when present, otherwise an
        exponential backoff.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    return float(2**attempt)


def _item_media_type(item: dict[str, Any], name: str) -> str:
    """Resolve a drive item's media type.

    Args:
        item: The Graph drive item.
        name: File name, used as the fallback signal.

    Returns:
        The MIME type reported by Graph, else one guessed from the name.
    """
    file_facet = item.get("file")
    if isinstance(file_facet, dict):
        mime = str(file_facet.get("mimeType") or "").strip()
        if mime:
            return mime
    return guess_media_type(name)


def _identity_display_name(identity_set: Any) -> str | None:
    """Extract a display name from a Graph ``identitySet``.

    Args:
        identity_set: A Graph ``identitySet`` object, or anything else.

    Returns:
        The user display name, or None.
    """
    if not isinstance(identity_set, dict):
        return None
    user = identity_set.get("user")
    if isinstance(user, dict):
        name = user.get("displayName")
        if name:
            return str(name)
    return None


def _parse_graph_datetime(value: Any) -> datetime | None:
    """Parse a Graph ISO-8601 timestamp.

    Args:
        value: Raw timestamp string such as ``"2026-01-05T09:12:44Z"``.

    Returns:
        An aware UTC datetime, or None when the value is absent or unparseable.
    """
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
