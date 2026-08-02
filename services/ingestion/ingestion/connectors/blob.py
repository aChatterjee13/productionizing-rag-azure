"""Azure Blob Storage and local-filesystem connectors.

These two share one class family on purpose: the local connector is what makes the
whole pipeline runnable and testable without a cloud account, and it must behave
identically to the blob connector so a bug found locally is the same bug in Azure.

Delta strategy for both: the listing already reports an ETag (blob ETag, or
``size-mtime_ns`` for a file), so an unchanged item is skipped without any download.
When the ETag *has* moved, the payload is fetched and the pipeline compares content
hashes — that is what makes an unchanged-but-touched file a skip rather than a
re-embed.

ACL resolution order, most specific first:

1. a ``<name>.acl.json`` sidecar next to the document,
2. blob metadata and blob index tags (tags win for the same key),
3. the source config defaults.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import anyio

from ingestion.acl import (
    SIDECAR_SUFFIX,
    access_control_from_metadata,
    access_control_from_sidecar,
    merge_with_source_defaults,
    sidecar_name_for,
)
from ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    FetchedContent,
    azure_credential,
    guess_media_type,
)
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import SourceConfig, SourceDocument
from ragcore.settings import Settings

__all__ = ["AzureBlobConnector", "LocalFilesystemConnector"]

_log = get_logger(__name__)

#: Text media types decoded to ``content_text`` instead of kept as bytes.
_TEXT_MEDIA_PREFIXES: tuple[str, ...] = ("text/",)
_TEXT_MEDIA_TYPES: frozenset[str] = frozenset(
    {"application/json", "application/xml", "application/xhtml+xml"}
)


def _is_text(media_type: str) -> bool:
    """Whether a media type should be decoded to text at fetch time.

    Args:
        media_type: MIME type of the payload.

    Returns:
        True for textual formats the parsers consume as ``str``.
    """
    if media_type.startswith(_TEXT_MEDIA_PREFIXES):
        return True
    return media_type in _TEXT_MEDIA_TYPES


def _decode(payload: bytes, media_type: str) -> FetchedContent:
    """Wrap raw bytes as a :class:`FetchedContent`, decoding text formats.

    Args:
        payload: Raw bytes read from the source.
        media_type: MIME type used to decide bytes vs text.

    Returns:
        The fetched content with exactly one of bytes/text populated.
    """
    if _is_text(media_type):
        return FetchedContent(
            content_text=payload.decode("utf-8", errors="replace"),
            media_type=media_type,
            size_bytes=len(payload),
        )
    return FetchedContent(
        content_bytes=payload, media_type=media_type, size_bytes=len(payload)
    )


class _BlobFamilyConnector(BaseConnector):
    """Behaviour shared by the blob and local-filesystem connectors."""

    supports_delta = True

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Initialise shared sidecar bookkeeping.

        Args:
            source: Source configuration.
            settings: Process settings.
        """
        super().__init__(source, settings)
        self._sidecars: set[str] = set()

    @staticmethod
    def _is_sidecar(name: str) -> bool:
        """Whether a name is an ACL sidecar rather than a document.

        Args:
            name: Blob or file name.

        Returns:
            True when the name ends with the sidecar suffix.
        """
        return name.endswith(SIDECAR_SUFFIX)

    async def _read_sidecar(self, name: str) -> bytes | None:
        """Read the sidecar for a document, when one exists.

        Args:
            name: Document name (not the sidecar name).

        Returns:
            The sidecar bytes, or None when absent.
        """
        raise NotImplementedError

    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Resolve the ACL from sidecar, then metadata/tags, then source defaults.

        Args:
            doc: Descriptor produced by ``list_documents``.

        Returns:
            The effective :class:`AccessControl`.
        """
        name = str(doc.metadata.get("blob_name") or doc.filename or "")
        item: AccessControl | None = None
        if name and sidecar_name_for(name) in self._sidecars:
            raw = await self._read_sidecar(name)
            if raw is not None:
                item = access_control_from_sidecar(raw, self.source)
        if item is None:
            item = access_control_from_metadata(
                doc.metadata.get("blob_metadata"),
                doc.metadata.get("blob_tags"),
                self.source,
            )
        return merge_with_source_defaults(item, self.source)


class LocalFilesystemConnector(_BlobFamilyConnector):
    """Ingest documents from a directory tree.

    Options: ``root`` (required), ``include_globs`` (default ``["**/*"]``) and
    ``exclude_globs``. Globs are matched against the POSIX path relative to ``root``
    with :meth:`pathlib.PurePath.full_match`, so ``**`` spans zero or more directory
    segments and ``"**/*.pdf"`` matches both ``report.pdf`` and ``hr/2026/report.pdf``.
    """

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Resolve the root directory and glob filters.

        Args:
            source: Source configuration; ``options["root"]`` is required.
            settings: Process settings; ``ingest_local_root`` is the fallback root.
        """
        super().__init__(source, settings)
        self.source_type = SourceType.LOCAL
        root = source.option("root") or settings.ingest_local_root
        self.root = Path(str(root)).expanduser()
        self.include_globs: list[str] = [
            str(pattern) for pattern in (source.option("include_globs") or ["**/*"])
        ]
        self.exclude_globs: list[str] = [
            str(pattern) for pattern in (source.option("exclude_globs") or [])
        ]

    def _matches(self, relative: str) -> bool:
        """Apply the include/exclude glob filters.

        Args:
            relative: Path relative to :attr:`root`, POSIX separators.

        Returns:
            True when the path should be ingested.
        """
        candidate = PurePosixPath(relative)
        if not any(candidate.full_match(pattern) for pattern in self.include_globs):
            return False
        return not any(candidate.full_match(pattern) for pattern in self.exclude_globs)

    def _uri(self, path: Path) -> str:
        """Build the canonical URI for a local file.

        Args:
            path: Absolute path to the file.

        Returns:
            A ``file://`` URI, stable across runs for the same file.
        """
        return path.resolve().as_uri()

    async def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Walk the tree and yield one descriptor per matching file.

        Args:
            since: Skip files whose modification time is older than this.

        Yields:
            Content-free descriptors.

        Raises:
            ConnectorError: If the configured root does not exist.
        """
        if not self.root.exists():
            msg = f"local source root does not exist: {self.root}"
            raise ConnectorError(msg, detail={"source_id": self.source.source_id})

        self._full_scan = since is None
        paths = sorted(path for path in self.root.rglob("*") if path.is_file())
        self._sidecars = {
            path.relative_to(self.root).as_posix()
            for path in paths
            if self._is_sidecar(path.name)
        }

        for path in paths:
            relative = path.relative_to(self.root).as_posix()
            if self._is_sidecar(path.name) or not self._matches(relative):
                continue
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if since is not None and modified < since:
                continue
            if not self.within_size_limit(stat.st_size):
                _log.warning(
                    "local.document_too_large",
                    source_id=self.source.source_id,
                    size_bytes=stat.st_size,
                )
                continue
            yield self.descriptor(
                source_uri=self._uri(path),
                title=path.stem.replace("_", " ").replace("-", " ").strip(),
                filename=path.name,
                etag=f"{stat.st_size}-{stat.st_mtime_ns}",
                source_modified_at=modified,
                size_bytes=stat.st_size,
                metadata={"blob_name": relative, "local_path": str(path)},
            )

    async def _read_sidecar(self, name: str) -> bytes | None:
        """Read a sidecar file from disk.

        Args:
            name: Document path relative to :attr:`root`.

        Returns:
            The sidecar bytes, or None when it disappeared between listing and read.
        """
        path = self.root / sidecar_name_for(name)
        if not await anyio.to_thread.run_sync(path.is_file):
            return None
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Read one file, skipping the read when the ETag has not moved.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The payload, or a ``not_modified`` result.

        Raises:
            ConnectorError: If the file vanished between listing and fetch.
        """
        known = self.known_etag(doc.document_id)
        if known and doc.etag and known == doc.etag:
            return FetchedContent(
                media_type=doc.media_type, etag=doc.etag, not_modified=True
            )
        path = Path(str(doc.metadata.get("local_path") or ""))
        if not await anyio.to_thread.run_sync(path.is_file):
            msg = "local file disappeared between listing and fetch"
            raise ConnectorError(msg, detail={"document_id": doc.document_id})
        payload = await anyio.to_thread.run_sync(path.read_bytes)
        fetched = _decode(payload, guess_media_type(doc.filename))
        fetched.etag = doc.etag
        fetched.source_modified_at = doc.source_modified_at
        return fetched


class AzureBlobConnector(_BlobFamilyConnector):
    """Ingest documents from an Azure Blob Storage container.

    Authentication prefers ``DefaultAzureCredential`` (managed identity in Azure,
    developer credentials locally); a connection string is used only when one is
    configured, which is the local-emulator path.

    Options: ``container`` (required), ``prefix`` and ``account_url``.
    """

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Resolve container, prefix and endpoint.

        Args:
            source: Source configuration; ``options["container"]`` is required.
            settings: Process settings supplying the account URL or connection
                string and the size limit.
        """
        super().__init__(source, settings)
        self.source_type = SourceType.BLOB
        self.container = str(
            source.option("container") or settings.azure_blob_container
        )
        self.prefix = str(source.option("prefix") or "")
        self.account_url = (
            source.option("account_url") or settings.azure_blob_account_url
        )
        self._service: Any = None
        self._credential: Any = None

    async def _client(self) -> Any:
        """Build (once) the async container client.

        Returns:
            An ``azure.storage.blob.aio.ContainerClient``.

        Raises:
            ConnectorError: If the SDK is missing or no endpoint is configured.
        """
        if self._service is None:
            try:
                from azure.storage.blob.aio import BlobServiceClient
            except ImportError as exc:  # pragma: no cover - optional install
                msg = "azure-storage-blob is required for the blob connector"
                raise ConnectorError(msg) from exc
            if self.settings.azure_blob_connection_string:
                self._service = BlobServiceClient.from_connection_string(
                    self.settings.azure_blob_connection_string
                )
            elif self.account_url:
                self._credential = azure_credential(self.settings)
                self._service = BlobServiceClient(
                    account_url=str(self.account_url), credential=self._credential
                )
            else:
                msg = (
                    "blob source needs options['account_url'], "
                    "RAG_AZURE_BLOB_ACCOUNT_URL or RAG_AZURE_BLOB_CONNECTION_STRING"
                )
                raise ConnectorError(msg, detail={"source_id": self.source.source_id})
        return self._service.get_container_client(self.container)

    async def close(self) -> None:
        """Close the service client and any credential it owns."""
        if self._service is not None:
            await self._service.close()
            self._service = None
        if self._credential is not None:
            await self._credential.close()
            self._credential = None

    def _uri(self, name: str) -> str:
        """Build the canonical URI for a blob.

        Args:
            name: Blob name within the container.

        Returns:
            An ``https://.../container/name`` URI, or a ``blob://`` form when no
            account URL is configured (connection-string/emulator runs).
        """
        if self.account_url:
            return f"{str(self.account_url).rstrip('/')}/{self.container}/{name}"
        return f"blob://{self.container}/{name}"

    async def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """List blobs under the configured prefix.

        Args:
            since: Skip blobs whose ``last_modified`` is older than this.

        Yields:
            Content-free descriptors carrying blob metadata and index tags.
        """
        container = await self._client()
        self._full_scan = since is None
        self._sidecars = set()
        pending: list[SourceDocument] = []

        async for blob in container.list_blobs(
            name_starts_with=self.prefix or None, include=["metadata", "tags"]
        ):
            name = str(blob.name)
            if self._is_sidecar(name):
                self._sidecars.add(name)
                continue
            modified = _as_utc(getattr(blob, "last_modified", None))
            if since is not None and modified is not None and modified < since:
                continue
            size = int(getattr(blob, "size", 0) or 0)
            if not self.within_size_limit(size):
                _log.warning(
                    "blob.document_too_large",
                    source_id=self.source.source_id,
                    size_bytes=size,
                )
                continue
            metadata = dict(getattr(blob, "metadata", None) or {})
            tags = dict(getattr(blob, "tags", None) or {})
            pending.append(
                self.descriptor(
                    source_uri=self._uri(name),
                    title=str(metadata.get("title") or "").strip(),
                    filename=name.rsplit("/", 1)[-1],
                    etag=_clean_etag(getattr(blob, "etag", None)),
                    source_modified_at=modified,
                    size_bytes=size,
                    author=metadata.get("author"),
                    metadata={
                        "blob_name": name,
                        "blob_metadata": metadata,
                        "blob_tags": tags,
                    },
                )
            )

        # Sidecars are only known after the full listing, so descriptors are emitted
        # afterwards; resolve_acl then sees a complete sidecar index.
        for doc in pending:
            yield doc

    async def _read_sidecar(self, name: str) -> bytes | None:
        """Download the sidecar blob for a document.

        Args:
            name: Blob name of the document.

        Returns:
            The sidecar bytes, or None when it could not be read.
        """
        container = await self._client()
        try:
            downloader = await container.download_blob(sidecar_name_for(name))
            return await downloader.readall()
        except Exception as exc:
            _log.warning(
                "blob.sidecar_read_failed",
                source_id=self.source.source_id,
                error=str(exc),
            )
            return None

    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Download one blob, skipping the download when the ETag has not moved.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The payload, or a ``not_modified`` result.

        Raises:
            ConnectorError: If the download fails.
        """
        known = self.known_etag(doc.document_id)
        if known and doc.etag and known == doc.etag:
            return FetchedContent(
                media_type=doc.media_type, etag=doc.etag, not_modified=True
            )
        container = await self._client()
        name = str(doc.metadata.get("blob_name") or "")
        try:
            downloader = await container.download_blob(name)
            payload = await downloader.readall()
        except Exception as exc:
            msg = "blob download failed"
            raise ConnectorError(
                msg, detail={"document_id": doc.document_id, "error": str(exc)}
            ) from exc
        media_type = guess_media_type(name, default=doc.media_type)
        fetched = _decode(bytes(payload), media_type)
        properties = getattr(downloader, "properties", None)
        fetched.etag = _clean_etag(getattr(properties, "etag", None)) or doc.etag
        fetched.source_modified_at = (
            _as_utc(getattr(properties, "last_modified", None))
            or doc.source_modified_at
        )
        return fetched


def _clean_etag(value: Any) -> str | None:
    """Normalise an ETag by stripping quotes and weak-validator markers.

    Args:
        value: Raw ETag from Azure or an HTTP header.

    Returns:
        The bare ETag value, or None.
    """
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("W/"):
        text = text[2:]
    return text.strip('"') or None


def _as_utc(value: Any) -> datetime | None:
    """Coerce a source timestamp into an aware UTC datetime.

    Args:
        value: A datetime, or anything else.

    Returns:
        The UTC datetime, or None when the value is not a datetime.
    """
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
