"""The connector contract every ingestion source implements.

A connector is deliberately small: it enumerates descriptors, fetches one document's
bytes, and resolves that document's ACL. Everything else — delta classification,
parsing, chunking, enrichment, embedding, dedupe, writing — happens in the pipeline,
so adding a source never means re-implementing the pipeline.

```
list_documents(since) -> AsyncIterator[SourceDocument]   # descriptors, no content
fetch(doc)            -> FetchedContent                  # bytes/text + content type
resolve_acl(doc)      -> AccessControl                   # merged with source defaults
supports_delta        -> bool                            # native incremental sync
```

Two delta styles are supported. A connector with a server-side cursor (SharePoint's
``deltaLink``, a SQL watermark) advances :attr:`BaseConnector.cursor` and the pipeline
persists it. A connector without one (blob, local, HTTP) is primed with the previous
run's manifest entries via :meth:`BaseConnector.prime_delta_state` and performs
conditional requests, returning ``FetchedContent.not_modified`` when nothing changed.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from ingestion.acl import access_control_from_mapping, merge_with_source_defaults
from ragcore.errors import RagError
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import IngestManifestEntry, SourceConfig, SourceDocument
from ragcore.settings import Settings

__all__ = [
    "BaseConnector",
    "ConnectorError",
    "FetchedContent",
    "SourceConnector",
    "azure_credential",
    "guess_media_type",
    "make_document_id",
    "resolve_secret",
    "utcnow",
]

_log = get_logger(__name__)

#: Extra extension -> media type mappings the stdlib table gets wrong or lacks.
_MEDIA_TYPE_OVERRIDES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".htm": "text/html",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class ConnectorError(RagError):
    """A source could not be enumerated or fetched.

    Raised for configuration problems and unrecoverable transport failures. The
    pipeline records the document as failed and continues with the rest of the run,
    so one bad source never aborts a nightly refresh.
    """

    status_code = 502
    code = "connector_error"


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime.

    Returns:
        The current moment in UTC.
    """
    return datetime.now(UTC)


def make_document_id(tenant_id: str, source_uri: str) -> str:
    """Derive the deterministic document id for a source URI.

    The id is a pure function of tenant and URI, which is what makes the whole
    pipeline idempotent: re-ingesting the same URI addresses the same document row,
    the same manifest entry and the same Qdrant points instead of creating duplicates.

    Args:
        tenant_id: Owning tenant id.
        source_uri: Canonical source URI, unique within the tenant.

    Returns:
        A 32-character lowercase hex id.
    """
    digest = hashlib.sha256(f"{tenant_id}\x00{source_uri}".encode())
    return digest.hexdigest()[:32]


def guess_media_type(
    name: str | None, *, default: str = "application/octet-stream"
) -> str:
    """Guess a document's media type from its file name.

    Args:
        name: File or blob name, possibly None.
        default: Media type returned when nothing matches.

    Returns:
        The resolved media type.
    """
    if not name:
        return default
    lowered = name.lower()
    for suffix, media_type in _MEDIA_TYPE_OVERRIDES.items():
        if lowered.endswith(suffix):
            return media_type
    guessed, _ = mimetypes.guess_type(lowered)
    return guessed or default


@dataclass(slots=True)
class FetchedContent:
    """The payload a connector returns for one document.

    Attributes:
        content_bytes: Raw bytes for binary formats (PDF, DOCX, XLSX, PPTX).
        content_text: Decoded text for textual formats (HTML, Markdown, SQL rows).
        media_type: MIME type used to select a parser.
        etag: ETag observed on this fetch, when the source provides one.
        source_modified_at: Modification time observed on this fetch.
        size_bytes: Payload size.
        metadata: Connector extras merged into the document metadata.
        not_modified: True when a conditional request proved the content is
            unchanged. The pipeline records a skip and performs no writes.
    """

    content_bytes: bytes | None = None
    content_text: str | None = None
    media_type: str = "application/octet-stream"
    etag: str | None = None
    source_modified_at: datetime | None = None
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    not_modified: bool = False

    @property
    def sha256(self) -> str:
        """Hex SHA-256 of the payload actually fetched.

        Returns:
            The digest of the bytes, or of the UTF-8 encoded text, or "" when the
            fetch produced no payload.
        """
        if self.content_bytes is not None:
            return hashlib.sha256(self.content_bytes).hexdigest()
        if self.content_text is not None:
            return hashlib.sha256(self.content_text.encode("utf-8")).hexdigest()
        return ""


@runtime_checkable
class SourceConnector(Protocol):
    """Structural contract for an ingestion source."""

    source_type: SourceType
    supports_delta: bool

    def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Enumerate document descriptors, newest state first is not required.

        Args:
            since: Only yield documents modified at or after this moment, when the
                source can filter on it. Connectors with a native cursor ignore it
                in favour of their own delta token.

        Returns:
            An async iterator of content-free :class:`SourceDocument` descriptors.
        """
        ...

    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Fetch one document's payload.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The payload plus the ETag and modification time observed.
        """
        ...

    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Resolve the effective ACL for one document.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The item ACL merged with the source defaults.
        """
        ...

    async def close(self) -> None:
        """Release transports, credentials and connection pools."""
        ...


class BaseConnector(ABC):
    """Shared behaviour for the four concrete connectors.

    Subclasses implement :meth:`list_documents` and :meth:`fetch`; everything else
    (descriptor construction, ACL merging, conditional-request state, cursor
    bookkeeping, async context management) is inherited.
    """

    #: Whether this connector performs a genuine incremental sync rather than a
    #: full enumeration plus hash comparison.
    supports_delta: bool = False

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Initialise the connector.

        Args:
            source: The source configuration to read.
            settings: Process settings supplying limits, timeouts and endpoints.
        """
        self.source = source
        self.settings = settings
        self.source_type: SourceType = source.source_type
        self._cursor: str | None = source.cursor
        self._known: dict[str, IngestManifestEntry] = {}
        self._full_scan = True

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> Self:
        """Enter an async context.

        Returns:
            This connector.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Leave an async context, closing transports.

        Args:
            exc_type: Exception class, when the body raised.
            exc: Exception instance, when the body raised.
            tb: Traceback, when the body raised.
        """
        await self.close()

    async def close(self) -> None:
        """Release transports and credentials. The default is a no-op."""
        return None

    # ---------------------------------------------------------------- delta state
    @property
    def cursor(self) -> str | None:
        """Delta token to persist after a successful run.

        Returns:
            The connector's next cursor, or None when it has no server-side token.
        """
        return self._cursor

    @property
    def performed_full_scan(self) -> bool:
        """Whether the last enumeration saw every live document.

        Deletion detection by set difference is only sound after a full scan; an
        incremental pass must not soft-delete documents it simply did not look at.

        Returns:
            True when :meth:`list_documents` enumerated the whole source.
        """
        return self._full_scan

    def prime_delta_state(self, entries: Mapping[str, IngestManifestEntry]) -> None:
        """Give the connector the previous run's manifest entries.

        Connectors without a server-side cursor use these for conditional requests
        (``If-None-Match`` / ``If-Modified-Since``) so an unchanged document costs one
        304 instead of a full download.

        Args:
            entries: Manifest entries keyed by document id.
        """
        self._known = dict(entries)

    def known_entry(self, document_id: str) -> IngestManifestEntry | None:
        """Look up what the last run recorded for a document.

        Args:
            document_id: Deterministic document id.

        Returns:
            The manifest entry, or None when the document is new.
        """
        return self._known.get(document_id)

    def known_etag(self, document_id: str) -> str | None:
        """ETag recorded by the last successful run.

        Args:
            document_id: Deterministic document id.

        Returns:
            The stored ETag, or None.
        """
        entry = self.known_entry(document_id)
        return entry.etag if entry else None

    def known_modified_at(self, document_id: str) -> datetime | None:
        """Modification time recorded by the last successful run.

        Args:
            document_id: Deterministic document id.

        Returns:
            The stored modification time, or None.
        """
        entry = self.known_entry(document_id)
        return entry.source_modified_at if entry else None

    # -------------------------------------------------------------- descriptors
    def descriptor(
        self,
        *,
        source_uri: str,
        document_id: str | None = None,
        title: str = "",
        filename: str | None = None,
        media_type: str | None = None,
        etag: str | None = None,
        source_modified_at: datetime | None = None,
        size_bytes: int = 0,
        author: str | None = None,
        deleted: bool = False,
        access_control: AccessControl | None = None,
        content_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SourceDocument:
        """Build a content-free descriptor for one source item.

        Args:
            source_uri: Canonical URI, unique within the tenant.
            document_id: Explicit document id. Defaults to
                ``make_document_id(tenant_id, source_uri)``. Connectors whose
                display URI can change while the item identity does not (SharePoint
                renames, for example) pass a stable id derived from the item id so a
                later deletion still addresses the same document.
            title: Document title; defaults to the file name when empty.
            filename: Original file name.
            media_type: MIME type; guessed from the file name when omitted.
            etag: Source ETag.
            source_modified_at: Source modification time.
            size_bytes: Payload size reported by the listing.
            author: Author, when the listing reports one.
            deleted: True when the listing reports the item was removed.
            access_control: Provisional ACL. Defaults to the source defaults;
                :meth:`resolve_acl` refines it before indexing.
            content_text: Text already materialised by the listing (SQL rows).
            metadata: Connector extras, including any raw ACL metadata.

        Returns:
            A :class:`SourceDocument` with no binary payload attached.
        """
        name = filename or source_uri.rsplit("/", 1)[-1]
        return SourceDocument(
            document_id=document_id
            or make_document_id(self.source.tenant_id, source_uri),
            tenant_id=self.source.tenant_id,
            source_id=self.source.source_id,
            source_type=self.source_type,
            source_uri=source_uri,
            title=title or name,
            content_text=content_text,
            media_type=media_type or guess_media_type(name),
            filename=name,
            size_bytes=max(0, size_bytes),
            etag=etag,
            source_modified_at=source_modified_at,
            access_control=access_control or self.source.default_access_control(),
            doc_type=self.source.doc_type,
            tags=list(self.source.tags),
            author=author,
            language=self.source.language,
            deleted=deleted,
            metadata=dict(metadata or {}),
        )

    def apply_fetched(
        self, doc: SourceDocument, fetched: FetchedContent
    ) -> SourceDocument:
        """Merge a fetch result into its descriptor.

        Args:
            doc: The descriptor.
            fetched: Payload returned by :meth:`fetch`.

        Returns:
            A copy of the descriptor carrying the payload, its hash, and the ETag and
            modification time actually observed on the wire.
        """
        metadata = dict(doc.metadata)
        metadata.update(fetched.metadata)
        return doc.model_copy(
            update={
                "content_bytes": fetched.content_bytes,
                "content_text": fetched.content_text,
                "media_type": fetched.media_type or doc.media_type,
                "size_bytes": fetched.size_bytes or doc.size_bytes,
                "etag": fetched.etag or doc.etag,
                "source_modified_at": fetched.source_modified_at
                or doc.source_modified_at,
                "content_sha256": fetched.sha256,
                "metadata": metadata,
                "fetched_at": utcnow(),
            }
        )

    # ------------------------------------------------------------------- the ACL
    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Resolve the effective ACL for a document.

        The default reads ACL keys out of ``doc.metadata`` (which connectors populate
        from sidecars, blob metadata, index tags or SQL columns) and merges them with
        the source defaults. SharePoint overrides this to call Graph ``/permissions``.

        Args:
            doc: The descriptor.

        Returns:
            The effective :class:`AccessControl`.
        """
        item = access_control_from_mapping(doc.metadata, self.source)
        return merge_with_source_defaults(item, self.source)

    # ------------------------------------------------------------------ abstract
    @abstractmethod
    def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Enumerate document descriptors.

        Args:
            since: Lower bound on modification time, when the source supports it.

        Returns:
            An async iterator of content-free descriptors.
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Fetch one document's payload.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The fetched payload.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------- guards
    def within_size_limit(self, size_bytes: int) -> bool:
        """Whether a document is small enough to ingest.

        Args:
            size_bytes: Reported payload size.

        Returns:
            True when the size is within ``settings.ingest_max_document_bytes``.
            Zero (unknown size) is allowed through and checked again after fetch.
        """
        return size_bytes <= self.settings.ingest_max_document_bytes


def azure_credential(settings: Settings) -> Any:
    """Build an async ``DefaultAzureCredential`` for Azure data-plane calls.

    Args:
        settings: Process settings; ``azure_client_id`` selects a user-assigned
            managed identity when set.

    Returns:
        An ``azure.identity.aio.DefaultAzureCredential``.

    Raises:
        ConnectorError: If ``azure-identity`` is not installed.
    """
    try:
        from azure.identity.aio import DefaultAzureCredential
    except ImportError as exc:  # pragma: no cover - depends on optional install
        msg = "azure-identity is required for Azure connectors"
        raise ConnectorError(msg) from exc
    if settings.azure_client_id:
        return DefaultAzureCredential(
            managed_identity_client_id=settings.azure_client_id
        )
    return DefaultAzureCredential()


async def resolve_secret(ref: str | None, settings: Settings) -> str | None:
    """Resolve a secret *reference* into its value.

    A source config never stores a secret, only its name. That name is looked up in
    Key Vault when ``azure_key_vault_url`` is configured, and otherwise in the process
    environment, which is how local runs and CI supply the same value.

    Args:
        ref: Key Vault secret name or environment variable name. None returns None.
        settings: Process settings.

    Returns:
        The secret value, or None when it cannot be resolved.
    """
    if not ref:
        return None
    if settings.azure_key_vault_url:
        try:
            from azure.keyvault.secrets.aio import SecretClient
        except ImportError:  # pragma: no cover - depends on optional install
            _log.warning("secret.keyvault_sdk_missing", ref=ref)
        else:
            credential = azure_credential(settings)
            try:
                async with (
                    credential,
                    SecretClient(
                        vault_url=settings.azure_key_vault_url, credential=credential
                    ) as client,
                ):
                    secret = await client.get_secret(ref)
                    if secret.value:
                        return secret.value
            except Exception as exc:
                # Key Vault is unreachable or the secret is absent: fall through to
                # the environment rather than failing the whole run.
                _log.warning("secret.keyvault_lookup_failed", ref=ref, error=str(exc))
    value = os.environ.get(ref)
    if value:
        return value
    _log.warning("secret.unresolved", ref=ref)
    return None
