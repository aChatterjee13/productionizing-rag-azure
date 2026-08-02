"""Connector registry: one factory, four real implementations.

``get_connector`` is the only place that maps a
:class:`~ragcore.models.document.SourceType` onto a connector class, so the pipeline
never imports a concrete connector and a new source type is a one-line change here.
"""

from __future__ import annotations

from ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    FetchedContent,
    SourceConnector,
    make_document_id,
)
from ingestion.connectors.blob import AzureBlobConnector, LocalFilesystemConnector
from ingestion.connectors.http_crawler import HttpCrawlerConnector
from ingestion.connectors.sharepoint import SharePointConnector
from ingestion.connectors.sql_source import SqlSourceConnector
from ragcore.errors import ConfigError
from ragcore.models.chunk import SourceType
from ragcore.models.document import SourceConfig
from ragcore.settings import Settings

__all__ = [
    "AzureBlobConnector",
    "BaseConnector",
    "ConnectorError",
    "FetchedContent",
    "HttpCrawlerConnector",
    "LocalFilesystemConnector",
    "SharePointConnector",
    "SourceConnector",
    "SqlSourceConnector",
    "get_connector",
    "make_document_id",
]

#: Source type -> connector class. ``UPLOAD`` has no connector: an API upload arrives
#: as bytes and enters the pipeline through
#: :func:`ingestion.pipeline.ingest_uploaded_document` instead of being enumerated.
_REGISTRY: dict[SourceType, type[BaseConnector]] = {
    SourceType.BLOB: AzureBlobConnector,
    SourceType.LOCAL: LocalFilesystemConnector,
    SourceType.SHAREPOINT: SharePointConnector,
    SourceType.HTTP: HttpCrawlerConnector,
    SourceType.SQL: SqlSourceConnector,
}


def get_connector(source: SourceConfig, settings: Settings) -> BaseConnector:
    """Build the connector that can read a source.

    Args:
        source: The source configuration to read.
        settings: Process settings.

    Returns:
        A ready-to-use connector. Callers should use it as an async context manager
        (or call ``close()``) so transports and credentials are released.

    Raises:
        ConfigError: If no connector handles the source type.
    """
    connector_class = _REGISTRY.get(source.source_type)
    if connector_class is None:
        msg = f"no connector implements source_type {source.source_type!r}"
        raise ConfigError(msg, detail={"source_id": source.source_id})
    return connector_class(source, settings)
