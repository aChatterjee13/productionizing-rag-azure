"""SQL connector: turn rows into documents, incrementally, with per-row ACLs.

Configuration lives entirely in ``SourceConfig.options``.

Required:

* ``dsn_secret_ref`` — Key Vault secret name (or env var) holding an **async**
  SQLAlchemy URL, e.g. ``postgresql+asyncpg://...``.
* ``query`` — a ``SELECT`` returning one row per document.
* ``watermark_column`` — monotonic column driving incremental reads.
* ``id_column`` — stable per-row identity.

Optional shaping:

* ``text_columns`` — columns rendered into the document body.
* ``title_column`` — column used as the document title.
* ``template`` — ``str.format``-style body template over the row's columns.
* ``uri_template`` — canonical URI template, e.g. ``"crm://accounts/{id}"``.
* ``media_type`` — parser selector for the rendered body (default
  ``text/markdown``).

Optional per-row metadata and access control:

* ``tenant_column`` — per-row tenant; a row from another tenant is refused.
* ``acl_groups_column``, ``acl_roles_column``, ``acl_users_column``,
  ``denied_users_column``, ``classification_column``.
* ``doc_type_column``, ``author_column``, ``language_column``,
  ``effective_from_column``, ``deleted_column``.

Two safety properties matter here:

* **Identifiers are validated, never interpolated blindly.** When the configured
  query does not already bind ``:watermark`` it is wrapped as a subquery and the
  watermark column name is checked against a strict identifier pattern first.
* **A row that declares a different tenant is dropped, not indexed.** A shared
  reporting view is a very easy way to leak across tenants; the connector refuses
  rather than trusting the query author.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ingestion.acl import (
    access_control_from_mapping,
    merge_with_source_defaults,
    parse_identifier_list,
)
from ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    FetchedContent,
    resolve_secret,
)
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import SourceConfig, SourceDocument
from ragcore.settings import Settings

__all__ = ["SqlSourceConnector"]

_log = get_logger(__name__)

#: Only plain SQL identifiers may be interpolated into the wrapper query.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Async SQLAlchemy drivers this connector accepts. A sync driver would block the
#: event loop for the whole scan, so it is refused with a clear message instead.
_ASYNC_DRIVERS: tuple[str, ...] = (
    "+asyncpg",
    "+aiosqlite",
    "+aiomysql",
    "+asyncmy",
    "+aioodbc",
    "+psycopg",
)


class SqlSourceConnector(BaseConnector):
    """Ingest rows from a relational database as documents."""

    supports_delta = True

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Validate the query configuration.

        Args:
            source: Source configuration.
            settings: Process settings supplying the row batch size.

        Raises:
            ValueError: If a required option is missing or the watermark column is
                not a plain SQL identifier.
        """
        super().__init__(source, settings)
        self.source_type = SourceType.SQL
        self.dsn_secret_ref = str(source.require_option("dsn_secret_ref"))
        self.query = str(source.require_option("query")).strip().rstrip(";")
        self.watermark_column = str(source.require_option("watermark_column"))
        self.id_column = str(source.require_option("id_column"))
        if not _IDENTIFIER_RE.match(self.watermark_column):
            msg = (
                "watermark_column must be a plain SQL identifier, got "
                f"{self.watermark_column!r}"
            )
            raise ValueError(msg)

        self.text_columns: list[str] = [
            str(column) for column in (source.option("text_columns") or [])
        ]
        self.title_column = source.option("title_column")
        self.template = source.option("template")
        self.tenant_column = source.option("tenant_column")
        self.uri_template = str(
            source.option("uri_template") or f"sql://{source.source_id}/{{id}}"
        )
        self.media_type = str(source.option("media_type") or "text/markdown")
        self.deleted_column = source.option("deleted_column")
        self._acl_columns: dict[str, Any] = {
            "allowed_groups": source.option("acl_groups_column"),
            "allowed_roles": source.option("acl_roles_column"),
            "allowed_users": source.option("acl_users_column"),
            "denied_users": source.option("denied_users_column"),
        }
        self.classification_column = source.option("classification_column")
        self.doc_type_column = source.option("doc_type_column")
        self.author_column = source.option("author_column")
        self.language_column = source.option("language_column")
        self.effective_from_column = source.option("effective_from_column")

        self._engine: AsyncEngine | None = None
        self._acl_cache: dict[str, AccessControl] = {}

    # ------------------------------------------------------------------- engine
    async def _get_engine(self) -> AsyncEngine:
        """Build (once) the async engine for the configured DSN.

        Returns:
            The engine.

        Raises:
            ConnectorError: If the DSN cannot be resolved or uses a sync driver.
        """
        if self._engine is not None:
            return self._engine
        dsn = await resolve_secret(self.dsn_secret_ref, self.settings)
        if not dsn:
            msg = "SQL source DSN could not be resolved from its secret reference"
            raise ConnectorError(
                msg,
                detail={
                    "source_id": self.source.source_id,
                    "dsn_secret_ref": self.dsn_secret_ref,
                },
            )
        if not any(driver in dsn for driver in _ASYNC_DRIVERS):
            msg = (
                "SQL source needs an async SQLAlchemy driver "
                "(e.g. postgresql+asyncpg://)"
            )
            raise ConnectorError(msg, detail={"source_id": self.source.source_id})
        self._engine = create_async_engine(dsn, pool_pre_ping=True)
        return self._engine

    def set_engine(self, engine: AsyncEngine) -> None:
        """Inject a pre-built engine.

        Used by tests to point the connector at an in-memory database instead of
        resolving a secret.

        Args:
            engine: The engine to use.
        """
        self._engine = engine

    async def close(self) -> None:
        """Dispose the engine and its connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # -------------------------------------------------------------------- query
    def build_statement(self, watermark: str | None) -> tuple[str, dict[str, Any]]:
        """Render the SQL to execute for one pass.

        Args:
            watermark: Cursor value from the previous successful run, or None for a
                full initial load.

        Returns:
            A ``(sql, parameters)`` pair. ``:watermark`` is always a bind parameter,
            never string-interpolated.
        """
        if ":watermark" in self.query:
            return self.query, {"watermark": watermark}
        column = self.watermark_column
        # `column` passed the strict identifier check in __init__ and `self.query`
        # is operator-supplied configuration, never request input; the watermark
        # *value* is always a bind parameter.
        if watermark is None:
            return (
                f"SELECT * FROM ({self.query}) AS delta_src "  # noqa: S608
                f"ORDER BY {column} ASC",
                {},
            )
        return (
            f"SELECT * FROM ({self.query}) AS delta_src "  # noqa: S608
            f"WHERE {column} > :watermark ORDER BY {column} ASC",
            {"watermark": watermark},
        )

    async def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Stream rows and yield one fully-materialised descriptor each.

        Rows already carry their text, so the descriptor is complete and
        :meth:`fetch` is a no-op read from it.

        Args:
            since: Ignored when a cursor exists; used as the initial watermark on a
                first run so a back-fill can be bounded.

        Yields:
            Descriptors with ``content_text`` populated.

        Raises:
            ConnectorError: If the query fails.
        """
        watermark = self._cursor or (since.isoformat() if since else None)
        self._full_scan = watermark is None
        sql, params = self.build_statement(watermark)
        engine = await self._get_engine()
        highest = watermark
        seen = 0

        try:
            async with engine.connect() as connection:
                statement = text(sql)
                if connection.dialect.supports_server_side_cursors:
                    # Stream rows so a multi-million-row source never materialises in
                    # the worker's memory.
                    streaming = statement.execution_options(
                        yield_per=self.settings.ingest_sql_batch_size
                    )
                    result = await connection.stream(streaming, params)
                    rows = result.mappings()
                else:
                    buffered = await connection.execute(statement, params)
                    rows = _as_async(buffered.mappings())
                async for row in rows:
                    document = self._descriptor_for(dict(row))
                    if document is None:
                        continue
                    seen += 1
                    highest = _max_watermark(highest, row.get(self.watermark_column))
                    yield document
        except ConnectorError:
            raise
        except Exception as exc:
            msg = "SQL source query failed"
            raise ConnectorError(
                msg, detail={"source_id": self.source.source_id, "error": str(exc)}
            ) from exc

        self._cursor = highest
        _log.info(
            "sql.scan_complete",
            source_id=self.source.source_id,
            rows=seen,
            full_scan=self._full_scan,
        )

    def _descriptor_for(self, row: Mapping[str, Any]) -> SourceDocument | None:
        """Convert one row into a descriptor.

        Args:
            row: Row mapping from the query.

        Returns:
            The descriptor, or None when the row must be dropped (missing id, or a
            tenant that does not match this source).
        """
        row_id = row.get(self.id_column)
        if row_id is None:
            _log.warning(
                "sql.row_missing_id",
                source_id=self.source.source_id,
                id_column=self.id_column,
            )
            return None

        if self.tenant_column:
            declared = row.get(str(self.tenant_column))
            if declared is not None and str(declared) != self.source.tenant_id:
                _log.warning(
                    "sql.row_tenant_mismatch",
                    source_id=self.source.source_id,
                    expected_tenant=self.source.tenant_id,
                )
                return None

        source_uri = self._render_uri(row, str(row_id))
        deleted = (
            bool(row.get(str(self.deleted_column))) if self.deleted_column else False
        )
        access_control = self._row_access_control(row)
        self._acl_cache[source_uri] = access_control

        body = "" if deleted else self._render_body(row)
        title = ""
        if self.title_column and row.get(str(self.title_column)) is not None:
            title = str(row[str(self.title_column)])

        document = self.descriptor(
            source_uri=source_uri,
            title=title,
            filename=f"{row_id}.md",
            media_type=self.media_type,
            etag=_row_etag(row.get(self.watermark_column)),
            source_modified_at=_as_datetime(row.get(self.watermark_column)),
            size_bytes=len(body.encode("utf-8")),
            author=(
                str(row[str(self.author_column)])
                if self.author_column and row.get(str(self.author_column))
                else None
            ),
            deleted=deleted,
            access_control=access_control,
            content_text=body or None,
            metadata={"row_id": str(row_id)},
        )
        updates: dict[str, Any] = {}
        if self.doc_type_column and row.get(str(self.doc_type_column)):
            updates["doc_type"] = str(row[str(self.doc_type_column)])
        if self.language_column and row.get(str(self.language_column)):
            updates["language"] = str(row[str(self.language_column)])
        if self.effective_from_column:
            effective = _as_datetime(row.get(str(self.effective_from_column)))
            if effective is not None:
                updates["effective_from"] = effective
        return document.model_copy(update=updates) if updates else document

    def _render_uri(self, row: Mapping[str, Any], row_id: str) -> str:
        """Render the canonical URI for a row.

        Args:
            row: Row mapping.
            row_id: Stringified value of the id column.

        Returns:
            The rendered URI. An unknown placeholder falls back to the default
            ``sql://<source_id>/<id>`` form rather than raising.
        """
        values = {str(key): value for key, value in row.items()}
        values.setdefault("id", row_id)
        try:
            return self.uri_template.format_map(values)
        except (KeyError, IndexError):
            _log.warning("sql.uri_template_unresolved", source_id=self.source.source_id)
            return f"sql://{self.source.source_id}/{row_id}"

    def _render_body(self, row: Mapping[str, Any]) -> str:
        """Render a row's document body.

        Args:
            row: Row mapping.

        Returns:
            The rendered text: ``template`` when configured, else the configured
            ``text_columns`` as ``"Column: value"`` lines, else every non-control
            column in query order.
        """
        values = {
            str(key): ("" if value is None else value) for key, value in row.items()
        }
        if self.template:
            try:
                return str(self.template).format_map(values)
            except (KeyError, IndexError):
                _log.warning("sql.template_unresolved", source_id=self.source.source_id)
        columns = self.text_columns or [
            key for key in values if key not in self._control_columns()
        ]
        lines: list[str] = []
        for column in columns:
            value = values.get(column)
            if value in (None, ""):
                continue
            label = column.replace("_", " ").strip().capitalize()
            lines.append(f"## {label}\n\n{value}")
        return "\n\n".join(lines)

    def _control_columns(self) -> set[str]:
        """Columns that carry control metadata rather than document text.

        Returns:
            The set of configured control column names.
        """
        candidates = [
            self.id_column,
            self.watermark_column,
            self.tenant_column,
            self.classification_column,
            self.deleted_column,
            *self._acl_columns.values(),
        ]
        return {str(name) for name in candidates if name}

    def _row_access_control(self, row: Mapping[str, Any]) -> AccessControl:
        """Build the effective ACL for one row.

        Args:
            row: Row mapping.

        Returns:
            The per-row ACL merged with the source defaults.
        """
        lists = {
            field: parse_identifier_list(row.get(str(column)))
            for field, column in self._acl_columns.items()
            if column
        }
        classification = None
        if self.classification_column:
            classification = row.get(str(self.classification_column))
        if not any(lists.values()) and classification is None:
            return self.source.default_access_control()

        payload: dict[str, Any] = dict(lists)
        if classification is not None:
            payload["classification"] = classification
        item = access_control_from_mapping(payload, self.source)
        return merge_with_source_defaults(item, self.source)

    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Return the text already materialised during enumeration.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The row's rendered text.
        """
        body = doc.content_text or ""
        return FetchedContent(
            content_text=body,
            media_type=doc.media_type or self.media_type,
            etag=doc.etag,
            source_modified_at=doc.source_modified_at,
            size_bytes=len(body.encode("utf-8")),
        )

    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Return the ACL resolved from the row's own columns.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The per-row ACL, or the source defaults when the row carried none.
        """
        return self._acl_cache.get(doc.source_uri, self.source.default_access_control())


async def _as_async(rows: Iterable[Any]) -> AsyncIterator[Any]:
    """Adapt a buffered result set to the async iteration protocol.

    Dialects without server-side cursors (aiosqlite, and any driver where
    ``supports_server_side_cursors`` is False) return a buffered result. Wrapping it
    keeps one iteration path in :meth:`SqlSourceConnector.list_documents`.

    Args:
        rows: Buffered rows.

    Yields:
        Each row in order.
    """
    for row in rows:
        yield row


def _row_etag(value: Any) -> str | None:
    """Derive an ETag from a row's watermark value.

    Args:
        value: Watermark column value.

    Returns:
        A string form of the watermark, or None.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _as_datetime(value: Any) -> datetime | None:
    """Coerce a column value into an aware UTC datetime.

    Args:
        value: A datetime, an ISO-8601 string, or anything else.

    Returns:
        The datetime, or None when the value is not date-like.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _max_watermark(current: str | None, candidate: Any) -> str | None:
    """Advance the watermark cursor monotonically.

    Args:
        current: Highest watermark seen so far, as a string.
        candidate: Watermark value from the row just read.

    Returns:
        The larger of the two, compared as datetimes when both parse as dates and
        lexicographically otherwise (which is correct for zero-padded ids and for
        ISO-8601 timestamps).
    """
    rendered = _row_etag(candidate)
    if rendered is None:
        return current
    if current is None:
        return rendered
    left, right = _as_datetime(current), _as_datetime(rendered)
    if left is not None and right is not None:
        return rendered if right > left else current
    if current.isdigit() and rendered.isdigit():
        return rendered if int(rendered) > int(current) else current
    return max(current, rendered)
