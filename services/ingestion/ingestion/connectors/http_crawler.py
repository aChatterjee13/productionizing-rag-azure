"""HTTP / sitemap crawler with conditional GET and robots.txt compliance.

Two enumeration modes, chosen by the source options:

* ``sitemap_url`` — fetch ``sitemap.xml`` (following ``<sitemapindex>`` entries one
  level deep), and use each ``<lastmod>`` as the modification time. Nothing is
  downloaded during enumeration.
* ``start_urls`` + ``max_depth`` — a real breadth-first crawl. Pages are fetched
  during enumeration (that is the only way to discover links), so the responses are
  cached and :meth:`HttpCrawlerConnector.fetch` serves them without a second request.

Politeness and safety are not optional: ``robots.txt`` is fetched and honoured per
host, the crawl never leaves ``allow_domains``, concurrency is capped by
``settings.ingest_http_concurrency``, and the page count by
``settings.ingest_http_max_pages``.

Delta is by conditional GET. The previous run's ETag and ``Last-Modified`` come from
the manifest via :meth:`~ingestion.connectors.base.BaseConnector.prime_delta_state`
and are sent as ``If-None-Match`` / ``If-Modified-Since``; a 304 becomes
``FetchedContent(not_modified=True)`` and the pipeline records a skip.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from defusedxml import ElementTree as SafeElementTree

from ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    FetchedContent,
    make_document_id,
)
from ragcore.logging import get_logger
from ragcore.models.acl import AccessControl
from ragcore.models.chunk import SourceType
from ragcore.models.document import SourceConfig, SourceDocument
from ragcore.settings import Settings

__all__ = ["HttpCrawlerConnector"]

_log = get_logger(__name__)

#: Namespace used by the sitemap protocol.
_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

#: Fallback link extractor for when selectolax is not installed.
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#>]+)""", re.IGNORECASE)

#: Media types the crawler will index. Anything else is skipped rather than fed to
#: a parser that cannot read it.
_CRAWLABLE_PREFIXES: tuple[str, ...] = ("text/", "application/pdf", "application/xhtml")


class HttpCrawlerConnector(BaseConnector):
    """Crawl a website or consume its sitemap.

    Options: ``start_urls`` (list of URLs) **or** ``sitemap_url``; optional
    ``allow_domains`` (defaults to the hosts of the seeds) and ``max_depth``
    (defaults to 1, meaning seeds plus one level of links).
    """

    supports_delta = True

    def __init__(self, source: SourceConfig, settings: Settings) -> None:
        """Resolve seeds, domain allow-list and depth.

        Args:
            source: Source configuration.
            settings: Process settings supplying the user agent, timeout, page cap
                and concurrency.

        Raises:
            ValueError: If neither ``start_urls`` nor ``sitemap_url`` is configured.
        """
        super().__init__(source, settings)
        self.source_type = SourceType.HTTP
        raw_starts = source.option("start_urls") or []
        self.start_urls: list[str] = [str(url).strip() for url in raw_starts if url]
        self.sitemap_url: str | None = (
            str(source.option("sitemap_url")) if source.option("sitemap_url") else None
        )
        if not self.start_urls and not self.sitemap_url:
            msg = (
                f"http source {source.source_id!r} requires options['start_urls'] "
                "or options['sitemap_url']"
            )
            raise ValueError(msg)
        self.max_depth = int(source.option("max_depth") or 1)
        self.allow_domains: set[str] = {
            str(domain).lower().lstrip(".")
            for domain in (source.option("allow_domains") or [])
        } or {
            host
            for host in (
                _host(url) for url in [*self.start_urls, self.sitemap_url or ""] if url
            )
            if host
        }

        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(settings.ingest_http_concurrency)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._cache: dict[str, FetchedContent] = {}

    # --------------------------------------------------------------------- http
    def _http(self) -> httpx.AsyncClient:
        """Build (once) the crawler's HTTP client.

        Returns:
            A shared :class:`httpx.AsyncClient` carrying the configured user agent.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.ingest_http_timeout_seconds,
                headers={"User-Agent": self.settings.ingest_http_user_agent},
                follow_redirects=True,
            )
        return self._client

    def set_client(self, client: httpx.AsyncClient) -> None:
        """Inject a pre-built HTTP client.

        Used by tests to supply an ``httpx.MockTransport`` so crawler behaviour can be
        verified without network access.

        Args:
            client: The client to use for every request.
        """
        self._client = client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------- robots
    async def _robots_for(self, url: str) -> RobotFileParser | None:
        """Fetch and cache ``robots.txt`` for a URL's host.

        Args:
            url: Any URL on the host.

        Returns:
            A parser, or None when robots.txt is absent or unreadable (in which case
            crawling is permitted, per the robots exclusion standard).
        """
        parsed = urlparse(url)
        key = f"{parsed.scheme}://{parsed.netloc}"
        if key in self._robots:
            return self._robots[key]
        parser: RobotFileParser | None = None
        try:
            response = await self._http().get(f"{key}/robots.txt")
            if response.status_code == httpx.codes.OK:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
        except httpx.HTTPError as exc:
            _log.info("crawler.robots_unavailable", host=key, error=str(exc))
        self._robots[key] = parser
        return parser

    async def _allowed(self, url: str) -> bool:
        """Whether the crawler may fetch a URL.

        Args:
            url: Candidate URL.

        Returns:
            True when the host is in the allow-list and robots.txt permits it.
        """
        host = _host(url)
        if not host or not self._domain_allowed(host):
            return False
        parser = await self._robots_for(url)
        if parser is None:
            return True
        return parser.can_fetch(self.settings.ingest_http_user_agent, url)

    def _domain_allowed(self, host: str) -> bool:
        """Check a host against the configured allow-list.

        Args:
            host: Lower-cased host name.

        Returns:
            True when the host equals or is a subdomain of an allowed domain.
        """
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.allow_domains
        )

    # -------------------------------------------------------------- enumeration
    async def list_documents(
        self, since: datetime | None = None
    ) -> AsyncIterator[SourceDocument]:
        """Enumerate pages from the sitemap or by crawling the seeds.

        Args:
            since: Skip sitemap entries whose ``<lastmod>`` is older than this.

        Yields:
            Descriptors. Crawled pages carry their payload in the connector's cache
            so :meth:`fetch` does not re-download them.
        """
        self._full_scan = since is None
        self._cache.clear()
        if self.sitemap_url:
            async for doc in self._list_from_sitemap(since):
                yield doc
            return
        async for doc in self._crawl():
            yield doc

    async def _list_from_sitemap(
        self, since: datetime | None
    ) -> AsyncIterator[SourceDocument]:
        """Yield descriptors listed by a sitemap.

        Args:
            since: Lower bound on ``<lastmod>``.

        Yields:
            Descriptors, without fetching any page body.

        Raises:
            ConnectorError: If the sitemap itself cannot be retrieved.
        """
        urls = await self._read_sitemap(str(self.sitemap_url), depth=0)
        emitted = 0
        for url, lastmod in urls:
            if emitted >= self.settings.ingest_http_max_pages:
                _log.warning(
                    "crawler.page_cap_reached",
                    source_id=self.source.source_id,
                    cap=self.settings.ingest_http_max_pages,
                )
                break
            if since is not None and lastmod is not None and lastmod < since:
                continue
            if not await self._allowed(url):
                continue
            emitted += 1
            yield self.descriptor(
                source_uri=url,
                title=_title_from_url(url),
                filename=_filename_from_url(url),
                media_type="text/html",
                source_modified_at=lastmod,
                etag=self.known_etag(self.descriptor_document_id(url)),
            )

    def descriptor_document_id(self, url: str) -> str:
        """Document id a URL will map to.

        Args:
            url: Canonical page URL.

        Returns:
            The deterministic document id, so the connector can look up the previous
            run's ETag before building the descriptor.
        """
        return make_document_id(self.source.tenant_id, url)

    async def _read_sitemap(
        self, url: str, *, depth: int
    ) -> list[tuple[str, datetime | None]]:
        """Parse a sitemap, following one level of ``<sitemapindex>``.

        Args:
            url: Sitemap URL.
            depth: Recursion depth; index files are followed only at depth 0.

        Returns:
            ``(url, lastmod)`` pairs.

        Raises:
            ConnectorError: If the sitemap cannot be fetched or parsed.
        """
        try:
            response = await self._http().get(url)
        except httpx.HTTPError as exc:
            msg = "sitemap fetch failed"
            raise ConnectorError(
                msg, detail={"source_id": self.source.source_id, "error": str(exc)}
            ) from exc
        if response.status_code != httpx.codes.OK:
            msg = f"sitemap fetch returned HTTP {response.status_code}"
            raise ConnectorError(msg, detail={"source_id": self.source.source_id})
        try:
            root = SafeElementTree.fromstring(response.text)
        except SafeElementTree.ParseError as exc:
            msg = "sitemap is not well-formed XML"
            raise ConnectorError(
                msg, detail={"source_id": self.source.source_id}
            ) from exc

        out: list[tuple[str, datetime | None]] = []
        for sitemap in root.findall(f"{_SITEMAP_NS}sitemap"):
            location = sitemap.findtext(f"{_SITEMAP_NS}loc")
            if location and depth == 0:
                out.extend(await self._read_sitemap(location.strip(), depth=depth + 1))
        for entry in root.findall(f"{_SITEMAP_NS}url"):
            location = entry.findtext(f"{_SITEMAP_NS}loc")
            if not location:
                continue
            out.append(
                (
                    urldefrag(location.strip()).url,
                    _parse_iso_or_http_date(entry.findtext(f"{_SITEMAP_NS}lastmod")),
                )
            )
        return out

    async def _crawl(self) -> AsyncIterator[SourceDocument]:
        """Breadth-first crawl from the configured seeds.

        Yields:
            Descriptors for every fetched page, in discovery order.
        """
        seen: set[str] = set()
        frontier = [urldefrag(url).url for url in self.start_urls]
        depth = 0

        while frontier and depth <= self.max_depth:
            targets: list[str] = []
            cap = self.settings.ingest_http_max_pages
            for url in frontier:
                if url in seen or len(seen) + len(targets) >= cap:
                    continue
                if await self._allowed(url):
                    targets.append(url)
            if not targets:
                break

            results = await asyncio.gather(
                *(self._conditional_get(url) for url in targets),
                return_exceptions=True,
            )
            next_frontier: list[str] = []
            for url, result in zip(targets, results, strict=True):
                seen.add(url)
                if isinstance(result, BaseException):
                    _log.warning(
                        "crawler.fetch_failed",
                        source_id=self.source.source_id,
                        error=str(result),
                    )
                    continue
                document_id = self.descriptor_document_id(url)
                self._cache[document_id] = result
                yield self.descriptor(
                    source_uri=url,
                    title=_title_from_url(url),
                    filename=_filename_from_url(url),
                    media_type=result.media_type,
                    etag=result.etag,
                    source_modified_at=result.source_modified_at,
                    size_bytes=result.size_bytes,
                )
                if depth < self.max_depth and result.content_text:
                    next_frontier.extend(
                        link
                        for link in extract_links(result.content_text, url)
                        if link not in seen
                    )
            frontier = next_frontier
            depth += 1

    # -------------------------------------------------------------------- fetch
    async def fetch(self, doc: SourceDocument) -> FetchedContent:
        """Return a crawled page's payload, or issue a conditional GET.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The payload, or a ``not_modified`` result when the server answered 304.
        """
        cached = self._cache.pop(doc.document_id, None)
        if cached is not None:
            return cached
        return await self._conditional_get(doc.source_uri, doc.document_id)

    async def _conditional_get(
        self, url: str, document_id: str | None = None
    ) -> FetchedContent:
        """Fetch one URL with ``If-None-Match`` / ``If-Modified-Since``.

        Args:
            url: Page URL.
            document_id: Document id whose manifest entry supplies the validators.
                Derived from the URL when omitted.

        Returns:
            The payload, or ``not_modified`` on a 304.

        Raises:
            ConnectorError: If the request fails or returns an error status.
        """
        key = document_id or self.descriptor_document_id(url)
        headers: dict[str, str] = {}
        etag = self.known_etag(key)
        if etag:
            headers["If-None-Match"] = etag
        modified = self.known_modified_at(key)
        if modified is not None:
            headers["If-Modified-Since"] = format_datetime(
                modified.astimezone(UTC), usegmt=True
            )

        async with self._semaphore:
            try:
                response = await self._http().get(url, headers=headers)
            except httpx.HTTPError as exc:
                msg = "HTTP fetch failed"
                raise ConnectorError(
                    msg, detail={"url": url, "error": str(exc)}
                ) from exc

        if response.status_code == httpx.codes.NOT_MODIFIED:
            return FetchedContent(etag=etag, not_modified=True, media_type="text/html")
        if response.status_code != httpx.codes.OK:
            msg = f"HTTP fetch returned {response.status_code}"
            raise ConnectorError(msg, detail={"url": url})

        media_type = (
            response.headers.get("content-type", "").split(";")[0].strip()
            or "text/html"
        )
        if not media_type.startswith(_CRAWLABLE_PREFIXES):
            msg = f"unsupported content type {media_type}"
            raise ConnectorError(msg, detail={"url": url})

        payload = response.content
        if len(payload) > self.settings.ingest_max_document_bytes:
            msg = "page exceeds ingest_max_document_bytes"
            raise ConnectorError(msg, detail={"url": url})

        fetched = FetchedContent(
            media_type=media_type,
            etag=_clean_header_etag(response.headers.get("etag")),
            source_modified_at=_parse_iso_or_http_date(
                response.headers.get("last-modified")
            ),
            size_bytes=len(payload),
            metadata={"http_status": response.status_code},
        )
        if media_type.startswith(("text/", "application/xhtml")):
            fetched.content_text = response.text
        else:
            fetched.content_bytes = payload
        return fetched

    async def resolve_acl(self, doc: SourceDocument) -> AccessControl:
        """Return the source-configured ACL.

        Crawled pages carry no per-item permissions, so the source config is the only
        authority. An operator who crawls an intranet marks the whole source
        ``internal`` (or higher) there.

        Args:
            doc: Descriptor produced by :meth:`list_documents`.

        Returns:
            The source default :class:`AccessControl`.
        """
        return self.source.default_access_control()


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute, fragment-free links from an HTML page.

    Uses selectolax when available (fast, spec-compliant) and a regex fallback
    otherwise, so link discovery never depends on an optional install.

    Args:
        html: Page markup.
        base_url: URL the page was fetched from, for relative-link resolution.

    Returns:
        De-duplicated absolute ``http(s)`` URLs in document order.
    """
    hrefs: Iterable[str]
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        hrefs = _HREF_RE.findall(html)
    else:
        tree = HTMLParser(html)
        hrefs = [
            node.attributes.get("href") or ""
            for node in tree.css("a")
            if node.attributes
        ]

    out: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        candidate = (href or "").strip()
        if not candidate or candidate.startswith(("mailto:", "javascript:", "tel:")):
            continue
        absolute = urldefrag(urljoin(base_url, candidate)).url
        if not absolute.startswith(("http://", "https://")) or absolute in seen:
            continue
        seen.add(absolute)
        out.append(absolute)
    return out


def _host(url: str) -> str:
    """Extract the lower-cased host from a URL.

    Args:
        url: Any URL.

    Returns:
        The host name, or "" when the URL has none.
    """
    return (urlparse(url).hostname or "").lower()


def _filename_from_url(url: str) -> str:
    """Derive a file name from a URL path.

    Args:
        url: Page URL.

    Returns:
        The last path segment, or ``"index.html"`` for a bare host.
    """
    path = urlparse(url).path.rstrip("/")
    if not path:
        return "index.html"
    name = path.rsplit("/", 1)[-1]
    return name or "index.html"


def _title_from_url(url: str) -> str:
    """Derive a provisional title from a URL path.

    The parser replaces this with the document's ``<title>`` when one exists.

    Args:
        url: Page URL.

    Returns:
        A human-ish title such as ``"Leave policy"``.
    """
    name = _filename_from_url(url)
    stem = name.rsplit(".", 1)[0]
    cleaned = stem.replace("-", " ").replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else url


def _clean_header_etag(value: str | None) -> str | None:
    """Normalise an HTTP ``ETag`` header value.

    Args:
        value: Raw header value.

    Returns:
        The bare validator, or None.
    """
    if not value:
        return None
    text = value.strip()
    if text.startswith("W/"):
        text = text[2:]
    return text.strip('"') or None


def _parse_iso_or_http_date(value: str | None) -> datetime | None:
    """Parse a timestamp that may be ISO-8601 or an HTTP date.

    Args:
        value: ``<lastmod>`` text or a ``Last-Modified`` header.

    Returns:
        An aware UTC datetime, or None when the value is absent or unparseable.
    """
    if not value:
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
