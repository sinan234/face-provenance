"""Web content fetching utilities.

Only publicly accessible content is fetched, subject to:
- HTTP/HTTPS schemes only,
- robots.txt directives (when enabled),
- explicit timeouts,
- response size limits,
- Content-Type checks.

Exceptions raised here are surfaced to the pipeline, which reports them
instead of fabricating results.
"""

from __future__ import annotations

import hashlib
import logging
import re
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "FaceProvenanceBot/1.0 (+https://example.invalid; contact: owner@example.invalid)"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Base error for web fetching / search providers."""


class RobotsDisallowed(FetchError):
    """robots.txt forbids fetching the requested URL."""


class RateLimitError(FetchError):
    """The remote service rate-limited the request (HTTP 429)."""


class SearchProviderError(FetchError):
    """A search provider returned an unusable response."""


class InvalidUrlError(FetchError):
    """The URL failed validation (scheme, host, ...)."""


# ---------------------------------------------------------------------------
# Page model + title extraction
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    url: str
    final_url: str
    title: str | None
    html: str
    text: str


class _TitleParser(HTMLParser):
    """Collects <title> and Open Graph <meta> tags."""

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._title_parts: list[str] = []
        self.og_title: str | None = None
        self.og_image: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            prop = (attrs.get("property") or "").lower()
            name = (attrs.get("name") or "").lower()
            content = attrs.get("content")
            if prop == "og:title" and content:
                self.og_title = content.strip()
            elif name == "og:title" and content:
                self.og_title = content.strip()
            elif prop == "og:image" and content:
                self.og_image = content.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)


def extract_title(html: str) -> str | None:
    parser = _TitleParser()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML must never crash the pipeline
        logger.debug("Title extraction failed on malformed HTML")
        return None
    title = (parser.og_title or "".join(parser._title_parts)).strip()
    return title or None


def extract_og_image(html: str) -> str | None:
    parser = _TitleParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    return parser.og_image


def normalize_text(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


class ContentFetcher(Protocol):
    def fetch_page(self, url: str) -> PageContent:
        ...

    def fetch_image(self, url: str) -> bytes:
        ...


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidUrlError(f"Refusing to fetch non-HTTP(S) URL: {url}")


class HttpContentFetcher:
    """Real HTTP fetcher with robots.txt respect and strict limits."""

    def __init__(
        self,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        max_page_bytes: int = 2 * 1024 * 1024,
        max_image_bytes: int = 5 * 1024 * 1024,
        session: httpx.Client | None = None,
    ) -> None:
        self._timeout = timeout
        self._user_agent = user_agent
        self._respect_robots = respect_robots
        self._max_page_bytes = max_page_bytes
        self._max_image_bytes = max_image_bytes
        self._session = session or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self._robots_cache: dict[str, RobotFileParser] = {}

    # -- robots.txt ---------------------------------------------------------
    def _robots_allowed(self, url: str) -> bool:
        if not self._respect_robots:
            return True
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = self._robots_cache.get(robots_url)
        if parser is None:
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                resp = self._session.get(robots_url)
                if resp.status_code in (200, 401, 403):
                    parser.parse(resp.text.splitlines())
                else:
                    # Missing/other -> assume robots.txt does not restrict.
                    parser.parse([])
            except Exception as exc:
                logger.debug("Could not fetch robots.txt for %s: %s", robots_url, exc)
                parser.parse([])  # fail-open is NOT used for the page itself
                parser = self._robots_cache.setdefault(robots_url, parser)
                return True
            self._robots_cache[robots_url] = parser
        try:
            return parser.can_fetch(self._user_agent, url)
        except Exception:
            return True

    # -- pages --------------------------------------------------------------
    def fetch_page(self, url: str) -> PageContent:
        _validate_http_url(url)
        if not self._robots_allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows fetching {url}")
        try:
            resp = self._session.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise FetchError(f"Network error fetching {url}: {exc}") from exc
        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited while fetching {url}")
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} fetching {url}")
        if len(resp.content) > self._max_page_bytes:
            raise FetchError(f"Page exceeds size limit: {url}")
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type.lower():
            raise FetchError(f"Not an HTML page ({content_type or 'unknown'}): {url}")
        html = resp.text
        text = normalize_text(_html_to_text(html))
        return PageContent(
            url=url,
            final_url=str(resp.url),
            title=extract_title(html),
            html=html,
            text=text,
        )

    # -- images -------------------------------------------------------------
    def fetch_image(self, url: str) -> bytes:
        _validate_http_url(url)
        if not self._robots_allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows fetching {url}")
        try:
            resp = self._session.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise FetchError(f"Network error fetching image {url}: {exc}") from exc
        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited while fetching image {url}")
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} fetching image {url}")
        content_type = resp.headers.get("content-type", "")
        if not content_type.lower().startswith("image/"):
            raise FetchError(f"Not an image ({content_type or 'unknown'}): {url}")
        if len(resp.content) > self._max_image_bytes:
            raise FetchError(f"Image exceeds size limit: {url}")
        return resp.content

    def close(self) -> None:
        self._session.close()


def _html_to_text(html: str) -> str:
    """Naive-but-robust HTML -> text conversion for content hashing."""
    text = re.sub(r"<head[\s\S]*?</head>", " ", html)
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    return text
