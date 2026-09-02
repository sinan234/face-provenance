"""Reverse-image / provenance search providers.

The ``ReverseImageSearchProvider`` protocol is the pluggable seam. Providers
return typed ``SearchResult`` objects; they never fabricate matches.

Implemented providers:
- ``SerperLensSearchProvider`` — genuine reverse-image search through the
  Serper (https://serper.dev) Google Lens API. Requires ``SEARCH_API_KEY``.
- ``FixtureSearchProvider``    — deterministic *demo* provider that replays a
  locally stored fixture. It is always labelled ``demo=True`` and is never
  presented as a real web search.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Protocol

import httpx

from src.models.schemas import SearchCandidate, SearchResult
from src.search.fixture import DEFAULT_FIXTURE_PATH, load_fixture
from src.search.publish import ImagePublisher, UguuImagePublisher
from src.search.web_search import (
    FetchError,
    RateLimitError,
    SearchProviderError,
)

logger = logging.getLogger(__name__)

SERPER_LENS_ENDPOINT = "https://google.serper.dev/lens"


class ReverseImageSearchProvider(Protocol):
    """Submits an image and returns candidate results."""

    name: str

    def search(self, image_bytes: bytes, mime: str, image_url: str | None = None) -> SearchResult:
        ...


# ---------------------------------------------------------------------------
# Real provider: Serper (Google Lens API)
# ---------------------------------------------------------------------------


class SerperLensSearchProvider:
    """Reverse-image search via the permitted Serper Lens API.

    API contract (see https://serper.dev):
    POST https://google.serper.dev/lens
    Header: X-API-KEY: <key>
    JSON body: ``{"url": "<public image URL>"}``

    Serper requires the image to be reachable at a public HTTP URL; it does
    not accept raw bytes or base64. The provider therefore resolves a public
    URL before calling the API:

    1. ``image_url`` passed to ``search()`` wins, or
    2. ``image_url`` provided to the constructor (e.g. from ``SEARCH_IMAGE_URL``), or
    3. the local image is uploaded via the injected ``publisher`` (default
       ``UguuImagePublisher``).

    The API key is read from configuration at construction time — never
    hardcoded. Requests carry an explicit timeout; HTTP 429 raises
    ``RateLimitError`` so callers can back off.
    """

    name = "serper"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        max_candidates: int = 20,
        session: httpx.Client | None = None,
        image_url: str | None = None,
        publisher: ImagePublisher | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "SerperLensSearchProvider requires an API key (SEARCH_API_KEY)"
            )
        self._api_key = api_key
        self._timeout = timeout
        self._max_candidates = max_candidates
        self._session = session or httpx.Client(timeout=timeout)
        self._image_url = image_url
        self._publisher = publisher if publisher is not None else UguuImagePublisher()

    def _resolve_image_url(self, image_bytes: bytes, mime: str, image_url: str | None) -> str:
        """Return a publicly reachable URL for the input image."""
        candidate = image_url or self._image_url
        if candidate:
            if not candidate.startswith(("http://", "https://")):
                raise SearchProviderError(
                    "SEARCH_IMAGE_URL must be a public http(s) URL of the input image"
                )
            return candidate
        # No user-supplied URL: publish the local bytes to a transient host.
        return self._publisher.publish(image_bytes, mime)

    def search(self, image_bytes: bytes, mime: str, image_url: str | None = None) -> SearchResult:
        public_url = self._resolve_image_url(image_bytes, mime, image_url)
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = self._session.post(
                SERPER_LENS_ENDPOINT, json={"url": public_url}, headers=headers
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise FetchError(f"Serper request failed: {exc}") from exc

        if resp.status_code == 429:
            raise RateLimitError("Serper rate limit reached (HTTP 429)")
        if resp.status_code != 200:
            raise SearchProviderError(
                f"Serper returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SearchProviderError("Serper returned malformed JSON") from exc

        # Serper's lens endpoint returns matches under `organic` (pages) and
        # `image` (images) in different response versions; accept both.
        items = payload.get("organic") or payload.get("image") or []
        candidates: list[SearchCandidate] = []
        for item in items[: self._max_candidates]:
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not link:
                continue
            candidates.append(
                SearchCandidate(
                    url=str(link),
                    title=item.get("title") or item.get("source"),
                    image_url=item.get("imageUrl"),
                    source=item.get("source") or self.name,
                    snippet=item.get("snippet") or item.get("description"),
                )
            )
        if not candidates:
            return SearchResult(
                match_found=False,
                candidates=[],
                provider=self.name,
                reason="No permitted matching public result found",
            )
        return SearchResult(
            match_found=True,
            candidates=candidates,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# Demo provider: deterministic local fixture
# ---------------------------------------------------------------------------


class FixtureSearchProvider:
    """Replays a locally stored demo result.

    This provider exists so the pipeline can be demonstrated end-to-end
    without external credentials. Every result it returns is explicitly
    labelled ``demo=True`` and the candidate is tagged so no consumer can
    mistake it for a real web search.
    """

    name = "demo-fixture"

    def __init__(
        self,
        fixture_path: Path = DEFAULT_FIXTURE_PATH,
        fixture: dict | None = None,
    ) -> None:
        self._fixture_path = Path(fixture_path)
        self._fixture = fixture if fixture is not None else load_fixture(self._fixture_path)

    @property
    def fixture(self) -> dict:
        return self._fixture

    @property
    def image_bytes(self) -> bytes:
        return base64.b64decode(self._fixture["image_b64"])

    def search(
        self, image_bytes: bytes, mime: str, image_url: str | None = None
    ) -> SearchResult:
        candidate = self._fixture["candidate"]
        return SearchResult(
            match_found=True,
            candidates=[
                SearchCandidate(
                    url=candidate["url"],
                    title=candidate["title"],
                    image_url=candidate["image_url"],
                    source=candidate["source"],
                    snippet=candidate.get("snippet"),
                    demo=True,
                )
            ],
            provider=self.name,
            demo=True,
        )
