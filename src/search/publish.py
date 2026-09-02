"""Image publishing: turn local image bytes into a publicly reachable URL.

Serper's (and most Lens APIs') reverse-image endpoint requires the image to be
reachable at a public HTTP URL — it will not accept raw bytes or base64
uploads. ``ImagePublisher`` is the pluggable seam for making a local image
briefly public so a real reverse-image search can be performed.

Implemented host: ``UguuImagePublisher`` (uguu.se). The upload must be
available to the search provider only long enough to run the search; the
returned URL is treated as transient and is never stored.

Privacy note: publishing sends the *image to a third-party host for the
purpose of a reverse-image search* — exactly what a reverse-image search
requires. Users who would rather control the URL themselves can set
``SEARCH_IMAGE_URL`` to point their input image at their own public copy, in
which case no upload happens.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from src.search.web_search import FetchError

logger = logging.getLogger(__name__)

UGUU_UPLOAD_ENDPOINT = "https://uguu.se/upload"


class ImagePublisher(Protocol):
    """Uploads an image and returns a publicly reachable URL."""

    def publish(self, image_bytes: bytes, mime: str) -> str:
        ...


class UguuImagePublisher:
    """Publishes an image via uguu.se (no account, returns a direct URL).

    The endpoint returns JSON where ``files[0].url`` is a direct, publicly
    fetchable image URL.
    """

    name = "uguu"

    def __init__(
        self,
        timeout: float = 60.0,
        endpoint: str = UGUU_UPLOAD_ENDPOINT,
        session: httpx.Client | None = None,
    ) -> None:
        self._timeout = timeout
        self._endpoint = endpoint
        self._session = session or httpx.Client(timeout=timeout, follow_redirects=True)

    def publish(self, image_bytes: bytes, mime: str) -> str:
        ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
            mime or "", "jpg"
        )
        try:
            resp = self._session.post(
                self._endpoint,
                files={"files[]": (f"upload.{ext}", image_bytes, mime or "image/jpeg")},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise FetchError(f"Image upload to uguu.se failed: {exc}") from exc
        if resp.status_code != 200:
            raise FetchError(
                f"Image upload to uguu.se returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            payload = resp.json()
            url = payload["files"][0]["url"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise FetchError("Image upload to uguu.se returned malformed JSON") from exc
        if not isinstance(url, str) or not url.startswith("http"):
            raise FetchError("Image upload to uguu.se returned an invalid URL")
        logger.debug("Published image to transient public URL: %s", url)
        return url