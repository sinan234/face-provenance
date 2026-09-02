"""Provenance extraction.

Turns a validated search match into the canonical ``ProvenanceRecord`` by
fetching the discovered public page, extracting its title / text / image and
hashing them. Extraction never stores private or personally identifying
data — only content-derived hashes and page metadata.
"""

from __future__ import annotations

import hashlib
import io
import logging
from urllib.parse import urlparse

from PIL import Image

from src.models.schemas import ProvenanceRecord, utc_now_iso
from src.search.phash import compute_ahash, compute_phash
from src.search.web_search import (
    ContentFetcher,
    extract_og_image,
    normalize_text,
    sha256_hex,
)

logger = logging.getLogger(__name__)


class ProvenanceExtractor:
    """Builds canonical provenance records from discovered content."""

    def __init__(self, fetcher: ContentFetcher) -> None:
        self._fetcher = fetcher

    def extract(
        self,
        source_url: str,
        title_hint: str | None = None,
        image_url: str | None = None,
        search_provider: str = "",
        match_type: str = "",
        retrieved_at: str | None = None,
    ) -> ProvenanceRecord:
        page = self._fetcher.fetch_page(source_url)
        title = (page.title or title_hint or source_url).strip()

        # Content hash: normalized page text (falls back to raw HTML).
        content_bytes = (
            normalize_text(page.text).encode("utf-8")
            if page.text
            else page.html.encode("utf-8")
        )
        content_sha256 = sha256_hex(content_bytes)

        # Image hash: prefer the explicit candidate image URL, else og:image.
        image_sha256: str | None = None
        image_phash: str | None = None
        image_url = image_url or self._resolve_og_image(page.html, source_url)
        if image_url:
            try:
                image_bytes = self._fetcher.fetch_image(image_url)
            except Exception as exc:
                logger.warning("Could not fetch provenance image %s: %s", image_url, exc)
            else:
                image_sha256 = sha256_hex(image_bytes)
                image_phash = self._phash(image_bytes)

        return ProvenanceRecord(
            source_url=source_url,
            title=title,
            retrieved_at=retrieved_at or utc_now_iso(),
            content_sha256=content_sha256,
            image_sha256=image_sha256,
            image_phash=image_phash,
            search_provider=search_provider,
            match_type=match_type,
            image_url=image_url,
        )

    @staticmethod
    def _resolve_og_image(html: str, page_url: str) -> str | None:
        og = extract_og_image(html)
        if not og:
            return None
        if og.startswith("//"):
            scheme = urlparse(page_url).scheme
            return f"{scheme}:{og}"
        if og.startswith(("http://", "https://")):
            return og
        return None

    @staticmethod
    def _phash(data: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            return compute_phash(image)
        except Exception:
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
                return compute_ahash(image)
            except Exception:
                return ""
