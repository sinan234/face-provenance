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

class ContentUnstableError(Exception):
    """The page content could not be reproduced across repeated fetches."""


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
        require_stable: bool = True,
    ) -> ProvenanceRecord:
        page = self._fetcher.fetch_page(source_url)
        # The search-provider title is stable across fetches; pages can serve
        # a different <title> on every request (bot-challenge pages), so when
        # a hint is available it is the canonical title.
        title = (title_hint or page.title or source_url).strip()

        def _page_hash(p) -> str:
            content_bytes = (
                normalize_text(p.text).encode("utf-8")
                if p.text
                else p.html.encode("utf-8")
            )
            return sha256_hex(content_bytes)

        # Content hash: normalized page text (falls back to raw HTML).
        if require_stable:
            content_sha256 = self._stable_content_hash(source_url, _page_hash)
        else:
            content_sha256 = _page_hash(page)

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

    def _stable_content_hash(self, source_url: str, hash_fn, attempts: int = 3) -> str:
        """Hash only content that reproduces across fetches.

        JS-heavy / bot-guarded pages (eBay, MercadoLibre, TikTok) serve
        different HTML on almost every request; hashing a single fetch would
        make the fingerprint impossible to re-verify seconds later. We sample
        up to attempts fetches and pin the hash that repeats. If nothing
        repeats, the candidate is unusable for integrity recording.
        """
        samples = []
        for _ in range(attempts):
            try:
                page = self._fetcher.fetch_page(source_url)
            except Exception as exc:
                logger.warning("Stability fetch failed for %s: %s", source_url, exc)
                break
            samples.append(hash_fn(page))
        if not samples:
            raise ContentUnstableError(f"Could not fetch content for {source_url}")
        from collections import Counter

        pinned, count = Counter(samples).most_common(1)[0]
        if count < 2:
            raise ContentUnstableError(
                f"Page content is not reproducible across fetches: {source_url}"
            )
        logger.info(
            "Pinned reproducible content hash for %s (agreement %s/%s)",
            source_url, count, len(samples),
        )
        return pinned

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
