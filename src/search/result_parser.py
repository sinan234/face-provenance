"""Candidate validation.

Search results are never trusted blindly. Each candidate is scored with
multiple independent signals:

- image perceptual hash (pHash) distance,
- SHA-256 byte equality,
- image dimensions,
- page title / source URL plausibility.

Outcomes are clearly distinguished:

- ``EXACT_MATCH`` — candidate image bytes are identical (or pHash distance
  and dimensions agree within a tight threshold),
- ``NEAR_MATCH``  — candidate image is perceptually very similar,
- ``PAGE_MATCH``  — the candidate *page* matches but its image could not be
  fetched; lowest confidence,
- ``NO_MATCH``    — nothing usable.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Protocol

from PIL import Image

from src.models.schemas import MatchType, SearchCandidate, ValidatedMatch
from src.search.phash import compute_ahash, compute_phash, hamming_distance
from src.search.web_search import ContentFetcher, FetchError

logger = logging.getLogger(__name__)


class MatchValidatorConfig:
    def __init__(
        self,
        exact_phash_max: int = 8,
        near_phash_max: int = 28,
    ) -> None:
        self.exact_phash_max = exact_phash_max
        self.near_phash_max = near_phash_max


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MatchValidator:
    """Validates search candidates against the input image."""

    def __init__(
        self,
        fetcher: ContentFetcher,
        config: MatchValidatorConfig | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._config = config or MatchValidatorConfig()

    def validate(
        self,
        candidates: list[SearchCandidate],
        input_image_bytes: bytes,
    ) -> ValidatedMatch | None:
        """Return the single best validated candidate (or None)."""
        matches = self.validate_all(candidates, input_image_bytes)
        return matches[0] if matches else None

    def validate_all(
        self,
        candidates: list[SearchCandidate],
        input_image_bytes: bytes,
    ) -> list[ValidatedMatch]:
        """Validate every candidate; return all that pass, best-ranked first.

        The pipeline may need to fall back to the next candidate when the
        best one's page becomes inaccessible at extraction time (robots.txt,
        403, network). Keeping every passing candidate lets it do that instead
        of failing the whole run on one flaky page.
        """
        if not candidates:
            return []

        input_phash = self._image_phash(input_image_bytes)
        input_sha = _sha256(input_image_bytes)
        input_dims = self._image_dims(input_image_bytes)

        matches: list[ValidatedMatch] = []
        for candidate in candidates:
            match = self._validate_one(
                candidate, input_image_bytes, input_phash, input_sha, input_dims
            )
            if match is not None:
                matches.append(match)
        # Best match type first (EXACT > NEAR > PAGE).
        matches.sort(key=lambda m: _rank(m.match_type), reverse=True)
        return matches

    # -- internals ----------------------------------------------------------
    def _validate_one(
        self,
        candidate: SearchCandidate,
        input_image_bytes: bytes,
        input_phash: str,
        input_sha: str,
        input_dims: tuple[int, int] | None,
    ) -> ValidatedMatch | None:
        signals: dict = {}
        try:
            candidate_bytes = self._fetcher.fetch_image(candidate.image_url or "")
        except (FetchError, ValueError) as exc:
            signals["image_fetch_error"] = str(exc)
            return self._page_fallback(candidate, signals)

        candidate_sha = _sha256(candidate_bytes)
        candidate_phash = self._image_phash(candidate_bytes)
        candidate_dims = self._image_dims(candidate_bytes)

        signals["candidate_sha256"] = candidate_sha
        signals["candidate_phash"] = candidate_phash
        signals["candidate_dimensions"] = candidate_dims
        signals["input_sha256"] = input_sha
        signals["input_phash"] = input_phash
        signals["input_dimensions"] = input_dims

        if not candidate_phash:
            # Corrupt/undecodable candidate image: cannot compare bytes, so
            # fall back to a page-level match if the page itself is reachable.
            signals["image_decode_error"] = "Candidate image could not be decoded"
            return self._page_fallback(candidate, signals)

        phash_dist = hamming_distance(input_phash, candidate_phash)
        signals["phash_distance"] = phash_dist
        sha_equal = candidate_sha == input_sha
        signals["sha256_equal"] = sha_equal
        dims_equal = input_dims is not None and candidate_dims == input_dims
        signals["dimensions_equal"] = dims_equal

        if sha_equal:
            return self._accepted(
                MatchType.EXACT, candidate, signals,
                "Candidate image bytes are byte-identical to the input (SHA-256 equal)",
            )
        if phash_dist <= self._config.exact_phash_max and dims_equal:
            return self._accepted(
                MatchType.EXACT, candidate, signals,
                f"pHash distance {phash_dist} (<= {self._config.exact_phash_max}) "
                "and identical dimensions",
            )
        if phash_dist <= self._config.near_phash_max:
            return self._accepted(
                MatchType.NEAR, candidate, signals,
                f"pHash distance {phash_dist} (<= {self._config.near_phash_max}) "
                "indicates perceptual similarity",
            )
        return None

    def _page_fallback(
        self, candidate: SearchCandidate, signals: dict
    ) -> ValidatedMatch | None:
        """Page-level match: provider identified the page, image unfetchable.

        The page itself must be fetchable — a candidate whose page is
        robots-blocked or unreachable can never yield provenance, so it is
        rejected outright instead of being handed to extraction.
        """
        if not (candidate.url and (candidate.title or candidate.snippet)):
            return None
        try:
            self._fetcher.fetch_page(candidate.url)
        except FetchError as exc:
            signals["page_fetch_error"] = str(exc)
            logger.info("Candidate page not fetchable, rejected: %s (%s)", candidate.url, exc)
            return None
        return self._accepted(
            MatchType.PAGE, candidate, signals,
            "Candidate page identified by the search provider, but its image "
            "could not be fetched (page-level match, lowest confidence)",
        )

    def _accepted(
        self,
        match_type: MatchType,
        candidate: SearchCandidate,
        signals: dict,
        rationale: str,
    ) -> ValidatedMatch:
        return ValidatedMatch(
            match_type=match_type,
            candidate=candidate,
            signals=signals,
            rationale=rationale,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _image_phash(data: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            return compute_phash(image)
        except Exception:
            # Corrupt/undecodable candidate image — no hash to compare.
            return ""

    @staticmethod
    def _image_dims(data: bytes) -> tuple[int, int] | None:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return image.size
        except Exception:
            return None


def _rank(match_type: MatchType) -> int:
    return {MatchType.EXACT: 3, MatchType.NEAR: 2, MatchType.PAGE: 1, MatchType.NONE: 0}[
        match_type
    ]
