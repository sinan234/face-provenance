"""End-to-end pipeline.

Stages:
    input image -> face detection/encoding -> reverse-image search ->
    candidate validation -> provenance extraction -> fingerprint ->
    blockchain record -> re-fetch -> recompute -> verify -> PASS/FAIL

The pipeline never fabricates results: if no permitted match is found, the
``match`` stage reports ``NO_MATCH`` with a reason and the chain is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.blockchain.client import (
    BlockchainClient,
    DuplicateRecordError,
)
from src.models.schemas import BlockchainRecordResult
from src.face.service import FaceService, load_image_bgr
from src.models.schemas import (
    BlockchainRecordResult,
    FaceDetectionResult,
    PipelineConfig,
    ProvenanceRecord,
    SearchResult,
    ValidatedMatch,
    VerificationResult,
)
from src.provenance.extractor import ContentUnstableError, ProvenanceExtractor
from src.provenance.fingerprint import provenance_fingerprint
from src.search.result_parser import MatchValidator
from src.search.reverse_image import ReverseImageSearchProvider
from src.search.web_search import FetchError

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Fatal pipeline error (unreadable image, no face, no match, ...)."""


@dataclass
class PipelineResult:
    input_path: str
    face: FaceDetectionResult | None = None
    search: SearchResult | None = None
    match: ValidatedMatch | None = None
    provenance: ProvenanceRecord | None = None
    fingerprint: str | None = None
    chain: BlockchainRecordResult | None = None
    verification: VerificationResult | None = None
    no_match_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.verification is not None and self.verification.verified

    def to_dict(self) -> dict:
        """JSON-serializable dict for the web API and reports."""

        def dump(value):
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value

        return {
            "input_path": self.input_path,
            "face": dump(self.face),
            "search": dump(self.search),
            "match": dump(self.match),
            "provenance": dump(self.provenance),
            "fingerprint": self.fingerprint,
            "chain": dump(self.chain),
            "verification": dump(self.verification),
            "no_match_reason": self.no_match_reason,
            "warnings": self.warnings,
            "completed": self.completed,
        }


class Pipeline:
    """Orchestrates the full pipeline with injected dependencies."""

    def __init__(
        self,
        face_service: FaceService,
        search_provider: ReverseImageSearchProvider,
        validator: MatchValidator,
        extractor: ProvenanceExtractor,
        blockchain_client: BlockchainClient,
        config: PipelineConfig,
    ) -> None:
        self._face = face_service
        self._search = search_provider
        self._validator = validator
        self._extractor = extractor
        self._chain = blockchain_client
        self._config = config

    # -- entry point --------------------------------------------------------
    def run(self, image_path: str) -> PipelineResult:
        path = Path(image_path)
        if not path.exists():
            raise PipelineError(f"Input image not found: {path}")

        image_bgr = load_image_bgr(str(path))
        with open(path, "rb") as handle:
            image_bytes = handle.read()
        mime = _guess_mime(path)

        result = PipelineResult(input_path=str(path))

        # [1] Face detection / encoding (privacy-preserving, no embeddings out).
        result.face = self._face.process(image_bgr, include_embeddings=False)
        logger.info(
            "Face detection: detected=%s count=%s",
            result.face.face_detected,
            result.face.face_count,
        )
        if self._config.require_face and not result.face.face_detected:
            raise PipelineError(
                "No face detected in the input image (require_face=True). "
                "Refusing to continue the provenance pipeline without a face."
            )

        # [2]+[3]+[4] Search, validate, extract - with recovery.
        # Serper returns a FRESH candidate set per call and its contents vary
        # (ads, blocked hosts, TikTok/X posts). Probes: first the full image,
        # then (for faces) a face crop, which matches celebrity photos far
        # better. A probe that yields an extractable, reproducible candidate
        # wins; otherwise keep probing before honestly reporting NO MATCH.
        attempts = 1 + self._config.search_retries
        probes: list[tuple[bytes, str]] = [(image_bytes, mime)] * attempts
        crop_jpeg = _face_crop_jpeg(path, result.face.faces[0].bbox) if (
            result.face is not None
            and result.face.face_detected
            and result.face.faces
        ) else None
        if crop_jpeg is not None and self._config.face_crop_retries > 0:
            probes += [(crop_jpeg, "image/jpeg")] * self._config.face_crop_retries
        last_error: str | None = None
        saw_matches = False
        for probe_bytes, probe_mime in probes:
            try:
                result.search = self._search.search(
                    probe_bytes, probe_mime, image_url=self._config.image_url
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Search attempt failed: %s", exc)
                continue
            if not (result.search and result.search.match_found):
                continue
            matches = self._validator.validate_all(
                result.search.candidates, probe_bytes,
            )
            if not matches:
                logger.warning("No candidate passed validation; next probe")
                continue
            saw_matches = True
            for match in matches:
                try:
                    result.provenance = self._extractor.extract(
                        source_url=match.candidate.url,
                        title_hint=match.candidate.title,
                        image_url=match.candidate.image_url,
                        search_provider=result.search.provider,
                        match_type=match.match_type.value,
                    )
                except (ContentUnstableError, FetchError) as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Extraction failed for candidate %s: %s",
                        match.candidate.url, exc,
                    )
                    continue
                result.match = match
                break
            if result.match is not None and result.provenance is not None:
                break
            logger.warning("All validated candidates failed extraction; next probe")

        if result.search is None or not result.search.match_found:
            result.no_match_reason = (
                (result.search.reason if result.search else None)
                or "No permitted matching public result found"
            )
            logger.warning("Search found no match: %s", result.no_match_reason)
            return result
        if result.match is None or result.provenance is None:
            if not saw_matches:
                result.no_match_reason = (
                    "No permitted matching public result found "
                    "(candidates failed validation)"
                )
            else:
                result.no_match_reason = (
                    "No permitted matching public result found "
                    f"(all candidates failed extraction: {last_error})"
                )
            logger.warning("%s", result.no_match_reason)
            return result
        # [5] Cryptographic fingerprint.
        result.fingerprint = provenance_fingerprint(result.provenance)
        logger.info("Provenance fingerprint: %s", result.fingerprint)

        # [6] Blockchain record.
        if self._config.record_on_chain:
            try:
                result.chain = self._chain.record(
                    result.fingerprint, source_id=result.provenance.source_url
                )
                logger.info(
                    "Recorded on chain: tx=%s block=%s",
                    result.chain.transaction_hash,
                    result.chain.block_number,
                )
            except DuplicateRecordError as exc:
                # Same content recorded earlier: not an error. Report the
                # existing on-chain record and let verification confirm it.
                existing = self._chain.get_record(result.fingerprint)
                result.chain = BlockchainRecordResult(
                    blockchain=getattr(self._chain, "blockchain_name", "ethereum"),
                    contract_address=getattr(
                        self._chain, "contract_address", ""
                    )
                    or "",
                    transaction_hash=(
                        existing.transaction_hash if existing else "0x" + "0" * 64
                    ),
                    block_number=existing.block_number if existing else 0,
                    fingerprint=result.fingerprint,
                    simulated=type(self._chain).__name__ == "InMemoryBlockchainClient",
                )
                result.warnings.append(
                    f"Fingerprint was already recorded on chain ({exc}); "
                    "reusing the existing record."
                )
                logger.info("Fingerprint already on chain: %s", result.fingerprint)

        # [7] Verification (re-fetch, recompute, compare).
        if self._config.verify_after_record:
            from src.blockchain.verifier import ProvenanceVerifier

            verifier = ProvenanceVerifier(self._chain, self._extractor)
            result.verification = verifier.verify_provenance(
                result.provenance,
                original_fingerprint=result.fingerprint,
            )
            logger.info(
                "Verification: verified=%s (%s)",
                result.verification.verified,
                result.verification.reason,
            )

        return result


def _face_crop_jpeg(image_path: Path, bbox: list[int]) -> bytes | None:
    """Crop the detected face region and re-encode as JPEG.

    Search-by-face matches celebrity photos far better than the full
    image (background/watermarks dilute Lens results). Privacy-safe:
    the crop is used only for a transient reverse-image search, never
    stored or identified.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(image_path).convert("RGB")
        x, y, w, h = (int(v) for v in bbox[:4])
        pad = max(w, h) // 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img.width, x + w + pad)
        y1 = min(img.height, y + h + pad)
        face = img.crop((x0, y0, x1, y1))
        face.thumbnail((512, 512))
        buf = io.BytesIO()
        face.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Face crop failed: %s", exc)
        return None


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
