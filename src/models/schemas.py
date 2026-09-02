"""Shared Pydantic data models.

All pipeline stages exchange strongly-typed, validated objects. Fields are
kept minimal on purpose: the system never stores or exposes personally
identifying information about the people in the processed images.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Sentinel wording surfaced in every face-processing result.
PRIVACY_NOTE = (
    "This system never identifies, names, profiles, or tracks people from faces. "
    "It only reports face presence, location, and technical similarity scores."
)

# Sentinel wording surfaced in every similarity result.
SIMILARITY_NOTE = (
    "Similarity is a technical signal only; it does not establish identity."
)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Face processing
# ---------------------------------------------------------------------------


class FaceInfo(BaseModel):
    """A single detected face.

    `embedding` is a *local, technical-demonstration* encoding. It is
    ``None`` unless the caller explicitly requests embeddings and is never
    written to the blockchain or to provenance records.
    """

    bbox: list[int] = Field(description="[x, y, width, height] in pixels")
    embedding_dimension: int
    embedding: list[float] | None = None


class FaceDetectionResult(BaseModel):
    face_detected: bool
    face_count: int
    faces: list[FaceInfo] = Field(default_factory=list)
    privacy_note: str = PRIVACY_NOTE


class SimilarityResult(BaseModel):
    """Result of comparing two images supplied by the same user."""

    image_a: str
    image_b: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    note: str = SIMILARITY_NOTE


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchCandidate(BaseModel):
    """A candidate result returned by a reverse-image search provider."""

    url: str
    title: str | None = None
    image_url: str | None = None
    source: str = Field(description="Provider / site that produced the candidate")
    snippet: str | None = None
    demo: bool = False


class SearchResult(BaseModel):
    match_found: bool
    candidates: list[SearchCandidate] = Field(default_factory=list)
    provider: str
    demo: bool = False
    reason: str | None = None


class MatchType(str, Enum):
    EXACT = "EXACT_MATCH"
    NEAR = "NEAR_MATCH"
    PAGE = "PAGE_MATCH"
    NONE = "NO_MATCH"


class ValidatedMatch(BaseModel):
    """The accepted candidate plus the evidence used to accept it."""

    match_type: MatchType
    candidate: SearchCandidate
    signals: dict[str, Any] = Field(default_factory=dict)
    rationale: str


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class ProvenanceRecord(BaseModel):
    """Canonical provenance object.

    Content-derived fields (title, page-text hash, image hash/pHash) are the
    basis of the SHA-256 fingerprint. ``retrieved_at`` is display metadata
    (when the content was captured) and is excluded from the fingerprint so
    that re-fetching *unchanged* content reproduces the identical hash.
    """

    source_url: str
    title: str
    retrieved_at: str
    content_sha256: str
    image_sha256: str | None = None
    image_phash: str | None = None
    search_provider: str
    match_type: str
    image_url: str | None = Field(
        default=None,
        description="Image URL used at extraction time. Recorded so re-verification "
        "re-fetches the exact same asset; excluded from the fingerprint because the "
        "fingerprint commits to the image *content* hashes, not its location.",
    )


# ---------------------------------------------------------------------------
# Blockchain
# ---------------------------------------------------------------------------


class BlockchainRecordResult(BaseModel):
    """Result of submitting a fingerprint to the chain."""

    blockchain: str
    contract_address: str
    transaction_hash: str
    block_number: int
    fingerprint: str
    simulated: bool = Field(
        default=False,
        description="True when the record lives on the offline in-memory chain "
        "(demo/testing only) instead of a real EVM chain.",
    )


class OnChainRecord(BaseModel):
    """A record read back from the chain."""

    fingerprint: str
    submitter: str
    timestamp: int
    block_number: int
    source_id: str | None = None
    transaction_hash: str | None = None


class VerificationResult(BaseModel):
    verified: bool
    calculated_hash: str
    on_chain_hash: str | None = None
    transaction_hash: str | None = None
    reason: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    mode: str = Field(description="'real' or 'demo'")
    require_face: bool = True
    record_on_chain: bool = True
    verify_after_record: bool = True
    max_candidates: int = 5
    image_url: str | None = Field(
        default=None,
        description="Optional public http(s) URL of the input image for real-mode "
        "reverse-image search; when unset, the search provider publishes the "
        "local image to a transient public host.",
    )
