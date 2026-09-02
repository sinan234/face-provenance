"""Cryptographic provenance fingerprint.

The fingerprint is the SHA-256 of a deterministic canonical JSON rendering of
the provenance record:

- object keys sorted lexicographically,
- no insignificant whitespace (compact separators),
- UTF-8 encoded,
- ``ensure_ascii=False`` so non-ASCII text hashes identically everywhere.

Same content + same metadata ==> same fingerprint, on any machine.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.models.schemas import ProvenanceRecord

FINGERPRINT_SCHEMA_VERSION = "1"


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact, UTF-8."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def record_to_canonical_dict(record: ProvenanceRecord) -> dict[str, Any]:
    """The exact key set that participates in the fingerprint.

    ``retrieved_at`` is intentionally EXCLUDED: it is display metadata about
    when the content was captured, not content. Including it would make every
    re-verification of unchanged content produce a different hash (because a
    new fetch gets a new timestamp), which would defeat tamper detection.
    Content-derived fields (title, page text hash, image hash/pHash) are what
    the fingerprint commits to.
    """
    data = record.model_dump(exclude_none=True)
    data.pop("retrieved_at", None)
    data.pop("image_url", None)
    data["schema_version"] = FINGERPRINT_SCHEMA_VERSION
    return data


def provenance_fingerprint(record: ProvenanceRecord) -> str:
    """SHA-256 hex digest of the canonical provenance object."""
    canonical = canonical_json_bytes(record_to_canonical_dict(record))
    return hashlib.sha256(canonical).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
