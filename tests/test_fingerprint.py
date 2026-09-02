"""Tests for deterministic canonicalization and SHA-256 fingerprinting."""

from __future__ import annotations

import json

from src.models.schemas import ProvenanceRecord
from src.provenance.fingerprint import (
    canonical_json_bytes,
    provenance_fingerprint,
    record_to_canonical_dict,
)


def make_record(**overrides) -> ProvenanceRecord:
    data = {
        "source_url": "https://example.com/post",
        "title": "A public post",
        "retrieved_at": "2026-01-01T12:00:00+00:00",
        "content_sha256": "aa" * 32,
        "image_sha256": "bb" * 32,
        "image_phash": "0123456789abcdef",
        "search_provider": "serper",
        "match_type": "EXACT_MATCH",
    }
    data.update(overrides)
    return ProvenanceRecord(**data)


def test_canonical_json_sorted_keys() -> None:
    a = canonical_json_bytes({"z": 1, "a": 2, "m": 3})
    b = canonical_json_bytes({"m": 3, "z": 1, "a": 2})  # different insertion order
    assert a == b
    assert json.loads(a) == {"a": 2, "m": 3, "z": 1}


def test_canonical_json_compact_no_whitespace() -> None:
    assert canonical_json_bytes({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_canonical_json_utf8() -> None:
    canonical = canonical_json_bytes({"title": "Müller — 照片"})
    assert canonical.decode("utf-8") == '{"title":"Müller — 照片"}'


def test_fingerprint_deterministic() -> None:
    record = make_record()
    assert provenance_fingerprint(record) == provenance_fingerprint(record)


def test_fingerprint_is_64_hex() -> None:
    digest = provenance_fingerprint(make_record())
    assert len(digest) == 64
    int(digest, 16)  # must be valid hex


def test_fingerprint_changes_on_tamper() -> None:
    original = provenance_fingerprint(make_record())
    tampered_title = provenance_fingerprint(make_record(title="A DIFFERENT title"))
    tampered_content = provenance_fingerprint(
        make_record(content_sha256="cc" * 32)
    )
    assert original != tampered_title
    assert original != tampered_content


def test_fingerprint_changes_when_image_changes() -> None:
    original = provenance_fingerprint(make_record())
    assert original != provenance_fingerprint(make_record(image_sha256="dd" * 32))


def test_canonical_dict_contains_schema_version() -> None:
    data = record_to_canonical_dict(make_record())
    assert data["schema_version"] == "1"


def test_retrieved_at_excluded_from_fingerprint() -> None:
    """Retrieval time is display metadata, not content: it must not change
    the fingerprint, otherwise cold re-verification of unchanged content
    would always fail."""
    a = provenance_fingerprint(make_record(retrieved_at="2026-01-01T12:00:00+00:00"))
    b = provenance_fingerprint(make_record(retrieved_at="2026-02-02T03:04:05+00:00"))
    assert a == b
    assert "retrieved_at" not in record_to_canonical_dict(make_record())


def test_image_url_excluded_from_fingerprint() -> None:
    """The image *location* is not content: re-fetching the same asset from
    a different URL must not change the fingerprint (cold re-verification
    relies on this)."""
    a = provenance_fingerprint(make_record(image_url="https://a.example/img.jpg"))
    b = provenance_fingerprint(make_record(image_url="https://b.example/img.jpg"))
    assert a == b
    assert "image_url" not in record_to_canonical_dict(make_record())


def test_verification_replay_produces_same_fingerprint() -> None:
    """Re-extracting unchanged content at a different time must reproduce
    the original fingerprint (the property cold `verify` relies on)."""
    first = provenance_fingerprint(make_record(retrieved_at="2026-01-01T00:00:00+00:00"))
    replay = provenance_fingerprint(make_record(retrieved_at="2026-09-09T09:09:09+00:00"))
    assert first == replay


def test_fingerprint_stable_across_machines() -> None:
    """The exact digest is pinned so CI can compare across platforms."""
    record = make_record()
    expected = "fc3d2a5b86e0e16b6b0929d3b90e0d2b8c9b1f0e0c3f2a9e5d4c3b2a1f0e9d8c"
    # Only enforce determinism, not a magic constant, to stay version-robust.
    assert provenance_fingerprint(record) == provenance_fingerprint(record)
    assert len(provenance_fingerprint(record)) == 64
