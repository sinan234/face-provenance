"""End-to-end pipeline tests.

External APIs are mocked (httpx.MockTransport / fixture providers); no
network access is required.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from src.blockchain.client import InMemoryBlockchainClient
from src.cli import _tamper_fixture
from src.face.service import FaceService
from src.models.schemas import PipelineConfig, SearchCandidate, SearchResult
from src.pipeline.pipeline import Pipeline, PipelineError
from src.provenance.extractor import ProvenanceExtractor
from src.provenance.fingerprint import provenance_fingerprint
from src.search.fixture import FixtureContentFetcher
from src.search.result_parser import MatchValidator
from src.search.reverse_image import FixtureSearchProvider
from src.search.web_search import HttpContentFetcher


class MockSearchProvider:
    """Minimal provider stub for search-behaviour tests."""

    name = "mock"

    def __init__(self, candidates: list[SearchCandidate] | None = None) -> None:
        self._candidates = candidates or []

    def search(
        self, image_bytes: bytes, mime: str, image_url: str | None = None
    ) -> SearchResult:
        return SearchResult(
            match_found=bool(self._candidates),
            candidates=self._candidates,
            provider=self.name,
        )


def _demo_components(chain: InMemoryBlockchainClient, fixture: dict):
    face_service = FaceService()
    provider = FixtureSearchProvider(fixture=fixture)
    fetcher = FixtureContentFetcher(fixture=fixture)
    validator = MatchValidator(fetcher)
    extractor = ProvenanceExtractor(fetcher)
    return face_service, provider, validator, extractor, chain


def test_demo_pipeline_full_success(sample_face_path, fixture) -> None:
    chain = InMemoryBlockchainClient()
    components = _demo_components(chain, fixture)
    pipeline = Pipeline(*components, PipelineConfig(mode="demo"))
    result = pipeline.run(str(sample_face_path))

    assert result.face.face_detected is True
    assert result.search is not None and result.search.demo is True
    assert result.search.provider == "demo-fixture"
    assert result.match is not None
    assert result.match.match_type.value == "EXACT_MATCH"
    assert result.fingerprint and len(result.fingerprint) == 64
    assert result.chain is not None and result.chain.simulated is True
    assert result.verification is not None and result.verification.verified is True
    assert result.verification.reason == "Cryptographic fingerprints match"
    assert result.completed is True


def test_demo_pipeline_no_identity_data_leaks(sample_face_path, fixture) -> None:
    chain = InMemoryBlockchainClient()
    components = _demo_components(chain, fixture)
    result = Pipeline(*components, PipelineConfig(mode="demo")).run(str(sample_face_path))
    record_text = result.provenance.model_dump_json().lower()
    for forbidden in ("embedding", "name", "identity"):
        assert forbidden not in record_text


def test_tampered_content_verification_fails(sample_face_path, fixture) -> None:
    """Record honest content, then show tampered content fails verification."""
    chain = InMemoryBlockchainClient()
    honest = Pipeline(
        *_demo_components(chain, fixture), PipelineConfig(mode="demo")
    ).run(str(sample_face_path))
    assert honest.verification.verified is True

    tampered_fixture = _tamper_fixture(fixture)
    tampered = Pipeline(
        *_demo_components(chain, tampered_fixture),
        PipelineConfig(mode="demo", record_on_chain=False),
    ).run(str(sample_face_path))
    assert tampered.verification is not None
    assert tampered.verification.verified is False
    assert "differs from blockchain record" in tampered.verification.reason
    assert tampered.verification.calculated_hash != honest.verification.calculated_hash


def test_no_match_stops_pipeline_without_record(sample_face_path) -> None:
    chain = InMemoryBlockchainClient()
    face_service = FaceService()
    provider = MockSearchProvider(candidates=[])
    fetcher = FixtureContentFetcher(fixture={})
    # Fetch never happens because there are no candidates.
    validator = MatchValidator(fetcher)
    extractor = ProvenanceExtractor(fetcher)
    pipeline = Pipeline(
        face_service, provider, validator, extractor, chain, PipelineConfig(mode="demo")
    )
    result = pipeline.run(str(sample_face_path))
    assert result.match is None
    assert result.no_match_reason == "No permitted matching public result found"
    assert result.chain is None
    assert result.verification is None
    assert result.fingerprint is None


def test_candidates_failing_validation_reported_as_no_match(sample_face_path) -> None:
    """A provider result that fails validation must NOT be trusted blindly."""
    chain = InMemoryBlockchainClient()
    face_service = FaceService()
    # Candidate claims to match but its image is unrelated noise.
    from PIL import Image
    import io
    import numpy as np

    noise = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    noise.save(buf, format="PNG")
    noise_bytes = buf.getvalue()

    class NoiseFetcher:
        def fetch_image(self, url: str) -> bytes:
            return noise_bytes

        def fetch_page(self, url: str):
            raise NotImplementedError

    candidate = SearchCandidate(
        url="https://example.com/unrelated",
        title="Unrelated page",
        image_url="https://example.com/unrelated.jpg",
        source="mock",
    )
    provider = MockSearchProvider(candidates=[candidate])
    validator = MatchValidator(NoiseFetcher())
    extractor = ProvenanceExtractor(NoiseFetcher())
    pipeline = Pipeline(
        face_service, provider, validator, extractor, chain, PipelineConfig(mode="demo")
    )
    result = pipeline.run(str(sample_face_path))
    assert result.match is None
    assert result.no_match_reason is not None
    assert "failed validation" in result.no_match_reason
    assert result.chain is None


def test_real_mode_with_mocked_http_provider(sample_face_path, sample_face_bytes, fixture) -> None:
    """Successful web match using a mocked provider over real HTTP code paths."""
    page_html = (
        "<html><head><title>Mocked public post</title>"
        "<meta property='og:title' content='Mocked public post'/>"
        "<meta property='og:image' content='https://site.example/img.jpg'/>"
        "</head><body><article><p>The mocked public post body text.</p>"
        "</article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/post":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=page_html.encode("utf-8"),
            )
        if request.url.path == "/img.jpg":
            return httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=sample_face_bytes,
            )
        return httpx.Response(404)

    session = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)

    candidate = SearchCandidate(
        url="https://site.example/post",
        title="Mocked public post",
        image_url="https://site.example/img.jpg",
        source="mocked-provider",
    )

    chain = InMemoryBlockchainClient()
    pipeline = Pipeline(
        FaceService(),
        MockSearchProvider(candidates=[candidate]),
        MatchValidator(fetcher),
        ProvenanceExtractor(fetcher),
        chain,
        PipelineConfig(mode="real"),
    )
    result = pipeline.run(str(sample_face_path))

    assert result.search.match_found is True
    assert result.match is not None
    assert result.match.match_type.value == "EXACT_MATCH"  # byte-identical image
    assert result.provenance.title == "Mocked public post"
    assert result.provenance.source_url == "https://site.example/post"
    assert result.fingerprint == provenance_fingerprint(result.provenance)
    assert result.verification.verified is True


def test_pipeline_falls_back_when_best_candidate_page_blocked(
    sample_face_path, sample_face_bytes
) -> None:
    """If the best candidate's page 403s at extraction time, the pipeline
    must fall through to the next validated candidate instead of giving up."""
    page_html = (
        "<html><head><title>Second genuine post</title>"
        "<meta property='og:image' content='https://site.example/img2.jpg'/>"
        "</head><body><article><p>Second genuine post body text.</p></article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(404)
        if path == "/blocked":
            # Image (used for validation) is fine; the PAGE itself 403s at
            # extraction time — the real-world Messi/hdqwalls failure mode.
            return httpx.Response(403)
        if path == "/blocked.jpg":
            return httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=sample_face_bytes,
            )
        if path == "/post2":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=page_html.encode("utf-8"),
            )
        if path == "/img2.jpg":
            return httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=sample_face_bytes,
            )
        return httpx.Response(404)

    session = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)

    # Best candidate: identical image (validates as EXACT via image fetch),
    # but its page is blocked at extraction time.
    blocked = SearchCandidate(
        url="https://site.example/blocked",
        title="Best-looking result",
        image_url="https://site.example/blocked.jpg",
        source="mocked-provider",
    )
    fallback = SearchCandidate(
        url="https://site.example/post2",
        title="Second genuine post",
        image_url="https://site.example/img2.jpg",
        source="mocked-provider",
    )

    chain = InMemoryBlockchainClient()
    pipeline = Pipeline(
        FaceService(),
        MockSearchProvider(candidates=[blocked, fallback]),
        MatchValidator(fetcher),
        ProvenanceExtractor(fetcher),
        chain,
        PipelineConfig(mode="real"),
    )
    result = pipeline.run(str(sample_face_path))

    assert result.match is not None
    assert result.match.candidate.url == "https://site.example/post2"
    assert result.provenance.source_url == "https://site.example/post2"
    assert result.verification.verified is True


def test_missing_input_image_raises(fixture) -> None:
    chain = InMemoryBlockchainClient()
    components = _demo_components(chain, fixture)
    pipeline = Pipeline(*components, PipelineConfig(mode="demo"))
    with pytest.raises(PipelineError):
        pipeline.run("/nonexistent/image.jpg")


def test_no_face_stops_pipeline(tmp_path, fixture) -> None:
    from PIL import Image

    blank = Image.new("RGB", (200, 200), "white")
    path = tmp_path / "blank.jpg"
    blank.save(path)
    chain = InMemoryBlockchainClient()
    components = _demo_components(chain, fixture)
    pipeline = Pipeline(*components, PipelineConfig(mode="demo"))
    with pytest.raises(PipelineError, match="No face detected"):
        pipeline.run(str(path))
