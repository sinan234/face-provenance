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


def test_extractor_pins_reproducible_content_and_rises_on_unstable() -> None:
    """Extraction must commit only to content that reproduces across fetches."""
    from src.provenance.extractor import ContentUnstableError

    class StablePage:
        def fetch_page(self, url: str):
            from src.search.web_search import PageContent
            return PageContent(url=url, final_url=url, title="Stable", html="<html><body>constant</body></html>", text="constant")
        def fetch_image(self, url: str) -> bytes:
            raise NotImplementedError

    stable = ProvenanceExtractor(StablePage())
    record = stable.extract(source_url="https://s.example/post", title_hint="Stable title")
    again = stable.extract(source_url="https://s.example/post", title_hint="Stable title")
    assert record.content_sha256 == again.content_sha256
    assert record.title == "Stable title"  # hint wins over page title

    class FlipFlopPage:
        def __init__(self) -> None:
            self._n = 0
        def fetch_page(self, url: str):
            from src.search.web_search import PageContent
            self._n += 1
            return PageContent(url=url, final_url=url, title="x", html=f"<html><body>tick {self._n}</body></html>", text=f"tick {self._n}")
        def fetch_image(self, url: str) -> bytes:
            raise NotImplementedError

    flaky = ProvenanceExtractor(FlipFlopPage())
    with pytest.raises(ContentUnstableError):
        flaky.extract(source_url="https://s.example/unstable", title_hint="T")


def test_pipeline_skips_unstable_candidate_and_verifies_fallback(sample_face_path, sample_face_bytes) -> None:
    """The TikTok/MercadoLibre failure mode: best candidate serves different
    HTML on every fetch, so the pipeline must fall through to a stable one."""

    class UnstableThenStableHandler:
        def __init__(self) -> None:
            self.calls = 0
        def __call__(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(404)
            if path == "/u.jpg":
                return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=sample_face_bytes)
            if path == "/s.jpg":
                return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=sample_face_bytes)
            if path == "/unstable":
                self.calls += 1
                html = f"<html><head><title>U{self.calls}</title></head><body>variant {self.calls}</body></html>"
                return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode("utf-8"))
            if path == "/stable":
                return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><head><title>S</title></head><body>fixed body</body></html>")
            return httpx.Response(404)

    session = httpx.Client(transport=httpx.MockTransport(UnstableThenStableHandler()), follow_redirects=True)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)
    unstable = SearchCandidate(url="https://s.example/unstable", title="Unstable", image_url="https://s.example/u.jpg", source="mocked-provider")
    stable = SearchCandidate(url="https://s.example/stable", title="Stable", image_url="https://s.example/s.jpg", source="mocked-provider")

    chain = InMemoryBlockchainClient()
    pipeline = Pipeline(
        FaceService(),
        MockSearchProvider(candidates=[unstable, stable]),
        MatchValidator(fetcher),
        ProvenanceExtractor(fetcher),
        chain,
        PipelineConfig(mode="real"),
    )
    result = pipeline.run(str(sample_face_path))

    assert result.match is not None
    assert result.match.candidate.url == "https://s.example/stable"
    assert result.provenance is not None
    assert result.provenance.source_url == "https://s.example/stable"
    assert result.verification is not None
    assert result.verification.verified is True


def test_verifier_retries_and_passes_when_recorded_content_reappears() -> None:
    """Verification retries: the first re-fetch may hit a bot-challenge
    variant, but once the recorded content reappears it must PASS."""
    from src.blockchain.verifier import ProvenanceVerifier

    recorded_html = "<html><head><title>Post</title></head><body>the real article text</body></html>"
    wrong_html = "<html><head><title>security check</title></head><body>verify you are human</body></html>"

    class RHandler:
        def __init__(self) -> None:
            self.pages: list[str] = []
            self.i = 0
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            if request.url.path == "/post":
                html = self.pages[min(self.i, len(self.pages) - 1)] if self.pages else wrong_html
                self.i += 1
                return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode("utf-8"))
            return httpx.Response(404)

    handler = RHandler()
    session = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    # Record phase: three identical fetches pin the real content.
    handler.pages = [recorded_html] * 3
    rec_extractor = ProvenanceExtractor(HttpContentFetcher(session=session, respect_robots=True))
    record = rec_extractor.extract(source_url="https://s.example/post", title_hint="Post")
    chain = InMemoryBlockchainClient()
    fingerprint = provenance_fingerprint(record)
    chain.record(fingerprint, source_id=record.source_url)

    # Verify phase: first fetch serves the challenge variant, then the real content.
    handler.pages = [wrong_html, recorded_html, recorded_html]
    handler.i = 0
    session2 = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    verifier = ProvenanceVerifier(chain, ProvenanceExtractor(HttpContentFetcher(session=session2, respect_robots=True)))
    result = verifier.verify_provenance(record, original_fingerprint=fingerprint)
    assert result.verified is True
    assert result.on_chain_hash == fingerprint


def test_pipeline_retries_search_when_first_candidates_fail_validation(
    sample_face_path, sample_face_bytes
) -> None:
    """Serper candidate sets vary per call: when nothing validates, the
    pipeline must retry the genuine search before reporting NO MATCH."""

    blocked = SearchCandidate(
        url="https://s.example/blocked", title="Blocked", image_url="https://s.example/b.jpg", source="mock"
    )
    good = SearchCandidate(
        url="https://s.example/good", title="Good", image_url="https://s.example/g.jpg", source="mock"
    )

    class RetryProvider:
        def __init__(self) -> None:
            self.calls = 0
        def search(self, image_bytes, mime, image_url=None):
            self.calls += 1
            cands = [blocked] if self.calls == 1 else [good]
            return SearchResult(match_found=True, candidates=cands, provider="mock-retry")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(404)
        if path == "/b.jpg" or path == "/blocked":
            return httpx.Response(403)
        if path == "/g.jpg":
            return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=sample_face_bytes)
        if path == "/good":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><head><title>Good</title></head><body>fixed content</body></html>")
        return httpx.Response(404)

    session = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)
    provider = RetryProvider()
    chain = InMemoryBlockchainClient()
    config = PipelineConfig(mode="real", search_retries=2)
    pipeline = Pipeline(
        FaceService(), provider, MatchValidator(fetcher), ProvenanceExtractor(fetcher), chain, config
    )
    result = pipeline.run(str(sample_face_path))

    assert provider.calls == 2
    assert result.match is not None
    assert result.match.candidate.url == "https://s.example/good"
    assert result.verification is not None and result.verification.verified is True


def test_pipeline_researches_when_all_candidates_fail_extraction(
    sample_face_path, sample_face_bytes
) -> None:
    """The Neymar failure mode: candidates validate but every page is
    unreproducible (TikTok) or robots-blocked (X). The pipeline must run
    a fresh genuine search and use the next candidate set."""

    tiktok = SearchCandidate(
        url="https://www.tiktok.com/@u/video/1", title="TikTok video",
        image_url="https://s.example/t.jpg", source="mock"
    )
    good = SearchCandidate(
        url="https://s.example/good", title="Good", image_url="https://s.example/g.jpg", source="mock"
    )

    class RetryProvider2:
        def __init__(self) -> None:
            self.calls = 0
        def search(self, image_bytes, mime, image_url=None):
            self.calls += 1
            return SearchResult(
                match_found=True,
                candidates=[tiktok] if self.calls == 1 else [good],
                provider="mock-retry",
            )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(404)
        if path == "/t.jpg":
            return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=sample_face_bytes)
        if path == "/g.jpg":
            return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=sample_face_bytes)
        if path == "/video/1":
            # validated (page fetchable) but never reproducible
            import time
            n = int(time.time() * 1000)
            html = f"<html><head><title>T</title></head><body>{n}</body></html>"
            return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode("utf-8"))
        if path == "/good":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><head><title>Good</title></head><body>fixed</body></html>")
        return httpx.Response(404)

    session = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)
    provider = RetryProvider2()
    pipeline = Pipeline(
        FaceService(), provider, MatchValidator(fetcher), ProvenanceExtractor(fetcher),
        InMemoryBlockchainClient(), PipelineConfig(mode="real", search_retries=2),
    )
    result = pipeline.run(str(sample_face_path))
    assert provider.calls == 2
    assert result.match is not None
    assert result.match.candidate.url == "https://s.example/good"
    assert result.verification is not None and result.verification.verified is True
