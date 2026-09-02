"""Tests for the search stage: providers, validation, hashing, fetching."""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from src.models.schemas import MatchType, SearchCandidate
from src.search.phash import compute_ahash, compute_phash, hamming_distance
from src.search.result_parser import MatchValidator
from src.search.reverse_image import SerperLensSearchProvider
from src.search.web_search import (
    FetchError,
    HttpContentFetcher,
    RateLimitError,
    RobotsDisallowed,
    SearchProviderError,
    extract_og_image,
    extract_title,
)


# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------


def _image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def test_phash_identical_images_zero_distance(sample_face_bytes) -> None:
    a = compute_phash(_image_from_bytes(sample_face_bytes))
    b = compute_phash(_image_from_bytes(sample_face_bytes))
    assert a == b
    assert hamming_distance(a, b) == 0


def test_phash_perturbed_image_close(sample_face_bytes) -> None:
    base = compute_phash(_image_from_bytes(sample_face_bytes))
    image = _image_from_bytes(sample_face_bytes)
    perturbed = image.resize((image.width - 40, image.height - 40))
    dist = hamming_distance(base, compute_phash(perturbed))
    assert dist <= 18, f"Expected small phash distance, got {dist}"


def test_phash_different_images_far_apart(sample_face_bytes) -> None:
    base = compute_phash(_image_from_bytes(sample_face_bytes))
    noise = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    )
    dist = hamming_distance(base, compute_phash(noise))
    assert dist > 18, f"Expected large phash distance, got {dist}"


def test_ahash_fallback(sample_face_bytes) -> None:
    assert len(compute_ahash(_image_from_bytes(sample_face_bytes))) == 16


# ---------------------------------------------------------------------------
# Serper provider (mocked HTTP)
# ---------------------------------------------------------------------------


class StubPublisher:
    """Never touches the network: returns a fixed public URL."""

    def __init__(self, url: str = "https://pub.example/uploads/abc.jpg") -> None:
        self.url = url
        self.calls: list[tuple[bytes, str]] = []

    def publish(self, image_bytes: bytes, mime: str) -> str:
        self.calls.append((image_bytes, mime))
        return self.url


def _serper_provider(payload, status=200, content_type="application/json"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": content_type},
            content=json.dumps(payload) if isinstance(payload, dict) else payload,
        )

    session = httpx.Client(transport=httpx.MockTransport(handler))
    return SerperLensSearchProvider(api_key="test-key", session=session, publisher=StubPublisher())


def test_serper_parses_candidates() -> None:
    payload = {
        "image": [
            {
                "title": "First result",
                "link": "https://site.example/first",
                "imageUrl": "https://site.example/first.jpg",
                "source": "example.com",
                "snippet": "snip",
            },
            {"title": "Second", "link": "https://site.example/second"},
        ]
    }
    provider = _serper_provider(payload)
    result = provider.search(b"fake-image-bytes", "image/jpeg")
    assert result.match_found is True
    assert result.demo is False
    assert len(result.candidates) == 2
    assert result.candidates[0].url == "https://site.example/first"
    assert result.candidates[0].image_url == "https://site.example/first.jpg"
    assert result.candidates[0].title == "First result"


def test_serper_parses_organic_results() -> None:
    """Serper's live lens API returns page matches under `organic`."""
    payload = {
        "searchParameters": {"type": "lens", "engine": "google", "url": "https://u"},
        "organic": [
            {
                "title": "Abraham Lincoln Portrait",
                "source": "Fine Art America",
                "link": "https://site.example/organic",
                "imageUrl": "https://site.example/o.jpg",
                "thumbnailUrl": "https://site.example/t.jpg",
            },
            {"title": "Second page", "link": "https://site.example/second"},
        ],
        "credits": 3,
    }
    provider = _serper_provider(payload)
    result = provider.search(b"bytes", "image/jpeg")
    assert result.match_found is True
    assert len(result.candidates) == 2
    assert result.candidates[0].url == "https://site.example/organic"
    assert result.candidates[0].image_url == "https://site.example/o.jpg"
    assert result.candidates[0].source == "Fine Art America"


def test_serper_sends_public_url_in_json_body() -> None:
    """The image must be submitted as a public URL, not raw bytes."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.read().decode("utf-8") if request.content else ""
        return httpx.Response(200, content=json.dumps({"image": []}))

    session = httpx.Client(transport=httpx.MockTransport(handler))
    publisher = StubPublisher(url="https://pub.example/u/img.jpg")
    provider = SerperLensSearchProvider(api_key="k", session=session, publisher=publisher)
    provider.search(b"raw-bytes", "image/jpeg")
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"]) == {"url": "https://pub.example/u/img.jpg"}
    assert publisher.calls == [(b"raw-bytes", "image/jpeg")]


def test_serper_uses_user_supplied_image_url_without_publishing() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode("utf-8") if request.content else ""
        return httpx.Response(200, content=json.dumps({"image": []}))

    session = httpx.Client(transport=httpx.MockTransport(handler))
    publisher = StubPublisher()
    provider = SerperLensSearchProvider(api_key="k", session=session, publisher=publisher)
    provider.search(
        b"raw-bytes", "image/jpeg", image_url="https://mine.example/photo.jpg"
    )
    assert json.loads(captured["body"]) == {"url": "https://mine.example/photo.jpg"}
    assert publisher.calls == []  # no upload happened


def test_serper_sends_api_key_header() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("X-API-KEY")
        return httpx.Response(200, content=json.dumps({"image": []}))

    session = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperLensSearchProvider(
        api_key="secret-key", session=session, publisher=StubPublisher()
    )
    result = provider.search(b"x", "image/jpeg")
    assert captured["key"] == "secret-key"
    assert result.match_found is False
    assert result.reason == "No permitted matching public result found"


def test_serper_rate_limit_raises() -> None:
    provider = _serper_provider({}, status=429)
    with pytest.raises(RateLimitError):
        provider.search(b"x", "image/jpeg")


def test_serper_server_error_raises() -> None:
    provider = _serper_provider({}, status=500)
    with pytest.raises(SearchProviderError):
        provider.search(b"x", "image/jpeg")


def test_serper_malformed_json_raises() -> None:
    provider = _serper_provider("this is not json", status=200)
    with pytest.raises(SearchProviderError):
        provider.search(b"x", "image/jpeg")


def test_serper_requires_api_key() -> None:
    with pytest.raises(ValueError):
        SerperLensSearchProvider(api_key="")


# ---------------------------------------------------------------------------
# Validation classification
# ---------------------------------------------------------------------------


class StubFetcher:
    def __init__(self, image_bytes: bytes) -> None:
        self._bytes = image_bytes

    def fetch_image(self, url: str) -> bytes:
        return self._bytes

    def fetch_page(self, url: str):
        raise NotImplementedError


class FailingImageFetcher:
    def __init__(self, page_error: Exception | None = None) -> None:
        self._page_error = page_error

    def fetch_image(self, url: str) -> bytes:
        raise FetchError("image unavailable")

    def fetch_page(self, url: str):
        if self._page_error is not None:
            raise self._page_error
        from src.search.web_search import PageContent

        return PageContent(url=url, final_url=url, title="T", html="<p>t</p>", text="t")


def test_validator_exact_match(sample_face_bytes) -> None:
    validator = MatchValidator(StubFetcher(sample_face_bytes))
    candidate = SearchCandidate(
        url="https://example.com/x",
        title="X",
        image_url="https://example.com/x.jpg",
        source="mock",
    )
    match = validator.validate([candidate], sample_face_bytes)
    assert match is not None
    assert match.match_type == MatchType.EXACT
    assert match.signals["sha256_equal"] is True
    assert "byte-identical" in match.rationale


def test_validator_near_match(sample_face_bytes) -> None:
    image = _image_from_bytes(sample_face_bytes)
    resized = image.resize((image.width - 40, image.height - 40))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=80)
    candidate_bytes = buf.getvalue()

    validator = MatchValidator(StubFetcher(candidate_bytes))
    candidate = SearchCandidate(
        url="https://example.com/y",
        title="Y",
        image_url="https://example.com/y.jpg",
        source="mock",
    )
    match = validator.validate([candidate], sample_face_bytes)
    assert match is not None
    assert match.match_type == MatchType.NEAR
    assert "perceptual similarity" in match.rationale


def test_validator_page_match_when_image_unavailable(sample_face_bytes) -> None:
    validator = MatchValidator(FailingImageFetcher())
    candidate = SearchCandidate(
        url="https://example.com/z",
        title="Z page",
        image_url="https://example.com/z.jpg",
        source="mock",
    )
    match = validator.validate([candidate], sample_face_bytes)
    assert match is not None
    assert match.match_type == MatchType.PAGE
    assert "page-level match" in match.rationale


def test_validator_rejects_robots_blocked_page(sample_face_bytes) -> None:
    """A candidate whose page is robots-blocked must be rejected, not accepted."""
    validator = MatchValidator(FailingImageFetcher(page_error=RobotsDisallowed("nope")))
    candidate = SearchCandidate(
        url="https://example.com/blocked",
        title="Blocked page",
        image_url="https://example.com/blocked.jpg",
        source="mock",
    )
    assert validator.validate([candidate], sample_face_bytes) is None


def test_validator_no_match_for_unrelated_image(sample_face_bytes) -> None:
    noise = Image.fromarray(
        np.random.default_rng(11).integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    noise.save(buf, format="PNG")
    validator = MatchValidator(StubFetcher(buf.getvalue()))
    candidate = SearchCandidate(
        url="https://example.com/n",
        title="N",
        image_url="https://example.com/n.jpg",
        source="mock",
    )
    assert validator.validate([candidate], sample_face_bytes) is None


def test_validator_empty_candidates(sample_face_bytes) -> None:
    validator = MatchValidator(StubFetcher(sample_face_bytes))
    assert validator.validate([], sample_face_bytes) is None


# ---------------------------------------------------------------------------
# HTTP fetching
# ---------------------------------------------------------------------------


def _site_session(page_html: str, image_bytes: bytes, robots_body: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            if robots_body is None:
                return httpx.Response(404)
            return httpx.Response(200, content=robots_body.encode("utf-8"))
        if path == "/post":
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=page_html.encode()
            )
        if path == "/img.jpg":
            return httpx.Response(
                200, headers={"content-type": "image/jpeg"}, content=image_bytes
            )
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_http_fetcher_page_and_image(sample_face_bytes) -> None:
    html = (
        "<html><head><title>Page Title</title>"
        "<meta property='og:image' content='https://site.example/img.jpg'/>"
        "</head><body><p>Hello   world</p></body></html>"
    )
    session = _site_session(html, sample_face_bytes)
    fetcher = HttpContentFetcher(session=session, respect_robots=True)

    page = fetcher.fetch_page("https://site.example/post")
    assert page.title == "Page Title"
    assert page.text == "Hello world"
    assert extract_title(page.html) == "Page Title"
    assert extract_og_image(page.html) == "https://site.example/img.jpg"

    image = fetcher.fetch_image("https://site.example/img.jpg")
    assert image == sample_face_bytes


def test_http_fetcher_respects_robots(sample_face_bytes) -> None:
    session = _site_session(
        "<html><body>x</body></html>",
        sample_face_bytes,
        robots_body="User-agent: *\nDisallow: /",
    )
    fetcher = HttpContentFetcher(session=session, respect_robots=True)
    with pytest.raises(RobotsDisallowed):
        fetcher.fetch_page("https://site.example/post")


def test_http_fetcher_ignores_robots_when_disabled(sample_face_bytes) -> None:
    session = _site_session(
        "<html><body>x</body></html>",
        sample_face_bytes,
        robots_body="User-agent: *\nDisallow: /",
    )
    fetcher = HttpContentFetcher(session=session, respect_robots=False)
    page = fetcher.fetch_page("https://site.example/post")
    assert page.html


def test_http_fetcher_rejects_bad_scheme() -> None:
    fetcher = HttpContentFetcher()
    with pytest.raises(FetchError):
        fetcher.fetch_page("file:///etc/passwd")
    with pytest.raises(FetchError):
        fetcher.fetch_page("ftp://example.com/file")
