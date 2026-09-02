"""Tests for the image publisher that makes local images public for search."""

from __future__ import annotations

import json

import httpx
import pytest

from src.search.publish import UguuImagePublisher
from src.search.web_search import FetchError


def _publisher(payload: dict | str, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=json.dumps(payload) if isinstance(payload, dict) else payload,
        )

    session = httpx.Client(transport=httpx.MockTransport(handler))
    return UguuImagePublisher(session=session)


def test_publisher_returns_public_url() -> None:
    pub = _publisher({"success": True, "files": [{"url": "https://h.uguu.se/abc.jpg"}]})
    url = pub.publish(b"bytes", "image/jpeg")
    assert url == "https://h.uguu.se/abc.jpg"


def test_publisher_error_on_malformed() -> None:
    pub = _publisher("not json")
    with pytest.raises(FetchError):
        pub.publish(b"bytes", "image/jpeg")


def test_publisher_error_on_http_failure() -> None:
    pub = _publisher({}, status=500)
    with pytest.raises(FetchError):
        pub.publish(b"bytes", "image/jpeg")