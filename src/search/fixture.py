"""Demo fixture support.

The fixture is a locally stored JSON file that stands in for a previously
retrieved public result. It is used ONLY in demo mode and every object it
produces is labelled ``demo=True``. Nothing here pretends to be a live web
search.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from src.search.web_search import FetchError, PageContent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "data" / "demo_fixture.json"


class FixtureContentFetcher:
    """A :class:`ContentFetcher` backed by the demo fixture."""

    def __init__(self, fixture: dict | None = None, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self._fixture = fixture if fixture is not None else load_fixture(fixture_path)

    @property
    def fixture(self) -> dict:
        return self._fixture

    def fetch_page(self, url: str) -> PageContent:
        candidate = self._fixture["candidate"]
        if url != candidate["url"]:
            raise FetchError(
                f"Fixture only serves its configured URL "
                f"({candidate['url']}), got {url}"
            )
        page = self._fixture["page"]
        return PageContent(
            url=candidate["url"],
            final_url=candidate["url"],
            title=page.get("title"),
            html=page["html"],
            text=page.get("text", ""),
        )

    def fetch_image(self, url: str) -> bytes:
        candidate = self._fixture["candidate"]
        if url != candidate["image_url"]:
            raise FetchError(
                f"Fixture only serves its configured image URL "
                f"({candidate['image_url']}), got {url}"
            )
        return base64.b64decode(self._fixture["image_b64"])


def load_fixture(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        fixture = json.load(handle)
    if not fixture.get("demo"):
        raise ValueError(f"Fixture {path} is not labelled demo=True")
    return fixture
