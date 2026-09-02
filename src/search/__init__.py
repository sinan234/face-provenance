"""Reverse-image search, web fetching, and candidate validation."""

from src.search.result_parser import MatchValidator, MatchValidatorConfig
from src.search.reverse_image import (
    DEFAULT_FIXTURE_PATH,
    FixtureSearchProvider,
    ReverseImageSearchProvider,
    SerperLensSearchProvider,
)
from src.search.web_search import (
    ContentFetcher,
    FetchError,
    HttpContentFetcher,
    InvalidUrlError,
    RateLimitError,
    RobotsDisallowed,
    SearchProviderError,
)

__all__ = [
    "DEFAULT_FIXTURE_PATH",
    "MatchValidator",
    "MatchValidatorConfig",
    "ReverseImageSearchProvider",
    "SerperLensSearchProvider",
    "FixtureSearchProvider",
    "ContentFetcher",
    "HttpContentFetcher",
    "FetchError",
    "RateLimitError",
    "RobotsDisallowed",
    "InvalidUrlError",
    "SearchProviderError",
]
