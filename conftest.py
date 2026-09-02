"""Shared pytest fixtures and import-path setup."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search.fixture import DEFAULT_FIXTURE_PATH, load_fixture  # noqa: E402


@pytest.fixture(scope="session")
def fixture() -> dict:
    """The committed demo fixture (explicitly labelled demo=True)."""
    return load_fixture(DEFAULT_FIXTURE_PATH)


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return DEFAULT_FIXTURE_PATH


@pytest.fixture(scope="session")
def sample_face_bytes(fixture) -> bytes:
    return base64.b64decode(fixture["image_b64"])


@pytest.fixture(scope="session")
def sample_face_path(tmp_path_factory, sample_face_bytes) -> Path:
    directory = tmp_path_factory.mktemp("images")
    path = directory / "sample_face.jpg"
    path.write_bytes(sample_face_bytes)
    return path
