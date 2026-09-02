"""Tests for the FastAPI web interface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.web.app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_index_serves_dashboard() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Face Provenance" in resp.text
    assert "/static/app.js" in resp.text


def test_sample_image_endpoint() -> None:
    resp = client.get("/api/sample-image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 1000


def test_process_demo_mode_success(sample_face_bytes) -> None:
    resp = client.post(
        "/api/process",
        files={"file": ("face.jpg", sample_face_bytes, "image/jpeg")},
        data={"mode": "demo", "chain": "memory"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["face"]["face_detected"] is True
    assert body["search"]["demo"] is True
    assert body["match"]["match_type"] == "EXACT_MATCH"
    assert len(body["fingerprint"]) == 64
    assert body["chain"]["simulated"] is True
    assert body["verification"]["verified"] is True
    assert body["completed"] is True


def test_process_rejects_blank_image(fixture) -> None:
    """A face-less image must stop the pipeline with a clear error."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (200, 200), "white").save(buf, format="JPEG")
    resp = client.post(
        "/api/process",
        files={"file": ("blank.jpg", buf.getvalue(), "image/jpeg")},
        data={"mode": "demo", "chain": "memory"},
    )
    assert resp.status_code == 422
    assert "No face detected" in resp.json()["detail"]


def test_verify_tamper_detection(sample_face_bytes) -> None:
    """Tampered demo content must fail verification through the API."""
    resp = client.post(
        "/api/verify",
        files={"file": ("face.jpg", sample_face_bytes, "image/jpeg")},
        data={"mode": "demo", "chain": "memory", "tamper": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verification"]["verified"] is False
    assert "differs from blockchain record" in body["verification"]["reason"]
