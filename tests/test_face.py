"""Tests for the face-processing stage.

Covers: single face, no face, multiple faces, deterministic encoding, and
the privacy guarantee that no identity fields leak into the output.
"""

from __future__ import annotations

import numpy as np

from src.face.detector import OpenCVFaceDetector
from src.face.encoder import LocalHistogramFaceEncoder, cosine_similarity
from src.face.service import FaceService, load_image_bgr


class StubFaceDetector:
    """Deterministic detector for tests that do not need a real cascade."""

    def __init__(self, boxes: list[list[int]]) -> None:
        self._boxes = boxes

    def detect(self, image_bgr: np.ndarray) -> list[list[int]]:
        return [list(b) for b in self._boxes]


def _blank_image(size: int = 240) -> np.ndarray:
    return (np.ones((size, size, 3), dtype=np.uint8) * 255)


def test_single_face_detected_on_sample(sample_face_path) -> None:
    service = FaceService()
    result = service.process(load_image_bgr(str(sample_face_path)))
    assert result.face_detected is True
    assert result.face_count >= 1
    x, y, w, h = result.faces[0].bbox
    assert x >= 0 and y >= 0 and w > 0 and h > 0
    # Embeddings are NOT included unless explicitly requested.
    assert result.faces[0].embedding is None


def test_no_face_on_blank_image() -> None:
    service = FaceService()
    result = service.process(_blank_image())
    assert result.face_detected is False
    assert result.face_count == 0
    assert result.faces == []


def test_multiple_faces_reported() -> None:
    service = FaceService(detector=StubFaceDetector([[0, 0, 50, 50], [100, 10, 40, 40]]))
    result = service.process(_blank_image())
    assert result.face_count == 2
    assert [f.bbox for f in result.faces] == [[0, 0, 50, 50], [100, 10, 40, 40]]


def test_multiple_faces_real_detector_composite(sample_face_path) -> None:
    """Two copies of the same face side by side must yield two detections."""
    image = load_image_bgr(str(sample_face_path))
    composite = np.hstack([image, image])
    detector = OpenCVFaceDetector()
    boxes = detector.detect(composite)
    assert len(boxes) >= 2, f"Expected >=2 faces, got {len(boxes)}: {boxes}"


def test_embedding_dimension_and_determinism(sample_face_path) -> None:
    encoder = LocalHistogramFaceEncoder()
    image = load_image_bgr(str(sample_face_path))
    service = FaceService(detector=StubFaceDetector([[0, 0, 64, 64]]))
    result = service.process(image, include_embeddings=True)
    assert result.faces[0].embedding_dimension == LocalHistogramFaceEncoder.DIM
    assert len(result.faces[0].embedding) == LocalHistogramFaceEncoder.DIM

    crop = image[0:64, 0:64]
    emb1 = encoder.encode(crop)
    emb2 = encoder.encode(crop)
    assert np.array_equal(emb1, emb2)  # deterministic for identical input


def test_embedding_similarity_same_face_vs_noise(sample_face_path) -> None:
    encoder = LocalHistogramFaceEncoder()
    image = load_image_bgr(str(sample_face_path))
    crop = image[100:360, 100:360]  # face region from the sample
    same = encoder.encode(crop)
    bright = encoder.encode(np.clip(crop.astype(np.int16) + 25, 0, 255).astype(np.uint8))
    noise = encoder.encode(
        np.random.default_rng(42).integers(0, 256, size=crop.shape, dtype=np.uint8)
    )
    same_vs_bright = cosine_similarity(same, bright)
    same_vs_noise = cosine_similarity(same, noise)
    # The same face under mild perturbation must be far closer than noise.
    assert same_vs_bright > 0.9
    assert same_vs_bright > same_vs_noise
    assert same_vs_noise < 0.95


def test_no_identity_fields_in_output(sample_face_path) -> None:
    """Face results expose only geometry + dimension — never identity."""
    service = FaceService()
    result = service.process(load_image_bgr(str(sample_face_path)))
    assert set(result.model_dump().keys()) == {
        "face_detected",
        "face_count",
        "faces",
        "privacy_note",
    }
    for face in result.faces:
        assert set(face.model_dump().keys()) == {
            "bbox",
            "embedding_dimension",
            "embedding",
        }
