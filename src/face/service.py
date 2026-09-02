"""High-level face processing service.

Combines detection + encoding and exposes:
- ``process``   -> structured face report (no identity),
- ``compare``   -> similarity between two images supplied by the same user.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.face.detector import FaceDetector, OpenCVFaceDetector
from src.face.encoder import FaceEncoder, LocalHistogramFaceEncoder, cosine_similarity
from src.models.schemas import (
    FaceDetectionResult,
    FaceInfo,
    SimilarityResult,
    SIMILARITY_NOTE,
)

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(
        self,
        detector: FaceDetector | None = None,
        encoder: FaceEncoder | None = None,
    ) -> None:
        self._detector = detector or OpenCVFaceDetector()
        self._encoder = encoder or LocalHistogramFaceEncoder()

    def process(
        self,
        image_bgr: np.ndarray,
        include_embeddings: bool = False,
    ) -> FaceDetectionResult:
        """Detect faces and optionally encode them.

        Embeddings are only included when explicitly requested and never leave
        the process (they are not written to provenance records or the chain).
        """
        boxes = self._detector.detect(image_bgr)
        faces: list[FaceInfo] = []
        for box in boxes:
            x, y, w, h = box
            crop = image_bgr[y : y + h, x : x + w]
            embedding: list[float] | None = None
            if include_embeddings:
                embedding = [
                    float(v) for v in self._encoder.encode(crop)
                ]
            faces.append(
                FaceInfo(
                    bbox=box,
                    embedding_dimension=self._encoder.embedding_dimension(),
                    embedding=embedding,
                )
            )
        return FaceDetectionResult(
            face_detected=len(faces) > 0,
            face_count=len(faces),
            faces=faces,
        )

    def compare(self, image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> SimilarityResult:
        """Compare the largest detected face in each user-supplied image."""
        result_a = self.process(image_a_bgr, include_embeddings=True)
        result_b = self.process(image_b_bgr, include_embeddings=True)
        if result_a.face_count == 0 or result_b.face_count == 0:
            raise ValueError(
                "Both images must contain a detectable face to be compared"
            )
        # Largest face = largest bounding-box area.
        face_a = max(result_a.faces, key=lambda f: f.bbox[2] * f.bbox[3])
        face_b = max(result_b.faces, key=lambda f: f.bbox[2] * f.bbox[3])
        emb_a = np.asarray(face_a.embedding, dtype=np.float64)
        emb_b = np.asarray(face_b.embedding, dtype=np.float64)
        score = cosine_similarity(emb_a, emb_b)
        return SimilarityResult(
            image_a="supplied-image-a",
            image_b="supplied-image-b",
            similarity_score=score,
            note=SIMILARITY_NOTE,
        )


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image with OpenCV (BGR). Raises ValueError when unreadable."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image file: {path}")
    return image
