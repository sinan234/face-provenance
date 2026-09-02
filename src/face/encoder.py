"""Local face encoding.

The encoder produces a *deterministic technical-demonstration embedding* from
a face crop. It is intentionally NOT a state-of-the-art identity embedding:
it exists so the assignment's encoding step can be exercised locally, offline
and reproducibly, while remaining useless for identifying a person.

Use cases (privacy-respecting):
- generate an embedding for a technical demo,
- compare two images *supplied by the same user*,
- report a similarity score without naming anyone.

The ``FaceEncoder`` protocol is pluggable: swap in InsightFace / FaceNet for
a higher-quality embedding if you accept the extra model-download dependency.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceEncoder(Protocol):
    """Encodes a face crop into a fixed-dimension vector."""

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        ...

    def embedding_dimension(self) -> int:
        ...


class LocalHistogramFaceEncoder:
    """128-d embedding from block means and standard deviations.

    Embeddings are L2-normalized so cosine similarity reduces to a dot
    product. Deterministic for identical input.
    """

    BLOCK = 8  # 64x64 crop -> 8x8 blocks
    DIM = 2 * BLOCK * BLOCK

    def embedding_dimension(self) -> int:
        return self.DIM

    def encode(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            raise ValueError("Cannot encode an empty face crop")
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        gray /= 255.0
        blocks = gray.reshape(
            self.BLOCK, 8, self.BLOCK, 8
        )  # (8,8,8,8) block layout
        means = blocks.mean(axis=(1, 3)).ravel()
        stds = blocks.std(axis=(1, 3)).ravel()
        embedding = np.concatenate([means, stds])
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding /= norm
        return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings (0..1 for normalized vectors)."""
    if a.shape != b.shape:
        raise ValueError(f"Embedding dimension mismatch: {a.shape} vs {b.shape}")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
