"""Face detection and local encoding (privacy-preserving)."""

from src.face.detector import FaceDetector, OpenCVFaceDetector
from src.face.encoder import (
    cosine_similarity,
    FaceEncoder,
    LocalHistogramFaceEncoder,
)
from src.face.service import FaceService, load_image_bgr

__all__ = [
    "FaceDetector",
    "OpenCVFaceDetector",
    "FaceEncoder",
    "LocalHistogramFaceEncoder",
    "cosine_similarity",
    "FaceService",
    "load_image_bgr",
]
