"""Face detection.

The detector only reports *whether* a face exists and *where* it is. It never
attempts to identify, name, profile, or track a person. The implementation
uses the classic Viola-Jones Haar cascade (a reputable, fully offline
detector shipped with OpenCV). The ``FaceDetector`` protocol is pluggable so
a different detector (e.g. InsightFace YuNet) can be swapped in without
touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector(Protocol):
    """Detects faces and returns bounding boxes."""

    def detect(self, image_bgr: np.ndarray) -> list[list[int]]:
        """Return a list of ``[x, y, width, height]`` boxes (pixels)."""
        ...


class OpenCVFaceDetector:
    """Viola-Jones face detector backed by an OpenCV Haar cascade."""

    def __init__(
        self,
        cascade_path: str | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        min_size: int = 40,
    ) -> None:
        if cascade_path is None:
            cascade_path = f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, image_bgr: np.ndarray) -> list[list[int]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_size, self._min_size),
        )
        return [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in rects]
