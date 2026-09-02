"""Perceptual image hashing.

Implements a DCT-based perceptual hash (pHash) with NumPy only — no SciPy
dependency. pHash captures visual similarity: two images that look the same
have a small Hamming distance even if their bytes differ (re-encoding,
resizing, slight compression).

pHash is a *similarity signal*, not a proof of identity or of content
integrity. Cryptographic integrity is handled by SHA-256 elsewhere.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 8
HIGHFREQ_FACTOR = 4


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix (C @ x @ C.T computes the 2-D DCT)."""
    k = np.arange(n, dtype=np.float64)[:, None]
    m = np.arange(n, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * k * (2.0 * m + 1.0) / (2.0 * n))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis * np.sqrt(2.0 / n)


def _dct_2d(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    basis = _dct_matrix(n)
    return basis @ arr @ basis.T


def compute_phash(
    image: Image.Image,
    hash_size: int = HASH_SIZE,
    highfreq_factor: int = HIGHFREQ_FACTOR,
) -> str:
    """Return a 64-bit perceptual hash as a 16-char hex string."""
    img_size = hash_size * highfreq_factor
    gray = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64)
    dct = _dct_2d(arr)
    low_freq = dct[:hash_size, :hash_size]
    median = float(np.median(low_freq))
    bits = (low_freq > median).ravel()
    # Pack 64 booleans into a 16-hex-char string.
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
    """Average hash — fast fallback for very small images."""
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64)
    mean = float(arr.mean())
    value = 0
    for px in arr.ravel():
        value = (value << 1) | int(px > mean)
    return f"{value:0{hash_size * hash_size // 4}x}"  # 64 bits -> 16 hex chars


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Number of differing bits between two hex perceptual hashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError(
            f"Hash length mismatch: {len(hash_a)} vs {len(hash_b)}"
        )
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    return bin(a ^ b).count("1")
