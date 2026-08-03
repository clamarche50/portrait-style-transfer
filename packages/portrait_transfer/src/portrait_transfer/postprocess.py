"""Final finite-range sanitation and confidence blending."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def blend_by_confidence(
    processed: ArrayLike, original: ArrayLike, confidence: ArrayLike
) -> NDArray[np.float32]:
    processed_value = np.asarray(processed, dtype=np.float32)
    original_value = np.asarray(original, dtype=np.float32)
    weights = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    if weights.ndim == 2:
        weights = weights[..., None]
    return (processed_value * weights + original_value * (1.0 - weights)).astype(
        np.float32
    )


def final_sanitize(image: ArrayLike) -> NDArray[np.float32]:
    value = np.asarray(image, dtype=np.float32)
    return np.clip(
        np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    ).astype(np.float32)
