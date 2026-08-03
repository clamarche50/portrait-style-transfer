"""Laplacian detail and residual reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import LaplacianStack


def reconstruct(
    bands: tuple[ArrayLike, ...] | list[ArrayLike], residual: ArrayLike
) -> NDArray[np.float32]:
    result = np.asarray(residual, dtype=np.float32).copy()
    for band in bands:
        value = np.asarray(band, dtype=np.float32)
        if value.shape != result.shape:
            raise ValueError("all bands and residual must share a shape")
        result += value
    return result.astype(np.float32)


def reconstruct_stack(stack: LaplacianStack) -> NDArray[np.float32]:
    return reconstruct(stack.bands, stack.residual)


def blend_residual(
    input_residual: ArrayLike, reference_residual: ArrayLike, strength: float
) -> NDArray[np.float32]:
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("residual strength must be in [0, 1]")
    input_value = np.asarray(input_residual, dtype=np.float32)
    reference_value = np.asarray(reference_residual, dtype=np.float32)
    if input_value.shape != reference_value.shape:
        raise ValueError("residual arrays must match")
    return ((1.0 - strength) * input_value + strength * reference_value).astype(
        np.float32
    )
