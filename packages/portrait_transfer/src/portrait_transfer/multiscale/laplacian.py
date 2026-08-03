"""Full-resolution paper and archived-source Laplacian stacks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import PAPER_SIGMAS, SOURCE_SIGMAS, AlgorithmProfile
from ..types import LaplacianStack
from .masked_gaussian import MaskBlurCache, masked_gaussian


def build_laplacian_stack(
    image: ArrayLike,
    mask: ArrayLike,
    *,
    profile: AlgorithmProfile = AlgorithmProfile.PAPER_EXACT,
    mask_epsilon: float = 1e-6,
    truncate: float = 3.0,
) -> LaplacianStack:
    value = np.asarray(image, dtype=np.float32)
    matte = np.asarray(mask, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("a Laplacian channel must be 2-D")
    sigmas = PAPER_SIGMAS if profile is AlgorithmProfile.PAPER_EXACT else SOURCE_SIGMAS
    cache = MaskBlurCache()
    blurred = tuple(
        np.asarray(
            masked_gaussian(
                value,
                matte,
                sigma,
                mask_epsilon=mask_epsilon,
                truncate=truncate,
                cache=cache,
            ),
            dtype=np.float32,
        )
        for sigma in sigmas
    )
    bands: list[NDArray[np.float32]] = [value - blurred[0]]
    bands.extend(
        blurred[index] - blurred[index + 1] for index in range(len(blurred) - 1)
    )
    return LaplacianStack(
        bands=tuple(np.asarray(band, dtype=np.float32) for band in bands),
        residual=blurred[-1].astype(np.float32),
        sigmas=tuple(sigmas),
        profile=profile.value,
    )


def build_masked_laplacian_stack(
    image: ArrayLike,
    mask: ArrayLike,
    levels: int = 6,
    *,
    profile: AlgorithmProfile = AlgorithmProfile.PAPER_EXACT,
) -> LaplacianStack:
    expected = 6 if profile is AlgorithmProfile.PAPER_EXACT else 5
    if levels != expected:
        raise ValueError(f"{profile.value} requires {expected} detail levels")
    return build_laplacian_stack(image, mask, profile=profile)
