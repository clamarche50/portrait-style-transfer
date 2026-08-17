"""Foreground-safe background extraction and compositing modes."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

from .config import BackgroundMode, TransferSettings
from .image_io import normalize_rgb


def _resize_to(
    image: NDArray[np.float32], shape: tuple[int, int]
) -> NDArray[np.float32]:
    if image.shape[:2] == shape:
        return image.copy()
    factors = (shape[0] / image.shape[0], shape[1] / image.shape[1], 1.0)
    return cast(
        NDArray[np.float32],
        zoom(image, factors, order=1, mode="reflect", prefilter=False).astype(
            np.float32
        ),
    )


def extract_reference_background(
    reference_rgb: ArrayLike, foreground_alpha: ArrayLike
) -> NDArray[np.float32]:
    """Remove foreground by nearest valid-background fill plus gentle smoothing."""

    reference = normalize_rgb(reference_rgb)
    alpha = np.clip(np.asarray(foreground_alpha, dtype=np.float32), 0.0, 1.0)
    if alpha.shape != reference.shape[:2]:
        raise ValueError("foreground alpha must match reference")
    missing = alpha > 0.05
    valid = ~missing
    if not valid.any():
        return np.full_like(reference, np.median(reference.reshape(-1, 3), axis=0))
    _, indices = distance_transform_edt(missing, return_indices=True)
    filled = reference.copy()
    filled[missing] = reference[indices[0][missing], indices[1][missing]]
    for channel in range(3):
        smoothed = gaussian_filter(filled[..., channel], sigma=2.0, mode="reflect")
        filled[..., channel] = np.where(missing, smoothed, filled[..., channel])
    return np.clip(filled, 0.0, 1.0).astype(np.float32)


def apply_background_mode(
    processed_foreground: ArrayLike,
    original_input: ArrayLike,
    reference_rgb: ArrayLike,
    foreground_alpha: ArrayLike,
    settings: TransferSettings,
    *,
    reference_alpha: ArrayLike | None = None,
) -> NDArray[np.float32]:
    processed = normalize_rgb(processed_foreground)
    original = normalize_rgb(original_input)
    alpha = np.clip(np.asarray(foreground_alpha, dtype=np.float32), 0.0, 1.0)
    if processed.shape != original.shape or alpha.shape != original.shape[:2]:
        raise ValueError("processed image, original, and alpha must align")
    if settings.background_mode is BackgroundMode.KEEP:
        background = original
    elif settings.background_mode is BackgroundMode.BLUR:
        background = np.stack(
            [
                gaussian_filter(
                    original[..., channel],
                    settings.background_blur_sigma,
                    mode="reflect",
                )
                for channel in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
    elif settings.background_mode is BackgroundMode.SOLID:
        background = np.broadcast_to(
            np.asarray(settings.background_color, dtype=np.float32), original.shape
        ).copy()
    else:
        reference = normalize_rgb(reference_rgb)
        ref_alpha = (
            np.zeros(reference.shape[:2], dtype=np.float32)
            if reference_alpha is None
            else np.asarray(reference_alpha, dtype=np.float32)
        )
        background = _resize_to(
            extract_reference_background(reference, ref_alpha),
            (original.shape[0], original.shape[1]),
        )
    matte = alpha[..., None]
    return np.clip(processed * matte + background * (1.0 - matte), 0.0, 1.0).astype(
        np.float32
    )
