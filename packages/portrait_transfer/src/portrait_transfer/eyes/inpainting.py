"""Small-component highlight detection and conservative nearest-fill inpainting."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from ..color.lab import rgb_to_lab
from ..image_io import normalize_rgb


def detect_existing_highlights(
    rgb: ArrayLike, iris_mask: ArrayLike
) -> NDArray[np.float32]:
    image = normalize_rgb(rgb)
    iris = np.asarray(iris_mask, dtype=np.float32) > 0.5
    lightness = rgb_to_lab(image)[..., 0]
    values = lightness[iris]
    if values.size == 0:
        return np.zeros(iris.shape, dtype=np.float32)
    threshold = max(0.60, float(np.percentile(values, 92)))
    candidate = iris & (lightness >= threshold)
    components, count = label(candidate)
    output = np.zeros(iris.shape, dtype=np.float32)
    maximum_area = max(2, int(0.08 * iris.sum()))
    for index in range(1, count + 1):
        component = components == index
        if 1 <= component.sum() <= maximum_area:
            output[component] = 1.0
    return cast(
        NDArray[np.float32],
        gaussian_filter(output, sigma=0.6, mode="constant").astype(np.float32),
    )


def remove_existing_highlights(
    rgb: ArrayLike, iris_mask: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    image = normalize_rgb(rgb)
    highlight = detect_existing_highlights(image, iris_mask)
    missing = highlight > 0.1
    valid = (np.asarray(iris_mask) > 0.5) & ~missing
    if not missing.any() or not valid.any():
        return image.copy(), highlight
    _, indices = distance_transform_edt(~valid, return_indices=True)
    filled = image.copy()
    replacement = image[indices[0], indices[1]]
    for channel in range(3):
        smooth = gaussian_filter(replacement[..., channel], sigma=0.8, mode="reflect")
        filled[..., channel] = np.where(missing, smooth, filled[..., channel])
    return np.clip(filled, 0.0, 1.0).astype(np.float32), highlight
