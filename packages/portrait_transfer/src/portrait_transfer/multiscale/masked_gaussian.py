"""Separable normalized convolution that prevents foreground/background leakage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter


@dataclass
class MaskBlurCache:
    values: dict[tuple[int, float, float], NDArray[np.float32]] = field(
        default_factory=dict
    )

    def blurred(
        self, mask: NDArray[np.float32], sigma: float, truncate: float
    ) -> NDArray[np.float32]:
        key = (id(mask), float(sigma), float(truncate))
        if key not in self.values:
            self.values[key] = gaussian_filter(
                mask, sigma=sigma, mode="reflect", truncate=truncate
            ).astype(np.float32)
        return self.values[key]


def masked_gaussian(
    image: ArrayLike,
    mask: ArrayLike,
    sigma: float,
    *,
    mask_epsilon: float = 1e-6,
    truncate: float = 3.0,
    cache: MaskBlurCache | None = None,
    return_support: bool = False,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.bool_]]:
    value = np.asarray(image, dtype=np.float32)
    matte = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if value.shape[:2] != matte.shape or matte.ndim != 2:
        raise ValueError("mask must be HxW and match the image")
    if sigma <= 0 or mask_epsilon <= 0 or truncate <= 0:
        raise ValueError("sigma, mask_epsilon, and truncate must be positive")
    blurred_mask = (cache or MaskBlurCache()).blurred(matte, sigma, truncate)
    support = blurred_mask >= mask_epsilon
    denominator = np.maximum(blurred_mask, mask_epsilon)
    if value.ndim == 2:
        numerator = gaussian_filter(
            value * matte, sigma=sigma, mode="reflect", truncate=truncate
        )
        normalized = numerator / denominator
        output = np.where(support, normalized, value)
    elif value.ndim == 3:
        channels = []
        for channel in range(value.shape[2]):
            numerator = gaussian_filter(
                value[..., channel] * matte,
                sigma=sigma,
                mode="reflect",
                truncate=truncate,
            )
            normalized = numerator / denominator
            channels.append(np.where(support, normalized, value[..., channel]))
        output = np.stack(channels, axis=-1)
    else:
        raise ValueError("image must be 2-D or HxWxC")
    result = np.asarray(output, dtype=np.float32)
    if return_support:
        return result, support
    return result
