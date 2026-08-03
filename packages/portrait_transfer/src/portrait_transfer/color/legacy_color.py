"""Numerical compatibility with the archived non-gamma-corrected Lab helper."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..image_io import normalize_rgb

_RGB_TO_XYZ = np.asarray(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.asarray(
    [
        [3.240479, -1.537150, -0.498535],
        [-0.969256, 1.875992, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ],
    dtype=np.float64,
)


def legacy_rgb_to_lab(rgb: ArrayLike) -> NDArray[np.float32]:
    value = normalize_rgb(rgb).astype(np.float64)
    xyz = value @ _RGB_TO_XYZ.T
    x = xyz[..., 0] / 0.950456
    y = xyz[..., 1]
    z = xyz[..., 2] / 1.088754
    threshold = 0.008856

    def f(channel: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(
            channel > threshold, np.cbrt(channel), 7.787 * channel + 16.0 / 116.0
        )

    fx, fy, fz = f(x), f(y), f(z)
    lightness = np.where(y > threshold, 116.0 * np.cbrt(y) - 16.0, 903.3 * y)
    return np.stack((lightness, 500.0 * (fx - fy), 200.0 * (fy - fz)), axis=-1).astype(
        np.float32
    )


def legacy_lab_to_rgb(lab: ArrayLike) -> NDArray[np.float32]:
    value = np.asarray(lab, dtype=np.float64)
    lightness, a, b = value[..., 0], value[..., 1], value[..., 2]
    fy_cubed = ((lightness + 16.0) / 116.0) ** 3
    y_test = fy_cubed > 0.008856
    y = np.where(y_test, fy_cubed, lightness / 903.3)
    fy = np.where(y_test, np.cbrt(y), 7.787 * y + 16.0 / 116.0)
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    x = np.where(fx > 0.206893, fx**3, (fx - 16.0 / 116.0) / 7.787) * 0.950456
    z = np.where(fz > 0.206893, fz**3, (fz - 16.0 / 116.0) / 7.787) * 1.088754
    xyz = np.stack((x, y, z), axis=-1)
    rgb = xyz @ _XYZ_TO_RGB.T
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
