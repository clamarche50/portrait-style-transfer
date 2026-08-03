"""sRGB/D65 CIE Lab with normalized channel ranges.

L is stored in [0, 1]. The a and b channels are stored as conventional Lab
values divided by 128, giving stable values around [-1, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..image_io import normalize_rgb

_RGB_TO_XYZ = np.asarray(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def srgb_to_linear(rgb: ArrayLike) -> NDArray[np.float32]:
    value = normalize_rgb(rgb).astype(np.float64)
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def linear_to_srgb(linear_rgb: ArrayLike) -> NDArray[np.float32]:
    value = np.asarray(linear_rgb, dtype=np.float64)
    positive = np.maximum(value, 0.0)
    srgb = np.where(
        positive <= 0.0031308, 12.92 * positive, 1.055 * positive ** (1.0 / 2.4) - 0.055
    )
    return srgb.astype(np.float32)


def _f_xyz(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3.0 * _DELTA**2) + 4.0 / 29.0)


def _f_inverse(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(t > _DELTA, t**3, 3.0 * _DELTA**2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: ArrayLike) -> NDArray[np.float32]:
    linear = srgb_to_linear(rgb).astype(np.float64)
    xyz = linear @ _RGB_TO_XYZ.T
    f = _f_xyz(xyz / _D65)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack((lightness / 100.0, a / 128.0, b / 128.0), axis=-1).astype(
        np.float32
    )


def lab_to_rgb(lab: ArrayLike, *, clip: bool = False) -> NDArray[np.float32]:
    value = np.asarray(lab, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Lab image must be HxWx3")
    lightness = value[..., 0] * 100.0
    a = value[..., 1] * 128.0
    b = value[..., 2] * 128.0
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack((_f_inverse(fx), _f_inverse(fy), _f_inverse(fz)), axis=-1) * _D65
    linear = xyz @ _XYZ_TO_RGB.T
    rgb = linear_to_srgb(linear)
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)


def lab_to_rgb_gamut_safe(lab: ArrayLike) -> NDArray[np.float32]:
    rgb = lab_to_rgb(lab, clip=False)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
