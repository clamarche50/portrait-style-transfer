"""Single-convention sampling: destination pixels map to absolute source x/y."""

from __future__ import annotations

from typing import Final, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates

# Out-of-bounds fill used by the archived MATLAB warpImage.m.
MATLAB_OOB_FILL: Final[float] = 0.6


def identity_map(shape: tuple[int, int] | tuple[int, int, int]) -> NDArray[np.float32]:
    height, width = shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((xx, yy), axis=-1)


def bilinear_sample(
    source: ArrayLike,
    mapping: ArrayLike,
    *,
    mode: str = "constant",
    cval: float = 0.0,
    return_validity: bool = False,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Sample a 2-D or channel-last source at absolute `(x, y)` coordinates."""

    image = np.asarray(source, dtype=np.float32)
    coordinates = np.asarray(mapping, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[2] != 2:
        raise ValueError("mapping must have shape HxWx2 with x/y coordinates")
    if image.ndim not in (2, 3):
        raise ValueError("source must be 2-D or HxWxC")
    source_h, source_w = image.shape[:2]
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= source_w - 1)
        & (y >= 0)
        & (y <= source_h - 1)
    )
    safe_x = np.nan_to_num(x, nan=-1.0, posinf=-1.0, neginf=-1.0)
    safe_y = np.nan_to_num(y, nan=-1.0, posinf=-1.0, neginf=-1.0)
    scipy_mode = {
        "constant": "constant",
        "border": "nearest",
        "reflect": "reflect",
    }.get(mode)
    if scipy_mode is None:
        raise ValueError("mode must be 'constant', 'border', or 'reflect'")

    def sample_plane(plane: NDArray[np.float32]) -> NDArray[np.float32]:
        return cast(
            NDArray[np.float32],
            map_coordinates(
                plane,
                (safe_y, safe_x),
                order=1,
                mode=scipy_mode,
                cval=float(cval),
                prefilter=False,
            ).astype(np.float32),
        )

    if image.ndim == 2:
        sampled = sample_plane(image)
    else:
        sampled = np.stack(
            [sample_plane(image[..., channel]) for channel in range(image.shape[2])],
            axis=-1,
        )
    if return_validity:
        return sampled, valid
    return sampled


def warp(
    source: ArrayLike,
    map_x: ArrayLike,
    map_y: ArrayLike | None = None,
    *,
    mode: str = "constant",
    cval: float = 0.0,
) -> NDArray[np.float32]:
    if map_y is None:
        mapping = np.asarray(map_x, dtype=np.float32)
    else:
        mapping = np.stack((np.asarray(map_x), np.asarray(map_y)), axis=-1).astype(
            np.float32
        )
    return np.asarray(
        bilinear_sample(source, mapping, mode=mode, cval=cval), dtype=np.float32
    )
