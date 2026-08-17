"""Validity and foldover diagnostics for absolute backward maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sampling import identity_map


@dataclass(frozen=True)
class MapValidity:
    valid: NDArray[np.bool_]
    jacobian: NDArray[np.float32]
    valid_fraction: float
    negative_jacobian_fraction: float
    displacement_p50: float
    displacement_p95: float


def jacobian_determinant(mapping: ArrayLike) -> NDArray[np.float32]:
    absolute = np.asarray(mapping, dtype=np.float32)
    if absolute.ndim != 3 or absolute.shape[2] != 2:
        raise ValueError("mapping must be HxWx2")
    dx_dx = np.gradient(absolute[..., 0], axis=1)
    dx_dy = np.gradient(absolute[..., 0], axis=0)
    dy_dx = np.gradient(absolute[..., 1], axis=1)
    dy_dy = np.gradient(absolute[..., 1], axis=0)
    return cast(
        NDArray[np.float32],
        (dx_dx * dy_dy - dx_dy * dy_dx).astype(np.float32),
    )


def map_validity(
    mapping: ArrayLike, source_shape: tuple[int, int] | tuple[int, int, int]
) -> MapValidity:
    absolute = np.asarray(mapping, dtype=np.float32)
    source_h, source_w = source_shape[:2]
    x, y = absolute[..., 0], absolute[..., 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= source_w - 1)
        & (y >= 0)
        & (y <= source_h - 1)
    )
    jacobian = jacobian_determinant(absolute)
    finite_jacobian = np.isfinite(jacobian)
    considered = valid & finite_jacobian
    negative = considered & (jacobian <= 0.0)
    destination_shape = (absolute.shape[0], absolute.shape[1])
    identity = identity_map(destination_shape)
    displacement = np.linalg.norm(
        cast(NDArray[np.float32], absolute - identity), axis=-1
    )
    valid_displacement = displacement[valid]
    p50 = (
        float(np.percentile(valid_displacement, 50))
        if valid_displacement.size
        else float("inf")
    )
    p95 = (
        float(np.percentile(valid_displacement, 95))
        if valid_displacement.size
        else float("inf")
    )
    return MapValidity(
        valid=valid,
        jacobian=jacobian,
        valid_fraction=float(valid.mean()),
        negative_jacobian_fraction=float(negative.sum() / max(considered.sum(), 1)),
        displacement_p50=p50,
        displacement_p95=p95,
    )


def replace_invalid_with_identity(
    mapping: ArrayLike, source_shape: tuple[int, int]
) -> NDArray[np.float32]:
    absolute = np.asarray(mapping, dtype=np.float32).copy()
    report = map_validity(absolute, source_shape)
    fallback = identity_map((absolute.shape[0], absolute.shape[1]))
    absolute[~report.valid] = fallback[~report.valid]
    return absolute
