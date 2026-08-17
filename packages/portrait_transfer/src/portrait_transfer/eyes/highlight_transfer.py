"""Pupil-centered, iris-clipped catchlight placement."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry.sampling import bilinear_sample, identity_map
from ..image_io import normalize_rgb
from ..types import EyeHighlightAsset
from .inpainting import remove_existing_highlights


def _mask_geometry(
    mask: NDArray[np.float32],
) -> tuple[tuple[float, float], float] | None:
    coordinates = np.argwhere(mask > 0.5)
    if len(coordinates) < 4:
        return None
    center = coordinates.mean(axis=0)
    radius = float(np.sqrt(len(coordinates) / np.pi))
    return (float(center[1]), float(center[0])), radius


def place_highlight(
    target_rgb: ArrayLike,
    target_iris_mask: ArrayLike,
    asset: EyeHighlightAsset,
    *,
    minimum_confidence: float = 0.15,
    scale_multiplier: float = 1.0,
    rotation_degrees: float = 0.0,
) -> NDArray[np.float32]:
    target = normalize_rgb(target_rgb)
    iris = np.clip(np.asarray(target_iris_mask, dtype=np.float32), 0.0, 1.0)
    geometry = _mask_geometry(iris)
    if (
        geometry is None
        or asset.confidence < minimum_confidence
        or asset.iris_radius <= 0
    ):
        return target.copy()
    (target_x, target_y), target_radius = geometry
    if not np.isfinite(scale_multiplier) or scale_multiplier <= 0.0:
        raise ValueError("scale_multiplier must be finite and positive")
    if not np.isfinite(rotation_degrees):
        raise ValueError("rotation_degrees must be finite")
    scale = target_radius / asset.iris_radius * scale_multiplier
    if not 0.35 <= scale <= 3.0:
        return target.copy()
    grid = identity_map((target.shape[0], target.shape[1]))
    relative_x = (grid[..., 0] - target_x) / scale
    relative_y = (grid[..., 1] - target_y) / scale
    placement_angle = asset.angle_radians + np.radians(rotation_degrees)
    cosine = np.cos(-placement_angle)
    sine = np.sin(-placement_angle)
    source_x = cosine * relative_x - sine * relative_y + asset.center[0]
    source_y = sine * relative_x + cosine * relative_y + asset.center[1]
    mapping = np.stack((source_x, source_y), axis=-1).astype(np.float32)
    foreground = np.asarray(
        bilinear_sample(asset.foreground_rgb, mapping, mode="constant"),
        dtype=np.float32,
    )
    alpha = (
        np.asarray(
            bilinear_sample(asset.alpha, mapping, mode="constant"), dtype=np.float32
        )
        * iris
    )
    alpha_coverage = float(np.mean(alpha > 0.01))
    iris_coverage = float(np.mean(iris > 0.5))
    if alpha_coverage > 0.30 * max(iris_coverage, 1e-6):
        return target.copy()
    return np.clip(
        target * (1.0 - alpha[..., None]) + foreground * alpha[..., None], 0.0, 1.0
    ).astype(np.float32)


def transfer_eye_highlights(
    target_rgb: ArrayLike,
    target_iris_masks: tuple[ArrayLike, ArrayLike],
    assets: tuple[EyeHighlightAsset | None, EyeHighlightAsset | None],
    *,
    scale_multipliers: tuple[float, float] = (1.0, 1.0),
    rotation_degrees: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float32]:
    output = normalize_rgb(target_rgb)
    for iris_mask, asset, scale, rotation in zip(
        target_iris_masks, assets, scale_multipliers, rotation_degrees
    ):
        if asset is None:
            continue
        output, _ = remove_existing_highlights(output, iris_mask)
        output = place_highlight(
            output,
            iris_mask,
            asset,
            scale_multiplier=scale,
            rotation_degrees=rotation,
        )
    return output
