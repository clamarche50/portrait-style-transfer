"""Paper-inspired multiscale energy features and reference ranking."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import zoom

from .alignment.anchors import normalized_landmark_shape
from .color.lab import rgb_to_lab
from .config import AlgorithmProfile
from .multiscale.energy import local_energy
from .multiscale.laplacian import build_laplacian_stack
from .quality import normalized_cross_correlation
from .types import PoseEstimate, RankedStyle, StyleFeature

_MIN_POSE_SIMILARITY = 0.10
_MIN_LANDMARK_SHAPE_SIMILARITY = 0.10
_MIN_MASK_QUALITY = 0.20


def _resize_2d(
    image: NDArray[np.float32], shape: tuple[int, int]
) -> NDArray[np.float32]:
    if image.shape == shape:
        return image
    return cast(
        NDArray[np.float32],
        zoom(
            image,
            (shape[0] / image.shape[0], shape[1] / image.shape[1]),
            order=1,
            mode="reflect",
            prefilter=False,
        ).astype(np.float32),
    )


def build_style_feature(
    identifier: str,
    rgb: ArrayLike,
    head_mask: ArrayLike,
    *,
    pose: PoseEstimate | None = None,
    landmarks: ArrayLike | None = None,
    mask_quality: float = 1.0,
) -> StyleFeature:
    image = np.asarray(rgb, dtype=np.float32)
    mask = np.clip(np.asarray(head_mask, dtype=np.float32), 0.0, 1.0)
    lightness = rgb_to_lab(image)[..., 0]
    stack = build_laplacian_stack(lightness, mask, profile=AlgorithmProfile.PAPER_EXACT)
    features: list[NDArray[np.float32]] = []
    for level, band in enumerate(stack.bands):
        energy = local_energy(band, mask, 2 ** (level + 1)) * mask
        features.append(_resize_2d(energy, (32, 32)).ravel())
    vector = np.concatenate(features).astype(np.float32)
    vector -= float(vector.mean())
    standard_deviation = float(vector.std())
    if standard_deviation > 1e-8:
        vector /= standard_deviation
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    shape = None if landmarks is None else normalized_landmark_shape(landmarks)
    stable_pixels = rgb_to_lab(image)[mask > 0.5]
    photometric = (
        np.median(stable_pixels, axis=0).astype(np.float32)
        if stable_pixels.size
        else np.zeros(3, dtype=np.float32)
    )
    return StyleFeature(
        identifier,
        vector,
        pose or PoseEstimate(),
        shape,
        photometric,
        float(np.clip(mask_quality, 0.0, 1.0)),
    )


def _pose_similarity(first: PoseEstimate | None, second: PoseEstimate | None) -> float:
    if first is None or second is None:
        return 0.5
    difference = np.linalg.norm(
        np.asarray(
            [
                first.yaw - second.yaw,
                first.pitch - second.pitch,
                first.roll - second.roll,
            ]
        )
    )
    return float(np.exp(-difference / 25.0))


def _shape_similarity(
    first: NDArray[np.float32] | None, second: NDArray[np.float32] | None
) -> float:
    if first is None or second is None or first.shape != second.shape:
        return 0.5
    return float(np.exp(-4.0 * np.mean(np.linalg.norm(first - second, axis=1))))


def _photometric_similarity(
    first: NDArray[np.float32] | None, second: NDArray[np.float32] | None
) -> float:
    if first is None or second is None:
        return 0.5
    return float(np.exp(-np.linalg.norm(first - second)))


def rank_style_examples(
    query: StyleFeature,
    candidates: list[StyleFeature] | tuple[StyleFeature, ...],
    top_k: int = 3,
) -> list[RankedStyle]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked: list[RankedStyle] = []
    for candidate in candidates:
        ncc = (normalized_cross_correlation(query.vector, candidate.vector) + 1.0) / 2.0
        pose = _pose_similarity(query.pose, candidate.pose)
        shape = _shape_similarity(query.landmark_shape, candidate.landmark_shape)
        photo = _photometric_similarity(
            query.photometric_lab, candidate.photometric_lab
        )
        if (
            (
                query.pose is not None
                and candidate.pose is not None
                and pose < _MIN_POSE_SIMILARITY
            )
            or (
                query.landmark_shape is not None
                and candidate.landmark_shape is not None
                and shape < _MIN_LANDMARK_SHAPE_SIMILARITY
            )
            or candidate.mask_quality < _MIN_MASK_QUALITY
        ):
            continue
        score = (
            0.65 * ncc
            + 0.15 * pose
            + 0.10 * shape
            + 0.05 * photo
            + 0.05 * candidate.mask_quality
        )
        ranked.append(
            RankedStyle(
                candidate.identifier,
                float(score),
                float(ncc),
                pose,
                shape,
                photo,
                candidate.mask_quality,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.identifier))
    return ranked[:top_k]


__all__ = [
    "build_style_feature",
    "normalized_cross_correlation",
    "rank_style_examples",
]
