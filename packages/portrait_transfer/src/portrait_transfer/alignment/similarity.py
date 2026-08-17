"""Robust reflection-free similarity alignment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TypedDict, Unpack

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import AlignmentFailure
from ..geometry.transforms import affine_backward_map, transform_points


@dataclass(frozen=True)
class SimilarityEstimate:
    matrix: NDArray[np.float64]
    scale: float
    rotation_degrees: float
    inliers: NDArray[np.bool_]
    normalized_error: float


class SimilarityOptions(TypedDict, total=False):
    ransac_threshold: float
    min_inliers: int
    scale_range: tuple[float, float]
    max_rotation_degrees: float


def _umeyama(
    source: NDArray[np.float64], destination: NDArray[np.float64]
) -> NDArray[np.float64]:
    if len(source) < 2:
        raise AlignmentFailure("At least two anchors are required")
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance < 1e-12:
        raise AlignmentFailure("Source anchors are degenerate")
    covariance = destination_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular * sign) / variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    return np.asarray(
        [
            [scale * rotation[0, 0], scale * rotation[0, 1], translation[0]],
            [scale * rotation[1, 0], scale * rotation[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_similarity(
    source_points: ArrayLike,
    destination_points: ArrayLike,
    *,
    ransac_threshold: float = 0.04,
    min_inliers: int = 3,
    scale_range: tuple[float, float] = (0.6, 1.7),
    max_rotation_degrees: float = 30.0,
) -> SimilarityEstimate:
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise AlignmentFailure("Anchor arrays must be matching Nx2 values")
    if len(source) < min_inliers:
        raise AlignmentFailure(
            "Too few anchors", count=len(source), required=min_inliers
        )
    destination_extent = max(
        float(np.ptp(destination[:, 0])), float(np.ptp(destination[:, 1])), 1.0
    )
    absolute_threshold = ransac_threshold * destination_extent

    best_inliers: NDArray[np.bool_] | None = None
    best_error = float("inf")
    for pair in combinations(range(len(source)), 2):
        try:
            candidate = _umeyama(source[list(pair)], destination[list(pair)])
        except AlignmentFailure:
            continue
        predicted = transform_points(source, candidate)
        errors = np.linalg.norm(predicted - destination, axis=1)
        inliers = errors <= absolute_threshold
        mean_error = float(errors[inliers].mean()) if inliers.any() else float("inf")
        if (
            best_inliers is None
            or inliers.sum() > best_inliers.sum()
            or (inliers.sum() == best_inliers.sum() and mean_error < best_error)
        ):
            best_inliers = inliers
            best_error = mean_error
    if best_inliers is None or int(best_inliers.sum()) < min_inliers:
        raise AlignmentFailure("Similarity RANSAC found too few inlier anchors")

    matrix = _umeyama(source[best_inliers], destination[best_inliers])
    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise AlignmentFailure("Similarity transform contains a reflection")
    scale = float(np.sqrt(determinant))
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    predicted = transform_points(source, matrix)
    errors = np.linalg.norm(predicted - destination, axis=1)
    normalized_error = float(errors[best_inliers].mean() / destination_extent)
    if not scale_range[0] <= scale <= scale_range[1]:
        raise AlignmentFailure("Similarity scale is outside limits", scale=scale)
    if abs(rotation) > max_rotation_degrees:
        raise AlignmentFailure(
            "Similarity rotation is outside limits", rotation=rotation
        )
    return SimilarityEstimate(matrix, scale, rotation, best_inliers, normalized_error)


def estimate_similarity_backward_map(
    source_points: ArrayLike,
    destination_points: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
    **kwargs: Unpack[SimilarityOptions],
) -> tuple[NDArray[np.float32], SimilarityEstimate]:
    estimate = estimate_similarity(source_points, destination_points, **kwargs)
    return affine_backward_map(estimate.matrix, destination_shape), estimate
