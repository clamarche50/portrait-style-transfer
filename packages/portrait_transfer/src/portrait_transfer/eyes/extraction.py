"""Deterministic Lab clustering for reference catchlight extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, label

from ..color.lab import rgb_to_lab
from ..image_io import normalize_rgb
from ..types import EyeHighlightAsset


def _kmeans(
    values: NDArray[np.float32], clusters: int = 3, iterations: int = 20
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    if len(values) < clusters:
        raise ValueError("too few iris pixels for highlight clustering")
    order = np.argsort(values[:, 0])
    indices = [
        order[round(position)] for position in np.linspace(0, len(values) - 1, clusters)
    ]
    centers = values[indices].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and _ > 0:
            break
        labels = updated
        for cluster in range(clusters):
            selected = values[labels == cluster]
            if len(selected):
                centers[cluster] = selected.mean(axis=0)
    return centers, labels


def extract_highlight_asset(
    reference_rgb: ArrayLike,
    iris_mask: ArrayLike,
    *,
    eye_angle_radians: float = 0.0,
    minimum_pixels: int = 12,
) -> EyeHighlightAsset | None:
    image = normalize_rgb(reference_rgb)
    iris = np.asarray(iris_mask, dtype=np.float32) > 0.5
    coordinates = np.argwhere(iris)
    if len(coordinates) < minimum_pixels:
        return None
    lab = rgb_to_lab(image)
    values = lab[iris]
    centers, labels_flat = _kmeans(values, clusters=3)
    scores = centers[:, 0] - 0.2 * np.linalg.norm(centers[:, 1:3], axis=1)
    highlight_cluster = int(np.argmax(scores))
    candidate = np.zeros(iris.shape, dtype=bool)
    candidate[iris] = labels_flat == highlight_cluster
    components, count = label(candidate)
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    selected = components == int(np.argmax(sizes))
    area_fraction = float(selected.sum() / max(iris.sum(), 1))
    if selected.sum() < 2 or area_fraction > 0.30:
        return None
    alpha = gaussian_filter(
        selected.astype(np.float32), sigma=0.8, mode="constant"
    ) * iris.astype(np.float32)
    if alpha.max() > 0:
        alpha /= alpha.max()
    iris_center_yx = coordinates.mean(axis=0)
    iris_radius = float(np.sqrt(iris.sum() / np.pi))
    confidence = float(
        np.clip(
            (centers[highlight_cluster, 0] - np.median(values[:, 0])) * 2.0
            + (0.3 - area_fraction),
            0.0,
            1.0,
        )
    )
    return EyeHighlightAsset(
        foreground_rgb=image.copy(),
        alpha=alpha.astype(np.float32),
        center=(float(iris_center_yx[1]), float(iris_center_yx[0])),
        iris_radius=iris_radius,
        angle_radians=float(eye_angle_radians),
        confidence=confidence,
    )
