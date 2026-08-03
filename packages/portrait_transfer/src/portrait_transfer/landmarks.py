"""Landmark validation and a deterministic reduced-68 synthetic template."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_landmarks(
    landmarks: ArrayLike,
    image_shape: tuple[int, int] | tuple[int, int, int],
    minimum: int = 68,
) -> NDArray[np.float32]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"landmarks must contain at least {minimum} x/y points")
    if not np.isfinite(points).all():
        raise ValueError("landmarks must be finite")
    height, width = image_shape[:2]
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] > width - 1)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] > height - 1)
    ):
        raise ValueError("landmarks must lie within the image")
    return points


def _ellipse(
    center: tuple[float, float],
    radii: tuple[float, float],
    count: int,
    start: float = 0.0,
) -> NDArray[np.float64]:
    angles = start + np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        (center[0] + radii[0] * np.cos(angles), center[1] + radii[1] * np.sin(angles))
    )


def canonical_68_landmarks(
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    """Create a standard-topology template for tests and explicit fallback use."""

    height, width = image_shape[:2]
    points: list[tuple[float, float]] = []
    jaw_angles = np.linspace(np.pi, 2.0 * np.pi, 17)
    for angle in jaw_angles:
        points.append((0.5 + 0.42 * np.cos(angle), 0.43 - 0.46 * np.sin(angle)))
    for x, y in zip(np.linspace(0.20, 0.42, 5), (0.34, 0.30, 0.29, 0.30, 0.33)):
        points.append((float(x), float(y)))
    for x, y in zip(np.linspace(0.58, 0.80, 5), (0.33, 0.30, 0.29, 0.30, 0.34)):
        points.append((float(x), float(y)))
    for y in np.linspace(0.36, 0.59, 4):
        points.append((0.50, float(y)))
    points.extend(
        ((0.40, 0.62), (0.45, 0.65), (0.50, 0.66), (0.55, 0.65), (0.60, 0.62))
    )
    right_eye = _ellipse((0.34, 0.43), (0.095, 0.045), 6, start=np.pi)
    left_eye = _ellipse((0.66, 0.43), (0.095, 0.045), 6, start=np.pi)
    points.extend(map(tuple, right_eye))
    points.extend(map(tuple, left_eye))
    outer_mouth = _ellipse((0.50, 0.74), (0.18, 0.075), 12, start=np.pi)
    inner_mouth = _ellipse((0.50, 0.74), (0.095, 0.035), 8, start=np.pi)
    points.extend(map(tuple, outer_mouth))
    points.extend(map(tuple, inner_mouth))
    normalized = np.asarray(points, dtype=np.float32)
    normalized[:, 0] *= max(width - 1, 1)
    normalized[:, 1] *= max(height - 1, 1)
    return normalized
