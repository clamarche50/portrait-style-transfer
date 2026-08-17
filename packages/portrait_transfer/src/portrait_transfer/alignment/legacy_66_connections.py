"""Documented topology of the archived 66-index facial contour graph.

The graph follows the familiar reduced 68-point face layout. The archived
inner-lip curve stopped at index 65, so indices 66 and 67 are intentionally not
part of the compatibility topology. Production callers may append a complete
inner-lip curve as an explicit extension.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

LEGACY_66_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
    *((i, i + 1) for i in range(16)),
    *((i, i + 1) for i in range(17, 21)),
    *((i, i + 1) for i in range(22, 26)),
    *((i, i + 1) for i in range(27, 30)),
    *((i, i + 1) for i in range(31, 35)),
    (36, 37),
    (37, 38),
    (38, 39),
    (39, 40),
    (40, 41),
    (41, 36),
    (42, 43),
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 42),
    (48, 49),
    (49, 50),
    (50, 51),
    (51, 52),
    (52, 53),
    (53, 54),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    (58, 59),
    (59, 48),
    (60, 65),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
)


def boundary_segments(
    shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    height, width = shape[:2]
    return np.asarray(
        [
            ((0.0, 0.0), (0.0, height - 1.0)),
            ((0.0, height - 1.0), (width - 1.0, height - 1.0)),
            ((width - 1.0, height - 1.0), (width - 1.0, 0.0)),
            ((width - 1.0, 0.0), (0.0, 0.0)),
        ],
        dtype=np.float32,
    )


def landmark_segments(
    landmarks: ArrayLike,
    *,
    connections: tuple[tuple[int, int], ...] = LEGACY_66_CONNECTIONS,
) -> NDArray[np.float32]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("landmarks must be Nx2")
    indices = np.asarray(connections, dtype=np.int64)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(points):
        raise ValueError("connections reference unavailable landmarks")
    return points[indices]


def validate_legacy_connections(landmarks: ArrayLike | None = None) -> bool:
    indices = np.asarray(LEGACY_66_CONNECTIONS, dtype=np.int64)
    if len(LEGACY_66_CONNECTIONS) != 61 or indices.min() != 0 or indices.max() != 65:
        return False
    if np.any(indices[:, 0] == indices[:, 1]):
        return False
    if landmarks is not None:
        segments = landmark_segments(landmarks)
        if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1e-6):
            return False
    return True
