"""Named groups and MediaPipe mapping for the reduced 68-landmark graph.

The production detector returns MediaPipe's 478-point face mesh.  The transfer
algorithm deliberately consumes a smaller, stable graph whose ordering matches
the familiar 68-point jaw/brow/nose/eye/lip topology used by the legacy line
morph.  Keeping the mapping here prevents detector-specific numeric indices
from leaking into the rest of the pipeline.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

LANDMARK_GROUPS: Final[dict[str, tuple[int, ...]]] = {
    "jaw": tuple(range(17)),
    "right_eyebrow": tuple(range(17, 22)),
    "left_eyebrow": tuple(range(22, 27)),
    "nose_ridge": tuple(range(27, 31)),
    "nose_base": tuple(range(31, 36)),
    "right_eye": tuple(range(36, 42)),
    "left_eye": tuple(range(42, 48)),
    "outer_lips": tuple(range(48, 60)),
    "inner_lips": tuple(range(60, 68)),
}

# MediaPipe Face Landmarker indices, grouped by their semantic role.  The eye
# names describe the subject's left/right side (not the viewer's side).
MEDIAPIPE_RIGHT_EYE_CONTOUR: Final[tuple[int, ...]] = (33, 160, 158, 133, 153, 144)
MEDIAPIPE_LEFT_EYE_CONTOUR: Final[tuple[int, ...]] = (362, 385, 387, 263, 373, 380)
MEDIAPIPE_RIGHT_IRIS: Final[tuple[int, ...]] = (469, 470, 471, 472)
MEDIAPIPE_LEFT_IRIS: Final[tuple[int, ...]] = (474, 475, 476, 477)

# The 68 entries are ordered as jaw (17), brows (5 + 5), nose (4 + 5), eyes
# (6 + 6), outer lips (12), and inner lips (8).  The lip entries intentionally
# use separate outer and inner MediaPipe rings; using the 20-point outer lip
# perimeter for both rings would create crossing Beier-Neely segments.
MEDIAPIPE_TO_REDUCED_68: Final[tuple[int, ...]] = (
    # Jaw.
    162,
    127,
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    152,
    377,
    400,
    378,
    379,
    # Subject-right and subject-left eyebrows.
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    # Nose ridge and base.
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    19,
    94,
    # Eyes.
    *MEDIAPIPE_RIGHT_EYE_CONTOUR,
    *MEDIAPIPE_LEFT_EYE_CONTOUR,
    # Outer lip ring.
    61,
    185,
    40,
    0,
    267,
    409,
    291,
    375,
    321,
    17,
    84,
    146,
    # Inner lip ring.
    78,
    81,
    13,
    311,
    308,
    402,
    14,
    178,
)

# Major features that must remain inside the frame for a production analysis.
MEDIAPIPE_REQUIRED_IN_FRAME: Final[tuple[int, ...]] = tuple(
    dict.fromkeys(
        (
            *MEDIAPIPE_RIGHT_EYE_CONTOUR,
            *MEDIAPIPE_LEFT_EYE_CONTOUR,
            *MEDIAPIPE_RIGHT_IRIS,
            *MEDIAPIPE_LEFT_IRIS,
            *MEDIAPIPE_TO_REDUCED_68[27:],
        )
    )
)

if len(MEDIAPIPE_TO_REDUCED_68) != 68:  # pragma: no cover - import invariant
    raise RuntimeError("MediaPipe reduced landmark mapping must contain 68 entries")


def group_points(landmarks: ArrayLike, name: str) -> NDArray[np.float32]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("landmarks must be Nx2")
    indices = LANDMARK_GROUPS[name]
    if max(indices) >= len(points):
        raise ValueError(f"landmark set does not contain group {name}")
    return cast(NDArray[np.float32], points[np.asarray(indices)])


def eye_centers(
    landmarks: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    points = np.asarray(landmarks, dtype=np.float32)
    right = group_points(points, "right_eye").mean(axis=0)
    left = group_points(points, "left_eye").mean(axis=0)
    return left.astype(np.float32), right.astype(np.float32)


def alignment_anchors(
    landmarks: ArrayLike, *, include_nose: bool = True
) -> NDArray[np.float32]:
    points = np.asarray(landmarks, dtype=np.float32)
    left_eye, right_eye = eye_centers(points)
    mouth_center = group_points(points, "outer_lips").mean(axis=0)
    anchors = [left_eye, right_eye, mouth_center]
    if include_nose:
        anchors.append(group_points(points, "nose_base").mean(axis=0))
    return cast(NDArray[np.float32], np.stack(anchors).astype(np.float32))


def normalized_landmark_shape(landmarks: ArrayLike) -> NDArray[np.float32]:
    points = np.asarray(landmarks, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale < 1e-9:
        raise ValueError("landmarks have no spatial extent")
    return cast(NDArray[np.float32], (centered / scale).astype(np.float32))


def mediapipe_to_reduced_68(
    normalized_landmarks: ArrayLike,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    """Map an exact 478-point normalized MediaPipe mesh into image pixels."""

    points = np.asarray(normalized_landmarks, dtype=np.float32)
    if points.shape != (478, 2):
        raise ValueError("MediaPipe Face Landmarker must return exactly 478 x/y points")
    if not np.isfinite(points).all():
        raise ValueError("MediaPipe landmarks must be finite")
    height, width = image_shape[:2]
    pixels = np.empty_like(points)
    pixels[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * max(width - 1, 1)
    pixels[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * max(height - 1, 1)
    return cast(
        NDArray[np.float32],
        pixels[np.asarray(MEDIAPIPE_TO_REDUCED_68)].astype(np.float32),
    )
