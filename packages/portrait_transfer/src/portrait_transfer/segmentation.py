"""Injectable segmentation plus deterministic mask-refinement helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from .alignment.anchors import group_points
from .exceptions import MaskFailure
from .types import BoundingBox, PortraitMasks


class SegmentationBackend(Protocol):
    segment: Callable[[NDArray[np.float32]], dict[str, NDArray[np.float32]]]


class MattingRefiner(Protocol):
    refine: Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float32]]


def ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], radii: tuple[float, float]
) -> NDArray[np.float32]:
    height, width = shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    rx, ry = max(radii[0], 1e-6), max(radii[1], 1e-6)
    return (
        (((xx - center[0]) / rx) ** 2 + ((yy - center[1]) / ry) ** 2) <= 1.0
    ).astype(np.float32)


def _component_at(
    mask: NDArray[np.bool_], point: tuple[float, float]
) -> NDArray[np.bool_]:
    components, count = label(mask)
    if count == 0:
        return mask
    x = int(np.clip(round(point[0]), 0, mask.shape[1] - 1))
    y = int(np.clip(round(point[1]), 0, mask.shape[0] - 1))
    selected = int(components[y, x])
    if selected == 0:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        selected = int(np.argmax(sizes))
    return cast(NDArray[np.bool_], components == selected)


def refine_head_mask(
    confidence: ArrayLike,
    face_box: BoundingBox,
    *,
    threshold: float = 0.35,
    feather_pixels: float = 4.0,
) -> NDArray[np.float32]:
    value = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    binary = binary_fill_holes(binary_closing(value >= threshold, iterations=2))
    binary = _component_at(binary, face_box.center)
    inside = distance_transform_edt(binary)
    outside = distance_transform_edt(~binary)
    signed = inside - outside
    feathered = np.clip(0.5 + signed / max(2.0 * feather_pixels, 1e-6), 0.0, 1.0)
    neural = gaussian_filter(value, sigma=1.0, mode="reflect")
    return cast(
        NDArray[np.float32],
        np.clip(0.75 * feathered + 0.25 * neural, 0.0, 1.0).astype(np.float32),
    )


def masks_from_landmarks(
    image_shape: tuple[int, int] | tuple[int, int, int],
    landmarks: ArrayLike,
    face_box: BoundingBox,
) -> PortraitMasks:
    height, width = image_shape[:2]
    center = face_box.center
    head_binary = ellipse_mask(
        (height, width),
        (center[0], center[1] - 0.08 * face_box.height),
        (0.68 * face_box.width, 0.78 * face_box.height),
    )
    head = refine_head_mask(head_binary, face_box)
    face_skin = (
        ellipse_mask(
            (height, width), center, (0.46 * face_box.width, 0.55 * face_box.height)
        )
        * head
    )
    hair = np.clip(head - face_skin * 0.75, 0.0, 1.0)
    person = np.maximum(
        head,
        ellipse_mask(
            (height, width),
            (center[0], face_box.y2),
            (0.9 * face_box.width, 0.65 * face_box.height),
        ),
    )
    eye_masks: list[NDArray[np.float32]] = []
    iris_masks: list[NDArray[np.float32]] = []
    for name in ("left_eye", "right_eye"):
        eye_points = group_points(landmarks, name)
        eye_center = tuple(eye_points.mean(axis=0))
        eye_width = float(np.ptp(eye_points[:, 0]))
        eye_height = max(float(np.ptp(eye_points[:, 1])), eye_width * 0.25)
        eye_masks.append(
            ellipse_mask(
                (height, width), eye_center, (eye_width * 0.65, eye_height * 0.85)
            )
        )
        iris_masks.append(
            ellipse_mask(
                (height, width), eye_center, (eye_height * 0.45, eye_height * 0.45)
            )
        )
    return PortraitMasks(
        person=person.astype(np.float32),
        head=head,
        face_skin=face_skin.astype(np.float32),
        hair=hair.astype(np.float32),
        eyes=(eye_masks[0], eye_masks[1]),
        irises=(iris_masks[0], iris_masks[1]),
        effective_transfer=head.copy(),
        foreground_alpha=head.copy(),
    )


def build_effective_mask(
    input_head_alpha: ArrayLike,
    warped_reference_head_alpha: ArrayLike,
    *,
    minimum_coverage: float = 0.03,
) -> NDArray[np.float32]:
    input_alpha = np.clip(np.asarray(input_head_alpha, dtype=np.float32), 0.0, 1.0)
    reference_alpha = np.clip(
        np.asarray(warped_reference_head_alpha, dtype=np.float32), 0.0, 1.0
    )
    if input_alpha.shape != reference_alpha.shape:
        raise ValueError("head alpha masks must share a shape")
    effective = input_alpha * reference_alpha
    coverage = float(np.mean(effective > 0.05))
    if coverage < minimum_coverage:
        raise MaskFailure(
            "Input and reference head masks do not overlap", coverage=coverage
        )
    return effective.astype(np.float32)
