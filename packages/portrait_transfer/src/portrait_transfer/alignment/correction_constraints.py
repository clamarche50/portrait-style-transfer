"""Non-destructive gain and correspondence correction constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class GainConstraintMode(StrEnum):
    LOCK_TO_ONE = "LOCK_TO_ONE"
    COPY_FROM_REGION = "COPY_FROM_REGION"
    LIMIT_RANGE = "LIMIT_RANGE"


def _normalized_points(
    value: Any,
    shape: tuple[int, int],
    *,
    name: str,
    minimum_points: int = 1,
) -> list[list[float]]:
    points = np.asarray(value, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < minimum_points
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{name} must contain finite normalized x/y points")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError(f"{name} normalized coordinates must be in [0, 1]")
    height, width = shape
    scaled = points.copy()
    scaled[:, 0] *= max(width - 1, 1)
    scaled[:, 1] *= max(height - 1, 1)
    normalized: list[list[float]] = []
    for index in range(len(scaled)):
        normalized.append([float(scaled[index, 0]), float(scaled[index, 1])])
    return normalized


def normalize_correction_operations(
    operations: Sequence[Mapping[str, Any]],
    shape: tuple[int, int],
) -> tuple[dict[str, Any], ...]:
    """Convert explicitly normalized correction geometry into crop pixels.

    Untagged and ``pixel`` operations are copied unchanged for backward
    compatibility.  Normalized radii use the crop's shorter pixel dimension,
    which is stable for both square and mildly portrait-shaped canonical crops.
    """

    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("correction crop shape must be positive")
    radius_scale = float(max(min(height - 1, width - 1), 1))
    normalized: list[dict[str, Any]] = []
    for original in operations:
        operation = deepcopy(dict(original))
        coordinate_space = str(operation.get("coordinate_space", "pixel")).lower()
        if coordinate_space in ("pixel", "pixels"):
            normalized.append(operation)
            continue
        if coordinate_space != "normalized":
            raise ValueError("correction coordinate_space must be pixel or normalized")

        operation_type = operation.get("type")
        if operation_type == "mask_stroke":
            operation["points"] = _normalized_points(
                operation.get("points", ()), shape, name="mask stroke points"
            )
            radius = float(operation.get("radius", 0.0))
            if not np.isfinite(radius) or not 0.0 < radius <= 1.0:
                raise ValueError("normalized mask stroke radius must be in (0, 1]")
            operation["radius"] = radius * radius_scale
        elif operation_type == "alignment_points":
            operation["input_points"] = _normalized_points(
                operation.get("input_points", ()),
                shape,
                name="input alignment points",
            )
            operation["reference_points"] = _normalized_points(
                operation.get("reference_points", ()),
                shape,
                name="reference alignment points",
            )
        elif operation_type == "gain_constraint":
            operation["polygon"] = _normalized_points(
                operation.get("polygon", ()),
                shape,
                name="gain polygon",
                minimum_points=3,
            )
            if "source_polygon" in operation:
                operation["source_polygon"] = _normalized_points(
                    operation["source_polygon"],
                    shape,
                    name="gain source polygon",
                    minimum_points=3,
                )
        elif operation_type == "eye_center":
            operation["center"] = _normalized_points(
                (operation.get("center", ()),),
                shape,
                name="eye center",
            )[0]
            if "radius" in operation:
                radius = float(operation["radius"])
                if not np.isfinite(radius) or not 0.0 < radius <= 1.0:
                    raise ValueError("normalized iris radius must be in (0, 1]")
                operation["radius"] = radius * radius_scale
        else:
            raise ValueError("unsupported normalized correction operation")
        operation["coordinate_space"] = "pixel"
        normalized.append(operation)
    return tuple(normalized)


def constrain_gain(
    gain: ArrayLike,
    region_mask: ArrayLike,
    mode: GainConstraintMode,
    *,
    source_region_mask: ArrayLike | None = None,
    limit_range: tuple[float, float] = (0.9, 2.8),
) -> NDArray[np.float32]:
    output = np.asarray(gain, dtype=np.float32).copy()
    region = np.asarray(region_mask, dtype=np.float32) > 0.5
    if region.shape != output.shape:
        raise ValueError("region mask must match gain")
    if mode is GainConstraintMode.LOCK_TO_ONE:
        output[region] = 1.0
    elif mode is GainConstraintMode.LIMIT_RANGE:
        low, high = limit_range
        if low > high:
            raise ValueError("invalid limit range")
        output[region] = np.clip(output[region], low, high)
    else:
        if source_region_mask is None:
            raise ValueError("COPY_FROM_REGION requires source_region_mask")
        source_region = np.asarray(source_region_mask, dtype=np.float32) > 0.5
        values = output[source_region]
        if values.size == 0:
            raise ValueError("source region contains no pixels")
        output[region] = float(np.median(values))
    return output


def apply_flow_correction(
    residual_flow: ArrayLike, correction: ArrayLike, weight: ArrayLike | float = 1.0
) -> NDArray[np.float32]:
    flow = np.asarray(residual_flow, dtype=np.float32)
    delta = np.asarray(correction, dtype=np.float32)
    if flow.shape != delta.shape or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow and correction must be matching HxWx2 arrays")
    weights = np.asarray(weight, dtype=np.float32)
    if weights.ndim == 2:
        weights = weights[..., None]
    return (flow + delta * weights).astype(np.float32)


def polyline_mask(
    shape: tuple[int, int],
    points: ArrayLike,
    radius: float,
) -> NDArray[np.bool_]:
    vertices = np.asarray(points, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) == 0:
        raise ValueError("stroke points must be a non-empty Nx2 array")
    if radius <= 0:
        raise ValueError("stroke radius must be positive")
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        indexing="ij",
    )
    selected = np.zeros(shape, dtype=bool)
    segments = (
        pairwise(vertices) if len(vertices) > 1 else ((vertices[0], vertices[0]),)
    )
    for start, end in segments:
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-12:
            closest_x = np.full(shape, start[0], dtype=np.float32)
            closest_y = np.full(shape, start[1], dtype=np.float32)
        else:
            projection = np.clip(
                ((xx - start[0]) * delta[0] + (yy - start[1]) * delta[1]) / denominator,
                0.0,
                1.0,
            )
            closest_x = start[0] + projection * delta[0]
            closest_y = start[1] + projection * delta[1]
        selected |= (xx - closest_x) ** 2 + (yy - closest_y) ** 2 <= radius**2
    return selected


def polygon_mask(shape: tuple[int, int], vertices: ArrayLike) -> NDArray[np.bool_]:
    polygon = np.asarray(vertices, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must contain at least three x/y vertices")
    yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        indexing="ij",
    )
    inside = np.zeros(shape, dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > yy) != (previous[1] > yy)
        boundary_x = (previous[0] - current[0]) * (yy - current[1]) / (
            previous[1] - current[1] + 1e-12
        ) + current[0]
        inside ^= crosses & (xx < boundary_x)
        previous = current
    return inside


def apply_mask_stroke_operations(
    mask: ArrayLike, operations: Sequence[Mapping[str, Any]], *, target: str = "head"
) -> NDArray[np.float32]:
    output = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0).copy()
    for operation in operations:
        if (
            operation.get("type") != "mask_stroke"
            or operation.get("target", "head") != target
        ):
            continue
        region = polyline_mask(
            (output.shape[0], output.shape[1]),
            operation.get("points", ()),
            float(operation.get("radius", 4.0)),
        )
        value = float(operation.get("value", 1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError("mask stroke value must be in [0, 1]")
        output[region] = value
    return output


def corrected_alignment_points(
    input_points: ArrayLike,
    reference_points: ArrayLike,
    operations: Sequence[Mapping[str, Any]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    destination = np.asarray(input_points, dtype=np.float32)
    source = np.asarray(reference_points, dtype=np.float32)
    for operation in operations:
        if operation.get("type") != "alignment_points":
            continue
        added_destination = np.asarray(
            operation.get("input_points", ()), dtype=np.float32
        )
        added_source = np.asarray(
            operation.get("reference_points", ()), dtype=np.float32
        )
        if (
            added_destination.shape != added_source.shape
            or added_destination.ndim != 2
            or added_destination.shape[1] != 2
        ):
            raise ValueError("alignment correction points must be matching Nx2 arrays")
        destination = np.concatenate((destination, added_destination), axis=0)
        source = np.concatenate((source, added_source), axis=0)
    return destination, source


def apply_gain_constraint_operations(
    gain: ArrayLike,
    operations: Sequence[Mapping[str, Any]],
    *,
    channel: int,
    level: int,
) -> NDArray[np.float32]:
    output = np.asarray(gain, dtype=np.float32)
    for operation in operations:
        if operation.get("type") != "gain_constraint":
            continue
        operation_channel = operation.get("channel", "*")
        operation_level = operation.get("level", "*")
        if operation_channel not in ("*", channel) or operation_level not in (
            "*",
            level,
        ):
            continue
        shape = (output.shape[0], output.shape[1])
        region = polygon_mask(shape, operation.get("polygon", ()))
        mode = GainConstraintMode(
            str(operation.get("mode", GainConstraintMode.LOCK_TO_ONE.value))
        )
        source_region = None
        if mode is GainConstraintMode.COPY_FROM_REGION:
            source_region = polygon_mask(shape, operation.get("source_polygon", ()))
        value_range = operation.get("range", (0.9, 2.8))
        output = constrain_gain(
            output,
            region,
            mode,
            source_region_mask=source_region,
            limit_range=(float(value_range[0]), float(value_range[1])),
        )
    return output


def apply_eye_center_operations(
    iris_masks: tuple[ArrayLike, ArrayLike],
    operations: Sequence[Mapping[str, Any]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    output = [np.asarray(mask, dtype=np.float32).copy() for mask in iris_masks]
    for operation in operations:
        if operation.get("type") != "eye_center":
            continue
        eye = str(operation.get("eye", "left")).lower()
        if eye not in ("left", "right"):
            raise ValueError("eye_center eye must be 'left' or 'right'")
        index = 0 if eye == "left" else 1
        center = np.asarray(operation.get("center", ()), dtype=np.float32)
        if center.shape != (2,):
            raise ValueError("eye_center center must be [x, y]")
        default_radius = float(np.sqrt(np.count_nonzero(output[index] > 0.5) / np.pi))
        radius = float(operation.get("radius", max(default_radius, 1.0)))
        yy, xx = np.meshgrid(
            np.arange(output[index].shape[0]),
            np.arange(output[index].shape[1]),
            indexing="ij",
        )
        output[index] = (
            ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius**2
        ).astype(np.float32)
    return output[0], output[1]


def eye_highlight_transforms(
    operations: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return per-eye highlight scale and rotation from the latest eye edits."""

    scales = [1.0, 1.0]
    rotations = [0.0, 0.0]
    for operation in operations:
        if operation.get("type") != "eye_center":
            continue
        eye = str(operation.get("eye", "left")).lower()
        if eye not in ("left", "right"):
            raise ValueError("eye_center eye must be 'left' or 'right'")
        index = 0 if eye == "left" else 1
        if "highlight_scale" in operation:
            scale = float(operation["highlight_scale"])
            if not np.isfinite(scale) or not 0.1 <= scale <= 10.0:
                raise ValueError("highlight_scale must be in [0.1, 10]")
            scales[index] = scale
        if "highlight_rotation_degrees" in operation:
            rotation = float(operation["highlight_rotation_degrees"])
            if not np.isfinite(rotation):
                raise ValueError("highlight_rotation_degrees must be finite")
            rotations[index] = rotation
    return (scales[0], scales[1]), (rotations[0], rotations[1])
