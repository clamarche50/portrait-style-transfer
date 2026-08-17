"""Clean-room scalar and chunked vectorized Beier-Neely backward maps."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import BeierNeelySettings
from ..exceptions import AlignmentFailure
from ..geometry.sampling import identity_map
from .legacy_66_connections import boundary_segments
from .map_composition import compose_backward_maps


def _validate_segments(
    destination_segments: ArrayLike,
    source_segments: ArrayLike,
    minimum_length: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    destination = np.asarray(destination_segments, dtype=np.float64)
    source = np.asarray(source_segments, dtype=np.float64)
    if (
        destination.shape != source.shape
        or destination.ndim != 3
        or destination.shape[1:] != (2, 2)
    ):
        raise ValueError("segment arrays must have matching Nx2x2 shapes")
    lengths = np.linalg.norm(destination[:, 1] - destination[:, 0], axis=1)
    source_lengths = np.linalg.norm(source[:, 1] - source[:, 0], axis=1)
    valid = (lengths > minimum_length) & (source_lengths > minimum_length)
    if not valid.any():
        raise AlignmentFailure("All Beier-Neely segments are degenerate")
    return destination[valid], source[valid]


def add_crop_boundaries(
    destination_segments: ArrayLike,
    source_segments: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    destination = np.asarray(destination_segments, dtype=np.float32)
    source = np.asarray(source_segments, dtype=np.float32)
    boundaries = boundary_segments(destination_shape)
    return np.concatenate((destination, boundaries)), np.concatenate(
        (source, boundaries)
    )


def _map_points(
    points: NDArray[np.float64],
    destination: NDArray[np.float64],
    source: NDArray[np.float64],
    *,
    a: float,
    b: float,
    p: float,
) -> NDArray[np.float64]:
    destination_p = destination[:, 0]
    destination_q = destination[:, 1]
    destination_delta = destination_q - destination_p
    destination_length = np.linalg.norm(destination_delta, axis=1)
    destination_perp = np.stack(
        (-destination_delta[:, 1], destination_delta[:, 0]), axis=1
    )

    relative = points[:, None, :] - destination_p[None, :, :]
    u = np.sum(relative * destination_delta[None, :, :], axis=2) / (
        destination_length[None, :] ** 2
    )
    v = (
        np.sum(relative * destination_perp[None, :, :], axis=2)
        / destination_length[None, :]
    )

    source_p = source[:, 0]
    source_q = source[:, 1]
    source_delta = source_q - source_p
    source_length = np.linalg.norm(source_delta, axis=1)
    source_perp = np.stack((-source_delta[:, 1], source_delta[:, 0]), axis=1)
    mapped = (
        source_p[None, :, :]
        + u[..., None] * source_delta[None, :, :]
        + v[..., None] * source_perp[None, :, :] / source_length[None, :, None]
    )

    distance_to_p = np.linalg.norm(
        points[:, None, :] - destination_p[None, :, :], axis=2
    )
    distance_to_q = np.linalg.norm(
        points[:, None, :] - destination_q[None, :, :], axis=2
    )
    distance = np.where(
        u < 0.0, distance_to_p, np.where(u > 1.0, distance_to_q, np.abs(v))
    )
    weights = (destination_length[None, :] ** p / (a + distance)) ** b
    weight_sum = weights.sum(axis=1, keepdims=True)
    if np.any(weight_sum <= np.finfo(np.float64).tiny):
        raise AlignmentFailure("Beier-Neely weights collapsed to zero")
    return cast(
        NDArray[np.float64],
        np.sum(mapped * weights[..., None], axis=1) / weight_sum,
    )


def beier_neely_map(
    destination_segments: ArrayLike,
    source_segments: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
    *,
    a: float = 10.0,
    b: float = 1.0,
    p: float = 1.0,
    chunk_rows: int = 64,
    minimum_segment_length: float = 1e-5,
    include_boundaries: bool = True,
    initial_map: ArrayLike | None = None,
) -> NDArray[np.float32]:
    if a <= 0 or b <= 0 or p < 0 or chunk_rows < 1:
        raise ValueError("invalid Beier-Neely parameters")
    destination = np.asarray(destination_segments, dtype=np.float32)
    source = np.asarray(source_segments, dtype=np.float32)
    if include_boundaries:
        destination, source = add_crop_boundaries(
            destination, source, destination_shape
        )
    destination64, source64 = _validate_segments(
        destination, source, minimum_segment_length
    )
    height, width = destination_shape[:2]
    output: NDArray[np.float32] = np.empty((height, width, 2), dtype=np.float32)
    x_coordinates = np.arange(width, dtype=np.float64)
    for row_start in range(0, height, chunk_rows):
        row_stop = min(height, row_start + chunk_rows)
        yy, xx = np.meshgrid(
            np.arange(row_start, row_stop, dtype=np.float64),
            x_coordinates,
            indexing="ij",
        )
        points = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        mapped = _map_points(points, destination64, source64, a=a, b=b, p=p)
        output[row_start:row_stop] = mapped.reshape(
            row_stop - row_start, width, 2
        ).astype(np.float32)
    if initial_map is not None:
        output = compose_backward_maps(initial_map, output)
    return output


def beier_neely_map_scalar(
    destination_segments: ArrayLike,
    source_segments: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
    *,
    a: float = 10.0,
    b: float = 1.0,
    p: float = 1.0,
    minimum_segment_length: float = 1e-5,
    include_boundaries: bool = True,
) -> NDArray[np.float32]:
    destination = np.asarray(destination_segments, dtype=np.float32)
    source = np.asarray(source_segments, dtype=np.float32)
    if include_boundaries:
        destination, source = add_crop_boundaries(
            destination, source, destination_shape
        )
    destination64, source64 = _validate_segments(
        destination, source, minimum_segment_length
    )
    height, width = destination_shape[:2]
    output: NDArray[np.float32] = identity_map(destination_shape)
    for y in range(height):
        for x in range(width):
            point = np.asarray([[float(x), float(y)]], dtype=np.float64)
            output[y, x] = _map_points(point, destination64, source64, a=a, b=b, p=p)[0]
    return output.astype(np.float32)


def build_beier_neely_backward_map(
    input_segments: ArrayLike,
    reference_segments: ArrayLike,
    input_shape: tuple[int, int] | tuple[int, int, int],
    *,
    initial_map: ArrayLike | None = None,
    settings: BeierNeelySettings | None = None,
) -> NDArray[np.float32]:
    settings = settings or BeierNeelySettings()
    return beier_neely_map(
        input_segments,
        reference_segments,
        input_shape,
        a=settings.a,
        b=settings.b,
        p=settings.p,
        chunk_rows=settings.chunk_rows,
        minimum_segment_length=settings.minimum_segment_length,
        initial_map=initial_map,
    )
