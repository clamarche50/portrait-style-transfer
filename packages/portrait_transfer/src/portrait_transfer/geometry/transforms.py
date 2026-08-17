"""Affine, point, and synthetic piecewise-affine transform helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay

from .sampling import identity_map


def as_homogeneous(matrix: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (2, 3):
        value = np.vstack((value, (0.0, 0.0, 1.0)))
    if value.shape != (3, 3):
        raise ValueError("transform must be 2x3 or 3x3")
    return value


def invert_transform(matrix: ArrayLike) -> NDArray[np.float64]:
    value = as_homogeneous(matrix)
    determinant = float(np.linalg.det(value))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("transform is singular")
    return cast(NDArray[np.float64], np.linalg.inv(value))


def transform_points(points: ArrayLike, matrix: ArrayLike) -> NDArray[np.float32]:
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("points must be Nx2")
    homogeneous = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    transformed = homogeneous @ as_homogeneous(matrix).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed.astype(np.float32)


def affine_backward_map(
    source_to_destination: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    """Build destination-to-source coordinates from a forward affine transform."""

    inverse = invert_transform(source_to_destination)
    grid = identity_map(destination_shape)
    points = grid.reshape(-1, 2)
    return transform_points(points, inverse).reshape((*grid.shape[:2], 2))


def piecewise_affine_backward_map(
    source_points: ArrayLike,
    destination_points: ArrayLike,
    destination_shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    """Map a triangulated destination mesh back to corresponding source vertices."""

    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            "source_points and destination_points must be matching Nx2 arrays"
        )
    if len(source) < 3:
        raise ValueError("at least three point pairs are required")
    triangulation = Delaunay(destination)
    grid = identity_map(destination_shape)
    flat = grid.reshape(-1, 2).astype(np.float64)
    simplex = triangulation.find_simplex(flat)
    output = flat.copy()
    inside_indices = np.flatnonzero(simplex >= 0)
    for triangle_index in np.unique(simplex[inside_indices]):
        selected = inside_indices[simplex[inside_indices] == triangle_index]
        transform = triangulation.transform[triangle_index]
        delta = flat[selected] - transform[2]
        barycentric_two = delta @ transform[:2].T
        barycentric = np.column_stack(
            (barycentric_two, 1.0 - barycentric_two.sum(axis=1))
        )
        vertices = triangulation.simplices[triangle_index]
        output[selected] = barycentric @ source[vertices]
    return output.reshape((*grid.shape[:2], 2)).astype(np.float32)
