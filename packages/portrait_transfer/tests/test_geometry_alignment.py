from __future__ import annotations

import numpy as np
from portrait_transfer.alignment.anchors import alignment_anchors
from portrait_transfer.alignment.beier_neely import (
    beier_neely_map,
    beier_neely_map_scalar,
)
from portrait_transfer.alignment.legacy_66_connections import (
    landmark_segments,
    validate_legacy_connections,
)
from portrait_transfer.alignment.map_composition import (
    compose_backward_maps,
    compose_with_residual,
    naive_add_offsets,
)
from portrait_transfer.alignment.similarity import estimate_similarity
from portrait_transfer.geometry.sampling import identity_map, warp
from portrait_transfer.geometry.transforms import (
    piecewise_affine_backward_map,
    transform_points,
)
from portrait_transfer.geometry.validity import jacobian_determinant, map_validity
from portrait_transfer.landmarks import canonical_68_landmarks


def test_identity_backward_warp() -> None:
    image = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    assert np.array_equal(warp(image, identity_map(image.shape)), image)


def test_piecewise_affine_maps_triangle_vertices() -> None:
    source = np.asarray(((1, 1), (7, 1), (1, 7)), dtype=np.float32)
    destination = np.asarray(((2, 2), (8, 2), (2, 8)), dtype=np.float32)
    mapping = piecewise_affine_backward_map(source, destination, (10, 10))
    for source_vertex, destination_vertex in zip(source, destination):
        x, y = destination_vertex.astype(int)
        assert np.allclose(mapping[y, x], source_vertex, atol=1e-6)


def test_composed_map_matches_two_step_warp() -> None:
    yy, xx = np.meshgrid(np.arange(32), np.arange(34), indexing="ij")
    image = (0.2 * xx + 0.1 * yy).astype(np.float32)
    outer = identity_map(image.shape)
    outer[..., 0] += 0.2 * np.sin(yy / 5.0)
    outer[..., 1] += 0.15 * np.cos(xx / 7.0)
    inner = identity_map(image.shape)
    inner[..., 0] += 0.35
    inner[..., 1] += 0.25
    composed = compose_backward_maps(outer, inner)
    one_sample = warp(image, composed, mode="border")
    two_step = warp(warp(image, outer, mode="border"), inner, mode="border")
    assert np.max(np.abs(one_sample[2:-2, 2:-2] - two_step[2:-2, 2:-2])) < 0.015


def test_exact_residual_composition_differs_from_naive_addition() -> None:
    base = identity_map((20, 20))
    base[..., 0] += 0.03 * base[..., 1] ** 2
    residual = np.zeros_like(base)
    residual[..., 1] = 1.25
    exact = compose_with_residual(base, residual)
    naive = naive_add_offsets(base, residual)
    assert np.max(np.abs(exact - naive)) > 0.05


def test_vectorized_beier_neely_matches_scalar() -> None:
    destination = np.asarray(
        [[[2, 2], [12, 3]], [[3, 11], [13, 10]], [[4, 3], [4, 10]]], dtype=np.float32
    )
    source = destination.copy()
    source[0] += np.asarray((1.5, -0.5), dtype=np.float32)
    source[2, 1] += np.asarray((2.0, 0.0), dtype=np.float32)
    vectorized = beier_neely_map(
        destination, source, (14, 16), include_boundaries=False, chunk_rows=3
    )
    scalar = beier_neely_map_scalar(
        destination, source, (14, 16), include_boundaries=False
    )
    assert np.allclose(vectorized, scalar, atol=2e-6)


def test_legacy_graph_is_valid_and_non_degenerate() -> None:
    landmarks = canonical_68_landmarks((256, 256))
    assert validate_legacy_connections(landmarks)
    segments = landmark_segments(landmarks)
    assert segments.shape == (61, 2, 2)
    assert np.all(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0)


def test_known_similarity_transform_is_recovered() -> None:
    source = alignment_anchors(canonical_68_landmarks((300, 300)))
    angle = np.deg2rad(12.0)
    scale = 1.15
    matrix = np.asarray(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), 7.0],
            [scale * np.sin(angle), scale * np.cos(angle), -4.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    destination = transform_points(source, matrix)
    estimate = estimate_similarity(source, destination, ransac_threshold=0.001)
    assert np.allclose(estimate.matrix, matrix, atol=2e-5)
    assert estimate.inliers.all()


def test_foldover_and_bounds_are_reported() -> None:
    mapping = identity_map((12, 14))
    assert np.allclose(jacobian_determinant(mapping), 1.0)
    mapping[:, 7:, 0] = mapping[:, 7:, 0][:, ::-1]
    mapping[0, 0] = (-10, -10)
    report = map_validity(mapping, (12, 14))
    assert report.valid_fraction < 1.0
    assert report.negative_jacobian_fraction > 0.0
