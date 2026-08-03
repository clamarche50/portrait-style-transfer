from __future__ import annotations

import numpy as np
import pytest

from portrait_transfer.alignment.correction_constraints import (
    apply_eye_center_operations,
    apply_gain_constraint_operations,
    apply_mask_stroke_operations,
    corrected_alignment_points,
    eye_highlight_transforms,
    normalize_correction_operations,
)


def test_canonical_mask_stroke_changes_only_local_region() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    operations = (
        {
            "type": "mask_stroke",
            "target": "head",
            "points": [[5, 5], [25, 25]],
            "radius": 2,
            "value": 1,
        },
    )
    corrected = apply_mask_stroke_operations(mask, operations)
    assert corrected[5, 5] == 1
    assert corrected[25, 25] == 1
    assert corrected[0, 31] == 0


def test_paired_alignment_points_are_appended() -> None:
    destination = np.asarray(((1, 1), (2, 2), (3, 3)), dtype=np.float32)
    source = destination + 1
    operations = (
        {
            "type": "alignment_points",
            "input_points": [[8, 9], [10, 11]],
            "reference_points": [[7, 8], [9, 10]],
        },
    )
    corrected_destination, corrected_source = corrected_alignment_points(
        destination, source, operations
    )
    assert corrected_destination.shape == (5, 2)
    assert corrected_source.shape == (5, 2)
    assert np.array_equal(corrected_destination[-1], (10, 11))


def test_gain_polygon_and_eye_center_operations() -> None:
    gain = np.full((24, 24), 2.0, dtype=np.float32)
    gain_operations = (
        {
            "type": "gain_constraint",
            "channel": 0,
            "level": 1,
            "mode": "LOCK_TO_ONE",
            "polygon": [[5, 5], [18, 5], [18, 18], [5, 18]],
        },
    )
    corrected = apply_gain_constraint_operations(
        gain, gain_operations, channel=0, level=1
    )
    assert corrected[10, 10] == 1.0
    assert corrected[1, 1] == 2.0

    empty = (np.zeros((24, 24), dtype=np.float32), np.zeros((24, 24), dtype=np.float32))
    eyes = apply_eye_center_operations(
        empty,
        ({"type": "eye_center", "eye": "left", "center": [9, 12], "radius": 3},),
    )
    assert eyes[0][12, 9] == 1.0
    assert eyes[1].sum() == 0


@pytest.mark.parametrize("shape", [(40, 40), (80, 120)])
def test_normalized_corrections_hit_same_relative_region(
    shape: tuple[int, int],
) -> None:
    operations = normalize_correction_operations(
        (
            {
                "type": "mask_stroke",
                "target": "head",
                "points": [[0.25, 0.50], [0.75, 0.50]],
                "radius": 0.05,
                "value": 1.0,
                "coordinate_space": "normalized",
            },
            {
                "type": "alignment_points",
                "input_points": [[0.25, 0.50]],
                "reference_points": [[0.75, 0.25]],
                "coordinate_space": "normalized",
            },
            {
                "type": "eye_center",
                "eye": "left",
                "center": [0.75, 0.50],
                "radius": 0.06,
                "coordinate_space": "normalized",
            },
        ),
        shape,
    )
    mask = apply_mask_stroke_operations(np.zeros(shape, dtype=np.float32), operations)
    center_y = round(0.50 * (shape[0] - 1))
    for fraction in (0.25, 0.50, 0.75):
        assert mask[center_y, round(fraction * (shape[1] - 1))] == 1.0

    destination, source = corrected_alignment_points(
        np.zeros((3, 2), dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
        operations,
    )
    assert destination[-1] == pytest.approx(
        (0.25 * (shape[1] - 1), 0.50 * (shape[0] - 1))
    )
    assert source[-1] == pytest.approx((0.75 * (shape[1] - 1), 0.25 * (shape[0] - 1)))

    eyes = apply_eye_center_operations(
        (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)),
        operations,
    )
    expected_x = round(0.75 * (shape[1] - 1))
    assert eyes[0][center_y, expected_x] == 1.0
    assert eyes[1].sum() == 0.0

    scales, rotations = eye_highlight_transforms(
        (
            {
                "type": "eye_center",
                "eye": "left",
                "highlight_scale": 1.4,
                "highlight_rotation_degrees": -25.0,
            },
        )
    )
    assert scales == (1.4, 1.0)
    assert rotations == (-25.0, 0.0)
