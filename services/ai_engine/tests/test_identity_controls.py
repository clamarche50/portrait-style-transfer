from __future__ import annotations

import numpy as np
from PIL import Image

from services.ai_engine.landmarks import FaceKeypoints, render_keypoints
from services.ai_engine.runtime import (
    cosine_similarity,
    denoise_strength_for,
    face_anchor_image,
    palette_transfer,
)


def test_denoise_strength_maps_structure_inversely() -> None:
    base, weight = 0.85, 0.25
    assert denoise_strength_for(1.0, base, weight) == 0.6
    assert denoise_strength_for(0.0, base, weight) == 0.85
    # Higher structure always repaints less.
    assert denoise_strength_for(0.9, base, weight) < denoise_strength_for(
        0.5, base, weight
    )


def test_denoise_strength_stays_inside_safe_bounds() -> None:
    for structure in np.linspace(0.0, 1.0, 21):
        strength = denoise_strength_for(float(structure), 0.85, 0.25)
        assert 0.5 <= strength <= 0.9


def test_cosine_similarity_matches_expected_values() -> None:
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(a, np.zeros(2, dtype=np.float32)) == 0.0


def test_keypoint_canvas_is_black_with_colored_markers() -> None:
    keypoints = FaceKeypoints(
        points=(
            (64.0, 64.0),
            (128.0, 64.0),
            (96.0, 96.0),
            (70.0, 128.0),
            (122.0, 128.0),
        ),
        det_score=0.97,
    )
    canvas = render_keypoints(256, 256, keypoints)
    assert canvas.size == (256, 256)
    array = np.asarray(canvas)
    assert array.shape == (256, 256, 3)
    # The canvas background stays black while the markers are brightened
    # (the reference renderer scales colors by 0.6).
    assert array.sum() > 0
    assert (array == 0).mean() > 0.98
    # A keypoint center carries its color channel.
    center = array[64, 64]
    assert center[0] > 0


def test_face_anchor_preserves_source_face_region() -> None:
    source = Image.new("RGB", (256, 256), (255, 0, 0))
    output = Image.new("RGB", (256, 256), (0, 0, 255))
    anchored = face_anchor_image(output, source, (64.0, 64.0, 192.0, 192.0))
    array = np.asarray(anchored)
    # The face center comes from the source; the corners keep the output.
    center = array[128, 128]
    assert center[0] > 200 and center[2] < 60
    corner = array[8, 8]
    assert corner[2] > 200 and corner[0] < 60


def test_face_anchor_follows_target_bbox_when_face_moved() -> None:
    source = Image.new("RGB", (256, 256), (255, 0, 0))
    output = Image.new("RGB", (256, 256), (0, 0, 255))
    anchored = face_anchor_image(
        output, source, (64.0, 64.0, 192.0, 192.0), (0.0, 0.0, 128.0, 128.0)
    )
    array = np.asarray(anchored)
    # The moved face lands in the top-left quadrant; the old spot stays output.
    moved_center = array[64, 64]
    assert moved_center[0] > 200 and moved_center[2] < 60
    old_spot = array[200, 200]
    assert old_spot[2] > 200 and old_spot[0] < 60


def test_palette_transfer_blend_zero_keeps_pixels() -> None:
    output = Image.new("RGB", (64, 64), (200, 100, 50))
    reference = Image.new("RGB", (64, 64), (10, 20, 30))
    result = palette_transfer(output, reference, 0.0)
    assert result.size == output.size
    assert np.array_equal(np.asarray(result), np.asarray(output))


def test_palette_transfer_full_blend_moves_toward_reference() -> None:
    output = Image.new("RGB", (64, 64), (220, 120, 60))
    reference = Image.new("RGB", (64, 64), (20, 120, 220))
    matched = palette_transfer(output, reference, 1.0)
    assert matched.size == output.size
    output_mean = np.asarray(output, dtype=np.float32).mean(axis=(0, 1))
    matched_mean = np.asarray(matched, dtype=np.float32).mean(axis=(0, 1))
    # A flat source has zero Lab variance, so the match lands on the
    # reference palette: red falls and blue rises.
    assert matched_mean[2] > output_mean[2] + 50
    assert matched_mean[0] < output_mean[0] - 50
