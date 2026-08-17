"""Scale-invariance of the inter-eye quality gate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from portrait_transfer.exceptions import QualityFailure
from portrait_transfer.preflight import HeuristicPortraitAnalyzer, analyze_portrait
from portrait_transfer.types import BoundingBox


class _StubAnalyzer:
    def __init__(self, analysis) -> None:
        self.analysis = analysis

    def analyze(self, rgb):
        return self.analysis


def _crafted(textured_rgb: np.ndarray, face_side: float, inter_eye: float):
    base = HeuristicPortraitAnalyzer().analyze(
        np.asarray(textured_rgb, dtype=np.float32)
    )
    return replace(
        base,
        face_box=BoundingBox(0.0, 0.0, face_side / 1.75, face_side / 1.75),
        quality=replace(base.quality, inter_eye_distance=inter_eye),
    )


def test_small_portrait_passes_default_gate(textured_rgb: np.ndarray) -> None:
    # A 128x128 portrait has ~46px between eyes; the old absolute 150px gate
    # rejected it, but the scaled gate judges the face proportion instead.
    analysis = analyze_portrait(textured_rgb)
    assert analysis.quality.inter_eye_distance < 150.0


@pytest.mark.parametrize(
    ("face_side", "inter_eye", "expected_minimum"),
    (
        (1280.0, 151.0, 150.0 * 1280.0 / 1280.0),
        (640.0, 76.0, 150.0 * 640.0 / 1280.0),
        (320.0, 38.0, 150.0 * 320.0 / 1280.0),
    ),
)
def test_gate_scales_with_crop_side(
    textured_rgb: np.ndarray,
    face_side: float,
    inter_eye: float,
    expected_minimum: float,
) -> None:
    analysis = _crafted(textured_rgb, face_side, inter_eye)
    result = analyze_portrait(
        textured_rgb, _StubAnalyzer(analysis), processing_long_edge=1280
    )
    assert result.quality.inter_eye_distance == inter_eye
    assert expected_minimum < inter_eye


@pytest.mark.parametrize(
    ("face_side", "inter_eye"),
    ((1280.0, 149.0), (640.0, 74.0), (320.0, 36.0)),
)
def test_gate_rejects_narrow_eyes_at_every_scale(
    textured_rgb: np.ndarray, face_side: float, inter_eye: float
) -> None:
    analysis = _crafted(textured_rgb, face_side, inter_eye)
    with pytest.raises(QualityFailure) as info:
        analyze_portrait(
            textured_rgb, _StubAnalyzer(analysis), processing_long_edge=1280
        )
    assert info.value.context["minimum"] == pytest.approx(
        150.0 * face_side / 1280.0, rel=1e-3
    )
