from __future__ import annotations

import numpy as np
import pytest
from portrait_transfer.background import (
    apply_background_mode,
    extract_reference_background,
)
from portrait_transfer.config import BackgroundMode, TransferSettings
from portrait_transfer.exceptions import MaskFailure
from portrait_transfer.segmentation import build_effective_mask
from portrait_transfer.selection import rank_style_examples
from portrait_transfer.types import PoseEstimate, StyleFeature


def test_effective_mask_handles_overlap_and_rejects_missing() -> None:
    first = np.zeros((20, 20), dtype=np.float32)
    second = np.zeros_like(first)
    first[2:12, 2:12] = 1.0
    second[7:17, 7:17] = 1.0
    effective = build_effective_mask(first, second, minimum_coverage=0.01)
    assert effective.sum() == 25
    with pytest.raises(MaskFailure):
        build_effective_mask(first, np.zeros_like(first), minimum_coverage=0.01)


def test_identical_style_feature_ranks_first(rng: np.random.Generator) -> None:
    vector = rng.normal(size=64).astype(np.float32)
    vector /= np.linalg.norm(vector)
    query = StyleFeature("query", vector, PoseEstimate())
    candidates = (
        StyleFeature("different", np.roll(vector, 17), PoseEstimate(yaw=10)),
        StyleFeature("identical", vector.copy(), PoseEstimate()),
    )
    ranked = rank_style_examples(query, candidates, top_k=2)
    assert ranked[0].identifier == "identical"
    assert ranked[0].energy_ncc > ranked[1].energy_ncc


def test_background_extraction_removes_foreground_color() -> None:
    reference = np.zeros((31, 31, 3), dtype=np.float32)
    reference[..., 2] = 0.8
    alpha = np.zeros((31, 31), dtype=np.float32)
    alpha[9:22, 9:22] = 1.0
    reference[alpha > 0.5] = (1.0, 0.0, 0.0)
    background = extract_reference_background(reference, alpha)
    assert background[15, 15, 2] > 0.6
    assert background[15, 15, 0] < 0.2


@pytest.mark.parametrize("mode", list(BackgroundMode))
def test_all_background_modes_return_finite_range(mode: BackgroundMode) -> None:
    processed = np.full((24, 24, 3), 0.8, dtype=np.float32)
    original = np.full_like(processed, 0.2)
    reference = np.full_like(processed, (0.1, 0.3, 0.7))
    alpha = np.zeros((24, 24), dtype=np.float32)
    alpha[5:19, 5:19] = 1.0
    settings = TransferSettings(
        background_mode=mode,
        background_color=(0.4, 0.5, 0.6) if mode is BackgroundMode.SOLID else None,
    )
    result = apply_background_mode(
        processed, original, reference, alpha, settings, reference_alpha=alpha
    )
    assert result.shape == processed.shape
    assert np.isfinite(result).all()
    assert 0.0 <= result.min() <= result.max() <= 1.0
