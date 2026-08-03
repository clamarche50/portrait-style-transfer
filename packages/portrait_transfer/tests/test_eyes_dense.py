from __future__ import annotations

import numpy as np
import pytest

from portrait_transfer.alignment import flow_optimization
from portrait_transfer.alignment.flow_optimization import CpuDenseCorrespondence
from portrait_transfer.config import DenseSettings
from portrait_transfer.eyes.extraction import extract_highlight_asset
from portrait_transfer.eyes.highlight_transfer import place_highlight
from portrait_transfer.geometry.sampling import identity_map
from portrait_transfer.segmentation import ellipse_mask


def test_highlight_extraction_and_placement_are_confidence_gated() -> None:
    image = np.full((45, 45, 3), (0.08, 0.16, 0.22), dtype=np.float32)
    iris = ellipse_mask((45, 45), (22, 22), (10, 10))
    image[17:20, 18:21] = 1.0
    asset = extract_highlight_asset(image, iris)
    assert asset is not None
    assert asset.confidence > 0.0
    target = np.full_like(image, (0.12, 0.20, 0.28))
    placed = place_highlight(target, iris, asset, minimum_confidence=0.0)
    scaled = place_highlight(
        target,
        iris,
        asset,
        minimum_confidence=0.0,
        scale_multiplier=0.7,
    )
    rotated = place_highlight(
        target,
        iris,
        asset,
        minimum_confidence=0.0,
        rotation_degrees=90.0,
    )
    assert placed.shape == target.shape
    assert np.max(np.abs(placed - target)) > 0.01
    assert not np.allclose(scaled, placed)
    assert not np.allclose(rotated, placed)


def test_cpu_dense_backend_is_finite_and_has_explicit_fallback() -> None:
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    input_image = np.stack((xx / 31, yy / 31, (xx + yy) / 62), axis=-1).astype(
        np.float32
    )
    reference = np.roll(input_image, 1, axis=1)
    mask = np.ones((32, 32), dtype=np.float32)
    backend = CpuDenseCorrespondence()
    result = backend.refine(
        input_crop=input_image,
        reference_rgb=reference,
        initial_backward_map=identity_map(input_image.shape),
        input_mask=mask,
        reference_mask=mask,
        settings=DenseSettings(
            iterations=(1,), min_loss_improvement=1.0, min_valid_fraction=0.5
        ),
    )
    assert result.mapping.shape == (32, 32, 2)
    assert np.isfinite(result.mapping).all()
    assert result.diagnostics.selected_stage in {"dense", "line"}


def test_cpu_dense_backend_uses_the_configured_coarse_to_fine_pyramid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: set[tuple[int, int]] = set()
    original = flow_optimization._resize_to

    def recording_resize(
        value: np.ndarray, shape: tuple[int, int], *, order: int = 1
    ) -> np.ndarray:
        seen.add(shape)
        return original(value, shape, order=order)

    monkeypatch.setattr(flow_optimization, "_resize_to", recording_resize)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    image = np.stack((xx / 63, yy / 63, (xx + yy) / 126), axis=-1).astype(np.float32)
    flow_optimization.optimize_residual_flow(
        image,
        np.roll(image, 1, axis=1),
        np.ones((64, 64), dtype=np.float32),
        DenseSettings(pyramid_scales=(0.25, 0.5), iterations=(1, 1)),
    )
    assert {(16, 16), (32, 32)}.issubset(seen)
