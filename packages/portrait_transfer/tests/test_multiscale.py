from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from portrait_transfer.config import AlgorithmProfile, LegacyColorMode
from portrait_transfer.geometry.sampling import identity_map
from portrait_transfer.multiscale.color import should_transfer_band, transfer_band
from portrait_transfer.multiscale.energy import (
    EnergyOrder,
    compute_energy_pair,
    local_energy,
)
from portrait_transfer.multiscale.gain import apply_transfer_strength, compute_gain
from portrait_transfer.multiscale.histogram import histogram_match
from portrait_transfer.multiscale.laplacian import build_laplacian_stack
from portrait_transfer.multiscale.masked_gaussian import masked_gaussian
from portrait_transfer.multiscale.reconstruction import (
    blend_residual,
    reconstruct_stack,
)


@given(st.integers(min_value=9, max_value=35), st.integers(min_value=9, max_value=35))
@settings(max_examples=8, deadline=None)
def test_masked_gaussian_preserves_constant(height: int, width: int) -> None:
    image = np.full((height, width), 0.37, dtype=np.float32)
    mask = np.zeros_like(image)
    mask[2:-2, 2:-2] = 1.0
    filtered = masked_gaussian(image, mask, 2.0)
    assert np.allclose(filtered[mask > 0.5], 0.37, atol=2e-6)


def test_masked_gaussian_does_not_leak_background() -> None:
    image = np.full((41, 41), 100.0, dtype=np.float32)
    mask = np.zeros_like(image)
    mask[10:31, 10:31] = 1.0
    image[mask > 0.5] = 2.0
    filtered = masked_gaussian(image, mask, 5.0)
    assert np.max(np.abs(filtered[12:29, 12:29] - 2.0)) < 1e-4


@pytest.mark.parametrize(
    ("profile", "expected_bands"),
    [(AlgorithmProfile.PAPER_EXACT, 6), (AlgorithmProfile.SOURCE_2014_COMPAT, 5)],
)
def test_stack_count_and_reconstruction(
    rng: np.random.Generator, profile: AlgorithmProfile, expected_bands: int
) -> None:
    image = rng.normal(size=(37, 29)).astype(np.float32)
    mask = np.zeros_like(image)
    mask[3:-4, 2:-3] = 1.0
    stack = build_laplacian_stack(image, mask, profile=profile)
    assert len(stack.bands) == expected_bands
    assert np.allclose(reconstruct_stack(stack), image, atol=4e-6)


def test_local_energy_nonnegative(rng: np.random.Generator) -> None:
    band = rng.normal(size=(31, 27)).astype(np.float32)
    energy = local_energy(band, np.ones_like(band), 4.0)
    assert np.isfinite(energy).all()
    assert float(energy.min()) >= 0.0


def test_identical_energy_gain_is_one() -> None:
    energy = np.full((19, 17), 2.0, dtype=np.float32)
    result = compute_gain(energy, energy + 1e-4, np.ones_like(energy), 2)
    assert np.allclose(result.raw, 1.0, atol=2e-4)


def test_gain_clamps_before_smoothing_and_paper_sigma() -> None:
    input_energy = np.ones((21, 21), dtype=np.float32)
    reference_energy = np.ones_like(input_energy)
    reference_energy[:, :7] = 1e-8
    reference_energy[:, 14:] = 100.0
    result = compute_gain(input_energy, reference_energy, np.ones_like(input_energy), 3)
    assert result.clipped.min() >= 0.9
    assert result.clipped.max() <= 2.8
    assert result.smoothing_sigma == 24.0


def test_source_gain_does_not_smooth() -> None:
    first = np.ones((15, 15), dtype=np.float32)
    second = first.copy()
    second[7, 7] = 16.0
    result = compute_gain(
        first,
        second,
        np.ones_like(first),
        1,
        profile=AlgorithmProfile.SOURCE_2014_COMPAT,
    )
    assert result.smoothing_sigma is None
    assert np.array_equal(result.smoothed, result.clipped)


def test_transfer_strength_zero_and_residual_zero() -> None:
    gain = np.linspace(0.9, 2.8, 25, dtype=np.float32).reshape(5, 5)
    band = np.arange(25, dtype=np.float32).reshape(5, 5)
    effective = apply_transfer_strength(gain, 0.0)
    assert np.array_equal(transfer_band(band, effective, enabled=True), band)
    assert np.array_equal(blend_residual(band, band + 10, 0.0), band)


def test_paper_chrominance_fine_bands_are_preserved() -> None:
    for channel in (1, 2):
        for level in range(3):
            assert not should_transfer_band(
                channel, level, profile=AlgorithmProfile.PAPER_EXACT
            )
        assert should_transfer_band(channel, 3, profile=AlgorithmProfile.PAPER_EXACT)
    assert should_transfer_band(0, 0, profile=AlgorithmProfile.PAPER_EXACT)


def test_style_names_cannot_control_color_policy() -> None:
    # The policy accepts only channel/profile/metadata; arbitrary artist strings
    # have no input path into the decision.
    baseline = should_transfer_band(
        1, 3, profile=AlgorithmProfile.PAPER_EXACT, monochrome_style=False
    )
    explicit_legacy = should_transfer_band(
        1,
        3,
        profile=AlgorithmProfile.SOURCE_2014_COMPAT,
        legacy_mode=LegacyColorMode.MONOCHROME_L_ONLY,
    )
    assert baseline is True
    assert explicit_legacy is False


def test_profile_energy_order_is_explicit(rng: np.random.Generator) -> None:
    input_band = rng.normal(size=(23, 25)).astype(np.float32)
    reference_band = np.roll(input_band, 1, axis=1)
    mask = np.ones_like(input_band)
    mapping = identity_map(input_band.shape)
    mapping[..., 0] += 0.5
    paper = compute_energy_pair(
        input_band,
        reference_band,
        mask,
        mask,
        mapping,
        0,
        profile=AlgorithmProfile.PAPER_EXACT,
    )
    source = compute_energy_pair(
        input_band,
        reference_band,
        mask,
        mask,
        mapping,
        0,
        profile=AlgorithmProfile.SOURCE_2014_COMPAT,
    )
    assert paper.order is EnergyOrder.BEFORE_WARP
    assert source.order is EnergyOrder.AFTER_WARP
    assert paper.sigma == 2.0
    assert source.sigma == 8.0
    assert not np.allclose(
        paper.warped_reference_energy, source.warped_reference_energy
    )


def test_histogram_match_supports_unequal_sample_counts() -> None:
    source = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    reference = np.linspace(10, 20, 63, dtype=np.float32).reshape(7, 9)
    matched = histogram_match(
        source, reference, np.ones_like(source), np.ones_like(reference)
    )
    assert matched.shape == source.shape
    assert matched.min() >= 10.0
    assert matched.max() <= 20.0
    assert np.all(np.diff(np.sort(matched.ravel())) >= 0)
