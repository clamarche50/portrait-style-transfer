"""Parity checks against archived MATLAB primitives (source_2014_compat)."""

from __future__ import annotations

import numpy as np
from portrait_transfer.color.legacy_color import legacy_lab_to_rgb, legacy_rgb_to_lab
from portrait_transfer.geometry.sampling import MATLAB_OOB_FILL, identity_map, warp
from portrait_transfer.multiscale.histogram import rank_transfer_one_dimensional
from portrait_transfer.multiscale.reconstruction import reconstruct


def test_oob_fill_matches_warp_image_constant() -> None:
    # warpImage.m fills out-of-bounds samples with 0.6.
    assert MATLAB_OOB_FILL == 0.6
    source = np.zeros((8, 8), dtype=np.float32)
    mapping = identity_map((8, 8))
    mapping[..., 0] += 50.0
    warped = warp(source, mapping, mode="constant", cval=MATLAB_OOB_FILL)
    assert np.allclose(warped, 0.6)


def test_legacy_lab_round_trip_matches_matlab_matrices() -> None:
    rng = np.random.default_rng(7)
    rgb = rng.uniform(0.0, 1.0, size=(6, 7, 3)).astype(np.float32)
    round_trip = legacy_lab_to_rgb(legacy_rgb_to_lab(rgb))
    assert np.allclose(round_trip, rgb, atol=1e-5)


def test_legacy_lab_gray_matches_matlab_values() -> None:
    gray = np.full((2, 2, 3), 0.6, dtype=np.float32)
    lab = legacy_rgb_to_lab(gray)[0, 0]
    # RGB2Lab.m: y = 0.6 -> L = 116 * y^(1/3) - 16, a = b = 0.
    assert abs(float(lab[0]) - (116.0 * 0.6 ** (1.0 / 3.0) - 16.0)) < 1e-3
    assert abs(float(lab[1])) < 1e-4
    assert abs(float(lab[2])) < 1e-4


def test_rank_transfer_preserves_order_like_hist_transfer_one_d() -> None:
    source = np.asarray([[0.1, 0.5, 0.9, 0.3]], dtype=np.float32)
    reference = np.asarray([[10.0, 30.0, 20.0, 40.0]], dtype=np.float32)
    matched = rank_transfer_one_dimensional(source, reference)
    order = np.argsort(source.ravel())
    assert np.all(np.diff(matched.ravel()[order]) >= 0)
    identity = rank_transfer_one_dimensional(source, source.copy())
    assert np.allclose(np.sort(identity.ravel()), np.sort(source.ravel()), atol=1e-6)


def test_sum_pyramid_reconstruction_is_plain_sum() -> None:
    bands = [np.full((3, 4), value, dtype=np.float32) for value in (1.0, -2.0, 0.5)]
    residual = np.full((3, 4), 7.0, dtype=np.float32)
    assert np.allclose(reconstruct(tuple(bands), residual), 7.0 + 1.0 - 2.0 + 0.5)
