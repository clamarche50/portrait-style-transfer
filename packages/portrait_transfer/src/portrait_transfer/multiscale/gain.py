"""Robust local-energy gain calculation and log-space strength control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import AlgorithmProfile, GainSettings
from .masked_gaussian import masked_gaussian


@dataclass(frozen=True)
class GainResult:
    raw: NDArray[np.float32]
    clipped: NDArray[np.float32]
    smoothed: NDArray[np.float32]
    effective: NDArray[np.float32]
    smoothing_sigma: float | None


def apply_transfer_strength(gain: ArrayLike, strength: float) -> NDArray[np.float32]:
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("transfer strength must be in [0, 1]")
    safe = np.maximum(np.asarray(gain, dtype=np.float32), 1e-6)
    return cast(
        NDArray[np.float32],
        np.exp(float(strength) * np.log(safe)).astype(np.float32),
    )


def compute_gain(
    input_energy: ArrayLike,
    reference_energy: ArrayLike,
    effective_mask: ArrayLike,
    level: int,
    *,
    profile: AlgorithmProfile = AlgorithmProfile.PAPER_EXACT,
    transfer_strength: float = 1.0,
    settings: GainSettings | None = None,
) -> GainResult:
    settings = settings or GainSettings()
    input_value = np.maximum(np.asarray(input_energy, dtype=np.float32), 0.0)
    reference_value = np.maximum(np.asarray(reference_energy, dtype=np.float32), 0.0)
    if input_value.shape != reference_value.shape:
        raise ValueError("energy maps must match")
    raw = np.sqrt(reference_value / (input_value + settings.epsilon_energy))
    both_flat = (input_value <= settings.epsilon_energy) & (
        reference_value <= settings.epsilon_energy
    )
    raw = np.where(both_flat, 1.0, raw).astype(np.float32)
    clipped = np.clip(raw, settings.theta_low, settings.theta_high).astype(np.float32)
    if profile is AlgorithmProfile.PAPER_EXACT:
        smoothing_sigma = float(settings.beta * (2**level))
        smoothed = np.asarray(
            masked_gaussian(
                clipped,
                effective_mask,
                smoothing_sigma,
                mask_epsilon=settings.mask_epsilon,
                truncate=settings.truncate,
            ),
            dtype=np.float32,
        )
        smoothed = np.clip(smoothed, settings.theta_low, settings.theta_high)
    else:
        smoothing_sigma = None
        smoothed = clipped.copy()
    effective = apply_transfer_strength(smoothed, transfer_strength)
    return GainResult(
        raw, clipped, smoothed.astype(np.float32), effective, smoothing_sigma
    )
