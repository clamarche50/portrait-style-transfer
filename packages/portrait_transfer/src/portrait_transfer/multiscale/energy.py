"""Profile-aware local detail energy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import SOURCE_ENERGY_SIGMAS, AlgorithmProfile
from ..geometry.sampling import MATLAB_OOB_FILL, warp
from .masked_gaussian import masked_gaussian


class EnergyOrder(StrEnum):
    BEFORE_WARP = "before_warp"
    AFTER_WARP = "after_warp"


@dataclass(frozen=True)
class EnergyPair:
    input_energy: NDArray[np.float32]
    warped_reference_energy: NDArray[np.float32]
    sigma: float
    order: EnergyOrder


def local_energy(band: ArrayLike, mask: ArrayLike, sigma: float) -> NDArray[np.float32]:
    value = np.asarray(band, dtype=np.float32)
    energy = masked_gaussian(value * value, mask, sigma)
    return np.maximum(np.asarray(energy, dtype=np.float32), 0.0)


def compute_energy_pair(
    input_band: ArrayLike,
    reference_band: ArrayLike,
    input_mask: ArrayLike,
    reference_mask: ArrayLike,
    mapping: ArrayLike,
    level: int,
    *,
    profile: AlgorithmProfile,
) -> EnergyPair:
    if profile is AlgorithmProfile.PAPER_EXACT:
        sigma = float(2 ** (level + 1))
        input_energy = local_energy(input_band, input_mask, sigma)
        reference_energy = local_energy(reference_band, reference_mask, sigma)
        warped_reference_energy = warp(
            reference_energy, mapping, mode="constant", cval=MATLAB_OOB_FILL
        )
        order = EnergyOrder.BEFORE_WARP
    else:
        if not 0 <= level < len(SOURCE_ENERGY_SIGMAS):
            raise ValueError("source-compatible energy level is out of range")
        sigma = SOURCE_ENERGY_SIGMAS[level]
        warped_band = warp(
            reference_band, mapping, mode="constant", cval=MATLAB_OOB_FILL
        )
        warped_mask = warp(reference_mask, mapping, mode="constant")
        input_energy = local_energy(input_band, input_mask, sigma)
        warped_reference_energy = local_energy(warped_band, warped_mask, sigma)
        order = EnergyOrder.AFTER_WARP
    return EnergyPair(input_energy, warped_reference_energy, sigma, order)
