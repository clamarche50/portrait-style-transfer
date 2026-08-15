"""Shared configuration for portrait analysis and style ranking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class AlgorithmProfile(StrEnum):
    """Multiscale sigma profile used for style energy ranking."""

    PAPER_EXACT = "paper_exact"
    SOURCE_2014_COMPAT = "source_2014_compat"


PAPER_SIGMAS: Final[tuple[float, ...]] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SOURCE_SIGMAS: Final[tuple[float, ...]] = (4.0, 8.0, 16.0, 32.0, 64.0)
SOURCE_ENERGY_SIGMAS: Final[tuple[float, ...]] = (8.0, 16.0, 32.0, 64.0, 128.0)


@dataclass(frozen=True)
class ImageLimits:
    max_encoded_bytes: int = 15 * 1024 * 1024
    max_decoded_pixels: int = 8_000_000
    max_original_long_edge: int = 8_000
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP")


@dataclass(frozen=True)
class PreflightThresholds:
    min_inter_eye_distance: float = 150.0
    max_abs_yaw: float = 25.0
    max_abs_pitch: float = 20.0
    max_abs_roll: float = 20.0
    min_head_height_fraction: float = 0.20
    severe_blur_variance: float = 2e-5
    min_mask_confidence: float = 0.20
    min_effective_coverage: float = 0.03
