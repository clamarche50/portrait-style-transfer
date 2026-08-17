"""Configuration for production and source-compatibility profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class AlgorithmProfile(StrEnum):
    PAPER_EXACT = "paper_exact"
    SOURCE_2014_COMPAT = "source_2014_compat"


class BackgroundMode(StrEnum):
    KEEP = "KEEP"
    BLUR = "BLUR"
    SOLID = "SOLID"
    REFERENCE = "REFERENCE"


class LegacyColorMode(StrEnum):
    ALL_CHANNELS = "legacy_color_all_channels"
    MONOCHROME_L_ONLY = "legacy_monochrome_l_only"


PAPER_SIGMAS: Final[tuple[float, ...]] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SOURCE_SIGMAS: Final[tuple[float, ...]] = (4.0, 8.0, 16.0, 32.0, 64.0)
SOURCE_ENERGY_SIGMAS: Final[tuple[float, ...]] = (8.0, 16.0, 32.0, 64.0, 128.0)


@dataclass(frozen=True)
class ImageLimits:
    max_encoded_bytes: int = 15 * 1024 * 1024
    max_decoded_pixels: int = 25_000_000
    max_original_long_edge: int = 8_000
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP")


@dataclass(frozen=True)
class PreflightThresholds:
    # Required eye-center separation in pixels at processing_long_edge; the
    # gate scales it by the canonical crop side so it is resolution invariant.
    min_inter_eye_distance: float = 150.0
    max_abs_yaw: float = 25.0
    max_abs_pitch: float = 20.0
    max_abs_roll: float = 20.0
    min_head_height_fraction: float = 0.20
    severe_blur_variance: float = 2e-5
    min_mask_confidence: float = 0.20
    min_effective_coverage: float = 0.03


@dataclass(frozen=True)
class BeierNeelySettings:
    a: float = 10.0
    b: float = 1.0
    p: float = 1.0
    chunk_rows: int = 64
    minimum_segment_length: float = 1e-5


@dataclass(frozen=True)
class DenseSettings:
    enabled: bool = True
    max_displacement: float = 32.0
    pyramid_scales: tuple[float, ...] = (0.125, 0.25, 0.5)
    iterations: tuple[int, ...] = (8, 6, 4)
    smoothness: float = 0.08
    magnitude: float = 0.01
    update_clip: float = 1.5
    min_loss_improvement: float = -1e-6
    min_valid_fraction: float = 0.75
    max_negative_jacobian_fraction: float = 0.02


@dataclass(frozen=True)
class GainSettings:
    epsilon_energy: float = 1e-4
    theta_low: float = 0.9
    theta_high: float = 2.8
    beta: float = 3.0
    mask_epsilon: float = 1e-6
    truncate: float = 3.0


@dataclass(frozen=True)
class TransferSettings:
    algorithm_profile: AlgorithmProfile = AlgorithmProfile.SOURCE_2014_COMPAT
    transfer_strength: float = 1.0
    residual_strength: float = 1.0
    global_range_mix: float = 0.25
    eye_highlights: bool = True
    background_mode: BackgroundMode = BackgroundMode.KEEP
    background_color: tuple[float, float, float] | None = None
    background_blur_sigma: float = 12.0
    dense_alignment: bool = True
    processing_long_edge: int = 1280
    debug_artifacts: bool = False
    random_seed: int = 0
    legacy_color_mode: LegacyColorMode = LegacyColorMode.ALL_CHANNELS
    preflight: PreflightThresholds = field(default_factory=PreflightThresholds)
    beier_neely: BeierNeelySettings = field(default_factory=BeierNeelySettings)
    dense: DenseSettings = field(default_factory=DenseSettings)
    gain: GainSettings = field(default_factory=GainSettings)

    def __post_init__(self) -> None:
        for name in ("transfer_strength", "residual_strength", "global_range_mix"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.processing_long_edge < 32:
            raise ValueError("processing_long_edge must be at least 32")
        if (
            self.background_mode is BackgroundMode.SOLID
            and self.background_color is None
        ):
            raise ValueError("background_color is required for SOLID mode")
        if self.background_color is not None and any(
            not 0.0 <= float(channel) <= 1.0 for channel in self.background_color
        ):
            raise ValueError("background_color channels must be in [0, 1]")
