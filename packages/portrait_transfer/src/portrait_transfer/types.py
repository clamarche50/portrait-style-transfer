"""Shared value types for portrait analysis and style ranking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def clamp(self, image_shape: tuple[int, int] | tuple[int, int, int]) -> BoundingBox:
        h, w = image_shape[:2]
        x1 = float(np.clip(self.x, 0, w))
        y1 = float(np.clip(self.y, 0, h))
        x2 = float(np.clip(self.x2, x1, w))
        y2 = float(np.clip(self.y2, y1, h))
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True)
class PoseEstimate:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class QualityReport:
    inter_eye_distance: float
    blur_variance: float
    luminance_mean: float
    luminance_std: float
    underexposed_fraction: float
    overexposed_fraction: float
    noise_estimate: float
    mask_confidence: float
    crop_truncation: float
    occlusion_proxy: float
    edge_density: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortraitMasks:
    person: FloatArray
    head: FloatArray
    face_skin: FloatArray
    hair: FloatArray
    eyes: tuple[FloatArray, FloatArray]
    irises: tuple[FloatArray, FloatArray]
    effective_transfer: FloatArray
    foreground_alpha: FloatArray

    def validate(self, shape: tuple[int, int]) -> None:
        arrays = (
            self.person,
            self.head,
            self.face_skin,
            self.hair,
            *self.eyes,
            *self.irises,
            self.effective_transfer,
            self.foreground_alpha,
        )
        for array in arrays:
            if array.shape != shape:
                raise ValueError(f"Mask shape {array.shape} does not match {shape}")
            if not np.isfinite(array).all():
                raise ValueError("Masks must be finite")


@dataclass(frozen=True)
class PortraitAnalysis:
    landmarks: FloatArray
    face_box: BoundingBox
    pose: PoseEstimate
    quality: QualityReport
    masks: PortraitMasks
    warnings: tuple[str, ...] = ()
    full_landmarks: FloatArray | None = None


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    score: float
    pose_similarity: float
    landmark_shape_similarity: float
    mask_overlap: float
    energy_ncc: float
    photometric_compatibility: float
    edge_similarity: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaplacianStack:
    bands: tuple[FloatArray, ...]
    residual: FloatArray
    sigmas: tuple[float, ...]
    profile: str


@dataclass(frozen=True)
class StyleFeature:
    identifier: str
    vector: FloatArray
    pose: PoseEstimate | None = field(default_factory=PoseEstimate)
    landmark_shape: FloatArray | None = None
    photometric_lab: FloatArray | None = None
    mask_quality: float = 1.0


@dataclass(frozen=True)
class RankedStyle:
    identifier: str
    score: float
    energy_ncc: float
    pose_similarity: float
    landmark_shape_similarity: float
    photometric_compatibility: float
    mask_quality: float
