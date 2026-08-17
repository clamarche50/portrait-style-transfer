"""Strongly typed data exchanged between portrait-transfer stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
ProgressCallback = Callable[["ProcessingStage", int, str], None]


class ProcessingStage(StrEnum):
    NORMALIZE = "normalize"
    PREFLIGHT = "preflight"
    CROP = "crop"
    ALIGNMENT = "alignment"
    DENSE_REFINEMENT = "dense_refinement"
    MULTISCALE = "multiscale"
    EYES = "eyes"
    BACKGROUND = "background"
    FINALIZE = "finalize"


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
class AlignmentDiagnostics:
    selected_stage: str
    anchor_error: float
    inlier_count: int
    valid_fraction: float
    negative_jacobian_fraction: float
    displacement_p50: float
    displacement_p95: float
    descriptor_loss_before: float | None = None
    descriptor_loss_after: float | None = None
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class CorrespondenceResult:
    map_x: FloatArray
    map_y: FloatArray
    aligned_reference: FloatArray
    diagnostics: AlignmentDiagnostics

    @property
    def mapping(self) -> FloatArray:
        return np.stack((self.map_x, self.map_y), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class TransferDiagnostics:
    input_quality: QualityReport
    reference_quality: QualityReport
    compatibility: CompatibilityReport
    alignment: AlignmentDiagnostics
    profile: str
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    resumed_from_stage: str | None = None


@dataclass(frozen=True)
class TransferResult:
    output_rgb: FloatArray
    diagnostics: TransferDiagnostics
    artifacts: dict[str, FloatArray]
    resume_artifacts: dict[str, FloatArray] = field(default_factory=dict)


@dataclass(frozen=True)
class LaplacianStack:
    bands: tuple[FloatArray, ...]
    residual: FloatArray
    sigmas: tuple[float, ...]
    profile: str


@dataclass(frozen=True)
class EyeHighlightAsset:
    foreground_rgb: FloatArray
    alpha: FloatArray
    center: tuple[float, float]
    iris_radius: float
    angle_radians: float
    confidence: float
    version: str = "clean-room-v1"


@dataclass(frozen=True)
class RuntimeContext:
    analyzer: Any = None
    dense_backend: Any = None
    progress_callback: ProgressCallback | None = None
    cancel_check: Callable[[], bool] | None = None
    eye_assets: tuple[EyeHighlightAsset | None, EyeHighlightAsset | None] | None = None
    corrections: Mapping[str, Any] = field(default_factory=dict)
    resume_artifacts: Mapping[str, FloatArray] = field(default_factory=dict)

    def progress(self, stage: ProcessingStage, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, int(np.clip(percent, 0, 100)), message)

    def cancelled(self) -> bool:
        return bool(self.cancel_check is not None and self.cancel_check())


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
