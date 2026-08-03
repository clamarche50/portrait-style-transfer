"""Portrait analysis protocol, heuristic fallback, and severe-failure checks."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .alignment.anchors import eye_centers
from .config import PreflightThresholds
from .exceptions import FaceDetectionError, PairCompatibilityError, QualityFailure
from .image_io import normalize_rgb
from .landmarks import canonical_68_landmarks, validate_landmarks
from .quality import analyze_quality, compare_portraits
from .segmentation import masks_from_landmarks
from .types import BoundingBox, CompatibilityReport, PortraitAnalysis, PoseEstimate


class PortraitAnalyzer(Protocol):
    def analyze(self, rgb: NDArray[np.float32]) -> PortraitAnalysis: ...


class HeuristicPortraitAnalyzer:
    """Deterministic fallback for synthetic tests and explicit degraded operation.

    This does not claim to detect a real face. Production deployments should
    inject a licensed landmark/segmentation backend.
    """

    def analyze(self, rgb: NDArray[np.float32]) -> PortraitAnalysis:
        image = normalize_rgb(rgb)
        height, width = image.shape[:2]
        image_shape = (image.shape[0], image.shape[1], image.shape[2])
        landmarks = canonical_68_landmarks(image_shape)
        face_box = BoundingBox(
            0.08 * width, 0.10 * height, 0.84 * width, 0.82 * height
        ).clamp(image_shape)
        masks = masks_from_landmarks(image_shape, landmarks, face_box)
        quality = analyze_quality(
            image, landmarks, face_box, masks.head, mask_confidence=0.5
        )
        return PortraitAnalysis(
            landmarks=landmarks,
            face_box=face_box,
            pose=PoseEstimate(),
            quality=quality,
            masks=masks,
            warnings=("heuristic_analysis_backend",),
        )


def _validate_analysis(
    analysis: PortraitAnalysis,
    image: NDArray[np.float32],
    thresholds: PreflightThresholds,
) -> None:
    image_shape = (image.shape[0], image.shape[1], image.shape[2])
    validate_landmarks(analysis.landmarks, image_shape)
    analysis.masks.validate((image.shape[0], image.shape[1]))
    pose = analysis.pose
    if (
        abs(pose.yaw) > thresholds.max_abs_yaw
        or abs(pose.pitch) > thresholds.max_abs_pitch
        or abs(pose.roll) > thresholds.max_abs_roll
    ):
        raise FaceDetectionError(
            "Face pose exceeds configured limits",
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
        )
    if analysis.face_box.height / image.shape[0] < thresholds.min_head_height_fraction:
        raise FaceDetectionError("Head occupies too little of the image")
    if analysis.quality.inter_eye_distance < thresholds.min_inter_eye_distance:
        raise QualityFailure(
            "Inter-eye distance is below the configured minimum",
            inter_eye_distance=analysis.quality.inter_eye_distance,
            minimum=thresholds.min_inter_eye_distance,
        )
    if analysis.quality.blur_variance < thresholds.severe_blur_variance:
        raise QualityFailure(
            "Face region is too blurred", blur_variance=analysis.quality.blur_variance
        )
    if analysis.quality.mask_confidence < thresholds.min_mask_confidence:
        raise QualityFailure(
            "Head mask confidence is too low",
            confidence=analysis.quality.mask_confidence,
        )
    left_eye, right_eye = eye_centers(analysis.landmarks)
    for eye in (left_eye, right_eye):
        if not (0 <= eye[0] < image.shape[1] and 0 <= eye[1] < image.shape[0]):
            raise FaceDetectionError("An eye center lies outside the image")


def analyze_portrait(
    rgb: NDArray[np.float32],
    analyzer: PortraitAnalyzer | None = None,
    thresholds: PreflightThresholds | None = None,
) -> PortraitAnalysis:
    image = normalize_rgb(rgb)
    backend = analyzer or HeuristicPortraitAnalyzer()
    analysis = backend.analyze(image)
    _validate_analysis(analysis, image, thresholds or PreflightThresholds())
    return analysis


def validate_pair(
    input_analysis: PortraitAnalysis,
    reference_analysis: PortraitAnalysis,
    input_rgb: NDArray[np.float32],
    reference_rgb: NDArray[np.float32],
) -> CompatibilityReport:
    report = compare_portraits(
        input_analysis, reference_analysis, input_rgb, reference_rgb
    )
    if not report.compatible:
        raise PairCompatibilityError(
            "Portrait pair is severely incompatible", score=report.score
        )
    return report
