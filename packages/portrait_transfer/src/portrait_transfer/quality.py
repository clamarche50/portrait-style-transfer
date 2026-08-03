"""Image quality and pair compatibility metrics without demographic inference."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, laplace, sobel

from .alignment.anchors import eye_centers, normalized_landmark_shape
from .color.lab import rgb_to_lab
from .types import BoundingBox, CompatibilityReport, PortraitAnalysis, QualityReport


def luminance(rgb: ArrayLike) -> NDArray[np.float32]:
    value = np.asarray(rgb, dtype=np.float32)
    return cast(
        NDArray[np.float32],
        (
            0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]
        ).astype(np.float32),
    )


def _roi_values(image: NDArray[np.float32], box: BoundingBox) -> NDArray[np.float32]:
    bounded = box.clamp((image.shape[0], image.shape[1]))
    x1, y1 = int(np.floor(bounded.x)), int(np.floor(bounded.y))
    x2, y2 = (
        max(x1 + 1, int(np.ceil(bounded.x2))),
        max(y1 + 1, int(np.ceil(bounded.y2))),
    )
    return image[y1:y2, x1:x2]


def edge_density(rgb: ArrayLike, mask: ArrayLike | None = None) -> float:
    gray = luminance(rgb)
    gradient = np.hypot(
        sobel(gray, axis=1, mode="reflect"), sobel(gray, axis=0, mode="reflect")
    )
    selection = (
        np.ones(gray.shape, dtype=bool) if mask is None else np.asarray(mask) > 0.5
    )
    values = gradient[selection]
    if values.size == 0:
        return 0.0
    threshold = float(np.percentile(values, 75))
    return float(np.mean(values > max(threshold, 1e-5)))


def analyze_quality(
    rgb: ArrayLike,
    landmarks: ArrayLike,
    face_box: BoundingBox,
    head_mask: ArrayLike,
    *,
    mask_confidence: float = 1.0,
    crop_truncation: float = 0.0,
    occlusion_proxy: float = 0.0,
) -> QualityReport:
    image = np.asarray(rgb, dtype=np.float32)
    gray = luminance(image)
    roi = _roi_values(gray, face_box)
    left_eye, right_eye = eye_centers(landmarks)
    inter_eye = float(np.linalg.norm(left_eye - right_eye))
    laplacian = laplace(roi, mode="reflect")
    high_frequency = roi - gaussian_filter(roi, sigma=1.0, mode="reflect")
    noise = float(
        1.4826 * np.median(np.abs(high_frequency - np.median(high_frequency)))
    )
    warnings: list[str] = []
    under = float(np.mean(roi < 0.04))
    over = float(np.mean(roi > 0.96))
    blur_variance = float(np.var(laplacian))
    if under > 0.25:
        warnings.append("underexposed_face")
    if over > 0.20:
        warnings.append("overexposed_face")
    if crop_truncation > 0.1:
        warnings.append("crop_truncation")
    return QualityReport(
        inter_eye_distance=inter_eye,
        blur_variance=blur_variance,
        luminance_mean=float(np.mean(roi)),
        luminance_std=float(np.std(roi)),
        underexposed_fraction=under,
        overexposed_fraction=over,
        noise_estimate=noise,
        mask_confidence=float(np.clip(mask_confidence, 0.0, 1.0)),
        crop_truncation=float(np.clip(crop_truncation, 0.0, 1.0)),
        occlusion_proxy=float(np.clip(occlusion_proxy, 0.0, 1.0)),
        edge_density=edge_density(image, head_mask),
        warnings=tuple(warnings),
    )


def normalized_cross_correlation(
    first: ArrayLike, second: ArrayLike, mask: ArrayLike | None = None
) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("NCC arrays must have equal shape")
    selected = np.ones(a.shape, dtype=bool) if mask is None else np.asarray(mask) > 0.5
    a_values, b_values = a[selected], b[selected]
    if a_values.size == 0:
        return 0.0
    a_values -= a_values.mean()
    b_values -= b_values.mean()
    denominator = float(np.linalg.norm(a_values) * np.linalg.norm(b_values))
    if denominator < 1e-12:
        return 1.0 if np.allclose(a[selected], b[selected]) else 0.0
    return float(np.clip(np.dot(a_values, b_values) / denominator, -1.0, 1.0))


def landmark_shape_similarity(first: ArrayLike, second: ArrayLike) -> float:
    first_shape = normalized_landmark_shape(first)
    second_shape = normalized_landmark_shape(second)
    if first_shape.shape != second_shape.shape:
        return 0.0
    covariance = second_shape.T @ first_shape
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    aligned = second_shape @ rotation
    distance = float(np.mean(np.linalg.norm(first_shape - aligned, axis=1)))
    return float(np.exp(-4.0 * distance))


def compare_portraits(
    input_analysis: PortraitAnalysis,
    reference_analysis: PortraitAnalysis,
    input_rgb: ArrayLike,
    reference_rgb: ArrayLike,
) -> CompatibilityReport:
    pose_difference = np.linalg.norm(
        np.asarray(
            [
                input_analysis.pose.yaw - reference_analysis.pose.yaw,
                input_analysis.pose.pitch - reference_analysis.pose.pitch,
                input_analysis.pose.roll - reference_analysis.pose.roll,
            ],
            dtype=np.float32,
        )
    )
    pose_similarity = float(np.exp(-pose_difference / 25.0))
    shape_similarity = landmark_shape_similarity(
        input_analysis.landmarks, reference_analysis.landmarks
    )
    input_head = input_analysis.masks.head
    reference_head = reference_analysis.masks.head
    if input_head.shape == reference_head.shape:
        intersection = float(np.minimum(input_head, reference_head).sum())
        union = float(np.maximum(input_head, reference_head).sum())
        overlap = intersection / max(union, 1e-6)
        energy_similarity = (
            normalized_cross_correlation(
                luminance(input_rgb),
                luminance(reference_rgb),
                np.minimum(input_head, reference_head),
            )
            + 1.0
        ) / 2.0
    else:
        overlap = 0.5
        energy_similarity = 0.5
    input_lab = rgb_to_lab(input_rgb)
    reference_lab = rgb_to_lab(reference_rgb)
    input_median = np.median(input_lab.reshape(-1, 3), axis=0)
    reference_median = np.median(reference_lab.reshape(-1, 3), axis=0)
    photometric = float(np.exp(-np.linalg.norm(input_median - reference_median)))
    edge_similarity = float(
        np.exp(
            -abs(
                input_analysis.quality.edge_density
                - reference_analysis.quality.edge_density
            )
            * 4.0
        )
    )
    score = (
        0.30 * energy_similarity
        + 0.20 * pose_similarity
        + 0.20 * shape_similarity
        + 0.15 * photometric
        + 0.10 * overlap
        + 0.05 * edge_similarity
    )
    warnings: list[str] = []
    if pose_similarity < 0.4:
        warnings.append("pose_mismatch")
    if shape_similarity < 0.4:
        warnings.append("landmark_shape_mismatch")
    return CompatibilityReport(
        compatible=score >= 0.20,
        score=float(score),
        pose_similarity=pose_similarity,
        landmark_shape_similarity=shape_similarity,
        mask_overlap=float(overlap),
        energy_ncc=float(energy_similarity),
        photometric_compatibility=photometric,
        edge_similarity=edge_similarity,
        warnings=tuple(warnings),
    )
