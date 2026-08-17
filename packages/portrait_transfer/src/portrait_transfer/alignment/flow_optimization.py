"""Deterministic CPU residual-flow refinement and explicit fallback result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, sobel, zoom

from ..config import DenseSettings
from ..geometry.sampling import identity_map, warp
from ..geometry.validity import map_validity
from ..types import AlignmentDiagnostics
from .dense_sift import dense_descriptor, grayscale
from .map_composition import compose_with_residual


@dataclass(frozen=True)
class DenseRefinementResult:
    mapping: NDArray[np.float32]
    residual_flow: NDArray[np.float32]
    valid: bool
    diagnostics: AlignmentDiagnostics


def _masked_loss(
    first: NDArray[np.float32], second: NDArray[np.float32], mask: NDArray[np.float32]
) -> float:
    difference = np.sqrt((first - second) ** 2 + 1e-6)
    if difference.ndim == 3:
        difference = difference.mean(axis=2)
    weights = np.clip(mask, 0.0, 1.0)
    return float(np.sum(difference * weights) / max(float(weights.sum()), 1e-6))


def _resize_to(
    value: NDArray[np.float32], shape: tuple[int, int], *, order: int = 1
) -> NDArray[np.float32]:
    if value.shape[:2] == shape:
        return value.astype(np.float32, copy=True)
    factors = (shape[0] / value.shape[0], shape[1] / value.shape[1])
    if value.ndim == 3:
        resized = zoom(
            value, (*factors, 1.0), order=order, mode="reflect", prefilter=False
        )
    else:
        resized = zoom(value, factors, order=order, mode="reflect", prefilter=False)
    return np.asarray(resized, dtype=np.float32)


def _resize_flow(
    flow: NDArray[np.float32], shape: tuple[int, int]
) -> NDArray[np.float32]:
    old_height, old_width = flow.shape[:2]
    resized = _resize_to(flow, shape)
    resized[..., 0] *= shape[1] / max(old_width, 1)
    resized[..., 1] *= shape[0] / max(old_height, 1)
    return resized


def optimize_residual_flow(
    input_image: ArrayLike,
    aligned_reference: ArrayLike,
    mask: ArrayLike,
    settings: DenseSettings | None = None,
) -> tuple[NDArray[np.float32], float, float]:
    """Refine an aligned preview with a robust demons/Horn-Schunck hybrid.

    This bounded CPU optimizer is a lawful fallback, not a claim of SIFT Flow
    parity. It minimizes local descriptor disagreement while Gaussian smoothing
    implements the spatial prior.
    """

    settings = settings or DenseSettings()
    target = grayscale(input_image)
    reference = grayscale(aligned_reference)
    support = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if target.shape != reference.shape or target.shape != support.shape:
        raise ValueError("input, aligned reference, and mask must share HxW")
    full_features = dense_descriptor(target)
    before = _masked_loss(full_features, dense_descriptor(reference), support)
    flow: NDArray[np.float32] | None = None
    configured_scales = tuple(float(scale) for scale in settings.pyramid_scales)
    scales = tuple(sorted(configured_scales))
    if (
        not scales
        or len(set(scales)) != len(scales)
        or any(not 0.0 < scale <= 1.0 for scale in scales)
    ):
        raise ValueError("pyramid scales must be unique values in (0, 1]")
    if not settings.iterations:
        raise ValueError("dense iteration schedule cannot be empty")

    for level, scale in enumerate(scales):
        shape = (
            max(16, min(target.shape[0], round(target.shape[0] * scale))),
            max(16, min(target.shape[1], round(target.shape[1] * scale))),
        )
        target_level = _resize_to(target, shape)
        reference_level = _resize_to(reference, shape)
        support_level = np.clip(_resize_to(support, shape), 0.0, 1.0)
        grid = identity_map(shape)
        flow = (
            np.zeros((*shape, 2), dtype=np.float32)
            if flow is None
            else _resize_flow(flow, shape)
        )
        target_features = dense_descriptor(target_level)
        best_flow = flow.copy()
        best_loss = _masked_loss(
            target_features, dense_descriptor(reference_level), support_level
        )
        iteration_count = max(
            1,
            int(settings.iterations[min(level, len(settings.iterations) - 1)]),
        )
        displacement_limit = settings.max_displacement * min(
            shape[0] / target.shape[0], shape[1] / target.shape[1]
        )

        for _ in range(iteration_count):
            warped = warp(reference_level, grid + flow, mode="border")
            difference = target_level - warped
            gx = sobel(warped, axis=1, mode="reflect") / 8.0
            gy = sobel(warped, axis=0, mode="reflect") / 8.0
            denominator = gx * gx + gy * gy + 0.01 + settings.magnitude
            update_x = np.clip(
                difference * gx / denominator,
                -settings.update_clip,
                settings.update_clip,
            )
            update_y = np.clip(
                difference * gy / denominator,
                -settings.update_clip,
                settings.update_clip,
            )
            update = np.stack((update_x, update_y), axis=-1) * support_level[..., None]
            flow = flow + update.astype(np.float32)
            for channel in range(2):
                flow[..., channel] = gaussian_filter(
                    flow[..., channel],
                    sigma=1.0 + settings.smoothness * 5.0,
                    mode="reflect",
                )
            magnitude = np.linalg.norm(flow, axis=-1)
            limiter = np.minimum(1.0, displacement_limit / np.maximum(magnitude, 1e-6))
            flow *= limiter[..., None]

            candidate = warp(reference_level, grid + flow, mode="border")
            candidate_loss = _masked_loss(
                target_features, dense_descriptor(candidate), support_level
            )
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_flow = flow.copy()
            elif candidate_loss > best_loss * 1.05:
                flow = 0.5 * (flow + best_flow)
        flow = best_flow

    assert flow is not None
    full_flow = _resize_flow(flow, (target.shape[0], target.shape[1]))
    final_candidate = warp(
        reference,
        identity_map((target.shape[0], target.shape[1])) + full_flow,
        mode="border",
    )
    after = _masked_loss(full_features, dense_descriptor(final_candidate), support)
    return full_flow.astype(np.float32), before, after


class CpuDenseCorrespondence:
    """CPU-safe, deterministic correspondence backend with strict validation."""

    def refine(
        self,
        *,
        input_crop: ArrayLike,
        reference_rgb: ArrayLike,
        initial_backward_map: ArrayLike,
        input_mask: ArrayLike,
        reference_mask: ArrayLike,
        settings: DenseSettings | None = None,
    ) -> DenseRefinementResult:
        settings = settings or DenseSettings()
        initial = np.asarray(initial_backward_map, dtype=np.float32)
        aligned_reference = warp(reference_rgb, initial, mode="border")
        warped_reference_mask = warp(reference_mask, initial, mode="constant")
        support = np.clip(
            np.asarray(input_mask, dtype=np.float32) * warped_reference_mask, 0.0, 1.0
        )
        residual, before, after = optimize_residual_flow(
            input_crop, aligned_reference, support, settings
        )
        candidate = compose_with_residual(initial, residual)
        reference = np.asarray(reference_rgb)
        reference_shape = (reference.shape[0], reference.shape[1])
        report = map_validity(candidate, reference_shape)
        improved = after <= before + settings.min_loss_improvement
        valid = bool(
            improved
            and report.valid_fraction >= settings.min_valid_fraction
            and report.negative_jacobian_fraction
            <= settings.max_negative_jacobian_fraction
        )
        selected = candidate if valid else initial
        selected_report = map_validity(selected, reference_shape)
        fallback_reason = None if valid else "dense_validation_failed"
        diagnostics = AlignmentDiagnostics(
            selected_stage="dense" if valid else "line",
            anchor_error=0.0,
            inlier_count=0,
            valid_fraction=selected_report.valid_fraction,
            negative_jacobian_fraction=selected_report.negative_jacobian_fraction,
            displacement_p50=selected_report.displacement_p50,
            displacement_p95=selected_report.displacement_p95,
            descriptor_loss_before=before,
            descriptor_loss_after=after,
            fallback_reason=fallback_reason,
            metadata={
                "legacy_alpha": 500.0,
                "legacy_gamma": 10.0,
                "legacy_d": 1_000_000.0,
                "legacy_cell_size": 7,
                "legacy_processing_scale": 0.25,
            },
        )
        return DenseRefinementResult(
            selected.astype(np.float32), residual, valid, diagnostics
        )


class NoOpDenseCorrespondence:
    """Explicitly skip dense refinement while preserving the multiscale stage."""

    def refine(
        self,
        *,
        input_crop: ArrayLike,
        reference_rgb: ArrayLike,
        initial_backward_map: ArrayLike,
        input_mask: ArrayLike,
        reference_mask: ArrayLike,
        settings: DenseSettings | None = None,
    ) -> DenseRefinementResult:
        mapping = np.asarray(initial_backward_map, dtype=np.float32)
        reference = np.asarray(reference_rgb)
        reference_shape = (reference.shape[0], reference.shape[1])
        report = map_validity(mapping, reference_shape)
        residual = np.zeros_like(mapping)
        diagnostics = AlignmentDiagnostics(
            selected_stage="line",
            anchor_error=0.0,
            inlier_count=0,
            valid_fraction=report.valid_fraction,
            negative_jacobian_fraction=report.negative_jacobian_fraction,
            displacement_p50=report.displacement_p50,
            displacement_p95=report.displacement_p95,
            fallback_reason="dense_disabled",
        )
        return DenseRefinementResult(mapping, residual, False, diagnostics)
