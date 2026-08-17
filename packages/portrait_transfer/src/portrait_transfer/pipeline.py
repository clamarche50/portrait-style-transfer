"""Side-effect-free end-to-end portrait style transfer pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .alignment.anchors import alignment_anchors
from .alignment.beier_neely import build_beier_neely_backward_map
from .alignment.correction_constraints import (
    apply_eye_center_operations,
    apply_gain_constraint_operations,
    apply_mask_stroke_operations,
    corrected_alignment_points,
    eye_highlight_transforms,
    normalize_correction_operations,
)
from .alignment.dense_sift import DenseCorrespondenceBackend, refine_with_dense_sift
from .alignment.diagnostics import collect_alignment_diagnostics
from .alignment.flow_optimization import CpuDenseCorrespondence, NoOpDenseCorrespondence
from .alignment.legacy_66_connections import landmark_segments
from .alignment.map_composition import compose_with_residual
from .alignment.similarity import SimilarityEstimate, estimate_similarity_backward_map
from .background import apply_background_mode
from .color.lab import lab_to_rgb_gamut_safe, rgb_to_lab
from .color.legacy_color import legacy_lab_to_rgb, legacy_rgb_to_lab
from .config import AlgorithmProfile, TransferSettings
from .crop import create_canonical_crop
from .diagnostics import ArtifactCollector
from .exceptions import AlignmentFailure, ProcessingCancelled
from .eyes.extraction import extract_highlight_asset
from .eyes.highlight_transfer import transfer_eye_highlights
from .geometry.sampling import MATLAB_OOB_FILL, warp
from .geometry.transforms import transform_points
from .geometry.validity import map_validity
from .image_io import normalize_rgb
from .multiscale.color import should_transfer_band, transfer_band
from .multiscale.energy import compute_energy_pair
from .multiscale.gain import compute_gain
from .multiscale.histogram import apply_global_range_mix, histogram_match
from .multiscale.laplacian import build_laplacian_stack
from .multiscale.reconstruction import blend_residual, reconstruct
from .postprocess import blend_by_confidence, final_sanitize
from .preflight import (
    HeuristicPortraitAnalyzer,
    PortraitAnalyzer,
    analyze_portrait,
    validate_pair,
)
from .resume import (
    RESUME_BACKGROUND,
    RESUME_MULTISCALE,
    build_resume_artifacts,
    load_resume_state,
    requested_resume_stage,
)
from .segmentation import build_effective_mask
from .types import (
    ProcessingStage,
    RuntimeContext,
    TransferDiagnostics,
    TransferResult,
)


def estimate_similarity_with_fallback(
    input_landmarks: ArrayLike,
    reference_landmarks: ArrayLike,
    operations: Sequence[Mapping[str, Any]],
    destination_shape: tuple[int, int],
) -> tuple[NDArray[np.float32], SimilarityEstimate, str]:
    class _Attempt(TypedDict):
        name: str
        include_nose: bool
        ransac_threshold: float
        min_inliers: int
        max_rotation_degrees: float

    attempts: list[_Attempt] = [
        {
            "name": "strict_4pt",
            "include_nose": True,
            "ransac_threshold": 0.04,
            "min_inliers": 3,
            "max_rotation_degrees": 30.0,
        },
        {
            "name": "strict_3pt",
            "include_nose": False,
            "ransac_threshold": 0.04,
            "min_inliers": 3,
            "max_rotation_degrees": 30.0,
        },
        {
            "name": "relaxed_3pt",
            "include_nose": False,
            "ransac_threshold": 0.06,
            "min_inliers": 3,
            "max_rotation_degrees": 35.0,
        },
        {
            "name": "fallback_2inlier",
            "include_nose": False,
            "ransac_threshold": 0.08,
            "min_inliers": 2,
            "max_rotation_degrees": 40.0,
        },
    ]

    last_exc = None

    for attempt in attempts:
        try:
            input_anchor_points = alignment_anchors(
                input_landmarks,
                include_nose=attempt["include_nose"],
            )
            reference_anchor_points = alignment_anchors(
                reference_landmarks,
                include_nose=attempt["include_nose"],
            )

            input_anchor_points, reference_anchor_points = corrected_alignment_points(
                input_anchor_points,
                reference_anchor_points,
                operations,
            )

            affine_map, similarity = estimate_similarity_backward_map(
                reference_anchor_points,
                input_anchor_points,
                destination_shape,
                ransac_threshold=attempt["ransac_threshold"],
                min_inliers=attempt["min_inliers"],
                max_rotation_degrees=attempt["max_rotation_degrees"],
            )

            return affine_map, similarity, attempt["name"]

        except AlignmentFailure as exc:
            last_exc = exc

    raise last_exc or AlignmentFailure("Failed to estimate similarity transform")


def create_default_runtime(
    *,
    enable_cpu_dense: bool = True,
    analyzer: PortraitAnalyzer | None = None,
) -> RuntimeContext:
    """Create a deterministic runtime with explicit lightweight fallbacks."""

    dense_backend = (
        CpuDenseCorrespondence() if enable_cpu_dense else NoOpDenseCorrespondence()
    )
    return RuntimeContext(
        analyzer=(analyzer if analyzer is not None else HeuristicPortraitAnalyzer()),
        dense_backend=dense_backend,
    )


def _check_cancel(runtime: RuntimeContext) -> None:
    if runtime.cancelled():
        raise ProcessingCancelled()


def _timed(stage_durations: dict[str, float], name: str, started: float) -> None:
    stage_durations[name] = (perf_counter() - started) * 1000.0


def _transfer_multiscale(
    input_rgb: NDArray[np.float32],
    reference_rgb: NDArray[np.float32],
    input_mask: NDArray[np.float32],
    reference_mask: NDArray[np.float32],
    effective_mask: NDArray[np.float32],
    mapping: NDArray[np.float32],
    settings: TransferSettings,
    artifacts: ArtifactCollector,
    runtime: RuntimeContext,
) -> NDArray[np.float32]:
    profile = settings.algorithm_profile
    if profile is AlgorithmProfile.PAPER_EXACT:
        input_color = rgb_to_lab(input_rgb)
        reference_color = rgb_to_lab(reference_rgb)
        to_rgb = lab_to_rgb_gamut_safe
    else:
        input_color = legacy_rgb_to_lab(input_rgb)
        reference_color = legacy_rgb_to_lab(reference_rgb)
        to_rgb = legacy_lab_to_rgb

    output_color = np.empty_like(input_color)
    monochrome_style = bool(runtime.corrections.get("monochrome_style", False))
    operations = tuple(runtime.corrections.get("operations", ()))
    for channel in range(3):
        input_stack = build_laplacian_stack(
            input_color[..., channel], input_mask, profile=profile
        )
        reference_stack = build_laplacian_stack(
            reference_color[..., channel], reference_mask, profile=profile
        )
        output_bands: list[NDArray[np.float32]] = []
        for level, input_band in enumerate(input_stack.bands):
            enabled = should_transfer_band(
                channel,
                level,
                profile=profile,
                legacy_mode=settings.legacy_color_mode,
                monochrome_style=monochrome_style,
            )
            if enabled:
                energy = compute_energy_pair(
                    input_band,
                    reference_stack.bands[level],
                    input_mask,
                    reference_mask,
                    mapping,
                    level,
                    profile=profile,
                )
                gain = compute_gain(
                    energy.input_energy,
                    energy.warped_reference_energy,
                    effective_mask,
                    level,
                    profile=profile,
                    transfer_strength=settings.transfer_strength,
                    settings=settings.gain,
                )
                effective_gain = gain.effective
                effective_gain = apply_gain_constraint_operations(
                    effective_gain,
                    operations,
                    channel=channel,
                    level=level,
                )
                correction_key = f"gain_{channel}_{level}"
                if correction_key in runtime.corrections:
                    correction = np.asarray(
                        runtime.corrections[correction_key], dtype=np.float32
                    )
                    if correction.shape != effective_gain.shape:
                        raise ValueError(f"{correction_key} has an incompatible shape")
                    effective_gain = effective_gain * correction
                output_band = transfer_band(input_band, effective_gain, enabled=True)
                artifacts.add(f"energy_input_c{channel}_l{level}", energy.input_energy)
                artifacts.add(
                    f"energy_reference_c{channel}_l{level}",
                    energy.warped_reference_energy,
                )
                artifacts.add(f"gain_raw_c{channel}_l{level}", gain.raw)
                artifacts.add(f"gain_clipped_c{channel}_l{level}", gain.clipped)
                artifacts.add(f"gain_smoothed_c{channel}_l{level}", gain.smoothed)
            else:
                output_band = input_band.copy()
            output_bands.append(output_band)
            artifacts.add(f"input_band_c{channel}_l{level}", input_band)
            artifacts.add(
                f"reference_band_c{channel}_l{level}", reference_stack.bands[level]
            )
        warped_reference_residual = warp(
            reference_stack.residual, mapping, mode="constant", cval=MATLAB_OOB_FILL
        )
        output_residual = blend_residual(
            input_stack.residual, warped_reference_residual, settings.residual_strength
        )
        output_color[..., channel] = reconstruct(tuple(output_bands), output_residual)
        artifacts.add(f"input_residual_c{channel}", input_stack.residual)
        artifacts.add(f"reference_residual_c{channel}", warped_reference_residual)
    return to_rgb(output_color)


def transfer_portrait_style(
    input_rgb: ArrayLike,
    reference_rgb: ArrayLike,
    settings: TransferSettings | None = None,
    runtime: RuntimeContext | None = None,
) -> TransferResult:
    """Transfer portrait style with no external writes or hidden model downloads."""

    settings = settings or TransferSettings()
    runtime = runtime or create_default_runtime()
    stage_durations: dict[str, float] = {}
    warnings: list[str] = []
    artifacts = ArtifactCollector(settings.debug_artifacts)
    resumed_from_stage: str | None = None

    started = perf_counter()
    runtime.progress(ProcessingStage.NORMALIZE, 2, "Normalizing images")
    input_image = normalize_rgb(input_rgb)
    reference_image = normalize_rgb(reference_rgb)
    _timed(stage_durations, ProcessingStage.NORMALIZE.value, started)
    _check_cancel(runtime)

    requested_stage = requested_resume_stage(runtime)
    resume_state = load_resume_state(
        input_image=input_image,
        reference_image=reference_image,
        settings=settings,
        runtime=runtime,
    )
    raw_operations = tuple(runtime.corrections.get("operations", ()))

    if resume_state is not None:
        resumed_from_stage = resume_state.stage
        input_context = resume_state.input_context
        input_crop = resume_state.input_crop
        reference_crop = resume_state.reference_crop
        input_masks = resume_state.input_masks
        reference_masks = resume_state.reference_masks
        input_base_irises = (
            input_masks.irises[0].copy(),
            input_masks.irises[1].copy(),
        )
        operations = normalize_correction_operations(
            raw_operations, (int(input_crop.shape[0]), int(input_crop.shape[1]))
        )
        input_masks = replace(
            input_masks,
            irises=apply_eye_center_operations(input_base_irises, operations),
        )
        mapping = resume_state.mapping
        input_quality = resume_state.input_quality
        reference_quality = resume_state.reference_quality
        compatibility = resume_state.compatibility
        alignment_diagnostics = resume_state.alignment
        warnings.extend(
            resume_state.warnings_after_eyes
            if resumed_from_stage == RESUME_BACKGROUND
            else resume_state.upstream_warnings
        )
        if operations:
            warnings.append("manual_corrections_applied")
        checkpoint_upstream_warnings = tuple(
            dict.fromkeys(
                (*resume_state.upstream_warnings,)
                + (("manual_corrections_applied",) if operations else ())
            )
        )
        warnings.append(f"resume_reused_{resumed_from_stage}")
    else:
        if requested_stage is not None:
            warnings.append("resume_cache_rejected")

        started = perf_counter()
        runtime.progress(ProcessingStage.PREFLIGHT, 8, "Analyzing portraits")
        input_analysis = analyze_portrait(
            input_image,
            runtime.analyzer,
            settings.preflight,
            settings.processing_long_edge,
        )
        reference_analysis = analyze_portrait(
            reference_image,
            runtime.analyzer,
            settings.preflight,
            settings.processing_long_edge,
        )
        compatibility = validate_pair(
            input_analysis, reference_analysis, input_image, reference_image
        )
        input_quality = input_analysis.quality
        reference_quality = reference_analysis.quality
        warnings.extend(input_analysis.warnings)
        warnings.extend(reference_analysis.warnings)
        warnings.extend(compatibility.warnings)
        _timed(stage_durations, ProcessingStage.PREFLIGHT.value, started)
        _check_cancel(runtime)

        started = perf_counter()
        runtime.progress(ProcessingStage.CROP, 16, "Building canonical crops")
        input_context = create_canonical_crop(
            input_image,
            input_analysis,
            processing_long_edge=settings.processing_long_edge,
        )
        reference_context = create_canonical_crop(
            reference_image,
            reference_analysis,
            processing_long_edge=settings.processing_long_edge,
            output_shape=input_context.crop_shape,
        )
        input_crop = input_context.extract(input_image)
        reference_crop = reference_context.extract(reference_image)
        input_masks = input_context.extract_masks(input_analysis.masks)
        reference_masks = reference_context.extract_masks(reference_analysis.masks)
        input_base_irises = (
            input_masks.irises[0].copy(),
            input_masks.irises[1].copy(),
        )
        operations = normalize_correction_operations(
            raw_operations, (int(input_crop.shape[0]), int(input_crop.shape[1]))
        )
        if operations:
            corrected_head = apply_mask_stroke_operations(
                input_masks.head, operations, target="head"
            )
            corrected_alpha = apply_mask_stroke_operations(
                input_masks.foreground_alpha,
                operations,
                target="foreground_alpha",
            )
            input_masks = replace(
                input_masks,
                head=corrected_head,
                effective_transfer=corrected_head.copy(),
                foreground_alpha=corrected_alpha,
                irises=apply_eye_center_operations(input_base_irises, operations),
            )
            warnings.append("manual_corrections_applied")
        _timed(stage_durations, ProcessingStage.CROP.value, started)
        _check_cancel(runtime)

        started = perf_counter()
        runtime.progress(ProcessingStage.ALIGNMENT, 24, "Aligning landmarks")
        input_anchor_points = alignment_anchors(input_context.landmarks)
        reference_anchor_points = alignment_anchors(reference_context.landmarks)
        input_anchor_points, reference_anchor_points = corrected_alignment_points(
            input_anchor_points,
            reference_anchor_points,
            operations,
        )
        affine_map, similarity, alignment_mode = estimate_similarity_with_fallback(
            input_context.landmarks,
            reference_context.landmarks,
            operations,
            (int(input_crop.shape[0]), int(input_crop.shape[1])),
        )

        if alignment_mode != "strict_4pt":
            warnings.append(f"alignment_fallback:{alignment_mode}")
        aligned_reference_landmarks = transform_points(
            reference_context.landmarks, similarity.matrix
        )
        input_segments = landmark_segments(input_context.landmarks)
        aligned_reference_segments = landmark_segments(aligned_reference_landmarks)
        reference_shape = (
            int(reference_crop.shape[0]),
            int(reference_crop.shape[1]),
            int(reference_crop.shape[2]),
        )
        try:
            line_map = build_beier_neely_backward_map(
                input_segments,
                aligned_reference_segments,
                (int(input_crop.shape[0]), int(input_crop.shape[1])),
                initial_map=affine_map,
                settings=settings.beier_neely,
            )
            line_report = map_validity(line_map, reference_shape)
            if (
                line_report.valid_fraction < 0.70
                or line_report.negative_jacobian_fraction > 0.05
            ):
                raise AlignmentFailure("Line map failed validity thresholds")
            mapping = line_map
            alignment_diagnostics = collect_alignment_diagnostics(
                mapping,
                reference_shape,
                selected_stage="line",
                anchor_error=similarity.normalized_error,
                inlier_count=int(similarity.inliers.sum()),
            )
        except AlignmentFailure as exc:
            mapping = affine_map
            alignment_diagnostics = collect_alignment_diagnostics(
                mapping,
                reference_shape,
                selected_stage="affine",
                anchor_error=similarity.normalized_error,
                inlier_count=int(similarity.inliers.sum()),
                fallback_reason=str(exc),
            )
            warnings.append("line_alignment_fallback")

        if "absolute_map" in runtime.corrections:
            corrected = np.asarray(
                runtime.corrections["absolute_map"], dtype=np.float32
            )
            if corrected.shape != mapping.shape:
                raise ValueError(
                    "absolute_map correction must match the canonical crop"
                )
            correction_report = map_validity(corrected, reference_shape)
            if (
                correction_report.valid_fraction < 0.70
                or correction_report.negative_jacobian_fraction > 0.05
            ):
                raise AlignmentFailure("Corrected absolute map is invalid")
            mapping = corrected
        if "residual_flow" in runtime.corrections:
            residual = np.asarray(
                runtime.corrections["residual_flow"], dtype=np.float32
            )
            if residual.shape != mapping.shape:
                raise ValueError(
                    "residual_flow correction must match the canonical crop"
                )
            mapping = compose_with_residual(mapping, residual)
        _timed(stage_durations, ProcessingStage.ALIGNMENT.value, started)
        _check_cancel(runtime)

        started = perf_counter()
        if settings.dense_alignment and settings.dense.enabled:
            runtime.progress(
                ProcessingStage.DENSE_REFINEMENT, 36, "Refining dense correspondence"
            )
            backend = cast(
                DenseCorrespondenceBackend,
                runtime.dense_backend or CpuDenseCorrespondence(),
            )
            dense = refine_with_dense_sift(
                input_crop=input_crop,
                reference_rgb=reference_crop,
                initial_backward_map=mapping,
                input_mask=input_masks.head,
                reference_mask=reference_masks.head,
                backend=backend,
                settings=settings.dense,
            )
            mapping = dense.mapping if dense.valid else mapping
            alignment_diagnostics = replace(
                dense.diagnostics,
                anchor_error=similarity.normalized_error,
                inlier_count=int(similarity.inliers.sum()),
            )
            if not dense.valid:
                warnings.append("dense_alignment_fallback")
        _timed(stage_durations, ProcessingStage.DENSE_REFINEMENT.value, started)
        _check_cancel(runtime)
        checkpoint_upstream_warnings = tuple(
            item for item in dict.fromkeys(warnings) if not item.startswith("resume_")
        )

    if resumed_from_stage in (None, RESUME_MULTISCALE):
        aligned_reference = warp(
            reference_crop, mapping, mode="constant", cval=MATLAB_OOB_FILL
        )
        warped_reference_head = warp(reference_masks.head, mapping, mode="constant")
        effective_mask = build_effective_mask(
            input_masks.head,
            warped_reference_head,
            minimum_coverage=settings.preflight.min_effective_coverage,
        )
        artifacts.add("mapping", mapping)
        artifacts.add("effective_mask", effective_mask)
        artifacts.add("aligned_reference", aligned_reference)

        started = perf_counter()
        runtime.progress(
            ProcessingStage.MULTISCALE, 48, "Transferring multiscale statistics"
        )
        local_rgb = _transfer_multiscale(
            input_crop,
            reference_crop,
            input_masks.head,
            reference_masks.head,
            effective_mask,
            mapping,
            settings,
            artifacts,
            runtime,
        )
        local_rgb = blend_by_confidence(local_rgb, input_crop, effective_mask)
        matched_rgb = histogram_match(
            local_rgb, aligned_reference, effective_mask, warped_reference_head
        )
        ranged_rgb = apply_global_range_mix(
            local_rgb, matched_rgb, settings.global_range_mix
        )
        artifacts.add("local_only", local_rgb)
        artifacts.add("histogram_only", matched_rgb)
        _timed(stage_durations, ProcessingStage.MULTISCALE.value, started)
        _check_cancel(runtime)
        pre_eye_rgb = ranged_rgb.copy()
    else:
        if resume_state is None or resume_state.pre_eye_rgb is None:
            raise RuntimeError("Validated eye/background resume is missing pre-eye RGB")
        pre_eye_rgb = resume_state.pre_eye_rgb.copy()
        ranged_rgb = (
            resume_state.post_eye_rgb.copy()
            if resumed_from_stage == RESUME_BACKGROUND
            and resume_state.post_eye_rgb is not None
            else pre_eye_rgb.copy()
        )

    if resumed_from_stage != RESUME_BACKGROUND:
        started = perf_counter()
        if settings.eye_highlights:
            runtime.progress(
                ProcessingStage.EYES, 82, "Applying confidence-gated eye highlights"
            )
            assets = runtime.eye_assets
            if assets is None:
                extracted = tuple(
                    extract_highlight_asset(reference_crop, iris)
                    for iris in reference_masks.irises
                )
                assets = (extracted[0], extracted[1])
            if any(asset is not None for asset in assets):
                highlight_scales, highlight_rotations = eye_highlight_transforms(
                    operations
                )
                ranged_rgb = transfer_eye_highlights(
                    ranged_rgb,
                    input_masks.irises,
                    assets,
                    scale_multipliers=highlight_scales,
                    rotation_degrees=highlight_rotations,
                )
            else:
                warnings.append("eye_highlights_skipped")
        _timed(stage_durations, ProcessingStage.EYES.value, started)
        _check_cancel(runtime)
    post_eye_rgb = ranged_rgb.copy()
    checkpoint_warnings_after_eyes = tuple(
        item for item in dict.fromkeys(warnings) if not item.startswith("resume_")
    )

    started = perf_counter()
    runtime.progress(ProcessingStage.BACKGROUND, 90, "Compositing background")
    composed_crop = apply_background_mode(
        ranged_rgb,
        input_crop,
        reference_crop,
        input_masks.foreground_alpha,
        settings,
        reference_alpha=reference_masks.foreground_alpha,
    )
    output = input_context.composite_back(
        original=input_image,
        processed_crop=composed_crop,
        alpha=input_masks.foreground_alpha,
    )
    _timed(stage_durations, ProcessingStage.BACKGROUND.value, started)

    started = perf_counter()
    runtime.progress(ProcessingStage.FINALIZE, 98, "Finalizing output")
    output = final_sanitize(output)
    artifacts.add("final_blend", output)
    _timed(stage_durations, ProcessingStage.FINALIZE.value, started)
    runtime.progress(ProcessingStage.FINALIZE, 100, "Complete")

    diagnostics = TransferDiagnostics(
        input_quality=input_quality,
        reference_quality=reference_quality,
        compatibility=compatibility,
        alignment=alignment_diagnostics,
        profile=settings.algorithm_profile.value,
        stage_durations_ms=stage_durations,
        warnings=tuple(dict.fromkeys(warnings)),
        resumed_from_stage=resumed_from_stage,
    )
    private_checkpoints = build_resume_artifacts(
        input_image=input_image,
        reference_image=reference_image,
        settings=settings,
        runtime=runtime,
        input_context=input_context,
        input_crop=input_crop,
        reference_crop=reference_crop,
        input_masks=input_masks,
        input_base_irises=input_base_irises,
        reference_masks=reference_masks,
        mapping=mapping,
        pre_eye_rgb=pre_eye_rgb,
        post_eye_rgb=post_eye_rgb,
        input_quality=input_quality,
        reference_quality=reference_quality,
        compatibility=compatibility,
        alignment=alignment_diagnostics,
        upstream_warnings=checkpoint_upstream_warnings,
        warnings_after_eyes=checkpoint_warnings_after_eyes,
    )
    return TransferResult(
        output,
        diagnostics,
        artifacts.snapshot(),
        resume_artifacts=private_checkpoints,
    )
