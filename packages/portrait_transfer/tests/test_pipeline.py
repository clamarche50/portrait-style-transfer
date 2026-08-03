from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from portrait_transfer import (
    RuntimeContext,
    TransferSettings,
    create_default_runtime,
    transfer_portrait_style,
)
from portrait_transfer.config import BackgroundMode, DenseSettings, PreflightThresholds
from portrait_transfer.exceptions import ProcessingCancelled
from portrait_transfer.image_io import encode_png
from portrait_transfer.preflight import HeuristicPortraitAnalyzer


class _ToggleAnalyzer:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False
        self.delegate = HeuristicPortraitAnalyzer()

    def analyze(self, rgb: np.ndarray):
        if self.fail:
            raise AssertionError("resume unexpectedly recomputed portrait analysis")
        self.calls += 1
        return self.delegate.analyze(rgb)


def _synthetic_settings(*, debug: bool = False) -> TransferSettings:
    return TransferSettings(
        processing_long_edge=64,
        dense_alignment=False,
        eye_highlights=False,
        global_range_mix=0.10,
        debug_artifacts=debug,
        preflight=PreflightThresholds(
            min_inter_eye_distance=8,
            severe_blur_variance=0,
            min_mask_confidence=0,
            min_effective_coverage=0.01,
        ),
        dense=DenseSettings(enabled=False),
    )


def test_end_to_end_is_deterministic_and_emits_diagnostics(
    textured_rgb: np.ndarray,
) -> None:
    reference = np.clip(
        textured_rgb * np.asarray((0.75, 0.90, 0.65), dtype=np.float32) + 0.08, 0, 1
    )
    settings = _synthetic_settings(debug=True)
    first = transfer_portrait_style(
        textured_rgb,
        reference,
        settings,
        create_default_runtime(enable_cpu_dense=False),
    )
    second = transfer_portrait_style(
        textured_rgb,
        reference,
        settings,
        create_default_runtime(enable_cpu_dense=False),
    )
    assert first.output_rgb.shape == textured_rgb.shape
    assert encode_png(first.output_rgb) == encode_png(second.output_rgb)
    assert first.diagnostics.profile == "paper_exact"
    assert first.diagnostics.alignment.selected_stage in {"line", "affine"}
    assert "effective_mask" in first.artifacts
    assert "final_blend" in first.artifacts


def test_cancellation_is_explicit_and_creates_no_files(
    textured_rgb: np.ndarray, tmp_path
) -> None:
    runtime = RuntimeContext(
        analyzer=create_default_runtime(enable_cpu_dense=False).analyzer,
        dense_backend=create_default_runtime(enable_cpu_dense=False).dense_backend,
        cancel_check=lambda: True,
    )
    with pytest.raises(ProcessingCancelled):
        transfer_portrait_style(
            textured_rgb, textured_rgb, _synthetic_settings(), runtime
        )
    assert list(tmp_path.iterdir()) == []


def test_progress_callback_is_monotonic(textured_rgb: np.ndarray) -> None:
    events: list[tuple[int, str]] = []
    default = create_default_runtime(enable_cpu_dense=False)
    runtime = RuntimeContext(
        analyzer=default.analyzer,
        dense_backend=default.dense_backend,
        progress_callback=lambda stage, percent, message: events.append(
            (percent, stage.value)
        ),
    )
    transfer_portrait_style(textured_rgb, textured_rgb, _synthetic_settings(), runtime)
    percentages = [event[0] for event in events]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100


def test_invalid_correction_map_fails_explicitly(textured_rgb: np.ndarray) -> None:
    default = create_default_runtime(enable_cpu_dense=False)
    runtime = RuntimeContext(
        analyzer=default.analyzer,
        dense_backend=default.dense_backend,
        corrections={"absolute_map": np.zeros((3, 3, 2), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="absolute_map"):
        transfer_portrait_style(
            textured_rgb, textured_rgb, _synthetic_settings(), runtime
        )


def _runtime_with_corrections(
    analyzer: _ToggleAnalyzer,
    corrections: dict,
    resume_artifacts: dict[str, np.ndarray] | None = None,
) -> RuntimeContext:
    return replace(
        create_default_runtime(enable_cpu_dense=False, analyzer=analyzer),
        corrections=corrections,
        resume_artifacts=resume_artifacts or {},
    )


def test_gain_rerun_reuses_correspondence_and_matches_full_run(
    textured_rgb: np.ndarray,
) -> None:
    reference = np.clip(textured_rgb * (0.7, 0.9, 0.8) + 0.05, 0.0, 1.0)
    settings = _synthetic_settings()
    analyzer = _ToggleAnalyzer()
    first = transfer_portrait_style(
        textured_rgb,
        reference,
        settings,
        _runtime_with_corrections(analyzer, {}),
    )
    assert analyzer.calls == 2
    gain_operation = {
        "type": "gain_constraint",
        "channel": "*",
        "level": "*",
        "mode": "LOCK_TO_ONE",
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "coordinate_space": "normalized",
    }
    analyzer.fail = True
    resumed = transfer_portrait_style(
        textured_rgb,
        reference,
        settings,
        _runtime_with_corrections(
            analyzer,
            {
                "resume_from_stage": "multiscale",
                "operations": (gain_operation,),
            },
            first.resume_artifacts,
        ),
    )
    full = transfer_portrait_style(
        textured_rgb,
        reference,
        settings,
        _runtime_with_corrections(_ToggleAnalyzer(), {"operations": (gain_operation,)}),
    )
    assert encode_png(resumed.output_rgb) == encode_png(full.output_rgb)
    assert resumed.diagnostics.resumed_from_stage == "multiscale"
    assert "resume_reused_multiscale" in resumed.diagnostics.warnings
    assert "preflight" not in resumed.diagnostics.stage_durations_ms


def test_eye_rerun_reuses_pre_eye_result_and_matches_full_run(
    textured_rgb: np.ndarray,
) -> None:
    reference = np.clip(textured_rgb * (0.8, 0.65, 0.95) + 0.03, 0.0, 1.0)
    initial_settings = _synthetic_settings()
    analyzer = _ToggleAnalyzer()
    first = transfer_portrait_style(
        textured_rgb,
        reference,
        initial_settings,
        _runtime_with_corrections(analyzer, {}),
    )
    eye_settings = replace(initial_settings, eye_highlights=True)
    eye_operation = {
        "type": "eye_center",
        "eye": "left",
        "center": [0.63, 0.43],
        "radius": 0.035,
        "highlight_scale": 0.8,
        "highlight_rotation_degrees": 20.0,
        "coordinate_space": "normalized",
    }
    analyzer.fail = True
    resumed = transfer_portrait_style(
        textured_rgb,
        reference,
        eye_settings,
        _runtime_with_corrections(
            analyzer,
            {"resume_from_stage": "eyes", "operations": (eye_operation,)},
            first.resume_artifacts,
        ),
    )
    full = transfer_portrait_style(
        textured_rgb,
        reference,
        eye_settings,
        _runtime_with_corrections(_ToggleAnalyzer(), {"operations": (eye_operation,)}),
    )
    assert encode_png(resumed.output_rgb) == encode_png(full.output_rgb)
    assert resumed.diagnostics.resumed_from_stage == "eyes"
    assert "multiscale" not in resumed.diagnostics.stage_durations_ms


def test_background_rerun_reuses_post_eye_result_and_matches_full_run(
    textured_rgb: np.ndarray,
) -> None:
    settings = _synthetic_settings()
    analyzer = _ToggleAnalyzer()
    first = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        settings,
        _runtime_with_corrections(analyzer, {}),
    )
    background_settings = replace(
        settings,
        background_mode=BackgroundMode.SOLID,
        background_color=(0.12, 0.18, 0.24),
    )
    analyzer.fail = True
    resumed = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        background_settings,
        _runtime_with_corrections(
            analyzer,
            {"resume_from_stage": "background"},
            first.resume_artifacts,
        ),
    )
    full = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        background_settings,
        _runtime_with_corrections(_ToggleAnalyzer(), {}),
    )
    assert encode_png(resumed.output_rgb) == encode_png(full.output_rgb)
    assert resumed.diagnostics.resumed_from_stage == "background"
    assert "eyes" not in resumed.diagnostics.stage_durations_ms


def test_corrupt_resume_bundle_falls_back_to_full_run(
    textured_rgb: np.ndarray,
) -> None:
    settings = _synthetic_settings()
    analyzer = _ToggleAnalyzer()
    first = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        settings,
        _runtime_with_corrections(analyzer, {}),
    )
    corrupt = dict(first.resume_artifacts)
    corrupt_mapping = corrupt["resume.correspondence_mapping"].copy()
    corrupt_mapping[0, 0, 0] = np.nan
    corrupt["resume.correspondence_mapping"] = corrupt_mapping
    rerun = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        settings,
        _runtime_with_corrections(
            analyzer,
            {"resume_from_stage": "multiscale"},
            corrupt,
        ),
    )
    assert analyzer.calls == 4
    assert encode_png(rerun.output_rgb) == encode_png(first.output_rgb)
    assert rerun.diagnostics.resumed_from_stage is None
    assert "resume_cache_rejected" in rerun.diagnostics.warnings


def test_stale_resume_signature_falls_back_to_full_run(
    textured_rgb: np.ndarray,
) -> None:
    settings = _synthetic_settings()
    analyzer = _ToggleAnalyzer()
    first = transfer_portrait_style(
        textured_rgb,
        textured_rgb,
        settings,
        _runtime_with_corrections(analyzer, {}),
    )
    changed_reference = textured_rgb.copy()
    changed_reference[0, 0] = (1.0, 0.0, 0.0)
    rerun = transfer_portrait_style(
        textured_rgb,
        changed_reference,
        settings,
        _runtime_with_corrections(
            analyzer,
            {"resume_from_stage": "background"},
            first.resume_artifacts,
        ),
    )
    assert rerun.output_rgb.shape == textured_rgb.shape
    assert analyzer.calls == 4
    assert rerun.diagnostics.resumed_from_stage is None
    assert "resume_cache_rejected" in rerun.diagnostics.warnings
