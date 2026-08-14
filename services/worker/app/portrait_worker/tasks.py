from __future__ import annotations

import hashlib
import io
import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Any, TypeVar, cast

import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from numpy.typing import NDArray
from portrait_api.metrics import JOBS, JOBS_RUNNING, STORAGE_ERRORS
from portrait_api.models import (
    ArtifactKind,
    Asset,
    AssetKind,
    JobStatus,
    ProcessingStage,
    Style,
    StyleExample,
)
from portrait_api.repositories import AssetRepository, JobRepository
from portrait_api.services.ranking import StyleRankingService
from portrait_api.time import is_expired
from portrait_transfer.alignment.anchors import eye_centers
from portrait_transfer.config import ImageLimits, PreflightThresholds
from portrait_transfer.exceptions import (
    InputValidationError,
    PortraitTransferError,
    ProcessingCancelled,
)
from portrait_transfer.image_io import decode_image, encode_jpeg, encode_png
from portrait_transfer.preflight import PortraitAnalyzer, analyze_portrait
from portrait_transfer.selection import build_style_feature
from portrait_transfer.style_ingestion import IngestedStyle, ingest_style
from portrait_transfer.types import EyeHighlightAsset, PortraitAnalysis, StyleFeature
from portrait_worker.ai_client import (
    AI_ENGINE_ID,
    AIEngineError,
    AIEngineResponse,
    get_ai_engine_client,
)
from portrait_worker.celery_app import celery_app
from portrait_worker.cleanup import purge_expired_records
from portrait_worker.infrastructure import WorkerInfrastructure, get_infrastructure
from portrait_worker.mediapipe_adapter import build_portrait_analyzer
from portrait_worker.progress import JobLeaseLost, JobProgressReporter
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy.exc import OperationalError


class StyleSelectionFailure(RuntimeError):
    """Raised when a style cannot provide a usable reference portrait."""


TaskCallable = TypeVar("TaskCallable", bound=Callable[..., object])
_AI_RESPONSE_MAX_ENCODED_BYTES = 96 * 1024 * 1024


def _celery_task(**options: object) -> Callable[[TaskCallable], TaskCallable]:
    """Type Celery's dynamically provided decorator without changing it."""

    return cast(Callable[[TaskCallable], TaskCallable], celery_app.task(**options))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "minimum": float(np.nanmin(value)) if value.size else None,
            "maximum": float(np.nanmax(value)) if value.size else None,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _decode_limits(settings: object, *, max_encoded_bytes: int) -> ImageLimits:
    return ImageLimits(
        max_encoded_bytes=max_encoded_bytes,
        max_decoded_pixels=int(getattr(settings, "max_decoded_pixels", 8_000_000)),
        max_original_long_edge=int(getattr(settings, "max_original_long_edge", 8_000)),
    )


def _bounded_diagnostic_strength(value: object, fallback: object) -> float:
    try:
        default = float(cast(Any, fallback))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(default) or not 0.0 <= default <= 1.0:
        default = 0.0
    try:
        candidate = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    return candidate if np.isfinite(candidate) and 0.0 <= candidate <= 1.0 else default


def _hex_color(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise InputValidationError("Invalid solid background color")
    red, green, blue = (int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    return red, green, blue


def _latest_background_correction(
    settings: dict[str, Any], corrections: list[dict[str, Any]]
) -> tuple[str, str | None]:
    mode = str(settings.get("background_mode", "KEEP"))
    color = settings.get("background_color")
    for correction in corrections:
        if correction.get("type") == "background":
            mode = str(correction["mode"])
            color = correction.get("color")
    return mode, color


def _ai_settings(
    job_settings: dict[str, Any], corrections: list[dict[str, Any]]
) -> dict[str, object]:
    background_mode, background_color = _latest_background_correction(job_settings, corrections)
    background_mode = background_mode.upper()
    if background_mode not in {"KEEP", "BLUR", "SOLID", "REFERENCE"}:
        raise InputValidationError("Unsupported background mode")
    if background_mode == "SOLID":
        if _hex_color(background_color) is None:
            raise InputValidationError("A solid background color is required")
    elif background_color is not None:
        raise InputValidationError("A background color is only valid for solid backgrounds")

    try:
        style_strength = float(job_settings.get("style_strength", 0.75))
        structure_strength = float(job_settings.get("structure_strength", 0.9))
        inference_steps = int(job_settings.get("inference_steps", 30))
        random_seed = int(job_settings.get("random_seed", 0))
    except (TypeError, ValueError) as exc:
        raise InputValidationError("AI transfer settings are invalid") from exc
    if not 0.0 <= style_strength <= 1.0:
        raise InputValidationError("style_strength must be in [0, 1]")
    if not 0.0 <= structure_strength <= 1.0:
        raise InputValidationError("structure_strength must be in [0, 1]")
    if not 10 <= inference_steps <= 50:
        raise InputValidationError("inference_steps must be in [10, 50]")
    if not 0 <= random_seed <= 2**31 - 1:
        raise InputValidationError("random_seed must be in [0, 2^31 - 1]")

    return {
        "algorithm_profile": AI_ENGINE_ID,
        "style_strength": style_strength,
        "structure_strength": structure_strength,
        "inference_steps": inference_steps,
        "random_seed": random_seed,
        "background_mode": background_mode,
        "background_color": background_color,
    }


def _engine_request_settings(settings: dict[str, object]) -> dict[str, object]:
    """Return only fields accepted by the isolated inference service."""

    return {
        key: settings[key]
        for key in (
            "algorithm_profile",
            "style_strength",
            "structure_strength",
            "inference_steps",
            "random_seed",
        )
    }


def _preflight_thresholds(image: NDArray[np.float32]) -> PreflightThresholds:
    minimum_eye_distance = min(150.0, max(24.0, min(image.shape[:2]) * 0.15))
    return PreflightThresholds(min_inter_eye_distance=minimum_eye_distance)


def _normalized_landmarks(analysis: PortraitAnalysis) -> NDArray[np.float32]:
    left_eye, right_eye = eye_centers(analysis.landmarks)
    midpoint = (left_eye + right_eye) * 0.5
    scale = float(np.linalg.norm(right_eye - left_eye))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("invalid inter-eye scale")
    return np.asarray((analysis.landmarks - midpoint) / scale, dtype=np.float32)


def _geometry_quality(
    source: PortraitAnalysis,
    generated_rgb: NDArray[np.float32],
    analyzer: PortraitAnalyzer,
) -> tuple[bool, dict[str, object]]:
    try:
        generated = analyze_portrait(
            generated_rgb,
            analyzer,
            _preflight_thresholds(generated_rgb),
        )
        source_landmarks = _normalized_landmarks(source)
        generated_landmarks = _normalized_landmarks(generated)
        if source_landmarks.shape != generated_landmarks.shape:
            raise ValueError("landmark shape changed")
        point_drift = np.linalg.norm(source_landmarks - generated_landmarks, axis=1)
        landmark_drift = float(np.mean(point_drift))
        landmark_drift_p95 = float(np.percentile(point_drift, 95))
        pose_components = {
            "yaw": abs(float(generated.pose.yaw - source.pose.yaw)),
            "pitch": abs(float(generated.pose.pitch - source.pose.pitch)),
            "roll": abs(float(generated.pose.roll - source.pose.roll)),
        }
        pose_drift = max(pose_components.values())
        passed = landmark_drift <= 0.08 and pose_drift <= 5.0
        return passed, {
            "passed": passed,
            "landmark_drift_mean": landmark_drift,
            "landmark_drift_p95": landmark_drift_p95,
            "pose_drift_degrees": pose_drift,
            "pose_components_degrees": pose_components,
        }
    except (PortraitTransferError, ValueError) as exc:
        return False, {
            "passed": False,
            "reason": "generated_face_analysis_failed",
            "failure_code": getattr(getattr(exc, "code", None), "value", None),
        }


def _composite_ai_background(
    generated_rgb: NDArray[np.float32],
    source_rgb: NDArray[np.float32],
    source_analysis: PortraitAnalysis,
    settings: dict[str, object],
) -> NDArray[np.float32]:
    mode = str(settings["background_mode"])
    if mode == "REFERENCE":
        return np.asarray(generated_rgb, dtype=np.float32).copy()
    if mode == "KEEP":
        background = source_rgb
    elif mode == "BLUR":
        from PIL import Image, ImageFilter

        source_u8 = np.rint(np.clip(source_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = Image.fromarray(source_u8, mode="RGB").filter(
            ImageFilter.GaussianBlur(radius=12.0)
        )
        background = np.asarray(blurred, dtype=np.float32) / 255.0
    elif mode == "SOLID":
        color = _hex_color(cast(str | None, settings["background_color"]))
        if color is None:
            raise InputValidationError("A solid background color is required")
        background = np.broadcast_to(np.asarray(color, dtype=np.float32), source_rgb.shape)
    else:  # _ai_settings validates this before any model work starts.
        raise InputValidationError("Unsupported background mode")
    matte = np.clip(source_analysis.masks.foreground_alpha, 0.0, 1.0)[..., None]
    return np.asarray(
        np.clip(generated_rgb * matte + background * (1.0 - matte), 0.0, 1.0),
        dtype=np.float32,
    )


def _select_reference(
    infrastructure: WorkerInfrastructure,
    job_id: uuid.UUID,
    query_feature: StyleFeature | None,
) -> Asset:
    with infrastructure.session_factory.begin() as db:
        job = JobRepository(db).get_for_worker(job_id, for_update=True)
        if job is None:
            raise ProcessingCancelled()
        if job.reference_asset is not None:
            return job.reference_asset
        if job.style is None:
            raise StyleSelectionFailure("No reference or style is available")
        examples = [
            example
            for example in job.style.examples
            if example.asset.deleted_at is None and not is_expired(example.asset.expires_at)
        ]
        if not examples:
            raise StyleSelectionFailure("The selected style has no available examples")
        if query_feature is None:
            raise StyleSelectionFailure("The input portrait was not analyzed for style selection")
        ranked = StyleRankingService(infrastructure.storage).rank_compatible(
            query_feature, examples, limit=3
        )
        if not ranked:
            raise StyleSelectionFailure("No compatible style example was found")
        selected = next(example for example in examples if example.id == ranked[0].example_id)
        job.selected_style_example_id = selected.id
        diagnostics = dict(job.diagnostics or {})
        diagnostics["selection"] = {
            "example_id": str(selected.id),
            "score": ranked[0].score,
            "metric": "weighted_compatibility_v1",
            "components": {
                "local_energy_ncc": ranked[0].energy_ncc,
                "pose_similarity": ranked[0].pose_similarity,
                "landmark_shape_similarity": ranked[0].landmark_shape_similarity,
                "photometric_compatibility": ranked[0].photometric_compatibility,
                "mask_quality": ranked[0].mask_quality,
            },
            "candidates": [
                {
                    "example_id": str(candidate.example_id),
                    "score": candidate.score,
                    "components": {
                        "local_energy_ncc": candidate.energy_ncc,
                        "pose_similarity": candidate.pose_similarity,
                        "landmark_shape_similarity": candidate.landmark_shape_similarity,
                        "photometric_compatibility": candidate.photometric_compatibility,
                        "mask_quality": candidate.mask_quality,
                    },
                }
                for candidate in ranked
            ],
        }
        job.diagnostics = diagnostics
        db.flush()
        return selected.asset


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, AIEngineError):
        return exc.retryable
    if isinstance(exc, (BotoCoreError, OperationalError, RedisConnectionError, RedisTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        status_code = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        return status_code >= 500 or status_code in {408, 429}
    return False


def _mark_cancelled(infrastructure: WorkerInfrastructure, job_id: uuid.UUID) -> None:
    with infrastructure.session_factory.begin() as db:
        repository = JobRepository(db)
        job = repository.get_for_worker(job_id, for_update=True)
        if job is None:
            return
        repository.mark_cancelled(job)
        event = {
            "job_id": str(job.id),
            "status": job.status.value,
            "stage": job.stage.value,
            "progress": job.progress,
            "message": "Job cancelled",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    infrastructure.set_progress(str(job_id), event)
    JOBS.labels("cancelled").inc()


def _mark_failed(
    infrastructure: WorkerInfrastructure,
    job_id: uuid.UUID,
    *,
    code: str,
    safe_message: str,
) -> None:
    with infrastructure.session_factory.begin() as db:
        repository = JobRepository(db)
        job = repository.get_for_worker(job_id, for_update=True)
        if job is None:
            return
        repository.mark_failed(job, code=code, safe_message=safe_message)
        event = {
            "job_id": str(job.id),
            "status": job.status.value,
            "stage": job.stage.value,
            "progress": job.progress,
            "message": safe_message,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    infrastructure.set_progress(str(job_id), event)
    JOBS.labels("failed").inc()


@_celery_task(
    bind=True,
    name="portrait_worker.process_transfer_job",
    max_retries=3,
    acks_late=True,
)
def process_transfer_job(self: Task, job_id: str) -> None:
    infrastructure = get_infrastructure()
    parsed_job_id = uuid.UUID(job_id)
    try:
        lock_token = infrastructure.acquire_job_lock(job_id)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2, max_retries=3) from exc
    if lock_token is None:
        return

    worker_id = str(getattr(self.request, "hostname", "worker"))
    reporter = JobProgressReporter(infrastructure, parsed_job_id, worker_id, lock_token)
    uploaded_keys: list[str] = []
    committed = False
    running_gauge = False
    portrait_analyzer: PortraitAnalyzer | None = None
    try:
        with infrastructure.session_factory.begin() as db:
            repository = JobRepository(db)
            job = repository.get_for_worker(parsed_job_id, for_update=True)
            if job is None or job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.EXPIRED,
            }:
                return
            if job.status == JobStatus.CANCEL_REQUESTED:
                raise ProcessingCancelled()
            if job.status != JobStatus.QUEUED:
                return
            repository.mark_running(job, worker_id=worker_id)
            input_asset_id = job.input_asset_id
            job_settings = dict(job.settings)
            persisted_corrections = list(job.corrections or [])
            existing_diagnostics = dict(job.diagnostics or {})
            attempt = job.attempt
            uses_style = job.style_id is not None
        JOBS_RUNNING.inc()
        running_gauge = True
        reporter.emit(ProcessingStage.VALIDATING, 1, "Validating job inputs")

        with TemporaryDirectory(prefix=f"portrait-job-{job_id}-"):
            with infrastructure.session_factory() as db:
                input_asset = db.get(Asset, input_asset_id)
                if input_asset is None or input_asset.deleted_at is not None:
                    raise InputValidationError("The input asset is unavailable")
            if reporter.cancel_requested():
                raise ProcessingCancelled()
            reporter.emit(ProcessingStage.DECODING, 3, "Decoding input portrait")
            input_bytes = infrastructure.storage.get_bytes(input_asset.object_key)
            upload_limits = _decode_limits(
                infrastructure.settings,
                max_encoded_bytes=int(infrastructure.settings.max_upload_bytes),
            )
            input_decoded = decode_image(input_bytes, upload_limits)

            ai_settings = _ai_settings(job_settings, persisted_corrections)
            portrait_analyzer = build_portrait_analyzer(infrastructure.settings)
            reporter.emit(ProcessingStage.FACE_LANDMARKS, 5, "Analyzing source portrait")
            input_analysis = analyze_portrait(
                input_decoded.rgb,
                portrait_analyzer,
                _preflight_thresholds(input_decoded.rgb),
            )
            query_feature: StyleFeature | None = None
            if uses_style:
                reporter.emit(ProcessingStage.REFERENCE_SELECTION, 8, "Selecting reference example")
                query_feature = build_style_feature(
                    "input",
                    input_decoded.rgb,
                    input_analysis.masks.head,
                    pose=input_analysis.pose,
                    landmarks=input_analysis.landmarks,
                    mask_quality=input_analysis.quality.mask_confidence,
                )
            reference_asset = _select_reference(infrastructure, parsed_job_id, query_feature)
            reference_bytes = infrastructure.storage.get_bytes(reference_asset.object_key)
            reference_decoded = decode_image(reference_bytes, upload_limits)
            reporter.emit(ProcessingStage.QUALITY_ANALYSIS, 10, "Validating style portrait")
            analyze_portrait(
                reference_decoded.rgb,
                portrait_analyzer,
                _preflight_thresholds(reference_decoded.rgb),
            )
            if reporter.cancel_requested():
                raise ProcessingCancelled()

            reporter.emit(
                ProcessingStage.MULTISCALE_TRANSFER,
                15,
                "Generating portrait with the AI style engine",
            )
            client = get_ai_engine_client(infrastructure.settings)
            request_settings = _engine_request_settings(ai_settings)
            engine_response: AIEngineResponse | None = None
            generated_rgb: NDArray[np.float32] | None = None
            quality_attempts: list[dict[str, object]] = []
            for quality_attempt in range(2):
                response = client.transfer(
                    # Preserve the bounded upload representation. Re-encoding a valid
                    # Re-encoding a valid compressed upload as PNG can expand it
                    # beyond the internal request cap.
                    content=input_bytes,
                    style=reference_bytes,
                    settings=request_settings,
                )
                if reporter.cancel_requested():
                    raise ProcessingCancelled()
                output_decoded = decode_image(
                    response.image_png,
                    _decode_limits(
                        infrastructure.settings,
                        max_encoded_bytes=_AI_RESPONSE_MAX_ENCODED_BYTES,
                    ),
                )
                if output_decoded.rgb.shape != input_decoded.rgb.shape:
                    raise AIEngineError(
                        "AI_ENGINE_INVALID_RESPONSE",
                        "The AI engine changed the portrait dimensions.",
                        retryable=False,
                    )
                reporter.emit(
                    ProcessingStage.POSTPROCESSING,
                    75 if quality_attempt == 0 else 92,
                    "Validating AI output geometry",
                )
                passed, quality = _geometry_quality(
                    input_analysis,
                    output_decoded.rgb,
                    portrait_analyzer,
                )
                quality_attempts.append(
                    {
                        "attempt": quality_attempt + 1,
                        "style_strength": _bounded_diagnostic_strength(
                            response.diagnostics.get("style_strength_applied"),
                            request_settings["style_strength"],
                        ),
                        "structure_strength": _bounded_diagnostic_strength(
                            response.diagnostics.get("structure_strength_applied"),
                            request_settings["structure_strength"],
                        ),
                        **quality,
                    }
                )
                if passed:
                    engine_response = response
                    generated_rgb = output_decoded.rgb
                    break
                if quality_attempt == 0:
                    request_settings = {
                        **request_settings,
                        "style_strength": min(
                            float(cast(Any, request_settings["style_strength"])), 0.6
                        ),
                        "structure_strength": max(
                            float(cast(Any, request_settings["structure_strength"])), 0.95
                        ),
                    }
                    reporter.emit(
                        ProcessingStage.MULTISCALE_TRANSFER,
                        80,
                        "Retrying with stronger identity preservation",
                    )
            if engine_response is None or generated_rgb is None:
                raise AIEngineError(
                    "AI_QUALITY_GUARD_FAILED",
                    "The generated portrait did not preserve the source face geometry.",
                    retryable=False,
                )

            reporter.emit(ProcessingStage.BACKGROUND, 96, "Compositing background")
            output_rgb = _composite_ai_background(
                generated_rgb,
                input_decoded.rgb,
                input_analysis,
                ai_settings,
            )
            result_diagnostics = dict(engine_response.diagnostics)
            result_diagnostics["worker_quality_guard"] = {
                "landmark_drift_limit": 0.08,
                "pose_drift_limit_degrees": 5.0,
                "retry_performed": len(quality_attempts) > 1,
                "attempts": quality_attempts,
            }

            reporter.emit(ProcessingStage.UPLOADING_OUTPUT, 99, "Uploading output")
            output_format = str(job_settings.get("output_format", "PNG"))
            if output_format == "JPEG":
                output_data = encode_jpeg(
                    output_rgb,
                    quality=int(job_settings.get("jpeg_quality", 95)),
                )
                output_mime, extension = "image/jpeg", "jpg"
            else:
                output_data = encode_png(output_rgb)
                output_mime, extension = "image/png", "png"
            output_key = f"outputs/{job_id}/attempt-{attempt}.{extension}"
            infrastructure.storage.put_bytes(output_key, output_data, output_mime)
            uploaded_keys.append(output_key)

            existing_manifest = existing_diagnostics.get("private_cache_manifest", {})
            old_cache_keys = (
                [
                    str(metadata["key"])
                    for metadata in existing_manifest.values()
                    if isinstance(metadata, dict) and isinstance(metadata.get("key"), str)
                ]
                if isinstance(existing_manifest, dict)
                else []
            )
            cache_manifest: dict[str, dict[str, Any]] = {}
            cache_bytes = 0
            diagnostics_data = _jsonable(result_diagnostics)
            summary = {
                "engine": engine_response.engine_id,
                "warnings": diagnostics_data.get("warnings", []),
                "requested_style_strength": ai_settings["style_strength"],
                "requested_structure_strength": ai_settings["structure_strength"],
                "effective_style_strength": _bounded_diagnostic_strength(
                    diagnostics_data.get("style_strength_applied"),
                    request_settings["style_strength"],
                ),
                "effective_structure_strength": _bounded_diagnostic_strength(
                    diagnostics_data.get("structure_strength_applied"),
                    request_settings["structure_strength"],
                ),
                "requested_inference_steps": ai_settings["inference_steps"],
                "random_seed": ai_settings["random_seed"],
                "quality_retry_performed": len(quality_attempts) > 1,
                "analysis_backend": type(portrait_analyzer).__name__,
            }
            with infrastructure.session_factory.begin() as db:
                repository = JobRepository(db)
                job = repository.get_for_worker(parsed_job_id, for_update=True)
                if job is None or job.status == JobStatus.CANCEL_REQUESTED:
                    raise ProcessingCancelled()
                output_asset = AssetRepository(db).create(
                    session_id=cast(uuid.UUID, job.session_id),
                    kind=AssetKind.OUTPUT,
                    object_key=output_key,
                    mime_type=output_mime,
                    width=int(output_rgb.shape[1]),
                    height=int(output_rgb.shape[0]),
                    byte_size=len(output_data),
                    sha256=hashlib.sha256(output_data).hexdigest(),
                    metadata={"job_id": str(job.id), "metadata_stripped": True},
                    expires_at=job.expires_at,
                )
                repository.add_artifact(job.id, output_asset.id, ArtifactKind.OUTPUT)
                final_diagnostics = dict(job.diagnostics or {})
                final_diagnostics.update(
                    {
                        "summary": summary,
                        "transfer": diagnostics_data,
                        "private_cache_manifest": cache_manifest,
                        "cache_bytes": cache_bytes,
                    }
                )
                repository.mark_succeeded(job, final_diagnostics)
                terminal_event = {
                    "job_id": str(job.id),
                    "status": JobStatus.SUCCEEDED.value,
                    "stage": ProcessingStage.COMPLETED.value,
                    "progress": 100,
                    "message": "Processing complete",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            committed = True
            new_cache_keys = {
                str(metadata["key"])
                for metadata in cache_manifest.values()
                if isinstance(metadata.get("key"), str)
            }
            for old_key in old_cache_keys:
                if old_key in new_cache_keys:
                    continue
                try:
                    infrastructure.storage.delete(old_key)
                except Exception:
                    STORAGE_ERRORS.labels("superseded_cache_delete").inc()
            for prior_attempt in range(1, attempt):
                try:
                    infrastructure.storage.delete_prefix(
                        f"jobs/{job_id}/cache/attempt-{prior_attempt}/"
                    )
                except Exception:
                    STORAGE_ERRORS.labels("superseded_cache_prefix_delete").inc()
            infrastructure.set_progress(job_id, terminal_event)
            JOBS.labels("succeeded").inc()
    except ProcessingCancelled:
        _mark_cancelled(infrastructure, parsed_job_id)
    except JobLeaseLost:
        # The current worker must not mutate state now owned by another lease holder.
        return
    except AIEngineError as exc:
        if exc.retryable and self.request.retries < self.max_retries:
            with infrastructure.session_factory.begin() as db:
                repository = JobRepository(db)
                job = repository.get_for_worker(parsed_job_id, for_update=True)
                if job is not None:
                    repository.mark_retry_queued(
                        job,
                        safe_message=(
                            "The AI engine is temporarily unavailable; the job will retry."
                        ),
                    )
            try:
                raise self.retry(
                    exc=exc,
                    countdown=min(60, 2 ** (self.request.retries + 1)),
                    max_retries=3,
                ) from exc
            except MaxRetriesExceededError:
                _mark_failed(
                    infrastructure,
                    parsed_job_id,
                    code=exc.code,
                    safe_message=exc.safe_message,
                )
                return
        _mark_failed(
            infrastructure,
            parsed_job_id,
            code=exc.code,
            safe_message=exc.safe_message,
        )
    except PortraitTransferError as exc:
        code = getattr(exc.code, "value", str(exc.code))
        _mark_failed(infrastructure, parsed_job_id, code=code, safe_message=str(exc))
    except StyleSelectionFailure as exc:
        _mark_failed(
            infrastructure,
            parsed_job_id,
            code="REFERENCE_SELECTION_FAILED",
            safe_message=str(exc),
        )
    except Exception as exc:
        if _is_transient(exc) and self.request.retries < self.max_retries:
            with infrastructure.session_factory.begin() as db:
                repository = JobRepository(db)
                job = repository.get_for_worker(parsed_job_id, for_update=True)
                if job is not None:
                    repository.mark_retry_queued(
                        job,
                        safe_message=(
                            "A temporary infrastructure error occurred; the job will retry."
                        ),
                    )
            try:
                raise self.retry(
                    exc=exc,
                    countdown=min(60, 2 ** (self.request.retries + 1)),
                    max_retries=3,
                ) from exc
            except MaxRetriesExceededError:
                _mark_failed(
                    infrastructure,
                    parsed_job_id,
                    code="PROCESSING_FAILED",
                    safe_message="The portrait could not be processed.",
                )
                return
        _mark_failed(
            infrastructure,
            parsed_job_id,
            code="PROCESSING_FAILED",
            safe_message="The portrait could not be processed.",
        )
        raise
    finally:
        if running_gauge:
            JOBS_RUNNING.dec()
        if not committed:
            for key in reversed(uploaded_keys):
                try:
                    infrastructure.storage.delete(key)
                except Exception:
                    STORAGE_ERRORS.labels("failed_job_cleanup").inc()
        with suppress(Exception):
            infrastructure.release_job_lock(job_id, lock_token)
        close_analyzer = getattr(portrait_analyzer, "close", None)
        if callable(close_analyzer):
            with suppress(Exception):
                close_analyzer()


def _npy_bytes(array: NDArray[Any]) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return output.getvalue()


def _eye_asset_bytes(asset: EyeHighlightAsset) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        foreground_rgb=np.asarray(asset.foreground_rgb, dtype=np.float32),
        alpha=np.asarray(asset.alpha, dtype=np.float32),
        center=np.asarray(asset.center, dtype=np.float32),
        iris_radius=np.asarray(asset.iris_radius, dtype=np.float32),
        angle_radians=np.asarray(asset.angle_radians, dtype=np.float32),
        confidence=np.asarray(asset.confidence, dtype=np.float32),
        version=np.asarray(asset.version),
    )
    return output.getvalue()


def _persist_ingested_style(
    infrastructure: WorkerInfrastructure,
    style_id: uuid.UUID,
    example: StyleExample,
    ingested: IngestedStyle,
    *,
    analysis_backend: str,
) -> tuple[str, dict[str, Any]]:
    prefix = f"styles/{style_id}/examples/{example.id}"
    infrastructure.storage.delete_prefix(f"{prefix}/")
    derived_assets = ["head_mask", "landmarks", "background"]
    try:
        infrastructure.storage.put_bytes(
            f"{prefix}/head-mask.npy",
            _npy_bytes(ingested.analysis.masks.head),
            "application/octet-stream",
        )
        infrastructure.storage.put_bytes(
            f"{prefix}/landmarks.npy",
            _npy_bytes(ingested.analysis.landmarks),
            "application/octet-stream",
        )
        infrastructure.storage.put_bytes(
            f"{prefix}/background.png", encode_png(ingested.background), "image/png"
        )
        for side, eye_asset in zip(("left", "right"), ingested.eye_assets, strict=True):
            if eye_asset is None:
                continue
            infrastructure.storage.put_bytes(
                f"{prefix}/eye-{side}.npz",
                _eye_asset_bytes(eye_asset),
                "application/octet-stream",
            )
            derived_assets.append(f"eye_{side}")
        feature_key = StyleRankingService(infrastructure.storage).index_vector(
            style_id,
            example,
            np.asarray(ingested.feature.vector, dtype=np.float32),
        )
    except Exception:
        with suppress(Exception):
            infrastructure.storage.delete_prefix(f"{prefix}/")
        raise

    analysis_quality = _jsonable(ingested.analysis.quality)
    quality = dict(analysis_quality) if isinstance(analysis_quality, dict) else {}
    quality.update(
        {
            "indexed": True,
            "indexed_at": datetime.now(UTC).isoformat(),
            "full_ingestion": "COMPLETED",
            "analysis_backend": analysis_backend,
            "pose": _jsonable(ingested.analysis.pose),
            "photometric_lab": (
                np.asarray(ingested.feature.photometric_lab, dtype=np.float32).tolist()
                if ingested.feature.photometric_lab is not None
                else None
            ),
            "face_box": _jsonable(ingested.analysis.face_box),
            "landmark_count": int(ingested.analysis.landmarks.shape[0]),
            "monochrome": bool(float(np.mean(np.ptp(ingested.rgb, axis=2))) < 0.025),
            "derived_assets": derived_assets,
            "warnings": list(
                dict.fromkeys([*quality.get("warnings", []), *ingested.analysis.warnings])
            ),
        }
    )
    return feature_key, quality


@_celery_task(bind=True, name="portrait_worker.index_style", max_retries=3)
def index_style(self: Task, style_id: str) -> None:
    infrastructure = get_infrastructure()
    parsed_style_id = uuid.UUID(style_id)
    try:
        with infrastructure.session_factory.begin() as db:
            style = db.get(Style, parsed_style_id)
            if style is None or style.deleted_at is not None:
                return
            examples = list(style.examples)
            analyzer = build_portrait_analyzer(infrastructure.settings)
            try:
                for example in examples:
                    if example.asset.deleted_at is not None or is_expired(example.asset.expires_at):
                        continue
                    try:
                        decoded = decode_image(
                            infrastructure.storage.get_bytes(example.asset.object_key),
                            _decode_limits(
                                infrastructure.settings,
                                max_encoded_bytes=int(infrastructure.settings.max_upload_bytes),
                            ),
                        )
                        ingested = ingest_style(str(example.id), decoded.rgb, analyzer)
                        old_key = example.feature_object_key
                        key, quality = _persist_ingested_style(
                            infrastructure,
                            style.id,
                            example,
                            ingested,
                            analysis_backend=type(analyzer).__name__,
                        )
                        example.feature_object_key = key
                        example.quality = quality
                        if old_key and old_key != key:
                            infrastructure.storage.delete(old_key)
                    except PortraitTransferError as exc:
                        code = getattr(exc.code, "value", str(exc.code))
                        example.quality = {
                            **(example.quality or {}),
                            "full_ingestion": "FAILED",
                            "error_code": code,
                            "warnings": [str(exc)],
                        }
                style.updated_at = datetime.now(UTC)
            finally:
                close_analyzer = getattr(analyzer, "close", None)
                if callable(close_analyzer):
                    with suppress(Exception):
                        close_analyzer()
    except Exception as exc:
        if _is_transient(exc):
            raise self.retry(exc=exc, countdown=5, max_retries=3) from exc
        raise


@_celery_task(bind=True, name="portrait_worker.purge_expired", max_retries=3)
def purge_expired(self: Task) -> None:
    infrastructure = get_infrastructure()
    try:
        purge_expired_records(infrastructure)
    except Exception as exc:
        if _is_transient(exc):
            raise self.retry(exc=exc, countdown=10, max_retries=3) from exc
        raise
