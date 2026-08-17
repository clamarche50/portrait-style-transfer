from __future__ import annotations

import hashlib
import hmac
import io
import re
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, is_dataclass, replace
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
from portrait_transfer import (
    AlgorithmProfile as CoreAlgorithmProfile,
)
from portrait_transfer import (
    BackgroundMode as CoreBackgroundMode,
)
from portrait_transfer import (
    TransferSettings,
    create_default_runtime,
    transfer_portrait_style,
)
from portrait_transfer.exceptions import (
    InputValidationError,
    PortraitTransferError,
    ProcessingCancelled,
)
from portrait_transfer.image_io import decode_image, encode_jpeg, encode_png
from portrait_transfer.preflight import PortraitAnalyzer, analyze_portrait
from portrait_transfer.selection import build_style_feature
from portrait_transfer.style_ingestion import IngestedStyle, ingest_style
from portrait_transfer.types import EyeHighlightAsset, StyleFeature
from portrait_worker.celery_app import celery_app
from portrait_worker.cleanup import purge_expired_records
from portrait_worker.gpu_dense import build_dense_backend
from portrait_worker.infrastructure import WorkerInfrastructure, get_infrastructure
from portrait_worker.mediapipe_adapter import build_portrait_analyzer
from portrait_worker.progress import JobLeaseLost, JobProgressReporter
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy.exc import OperationalError


class StyleSelectionFailure(RuntimeError):
    """Raised when a style cannot provide a usable reference portrait."""


TaskCallable = TypeVar("TaskCallable", bound=Callable[..., object])


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


def _hex_color(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Invalid solid background color")
    red, green, blue = (int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    return red, green, blue


def _translate_corrections(corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for correction in corrections:
        correction_type = correction.get("type")
        if correction_type == "mask":
            value = 1.0 if correction["operation"] == "ADD" else 0.0
            for target in ("head", "foreground_alpha"):
                operations.append(
                    {
                        "type": "mask_stroke",
                        "target": target,
                        "points": correction["points"],
                        "radius": correction["radius"],
                        "value": value,
                        "coordinate_space": "normalized",
                    }
                )
        elif correction_type == "alignment":
            operations.append(
                {
                    "type": "alignment_points",
                    "input_points": correction["input_points"],
                    "reference_points": correction["reference_points"],
                    "coordinate_space": "normalized",
                }
            )
        elif correction_type == "gain_copy":
            for channel in range(3):
                for level in correction.get("levels", range(6)):
                    operations.append(
                        {
                            "type": "gain_constraint",
                            "channel": channel,
                            "level": level,
                            "mode": "COPY_FROM_REGION",
                            "polygon": correction["target_polygon"],
                            "source_polygon": correction["source_polygon"],
                            "coordinate_space": "normalized",
                        }
                    )
        elif correction_type == "eye":
            operation: dict[str, Any] = {
                "type": "eye_center",
                "eye": str(correction["eye"]).lower(),
                "center": correction["pupil_center"],
                "coordinate_space": "normalized",
            }
            if correction.get("iris_radius") is not None:
                operation["radius"] = correction["iris_radius"]
            if correction.get("highlight_scale") is not None:
                operation["highlight_scale"] = correction["highlight_scale"]
            if correction.get("highlight_rotation_degrees") is not None:
                operation["highlight_rotation_degrees"] = correction["highlight_rotation_degrees"]
            operations.append(operation)
    return operations


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


def _core_settings(
    job_settings: dict[str, Any], corrections: list[dict[str, Any]]
) -> TransferSettings:
    background_mode, background_color = _latest_background_correction(job_settings, corrections)
    return TransferSettings(
        algorithm_profile=CoreAlgorithmProfile(
            str(job_settings.get("algorithm_profile", "source_2014_compat"))
        ),
        transfer_strength=float(job_settings.get("transfer_strength", 1.0)),
        residual_strength=float(job_settings.get("residual_strength", 1.0)),
        global_range_mix=float(job_settings.get("global_range_mix", 0.25)),
        eye_highlights=bool(job_settings.get("eye_highlights", True)),
        background_mode=CoreBackgroundMode(background_mode),
        background_color=_hex_color(background_color),
        dense_alignment=bool(job_settings.get("dense_alignment", True)),
        processing_long_edge=int(job_settings.get("processing_long_edge", 1280)),
        debug_artifacts=bool(job_settings.get("debug_artifacts", False)),
        random_seed=int(job_settings.get("random_seed", 0)),
    )


def _artifact_stage(name: str) -> ProcessingStage:
    lowered = name.lower()
    if lowered.startswith("resume."):
        if lowered == "resume.post_eye_rgb" or lowered.endswith(".background"):
            return ProcessingStage.EYE_HIGHLIGHTS
        if lowered == "resume.pre_eye_rgb" or lowered.endswith(".eyes"):
            return ProcessingStage.MULTISCALE_TRANSFER
        return ProcessingStage.DENSE_ALIGNMENT
    if "mask" in lowered:
        return ProcessingStage.SEGMENTATION
    if any(token in lowered for token in ("map", "flow", "align")):
        return ProcessingStage.DENSE_ALIGNMENT
    if any(token in lowered for token in ("band", "energy", "gain", "residual")):
        return ProcessingStage.MULTISCALE_TRANSFER
    if "eye" in lowered or "iris" in lowered:
        return ProcessingStage.EYE_HIGHLIGHTS
    if "background" in lowered:
        return ProcessingStage.BACKGROUND
    return ProcessingStage.POSTPROCESSING


def _requested_resume_stage(diagnostics: dict[str, Any]) -> str | None:
    resume = diagnostics.get("resume", {})
    if not isinstance(resume, dict):
        return None
    requested = str(resume.get("requested_stage", "")).upper()
    return {
        ProcessingStage.MULTISCALE_TRANSFER.value: "multiscale",
        ProcessingStage.EYE_HIGHLIGHTS.value: "eyes",
        ProcessingStage.BACKGROUND.value: "background",
    }.get(requested)


def _artifact_kind(name: str) -> ArtifactKind:
    lowered = name.lower()
    if "input" in lowered and "mask" in lowered:
        return ArtifactKind.INPUT_MASK
    if "reference" in lowered and "mask" in lowered:
        return ArtifactKind.REFERENCE_MASK
    if "gain" in lowered:
        return ArtifactKind.GAIN
    if "energy" in lowered:
        return ArtifactKind.ENERGY
    if "dense" in lowered or "flow" in lowered:
        return ArtifactKind.DENSE_PREVIEW
    if "affine" in lowered:
        return ArtifactKind.AFFINE_PREVIEW
    return ArtifactKind.OTHER


def _thumbnail(array: NDArray[Any]) -> tuple[bytes, int, int]:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 3 and value.shape[2] == 2:
        value = np.linalg.norm(value, axis=2)
    if value.ndim == 2:
        finite = value[np.isfinite(value)]
        low = float(np.percentile(finite, 1)) if finite.size else 0.0
        high = float(np.percentile(finite, 99)) if finite.size else 1.0
        normalized = np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)
        value = np.repeat(normalized[..., None], 3, axis=2)
    elif value.ndim == 3 and value.shape[2] >= 3:
        value = value[..., :3]
        finite = value[np.isfinite(value)]
        if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
            low, high = np.percentile(finite, (1, 99))
            value = (value - float(low)) / max(float(high - low), 1e-6)
        value = np.clip(value, 0.0, 1.0)
    else:
        value = np.zeros((32, 32, 3), dtype=np.float32)
    from PIL import Image

    image = Image.fromarray(np.rint(value * 255).astype(np.uint8), mode="RGB")
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, "PNG", optimize=False, compress_level=6)
    return output.getvalue(), image.width, image.height


def _load_resume_artifacts(
    infrastructure: WorkerInfrastructure, diagnostics: dict[str, Any]
) -> dict[str, NDArray[np.float32]]:
    manifest = diagnostics.get("private_cache_manifest", {})
    loaded: dict[str, NDArray[np.float32]] = {}
    if not isinstance(manifest, dict):
        return loaded
    loaded_bytes = 0
    for name, metadata in manifest.items():
        if (
            not isinstance(name, str)
            or not name.startswith("resume.")
            or not isinstance(metadata, dict)
            or "key" not in metadata
        ):
            continue
        try:
            payload = infrastructure.storage.get_bytes(str(metadata["key"]))
            loaded_bytes += len(payload)
            if loaded_bytes > infrastructure.settings.max_job_cache_bytes:
                break
            expected_sha256 = metadata.get("sha256")
            if (
                metadata.get("schema") != "ndarray-npy-v1"
                or not isinstance(expected_sha256, str)
                or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_sha256)
            ):
                continue
            value = np.asarray(np.load(io.BytesIO(payload), allow_pickle=False), dtype=np.float32)
            expected_shape = metadata.get("shape")
            if isinstance(expected_shape, list) and list(value.shape) != expected_shape:
                continue
            loaded[name] = value
        except Exception:
            continue
    return loaded


def _load_eye_asset(
    infrastructure: WorkerInfrastructure,
    key: str,
) -> EyeHighlightAsset | None:
    try:
        with np.load(
            io.BytesIO(infrastructure.storage.get_bytes(key)), allow_pickle=False
        ) as archive:
            return EyeHighlightAsset(
                foreground_rgb=np.asarray(archive["foreground_rgb"], dtype=np.float32),
                alpha=np.asarray(archive["alpha"], dtype=np.float32),
                center=(float(archive["center"][0]), float(archive["center"][1])),
                iris_radius=float(archive["iris_radius"]),
                angle_radians=float(archive["angle_radians"]),
                confidence=float(archive["confidence"]),
                version=str(archive["version"]),
            )
    except (KeyError, OSError, ValueError):
        return None


def _load_style_eye_assets(
    infrastructure: WorkerInfrastructure,
    style_id: uuid.UUID,
    example_id: uuid.UUID,
) -> tuple[EyeHighlightAsset | None, EyeHighlightAsset | None] | None:
    prefix = f"styles/{style_id}/examples/{example_id}"
    assets = (
        _load_eye_asset(infrastructure, f"{prefix}/eye-left.npz"),
        _load_eye_asset(infrastructure, f"{prefix}/eye-right.npz"),
    )
    return assets if any(asset is not None for asset in assets) else None


def _select_reference(
    infrastructure: WorkerInfrastructure,
    job_id: uuid.UUID,
    query_feature: StyleFeature | None,
) -> tuple[Asset, tuple[EyeHighlightAsset | None, EyeHighlightAsset | None] | None]:
    with infrastructure.session_factory.begin() as db:
        job = JobRepository(db).get_for_worker(job_id, for_update=True)
        if job is None:
            raise ProcessingCancelled()
        if job.reference_asset is not None:
            return job.reference_asset, None
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
        eye_assets = _load_style_eye_assets(
            infrastructure,
            selected.style_id,
            selected.id,
        )
        return selected.asset, eye_assets


def _is_transient(exc: BaseException) -> bool:
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
            input_decoded = decode_image(infrastructure.storage.get_bytes(input_asset.object_key))

            settings = _core_settings(job_settings, persisted_corrections)
            portrait_analyzer = build_portrait_analyzer(infrastructure.settings)
            query_feature: StyleFeature | None = None
            if uses_style:
                reporter.emit(ProcessingStage.REFERENCE_SELECTION, 5, "Selecting reference example")
                input_analysis = analyze_portrait(
                    input_decoded.rgb,
                    portrait_analyzer,
                    settings.preflight,
                    settings.processing_long_edge,
                )
                query_feature = build_style_feature(
                    "input",
                    input_decoded.rgb,
                    input_analysis.masks.head,
                    pose=input_analysis.pose,
                    landmarks=input_analysis.landmarks,
                    mask_quality=input_analysis.quality.mask_confidence,
                )
            reference_asset, indexed_eye_assets = _select_reference(
                infrastructure, parsed_job_id, query_feature
            )
            reference_decoded = decode_image(
                infrastructure.storage.get_bytes(reference_asset.object_key)
            )
            if reporter.cancel_requested():
                raise ProcessingCancelled()

            base_runtime = create_default_runtime(enable_cpu_dense=settings.dense_alignment)
            runtime_corrections: dict[str, Any] = {
                "operations": _translate_corrections(persisted_corrections)
            }
            requested_resume_stage = _requested_resume_stage(existing_diagnostics)
            if requested_resume_stage is not None:
                runtime_corrections["resume_from_stage"] = requested_resume_stage
            runtime = replace(
                base_runtime,
                analyzer=portrait_analyzer,
                dense_backend=build_dense_backend(
                    infrastructure.settings, enabled=settings.dense_alignment
                ),
                progress_callback=reporter.package_callback,
                cancel_check=reporter.cancel_requested,
                eye_assets=indexed_eye_assets,
                corrections=runtime_corrections,
                resume_artifacts=_load_resume_artifacts(infrastructure, existing_diagnostics),
            )
            result = transfer_portrait_style(
                input_decoded.rgb,
                reference_decoded.rgb,
                settings,
                runtime,
            )
            if reporter.cancel_requested():
                raise ProcessingCancelled()

            reporter.emit(ProcessingStage.UPLOADING_OUTPUT, 99, "Uploading output")
            output_format = str(job_settings.get("output_format", "PNG"))
            if output_format == "JPEG":
                output_data = encode_jpeg(
                    result.output_rgb,
                    quality=int(job_settings.get("jpeg_quality", 95)),
                )
                output_mime, extension = "image/jpeg", "jpg"
            else:
                output_data = encode_png(result.output_rgb)
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
            debug_records: list[tuple[uuid.UUID, str, ArtifactKind, bytes, int, int]] = []
            cache_bytes = 0
            resume_artifacts = getattr(result, "resume_artifacts", {})
            if isinstance(resume_artifacts, dict):
                for name, array in sorted(resume_artifacts.items()):
                    value = np.asarray(array, dtype=np.float32)
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    cache_data = buffer.getvalue()
                    if cache_bytes + len(cache_data) > infrastructure.settings.max_job_cache_bytes:
                        break
                    stage = _artifact_stage(str(name))
                    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))[:120]
                    cache_key = (
                        f"jobs/{job_id}/cache/attempt-{attempt}/"
                        f"{stage.value.lower()}/{safe_name}.npy"
                    )
                    infrastructure.storage.put_bytes(
                        cache_key, cache_data, "application/octet-stream"
                    )
                    uploaded_keys.append(cache_key)
                    cache_bytes += len(cache_data)
                    cache_manifest[str(name)] = {
                        "key": cache_key,
                        "stage": stage.value,
                        "shape": list(value.shape),
                        "bytes": len(cache_data),
                        "sha256": hashlib.sha256(cache_data).hexdigest(),
                        "schema": "ndarray-npy-v1",
                    }
            if settings.debug_artifacts:
                for index, (name, array) in enumerate(sorted(result.artifacts.items())):
                    if index >= infrastructure.settings.max_debug_artifacts:
                        break
                    buffer = io.BytesIO()
                    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
                    cache_data = buffer.getvalue()
                    stage = _artifact_stage(name)
                    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)[:120]
                    if (
                        name not in cache_manifest
                        and cache_bytes + len(cache_data)
                        <= infrastructure.settings.max_job_cache_bytes
                    ):
                        cache_key = (
                            f"jobs/{job_id}/cache/attempt-{attempt}/"
                            f"{stage.value.lower()}/{safe_name}.npy"
                        )
                        infrastructure.storage.put_bytes(
                            cache_key, cache_data, "application/octet-stream"
                        )
                        uploaded_keys.append(cache_key)
                        cache_bytes += len(cache_data)
                        cache_manifest[name] = {
                            "key": cache_key,
                            "stage": stage.value,
                            "shape": list(array.shape),
                            "bytes": len(cache_data),
                            "sha256": hashlib.sha256(cache_data).hexdigest(),
                            "schema": "ndarray-npy-v1",
                        }
                    thumbnail, thumbnail_width, thumbnail_height = _thumbnail(array)
                    debug_records.append(
                        (
                            uuid.uuid4(),
                            safe_name,
                            _artifact_kind(name),
                            thumbnail,
                            thumbnail_width,
                            thumbnail_height,
                        )
                    )

            diagnostics_data = _jsonable(result.diagnostics)
            summary = {
                "profile": diagnostics_data.get("profile"),
                "warnings": diagnostics_data.get("warnings", []),
                "compatibility_score": diagnostics_data.get("compatibility", {}).get("score"),
                "alignment_stage": diagnostics_data.get("alignment", {}).get("selected_stage"),
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
                    width=int(result.output_rgb.shape[1]),
                    height=int(result.output_rgb.shape[0]),
                    byte_size=len(output_data),
                    sha256=hashlib.sha256(output_data).hexdigest(),
                    metadata={"job_id": str(job.id), "metadata_stripped": True},
                    expires_at=job.expires_at,
                )
                repository.add_artifact(job.id, output_asset.id, ArtifactKind.OUTPUT)
                for (
                    asset_id,
                    name,
                    artifact_kind,
                    thumbnail,
                    thumbnail_width,
                    thumbnail_height,
                ) in debug_records:
                    debug_key = f"jobs/{job_id}/debug/attempt-{attempt}/{asset_id}.png"
                    infrastructure.storage.put_bytes(debug_key, thumbnail, "image/png")
                    uploaded_keys.append(debug_key)
                    debug_asset = AssetRepository(db).create(
                        asset_id=asset_id,
                        session_id=cast(uuid.UUID, job.session_id),
                        kind=AssetKind.DEBUG,
                        object_key=debug_key,
                        mime_type="image/png",
                        width=thumbnail_width,
                        height=thumbnail_height,
                        byte_size=len(thumbnail),
                        sha256=hashlib.sha256(thumbnail).hexdigest(),
                        metadata={"diagnostic_name": name},
                        expires_at=job.expires_at,
                    )
                    repository.add_artifact(job.id, debug_asset.id, artifact_kind)
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
                            infrastructure.storage.get_bytes(example.asset.object_key)
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
