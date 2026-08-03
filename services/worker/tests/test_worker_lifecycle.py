from __future__ import annotations

import hashlib
import io
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import WorkerHarness
from PIL import Image
from portrait_api.models import (
    ArtifactKind,
    Asset,
    AssetKind,
    Job,
    JobStatus,
    ProcessingStage,
    StyleExample,
)
from portrait_api.repositories import AssetRepository, JobRepository, StyleRepository
from portrait_transfer.exceptions import InputValidationError, OptionalDependencyError
from portrait_transfer.types import ProcessingStage as CoreProcessingStage
from portrait_worker.cleanup import purge_expired_records
from portrait_worker.gpu_dense import KorniaGpuDenseCorrespondence, build_dense_backend
from portrait_worker.mediapipe_adapter import build_portrait_analyzer
from portrait_worker.tasks import (
    _artifact_stage,
    _load_resume_artifacts,
    _translate_corrections,
    index_style,
    process_transfer_job,
)


def _successful_result(input_rgb: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        output_rgb=np.clip(input_rgb * 0.8 + 0.1, 0.0, 1.0).astype(np.float32),
        diagnostics={
            "profile": "paper_exact",
            "warnings": ["synthetic_pipeline"],
            "compatibility": {"score": 0.91},
            "alignment": {"selected_stage": "line"},
        },
        artifacts={"input_mask": np.ones(input_rgb.shape[:2], dtype=np.float32)},
        resume_artifacts={
            "resume.schema": np.asarray([1], dtype=np.float32),
            "resume.input_crop": np.asarray(input_rgb, dtype=np.float32),
        },
    )


def test_worker_executes_transfer_and_commits_output(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job(debug_artifacts=True)
    calls: list[dict[str, object]] = []

    def transfer(
        input_rgb: np.ndarray, reference_rgb: np.ndarray, settings: object, runtime: object
    ):
        assert input_rgb.shape == reference_rgb.shape == (256, 256, 3)
        runtime.progress_callback(CoreProcessingStage.PREFLIGHT, 8, "Synthetic preflight")
        calls.append({"settings": settings, "runtime": runtime})
        return _successful_result(input_rgb)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr("portrait_worker.tasks.transfer_portrait_style", transfer)
    process_transfer_job.run(job_id)

    assert len(calls) == 1
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED
        assert job.progress == 100
        assert job.diagnostics["summary"]["analysis_backend"] == "HeuristicPortraitAnalyzer"
        assert job.diagnostics["private_cache_manifest"]
        manifest = job.diagnostics["private_cache_manifest"]
        assert set(manifest) >= {"resume.schema", "resume.input_crop"}
        assert manifest["resume.schema"]["schema"] == "ndarray-npy-v1"
        artifacts = JobRepository(db).artifacts(job.id)
        assert {artifact.artifact_kind for artifact in artifacts} >= {
            ArtifactKind.OUTPUT,
            ArtifactKind.INPUT_MASK,
        }
        output = JobRepository(db).output_asset(job.id)
        assert output is not None
        assert output.object_key in worker.storage.objects
    assert worker.progress(job_id)["status"] == "SUCCEEDED"  # type: ignore[index]
    assert f"job:{job_id}:lock" not in worker.redis.values


def test_worker_loads_persisted_resume_bundle_and_exact_stage(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    checkpoint = io.BytesIO()
    np.save(checkpoint, np.asarray([1], dtype=np.float32), allow_pickle=False)
    checkpoint_key = f"jobs/{job_id}/cache/attempt-0/dense_alignment/resume.schema.npy"
    worker.storage.put_bytes(checkpoint_key, checkpoint.getvalue(), "application/octet-stream")
    with worker.session_factory.begin() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None
        job.diagnostics = {
            "private_cache_manifest": {
                "resume.schema": {
                    "key": checkpoint_key,
                    "stage": "DENSE_ALIGNMENT",
                    "schema": "ndarray-npy-v1",
                    "shape": [1],
                    "sha256": hashlib.sha256(checkpoint.getvalue()).hexdigest(),
                }
            },
            "resume": {"requested_stage": "BACKGROUND", "cache_reuse": True},
        }

    captured: dict[str, object] = {}

    def transfer(input_rgb: np.ndarray, _reference: np.ndarray, _settings: object, runtime: object):
        captured["resume"] = runtime.resume_artifacts
        captured["stage"] = runtime.corrections["resume_from_stage"]
        return _successful_result(input_rgb)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr("portrait_worker.tasks.transfer_portrait_style", transfer)
    process_transfer_job.run(job_id)
    assert captured["stage"] == "background"
    assert np.array_equal(captured["resume"]["resume.schema"], np.asarray([1], np.float32))


def test_resume_manifest_integrity_and_invalidation_stages(worker: WorkerHarness) -> None:
    key = "jobs/test/cache/attempt-1/dense_alignment/resume.schema.npy"
    payload = io.BytesIO()
    np.save(payload, np.asarray([1], dtype=np.float32), allow_pickle=False)
    worker.storage.put_bytes(key, payload.getvalue(), "application/octet-stream")
    diagnostics = {
        "private_cache_manifest": {
            "resume.schema": {
                "key": key,
                "schema": "ndarray-npy-v1",
                "shape": [1],
                "sha256": "0" * 64,
            }
        }
    }
    assert _load_resume_artifacts(worker.infrastructure, diagnostics) == {}
    assert _artifact_stage("resume.input_crop") == ProcessingStage.DENSE_ALIGNMENT
    assert _artifact_stage("resume.pre_eye_rgb") == ProcessingStage.MULTISCALE_TRANSFER
    assert _artifact_stage("resume.post_eye_rgb") == ProcessingStage.EYE_HIGHLIGHTS
    assert _artifact_stage("resume.integrity.background") == ProcessingStage.EYE_HIGHLIGHTS


def test_style_indexing_persists_private_analysis_assets(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = uuid.uuid4()
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    source_buffer = io.BytesIO()
    yy, xx = np.indices((768, 768))
    texture = (((xx // 12 + yy // 12) % 2) * 24 - 12)[..., None]
    base = np.asarray([175, 140, 110], dtype=np.int16)
    pixels = np.clip(base + texture, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(source_buffer, "PNG")
    source = source_buffer.getvalue()
    with worker.session_factory.begin() as db:
        asset = AssetRepository(db).create(
            session_id=session_id,
            kind=AssetKind.STYLE_EXAMPLE,
            object_key=f"styles/source/{session_id}.png",
            mime_type="image/png",
            width=768,
            height=768,
            byte_size=len(source),
            sha256="3" * 64,
            metadata={"normalized": True},
            expires_at=expires_at,
        )
        style = StyleRepository(db).create(
            session_id=session_id,
            name="Indexed test style",
            description="",
            rights_confirmed=True,
            is_public=False,
        )
        example = StyleRepository(db).add_example(style, asset)
        style_id, example_id = style.id, example.id
        source_key = asset.object_key
    worker.storage.put_bytes(source_key, source, "image/png")
    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)

    index_style.run(str(style_id))

    prefix = f"styles/{style_id}/examples/{example_id}/"
    derived = {key for key in worker.storage.objects if key.startswith(prefix)}
    assert derived >= {
        f"{prefix}head-mask.npy",
        f"{prefix}landmarks.npy",
        f"{prefix}background.png",
    }
    with worker.session_factory() as db:
        indexed = db.get(StyleExample, example_id)
        assert indexed is not None and indexed.feature_object_key in worker.storage.objects
        assert indexed.quality["indexed"] is True
        assert indexed.quality["landmark_count"] > 0
        assert set(indexed.quality["derived_assets"]) >= {
            "head_mask",
            "landmarks",
            "background",
        }


def test_worker_cancellation_and_deterministic_failure_do_not_retry(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled_id, _, _ = worker.create_job()
    worker.redis.values[f"job:{cancelled_id}:cancel"] = "1"
    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)

    def should_not_run(*_args: object) -> None:
        raise AssertionError("transfer should not start after cancellation")

    monkeypatch.setattr("portrait_worker.tasks.transfer_portrait_style", should_not_run)
    process_transfer_job.run(cancelled_id)
    with worker.session_factory() as db:
        cancelled = db.get(Job, uuid.UUID(cancelled_id))
        assert cancelled is not None and cancelled.status == JobStatus.CANCELLED

    failed_id, _, _ = worker.create_job()

    def invalid(*_args: object) -> None:
        raise InputValidationError("Synthetic deterministic validation failure")

    monkeypatch.setattr("portrait_worker.tasks.transfer_portrait_style", invalid)
    process_transfer_job.run(failed_id)
    with worker.session_factory() as db:
        failed = db.get(Job, uuid.UUID(failed_id))
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.error_code == "INVALID_INPUT"
        assert failed.attempt == 1


def test_idempotency_lock_prevents_duplicate_execution(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    worker.redis.values[f"job:{job_id}:lock"] = "another-worker"
    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.transfer_portrait_style",
        lambda *_args: pytest.fail("duplicate worker executed the transfer"),
    )
    process_transfer_job.run(job_id)
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None and job.status == JobStatus.QUEUED


def test_expiry_purges_objects_and_soft_deletes_records(worker: WorkerHarness) -> None:
    expired = datetime.now(UTC) - timedelta(hours=2)
    job_id, input_key, reference_key = worker.create_job(expires_at=expired)
    cache_key = f"jobs/{job_id}/cache/validating/cache.npy"
    worker.storage.put_bytes(cache_key, b"cache", "application/octet-stream")
    with worker.session_factory.begin() as db:
        job = JobRepository(db).get_for_worker(uuid.UUID(job_id), for_update=True)
        assert job is not None
        JobRepository(db).mark_failed(job, code="SYNTHETIC", safe_message="Synthetic")

    result = purge_expired_records(worker.infrastructure)
    assert result == {"deleted_assets": 2, "expired_jobs": 1, "deletion_failures": 0}
    assert input_key not in worker.storage.objects
    assert reference_key not in worker.storage.objects
    assert cache_key not in worker.storage.objects
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None and job.status == JobStatus.EXPIRED and job.deleted_at is not None
        assert all(asset.deleted_at is not None for asset in db.query(Asset).all())


def test_normalized_corrections_are_tagged_for_crop_scaling() -> None:
    operations = _translate_corrections(
        [
            {
                "type": "eye",
                "eye": "LEFT",
                "pupil_center": [0.4, 0.35],
                "iris_radius": 0.025,
                "highlight_scale": 1.2,
                "highlight_rotation_degrees": 15.0,
            },
            {
                "type": "mask",
                "operation": "ADD",
                "points": [[0.2, 0.3], [0.4, 0.5]],
                "radius": 0.02,
            },
        ]
    )
    assert all(item["coordinate_space"] == "normalized" for item in operations)
    eye = operations[0]
    assert eye["highlight_scale"] == 1.2
    assert eye["highlight_rotation_degrees"] == 15.0


def test_missing_models_fail_closed_unless_explicit_test_fallback(worker: WorkerHarness) -> None:
    analyzer = build_portrait_analyzer(worker.infrastructure.settings)
    assert type(analyzer).__name__ == "HeuristicPortraitAnalyzer"


def test_gpu_backend_selection_is_explicit_and_fails_without_gpu_extra(
    worker: WorkerHarness,
) -> None:
    disabled = build_dense_backend(worker.infrastructure.settings, enabled=False)
    assert type(disabled).__name__ == "NoOpDenseCorrespondence"
    gpu_settings = worker.infrastructure.settings.model_copy(
        update={"enable_gpu": True, "dense_alignment_device": "cuda"}
    )
    backend = build_dense_backend(gpu_settings, enabled=True)
    assert isinstance(backend, KorniaGpuDenseCorrespondence)
    with pytest.raises(OptionalDependencyError, match=r"GPU dense backend|CUDA"):
        backend._libraries()
