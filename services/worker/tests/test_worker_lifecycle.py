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
    StyleExample,
)
from portrait_api.repositories import AssetRepository, JobRepository, StyleRepository
from portrait_transfer.exceptions import OptionalDependencyError
from portrait_worker.ai_client import AIEngineError, AIEngineResponse
from portrait_worker.cleanup import purge_expired_records
from portrait_worker.gpu_dense import KorniaGpuDenseCorrespondence, build_dense_backend
from portrait_worker.mediapipe_adapter import build_portrait_analyzer
from portrait_worker.tasks import (
    _AI_RESPONSE_MAX_ENCODED_BYTES,
    _ai_settings,
    _engine_request_settings,
    index_style,
    process_transfer_job,
)


def _successful_ai_response(content: bytes) -> AIEngineResponse:
    return AIEngineResponse(
        image_png=content,
        diagnostics={
            "model": "DGPST",
            "warnings": ["synthetic_engine"],
            "style_strength_applied": 0.58,
            "structure_strength_applied": 0.97,
        },
        engine_id="ai_dgpst_v1",
    )


def test_worker_executes_transfer_and_commits_output(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    calls: list[dict[str, object]] = []

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        with Image.open(io.BytesIO(content)) as content_image:
            assert np.allclose(np.asarray(content_image).mean(axis=(0, 1)), (75, 110, 145))
        with Image.open(io.BytesIO(style)) as style_image:
            assert np.allclose(np.asarray(style_image).mean(axis=(0, 1)), (190, 150, 115))
        calls.append({"content": content, "style": style, "settings": settings})
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    process_transfer_job.run(job_id)

    assert len(calls) == 1
    assert calls[0]["settings"] == {
        "algorithm_profile": "ai_dgpst_v1",
        "style_strength": 0.75,
        "structure_strength": 0.9,
        "inference_steps": 30,
        "random_seed": 0,
    }
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED
        assert job.progress == 100
        assert job.diagnostics["summary"]["engine"] == "ai_dgpst_v1"
        assert job.diagnostics["summary"]["requested_style_strength"] == 0.75
        assert job.diagnostics["summary"]["effective_style_strength"] == 0.58
        assert job.diagnostics["summary"]["effective_structure_strength"] == 0.97
        assert job.diagnostics["private_cache_manifest"] == {}
        artifacts = JobRepository(db).artifacts(job.id)
        assert {artifact.artifact_kind for artifact in artifacts} == {ArtifactKind.OUTPUT}
        output = JobRepository(db).output_asset(job.id)
        assert output is not None
        assert output.object_key in worker.storage.objects
    assert worker.progress(job_id)["status"] == "SUCCEEDED"  # type: ignore[index]
    assert f"job:{job_id}:lock" not in worker.redis.values


def test_worker_uses_separate_external_and_ai_response_decode_limits(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    from portrait_transfer.image_io import decode_image as real_decode_image

    encoded_limits: list[int] = []

    def tracked_decode(payload: bytes, limits: object) -> object:
        encoded_limits.append(limits.max_encoded_bytes)  # type: ignore[attr-defined]
        return real_decode_image(payload, limits)  # type: ignore[arg-type]

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        del style, settings
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.decode_image", tracked_decode)
    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    process_transfer_job.run(job_id)

    assert encoded_limits == [
        worker.infrastructure.settings.max_upload_bytes,
        worker.infrastructure.settings.max_upload_bytes,
        _AI_RESPONSE_MAX_ENCODED_BYTES,
    ]


def test_worker_does_not_send_classical_resume_data_and_removes_old_cache(
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

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        del style
        captured["settings"] = settings
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    process_transfer_job.run(job_id)
    assert "resume" not in captured["settings"]
    assert checkpoint_key not in worker.storage.objects
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None
        assert job.diagnostics["private_cache_manifest"] == {}


def test_worker_retries_once_with_identity_preserving_settings(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    requests: list[dict[str, object]] = []
    quality_results = iter(
        [
            (False, {"passed": False, "landmark_drift_mean": 0.12}),
            (True, {"passed": True, "landmark_drift_mean": 0.03}),
        ]
    )

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        del style
        requests.append(dict(settings))  # type: ignore[arg-type]
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    monkeypatch.setattr(
        "portrait_worker.tasks._geometry_quality",
        lambda *_args: next(quality_results),
    )
    process_transfer_job.run(job_id)

    assert len(requests) == 2
    assert requests[0]["style_strength"] == 0.75
    assert requests[0]["structure_strength"] == 0.9
    assert requests[1]["style_strength"] == 0.6
    assert requests[1]["structure_strength"] == 0.95
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None and job.status == JobStatus.SUCCEEDED
        assert job.diagnostics["summary"]["quality_retry_performed"] is True
        guard = job.diagnostics["transfer"]["worker_quality_guard"]
        assert guard["retry_performed"] is True
        assert len(guard["attempts"]) == 2


def test_worker_fails_closed_when_both_geometry_checks_fail(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    call_count = 0

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        nonlocal call_count
        del style, settings
        call_count += 1
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    monkeypatch.setattr(
        "portrait_worker.tasks._geometry_quality",
        lambda *_args: (False, {"passed": False, "landmark_drift_mean": 0.2}),
    )
    process_transfer_job.run(job_id)

    assert call_count == 2
    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None and job.status == JobStatus.FAILED
        assert job.error_code == "AI_QUALITY_GUARD_FAILED"
        assert JobRepository(db).output_asset(job.id) is None


def test_worker_applies_solid_background_and_preserves_jpeg_output_setting(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    with worker.session_factory.begin() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None
        settings = dict(job.settings)
        settings.update(
            {
                "background_mode": "SOLID",
                "background_color": "#112233",
                "output_format": "JPEG",
                "jpeg_quality": 90,
            }
        )
        job.settings = settings

    def transfer(*, content: bytes, style: bytes, settings: object) -> AIEngineResponse:
        del style
        assert "background_mode" not in settings  # type: ignore[operator]
        return _successful_ai_response(content)

    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=transfer),
    )
    process_transfer_job.run(job_id)

    with worker.session_factory() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job is not None and job.status == JobStatus.SUCCEEDED
        output = JobRepository(db).output_asset(job.id)
        assert output is not None
        assert output.mime_type == "image/jpeg"
        assert output.object_key.endswith(".jpg")
        with Image.open(io.BytesIO(worker.storage.get_bytes(output.object_key))) as image:
            corner = np.asarray(image)[0, 0]
        assert np.allclose(corner, (17, 34, 51), atol=8)


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

    def should_not_run(**_kwargs: object) -> None:
        raise AssertionError("transfer should not start after cancellation")

    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=should_not_run),
    )
    process_transfer_job.run(cancelled_id)
    with worker.session_factory() as db:
        cancelled = db.get(Job, uuid.UUID(cancelled_id))
        assert cancelled is not None and cancelled.status == JobStatus.CANCELLED

    failed_id, _, _ = worker.create_job()

    def invalid(**_kwargs: object) -> None:
        raise AIEngineError(
            "AI_QUALITY_GUARD_FAILED",
            "The generated portrait did not preserve the source geometry.",
            retryable=False,
        )

    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: SimpleNamespace(transfer=invalid),
    )
    process_transfer_job.run(failed_id)
    with worker.session_factory() as db:
        failed = db.get(Job, uuid.UUID(failed_id))
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.error_code == "AI_QUALITY_GUARD_FAILED"
        assert failed.attempt == 1


def test_idempotency_lock_prevents_duplicate_execution(
    worker: WorkerHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _, _ = worker.create_job()
    worker.redis.values[f"job:{job_id}:lock"] = "another-worker"
    monkeypatch.setattr("portrait_worker.tasks.get_infrastructure", lambda: worker.infrastructure)
    monkeypatch.setattr(
        "portrait_worker.tasks.get_ai_engine_client",
        lambda _settings: pytest.fail("duplicate worker executed the transfer"),
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


def test_ai_settings_ignore_classical_corrections_but_apply_background_override() -> None:
    settings = _ai_settings(
        {
            "style_strength": 0.7,
            "structure_strength": 0.8,
            "inference_steps": 25,
            "random_seed": 42,
            "background_mode": "KEEP",
            "background_color": None,
        },
        [
            {"type": "eye", "pupil_center": [0.4, 0.35]},
            {"type": "mask", "points": [[0.2, 0.3]]},
            {"type": "background", "mode": "SOLID", "color": "#112233"},
        ],
    )
    request = _engine_request_settings(settings)
    assert request == {
        "algorithm_profile": "ai_dgpst_v1",
        "style_strength": 0.7,
        "structure_strength": 0.8,
        "inference_steps": 25,
        "random_seed": 42,
    }
    assert settings["background_mode"] == "SOLID"
    assert settings["background_color"] == "#112233"
    assert "operations" not in request


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
