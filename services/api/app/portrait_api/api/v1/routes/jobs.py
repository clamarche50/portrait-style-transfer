from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from portrait_api.config import Settings
from portrait_api.dependencies import (
    Principal,
    get_db,
    get_principal,
    get_progress_store,
    get_settings_from_app,
    get_storage,
    get_task_queue,
)
from portrait_api.errors import AppError
from portrait_api.metrics import JOBS, STORAGE_ERRORS
from portrait_api.models import (
    AlgorithmProfile,
    ArtifactKind,
    AssetKind,
    Job,
    JobStatus,
    ProcessingStage,
)
from portrait_api.models.enums import TERMINAL_JOB_STATUSES
from portrait_api.repositories import AssetRepository, JobRepository, StyleRepository
from portrait_api.schemas.assets import DownloadUrlResponse
from portrait_api.schemas.jobs import (
    CorrectionRequest,
    CreateJobRequest,
    DiagnosticArtifactResponse,
    JobDiagnosticsResponse,
    JobResponse,
)
from portrait_api.security import DownloadTokenSigner
from portrait_api.services.corrections import build_invalidation_plan
from portrait_api.services.queue import TaskQueue
from portrait_api.services.redis_gateway import ProgressStore
from portrait_api.services.storage import ObjectStorage
from portrait_api.time import is_expired
from portrait_api.urls import asset_content_url
from sqlalchemy.orm import Session

router = APIRouter()


def _owned_job(db: Session, principal: Principal, job_id: uuid.UUID, *, lock: bool = False) -> Job:
    job = JobRepository(db).get_owned(job_id, principal.session_id, for_update=lock)
    if job is None:
        raise AppError("JOB_NOT_FOUND", "The job was not found.", 404)
    return job


def _response(request: Request, db: Session, job: Job) -> JobResponse:
    config: Settings = request.app.state.settings
    output = JobRepository(db).output_asset(job.id)

    return JobResponse.from_entity(
        job,
        output_asset_id=output.id if output else None,
        output_url=asset_content_url(config, output.id) if output else None,
        input_preview_url=(
            asset_content_url(config, job.input_asset.id)
            if job.input_asset.deleted_at is None
            else None
        ),
        reference_preview_url=(
            asset_content_url(config, job.reference_asset.id)
            if job.reference_asset is not None and job.reference_asset.deleted_at is None
            else None
        ),
    )


def _public_diagnostics(value: dict[str, object] | None) -> dict[str, object]:
    """Remove internal object-store/cache coordinates from an API response."""

    if not value:
        return {}
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("private_") and key not in {"queue_task_id"}
    }


def _event(job: Job, message: str) -> dict[str, object]:
    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "stage": job.stage.value,
        "progress": job.progress,
        "message": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _enqueue(job: Job, queue: TaskQueue, config: Settings) -> str:
    # The CPU Celery worker orchestrates the isolated GPU inference sidecar.
    # Routing this task to the legacy in-process GPU queue would strand jobs
    # whenever ENABLE_GPU is set for unrelated dense-alignment tooling.
    del config
    return queue.enqueue_transfer(str(job.id), use_gpu=False)


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    payload: CreateJobRequest,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
    progress_store: ProgressStore = Depends(get_progress_store),
    queue: TaskQueue = Depends(get_task_queue),
) -> JobResponse:
    asset_repository = AssetRepository(db)
    input_asset = asset_repository.get_owned(payload.input_asset_id, principal.session_id)
    if input_asset is None or input_asset.kind != AssetKind.INPUT:
        raise AppError("INPUT_ASSET_NOT_FOUND", "The input portrait was not found.", 404)
    if is_expired(input_asset.expires_at):
        raise AppError("INPUT_ASSET_EXPIRED", "The input portrait has expired.", 410)

    if payload.reference_asset_id:
        reference = asset_repository.get_owned(payload.reference_asset_id, principal.session_id)
        if reference is None or reference.kind not in {
            AssetKind.REFERENCE,
            AssetKind.STYLE_EXAMPLE,
        }:
            raise AppError(
                "REFERENCE_ASSET_NOT_FOUND", "The reference portrait was not found.", 404
            )
        if is_expired(reference.expires_at):
            raise AppError("REFERENCE_ASSET_EXPIRED", "The reference portrait has expired.", 410)
    else:
        assert payload.style_id is not None
        style = StyleRepository(db).get_visible(payload.style_id, principal.session_id)
        if style is None:
            raise AppError("STYLE_NOT_FOUND", "The style was not found.", 404)
        if not style.examples:
            raise AppError("STYLE_EMPTY", "The style has no indexed examples.", 422)

    repository = JobRepository(db)
    if repository.active_count(principal.session_id) >= config.max_concurrent_jobs_per_user:
        raise AppError(
            "JOB_QUOTA_EXCEEDED",
            "The concurrent job quota has been reached.",
            429,
            {"max_concurrent_jobs": config.max_concurrent_jobs_per_user},
        )
    settings_data = payload.settings.model_dump(mode="json")
    job = repository.create(
        session_id=principal.session_id,
        input_asset_id=payload.input_asset_id,
        reference_asset_id=payload.reference_asset_id,
        style_id=payload.style_id,
        algorithm_profile=payload.settings.algorithm_profile,
        settings=settings_data,
        expires_at=datetime.now(UTC) + timedelta(hours=config.asset_ttl_hours),
    )
    db.commit()
    try:
        task_id = _enqueue(job, queue, config)
    except Exception as exc:
        failed_job = repository.get_for_worker(job.id, for_update=True)
        if failed_job:
            repository.mark_failed(
                failed_job,
                code="QUEUE_UNAVAILABLE",
                safe_message="The processing queue is temporarily unavailable.",
            )
            db.commit()
        request.app.state.logger.exception(
            "job_enqueue_failed",
            extra={"request_id": request.state.request_id, "job_id": str(job.id)},
        )
        raise AppError(
            "QUEUE_UNAVAILABLE", "The job could not be queued. Try again later.", 503
        ) from exc
    job.diagnostics = {"queue_task_id": task_id, "summary": {}}
    db.commit()
    await progress_store.set_progress(str(job.id), _event(job, "Job queued"))
    JOBS.labels("queued").inc()
    request.app.state.logger.info(
        "job_created",
        extra={
            "request_id": request.state.request_id,
            "job_id": str(job.id),
            "session_hash": request.state.session_hash,
            "algorithm_version": config.algorithm_version,
        },
    )
    return _response(request, db, job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobResponse:
    return _response(request, db, _owned_job(db, principal, job_id))


@router.get("/{job_id}/events")
async def job_events(
    job_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    progress_store: ProgressStore = Depends(get_progress_store),
) -> StreamingResponse:
    job = _owned_job(db, principal, job_id)
    initial_snapshot = _response(request, db, job)

    def load_snapshot() -> JobResponse | None:
        with request.app.state.session_factory() as stream_db:
            current = JobRepository(stream_db).get_owned(job_id, principal.session_id)
            if current is None:
                return None
            return _response(request, stream_db, current)

    def fingerprint(snapshot: JobResponse) -> tuple[object, ...]:
        return (
            snapshot.status,
            snapshot.stage,
            snapshot.progress,
            snapshot.output_asset_id,
            snapshot.error_code,
            snapshot.error_message_safe,
        )

    async def stream() -> AsyncIterator[str]:
        snapshot = initial_snapshot
        last_fingerprint: tuple[object, ...] | None = None
        last_progress_payload = ""
        last_database_poll = 0.0
        last_heartbeat = asyncio.get_running_loop().time()
        while True:
            if await request.is_disconnected():
                return
            current_fingerprint = fingerprint(snapshot)
            if current_fingerprint != last_fingerprint:
                yield f"data: {snapshot.model_dump_json()}\n\n"
                last_fingerprint = current_fingerprint
            status_value = str(snapshot.status)
            if status_value in {item.value for item in TERMINAL_JOB_STATUSES}:
                return
            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = now
            await asyncio.sleep(0.75)
            try:
                latest = await progress_store.get_progress(str(job_id))
            except Exception:
                latest = None
            latest_payload = json.dumps(latest, separators=(",", ":"), default=str)
            should_poll_database = (
                latest_payload != last_progress_payload or now - last_database_poll >= 5
            )
            if should_poll_database:
                loaded = await asyncio.to_thread(load_snapshot)
                if loaded is None:
                    return
                snapshot = loaded
                last_progress_payload = latest_payload
                last_database_poll = now

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    progress_store: ProgressStore = Depends(get_progress_store),
) -> JobResponse:
    job = _owned_job(db, principal, job_id, lock=True)
    if job.status in TERMINAL_JOB_STATUSES:
        return _response(request, db, job)
    JobRepository(db).request_cancel(job)
    db.commit()
    await progress_store.request_cancel(str(job.id))
    await progress_store.set_progress(str(job.id), _event(job, "Cancellation requested"))
    return _response(request, db, job)


def _delete_job_artifacts(
    request: Request,
    db: Session,
    job: Job,
    storage: ObjectStorage,
    *,
    kinds: frozenset[ArtifactKind] | None = None,
) -> None:
    repository = JobRepository(db)
    for artifact in repository.artifacts(job.id):
        if kinds is not None and artifact.artifact_kind not in kinds:
            continue
        try:
            storage.delete(artifact.asset.object_key)
        except Exception as exc:
            STORAGE_ERRORS.labels("job_artifact_delete").inc()
            request.app.state.logger.exception(
                "job_artifact_delete_failed",
                extra={
                    "request_id": request.state.request_id,
                    "job_id": str(job.id),
                    "asset_id": str(artifact.asset_id),
                    "error_type": type(exc).__name__,
                },
            )
            raise AppError(
                "STORAGE_DELETE_FAILED",
                "Job data could not be deleted. Try again later.",
                503,
            ) from exc
        artifact.asset.deleted_at = datetime.now(UTC)
        db.delete(artifact)
    db.flush()


def _delete_unreferenced_sources(
    request: Request,
    db: Session,
    job: Job,
    principal: Principal,
    storage: ObjectStorage,
) -> int:
    repository = AssetRepository(db)
    deleted = 0
    source_assets = [job.input_asset]
    if job.reference_asset is not None:
        source_assets.append(job.reference_asset)
    for asset in source_assets:
        if asset.deleted_at is not None or asset.session_id != principal.session_id:
            continue
        if repository.is_used_by_other_job(asset.id, excluding_job_id=job.id):
            continue
        if repository.is_style_example(asset.id):
            continue
        try:
            storage.delete(asset.object_key)
        except Exception as exc:
            STORAGE_ERRORS.labels("job_source_delete").inc()
            request.app.state.logger.exception(
                "job_source_delete_failed",
                extra={
                    "request_id": request.state.request_id,
                    "job_id": str(job.id),
                    "asset_id": str(asset.id),
                    "error_type": type(exc).__name__,
                },
            )
            raise AppError(
                "STORAGE_DELETE_FAILED",
                "Uploaded source data could not be deleted. Try again later.",
                503,
            ) from exc
        repository.mark_deleted(asset)
        deleted += 1
    return deleted


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    progress_store: ProgressStore = Depends(get_progress_store),
    storage: ObjectStorage = Depends(get_storage),
) -> Response:
    job = _owned_job(db, principal, job_id, lock=True)
    if job.status not in TERMINAL_JOB_STATUSES:
        JobRepository(db).request_cancel(job)
        await progress_store.request_cancel(str(job.id))
    _delete_job_artifacts(request, db, job, storage)
    storage.delete_prefix(f"jobs/{job.id}/cache/")
    deleted_sources = _delete_unreferenced_sources(request, db, job, principal, storage)
    job.deleted_at = datetime.now(UTC)
    db.commit()
    request.app.state.logger.info(
        "job_deleted",
        extra={
            "request_id": request.state.request_id,
            "job_id": str(job.id),
            "session_hash": request.state.session_hash,
            "deleted_source_assets": deleted_sources,
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{job_id}/download-url", response_model=DownloadUrlResponse)
def job_download_url(
    job_id: uuid.UUID,
    response: Response,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
) -> DownloadUrlResponse:
    job = _owned_job(db, principal, job_id)
    if job.status != JobStatus.SUCCEEDED:
        raise AppError("JOB_NOT_COMPLETE", "The job has no downloadable output.", 409)
    output = JobRepository(db).output_asset(job.id)
    if output is None:
        raise AppError("OUTPUT_NOT_FOUND", "The job output was not found.", 404)
    DownloadTokenSigner(config).set_cookie(response, output.id, principal.session_id)
    return DownloadUrlResponse(
        url=asset_content_url(config, output.id, download=True),
        expires_in_seconds=config.signed_url_ttl_seconds,
        expires_at=datetime.now(UTC) + timedelta(seconds=config.signed_url_ttl_seconds),
    )


@router.get("/{job_id}/diagnostics", response_model=JobDiagnosticsResponse)
def job_diagnostics(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
) -> JobDiagnosticsResponse:
    job = _owned_job(db, principal, job_id)
    artifacts = []
    for artifact in JobRepository(db).artifacts(job.id):
        if artifact.asset.deleted_at is not None or artifact.artifact_kind == ArtifactKind.OUTPUT:
            continue
        signed_url = None
        if config.output_signed_urls_in_diagnostics:
            signed_url = asset_content_url(config, artifact.asset.id)
        artifacts.append(
            DiagnosticArtifactResponse(
                asset_id=artifact.asset_id,
                kind=artifact.artifact_kind.value,
                download_url=signed_url,
            )
        )
    return JobDiagnosticsResponse(
        job_id=job.id,
        diagnostics=_public_diagnostics(job.diagnostics),
        artifacts=artifacts,
    )


@router.post("/{job_id}/corrections", response_model=JobResponse)
def save_corrections(
    job_id: uuid.UUID,
    payload: CorrectionRequest,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
) -> JobResponse:
    job = _owned_job(db, principal, job_id, lock=True)
    if job.status not in TERMINAL_JOB_STATUSES or job.status == JobStatus.EXPIRED:
        raise AppError("JOB_NOT_EDITABLE", "Corrections require a completed or failed job.", 409)
    new_corrections = [item.model_dump(mode="json") for item in payload.corrections]
    ai_job = job.algorithm_profile == AlgorithmProfile.AI_DGPST_V1
    if ai_job and any(correction.get("type") != "background" for correction in new_corrections):
        raise AppError(
            "AI_CORRECTION_UNSUPPORTED",
            "AI jobs support background corrections only; "
            "create a new job to change transfer controls.",
            422,
        )
    combined = [*(job.corrections or []), *new_corrections]
    if len(combined) > 256:
        raise AppError("TOO_MANY_CORRECTIONS", "Too many correction operations were supplied.", 422)
    plan = build_invalidation_plan(
        new_corrections,
        persisted_corrections=combined,
    )
    _delete_job_artifacts(
        request,
        db,
        job,
        storage,
        kinds=plan.invalidated_artifact_kinds,
    )
    diagnostics = dict(job.diagnostics or {})
    manifest = diagnostics.get("private_cache_manifest", {})
    invalidated_stage_names = {stage.value for stage in plan.invalidated_stages}
    retained_manifest: dict[str, object] = {}
    if isinstance(manifest, dict):
        for name, metadata in manifest.items():
            if not isinstance(metadata, dict):
                continue
            if metadata.get("stage") in invalidated_stage_names:
                key = metadata.get("key")
                if isinstance(key, str):
                    storage.delete(key)
            else:
                retained_manifest[str(name)] = metadata
    if ai_job:
        for metadata in retained_manifest.values():
            if not isinstance(metadata, dict):
                continue
            key = metadata.get("key")
            if isinstance(key, str):
                storage.delete(key)
        retained_manifest = {}
        storage.delete_prefix(f"jobs/{job.id}/cache/")
    for invalidated_stage in plan.invalidated_stages:
        storage.delete_prefix(f"jobs/{job.id}/cache/{invalidated_stage.value.lower()}/")
    diagnostics["private_cache_manifest"] = retained_manifest
    diagnostics["cache_invalidation"] = {
        "resume_from_stage": (
            ProcessingStage.VALIDATING.value if ai_job else plan.earliest_stage.value
        ),
        "invalidated_stages": [stage.value for stage in plan.invalidated_stages],
        "correction_hash": plan.correction_hash,
        "full_ai_rerun": ai_job,
    }
    job.corrections = combined
    job.diagnostics = diagnostics
    db.commit()
    return _response(request, db, job)


@router.post("/{job_id}/rerun", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def rerun_job(
    job_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
    progress_store: ProgressStore = Depends(get_progress_store),
    storage: ObjectStorage = Depends(get_storage),
    queue: TaskQueue = Depends(get_task_queue),
) -> JobResponse:
    repository = JobRepository(db)
    job = _owned_job(db, principal, job_id, lock=True)
    if job.status not in TERMINAL_JOB_STATUSES or job.status == JobStatus.EXPIRED:
        raise AppError("JOB_NOT_RERUNNABLE", "This job cannot be rerun yet.", 409)
    if repository.active_count(principal.session_id) >= config.max_concurrent_jobs_per_user:
        raise AppError("JOB_QUOTA_EXCEEDED", "The concurrent job quota has been reached.", 429)
    _delete_job_artifacts(request, db, job, storage)
    diagnostics = dict(job.diagnostics or {})
    invalidation = diagnostics.get("cache_invalidation", {})
    cache_manifest = diagnostics.get("private_cache_manifest", {})
    ai_job = job.algorithm_profile == AlgorithmProfile.AI_DGPST_V1
    resume_stage = (
        ProcessingStage.VALIDATING.value
        if ai_job
        else invalidation.get("resume_from_stage", ProcessingStage.VALIDATING.value)
    )
    diagnostics["resume"] = {
        "requested_stage": resume_stage,
        "cache_reuse": not ai_job and isinstance(cache_manifest, dict) and bool(cache_manifest),
    }
    repository.reset_for_rerun(job, diagnostics=diagnostics)
    job.expires_at = datetime.now(UTC) + timedelta(hours=config.asset_ttl_hours)
    db.commit()
    await progress_store.clear_cancel(str(job.id))
    try:
        task_id = _enqueue(job, queue, config)
    except Exception as exc:
        repository.mark_failed(
            job,
            code="QUEUE_UNAVAILABLE",
            safe_message="The processing queue is temporarily unavailable.",
        )
        db.commit()
        raise AppError(
            "QUEUE_UNAVAILABLE", "The job could not be queued. Try again later.", 503
        ) from exc
    diagnostics["queue_task_id"] = task_id
    job.diagnostics = diagnostics
    db.commit()
    await progress_store.set_progress(str(job.id), _event(job, "Job queued for rerun"))
    return _response(request, db, job)
