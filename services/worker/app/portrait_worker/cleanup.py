from __future__ import annotations

from datetime import UTC, datetime

from portrait_api.metrics import EXPIRED_ASSET_LAG, STORAGE_ERRORS
from portrait_api.models import JobStatus, StyleExample
from portrait_api.repositories import AssetRepository, JobRepository
from portrait_api.time import as_utc
from portrait_worker.infrastructure import WorkerInfrastructure
from sqlalchemy import select


def purge_expired_records(
    infrastructure: WorkerInfrastructure,
    *,
    batch_size: int = 100,
) -> dict[str, int]:
    deleted_assets = 0
    expired_jobs = 0
    deletion_failures = 0

    with infrastructure.session_factory.begin() as db:
        jobs = JobRepository(db).expired_batch(limit=batch_size)
        for job in jobs:
            try:
                infrastructure.storage.delete_prefix(f"jobs/{job.id}/cache/")
            except Exception:
                STORAGE_ERRORS.labels("expired_job_cache_delete").inc()
                deletion_failures += 1
                continue
            job.status = JobStatus.EXPIRED
            job.deleted_at = datetime.now(UTC)
            expired_jobs += 1

    with infrastructure.session_factory.begin() as db:
        repository = AssetRepository(db)
        for asset in repository.expired_batch(limit=batch_size):
            try:
                examples = list(
                    db.scalars(select(StyleExample).where(StyleExample.asset_id == asset.id))
                )
                for example in examples:
                    if example.feature_object_key:
                        infrastructure.storage.delete(example.feature_object_key)
                    infrastructure.storage.delete_prefix(
                        f"styles/{example.style_id}/examples/{example.id}/"
                    )
                    db.delete(example)
                infrastructure.storage.delete(asset.object_key)
            except Exception:
                STORAGE_ERRORS.labels("expired_asset_delete").inc()
                deletion_failures += 1
                continue
            lag = max((datetime.now(UTC) - as_utc(asset.expires_at)).total_seconds(), 0.0)
            EXPIRED_ASSET_LAG.observe(lag)
            repository.mark_deleted(asset)
            deleted_assets += 1

    return {
        "deleted_assets": deleted_assets,
        "expired_jobs": expired_jobs,
        "deletion_failures": deletion_failures,
    }
