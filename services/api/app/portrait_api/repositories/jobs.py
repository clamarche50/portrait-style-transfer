from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from portrait_api.models import (
    AlgorithmProfile,
    ArtifactKind,
    Asset,
    Job,
    JobArtifact,
    JobStatus,
    ProcessingStage,
)
from portrait_api.models.enums import TERMINAL_JOB_STATUSES
from sqlalchemy import func, select
from sqlalchemy.orm import Session


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        session_id: uuid.UUID,
        input_asset_id: uuid.UUID,
        reference_asset_id: uuid.UUID | None,
        style_id: uuid.UUID | None,
        algorithm_profile: AlgorithmProfile,
        settings: dict[str, Any],
        expires_at: datetime,
    ) -> Job:
        job = Job(
            session_id=session_id,
            status=JobStatus.QUEUED,
            stage=ProcessingStage.VALIDATING,
            progress=0,
            input_asset_id=input_asset_id,
            reference_asset_id=reference_asset_id,
            style_id=style_id,
            algorithm_profile=algorithm_profile,
            settings=settings,
            diagnostics={},
            attempt=0,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def get_owned(
        self,
        job_id: uuid.UUID,
        session_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> Job | None:
        statement = select(Job).where(
            Job.id == job_id,
            Job.session_id == session_id,
            Job.deleted_at.is_(None),
        )
        if for_update:
            statement = statement.with_for_update(of=Job)
        return self.session.scalar(statement)

    def get_for_worker(self, job_id: uuid.UUID, *, for_update: bool = False) -> Job | None:
        statement = select(Job).where(Job.id == job_id, Job.deleted_at.is_(None))
        if for_update:
            statement = statement.with_for_update(of=Job)
        return self.session.scalar(statement)

    def active_count(self, session_id: uuid.UUID) -> int:
        active = tuple(status for status in JobStatus if status not in TERMINAL_JOB_STATUSES)
        return int(
            self.session.scalar(
                select(func.count(Job.id)).where(
                    Job.session_id == session_id,
                    Job.status.in_(active),
                    Job.deleted_at.is_(None),
                )
            )
            or 0
        )

    def output_asset(self, job_id: uuid.UUID) -> Asset | None:
        return self.session.scalar(
            select(Asset)
            .join(JobArtifact, JobArtifact.asset_id == Asset.id)
            .where(
                JobArtifact.job_id == job_id,
                JobArtifact.artifact_kind == ArtifactKind.OUTPUT,
                Asset.deleted_at.is_(None),
            )
            .order_by(JobArtifact.created_at.desc())
        )

    def artifacts(self, job_id: uuid.UUID) -> list[JobArtifact]:
        return list(
            self.session.scalars(
                select(JobArtifact)
                .where(JobArtifact.job_id == job_id)
                .order_by(JobArtifact.created_at)
            )
        )

    def add_artifact(
        self, job_id: uuid.UUID, asset_id: uuid.UUID, kind: ArtifactKind
    ) -> JobArtifact:
        artifact = JobArtifact(
            job_id=job_id,
            asset_id=asset_id,
            artifact_kind=kind,
            created_at=datetime.now(UTC),
        )
        self.session.add(artifact)
        self.session.flush()
        return artifact

    def request_cancel(self, job: Job) -> None:
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
        elif job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCEL_REQUESTED
        self.session.flush()

    def update_progress(
        self,
        job: Job,
        *,
        stage: ProcessingStage,
        progress: int,
        worker_id: str | None = None,
    ) -> None:
        if job.status not in {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}:
            return
        job.stage = stage
        job.progress = max(job.progress, min(max(progress, 0), 99))
        if worker_id:
            job.worker_id = worker_id
        self.session.flush()

    def mark_running(self, job: Job, *, worker_id: str) -> None:
        if job.status != JobStatus.QUEUED:
            raise ValueError(f"Cannot start job in {job.status}")
        job.status = JobStatus.RUNNING
        job.stage = ProcessingStage.VALIDATING
        job.progress = 1
        job.worker_id = worker_id
        job.started_at = datetime.now(UTC)
        job.finished_at = None
        job.error_code = None
        job.error_message_safe = None
        job.attempt += 1
        self.session.flush()

    def mark_succeeded(self, job: Job, diagnostics: dict[str, Any]) -> None:
        job.status = JobStatus.SUCCEEDED
        job.stage = ProcessingStage.COMPLETED
        job.progress = 100
        job.diagnostics = diagnostics
        job.error_code = None
        job.error_message_safe = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def mark_failed(self, job: Job, *, code: str, safe_message: str) -> None:
        job.status = JobStatus.FAILED
        job.error_code = code[:100]
        job.error_message_safe = safe_message[:500]
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def mark_cancelled(self, job: Job) -> None:
        job.status = JobStatus.CANCELLED
        job.error_code = None
        job.error_message_safe = None
        job.finished_at = datetime.now(UTC)
        self.session.flush()

    def mark_retry_queued(self, job: Job, *, safe_message: str) -> None:
        job.status = JobStatus.QUEUED
        job.error_code = "TRANSIENT_RETRY"
        job.error_message_safe = safe_message[:500]
        job.worker_id = None
        self.session.flush()

    def reset_for_rerun(self, job: Job, *, diagnostics: dict[str, Any]) -> None:
        if job.status not in TERMINAL_JOB_STATUSES:
            raise ValueError(f"Cannot rerun job in {job.status}")
        job.status = JobStatus.QUEUED
        job.stage = ProcessingStage.VALIDATING
        job.progress = 0
        job.diagnostics = diagnostics
        job.error_code = None
        job.error_message_safe = None
        job.worker_id = None
        job.started_at = None
        job.finished_at = None
        self.session.flush()

    def expired_batch(self, *, limit: int = 100) -> list[Job]:
        return list(
            self.session.scalars(
                select(Job)
                .where(
                    Job.deleted_at.is_(None),
                    Job.expires_at <= datetime.now(UTC),
                    Job.status.in_(tuple(TERMINAL_JOB_STATUSES - {JobStatus.EXPIRED})),
                )
                .order_by(Job.expires_at)
                .limit(limit)
            )
        )
