from __future__ import annotations

import uuid
from datetime import UTC, datetime

from portrait_api.models import JobStatus, ProcessingStage
from portrait_api.repositories import JobRepository
from portrait_worker.infrastructure import WorkerInfrastructure


class JobLeaseLost(RuntimeError):
    """Raised when another worker owns the idempotency lease."""


class JobProgressReporter:
    def __init__(
        self,
        infrastructure: WorkerInfrastructure,
        job_id: uuid.UUID,
        worker_id: str,
        lock_token: str,
    ) -> None:
        self.infrastructure = infrastructure
        self.job_id = job_id
        self.worker_id = worker_id
        self.lock_token = lock_token

    def emit(self, stage: ProcessingStage, percent: int, message: str) -> None:
        with self.infrastructure.session_factory.begin() as db:
            repository = JobRepository(db)
            job = repository.get_for_worker(self.job_id, for_update=True)
            if job is None:
                return
            if job.status == JobStatus.CANCEL_REQUESTED:
                return
            repository.update_progress(
                job,
                stage=stage,
                progress=percent,
                worker_id=self.worker_id,
            )
            event = {
                "job_id": str(job.id),
                "status": job.status.value,
                "stage": job.stage.value,
                "progress": job.progress,
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        if not self.infrastructure.refresh_job_lock(str(self.job_id), self.lock_token):
            raise JobLeaseLost("The job execution lease was lost")
        self.infrastructure.set_progress(str(self.job_id), event)

    def cancel_requested(self) -> bool:
        if self.infrastructure.cancel_requested(str(self.job_id)):
            return True
        with self.infrastructure.session_factory() as db:
            job = JobRepository(db).get_for_worker(self.job_id)
            return job is None or job.status == JobStatus.CANCEL_REQUESTED
