from __future__ import annotations

from typing import Protocol

from celery import Celery
from portrait_api.config import Settings


class TaskQueue(Protocol):
    def enqueue_transfer(self, job_id: str, *, use_gpu: bool = False) -> str: ...

    def enqueue_style_index(self, style_id: str) -> str: ...

    def enqueue_expiry_purge(self) -> str: ...


class CeleryTaskQueue:
    def __init__(self, settings: Settings) -> None:
        self.client = Celery(
            "portrait_api_client",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
        )

    def enqueue_transfer(self, job_id: str, *, use_gpu: bool = False) -> str:
        queue = "portrait-gpu" if use_gpu else "portrait-cpu"
        result = self.client.send_task(
            "portrait_worker.process_transfer_job",
            args=[job_id],
            queue=queue,
        )
        return str(result.id)

    def enqueue_style_index(self, style_id: str) -> str:
        result = self.client.send_task(
            "portrait_worker.index_style",
            args=[style_id],
            queue="portrait-cpu",
        )
        return str(result.id)

    def enqueue_expiry_purge(self) -> str:
        result = self.client.send_task("portrait_worker.purge_expired", queue="maintenance")
        return str(result.id)


class MemoryTaskQueue:
    def __init__(self) -> None:
        self.transfers: list[tuple[str, bool]] = []
        self.style_indexes: list[str] = []
        self.purges = 0

    def enqueue_transfer(self, job_id: str, *, use_gpu: bool = False) -> str:
        self.transfers.append((job_id, use_gpu))
        return f"transfer-{len(self.transfers)}"

    def enqueue_style_index(self, style_id: str) -> str:
        self.style_indexes.append(style_id)
        return f"style-{len(self.style_indexes)}"

    def enqueue_expiry_purge(self) -> str:
        self.purges += 1
        return f"purge-{self.purges}"
