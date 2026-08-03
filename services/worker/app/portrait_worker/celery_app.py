from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue
from portrait_api.config import get_settings
from portrait_worker.telemetry import install_celery_tracing

settings = get_settings()
_celery_tracing = install_celery_tracing(settings.otel_exporter_otlp_endpoint)

celery_app = Celery(
    "portrait_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["portrait_worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.worker_task_time_limit_seconds,
    task_soft_time_limit=max(settings.worker_task_time_limit_seconds - 30, 30),
    task_default_queue="portrait-cpu",
    task_queues=(
        Queue("portrait-cpu", Exchange("portrait"), routing_key="cpu"),
        Queue("portrait-gpu", Exchange("portrait"), routing_key="gpu"),
        Queue("maintenance", Exchange("portrait"), routing_key="maintenance"),
    ),
    task_routes={
        "portrait_worker.process_transfer_job": {"queue": "portrait-cpu", "routing_key": "cpu"},
        "portrait_worker.index_style": {"queue": "portrait-cpu", "routing_key": "cpu"},
        "portrait_worker.purge_expired": {"queue": "maintenance", "routing_key": "maintenance"},
    },
    beat_schedule={
        "purge-expired-hourly": {
            "task": "portrait_worker.purge_expired",
            "schedule": 3600.0,
            "options": {"queue": "maintenance"},
        }
    },
)
