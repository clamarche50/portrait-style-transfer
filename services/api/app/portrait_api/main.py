from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from portrait_api.api.v1 import api_router
from portrait_api.config import Settings, get_settings
from portrait_api.db import Base, build_engine, build_session_factory
from portrait_api.errors import install_error_handlers
from portrait_api.logging import configure_logging
from portrait_api.middleware import (
    RateLimitMiddleware,
    RequestMetricsMiddleware,
    RequestSessionMiddleware,
    RequestTracingMiddleware,
    SecurityHeadersMiddleware,
)
from portrait_api.services.queue import CeleryTaskQueue, TaskQueue
from portrait_api.services.redis_gateway import ProgressStore, RedisGateway
from portrait_api.services.storage import Boto3ObjectStorage, ObjectStorage
from portrait_api.telemetry import TelemetryRuntime, create_telemetry
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import Engine


@dataclass(slots=True)
class ServiceOverrides:
    engine: Engine | None = None
    storage: ObjectStorage | None = None
    progress_store: ProgressStore | None = None
    task_queue: TaskQueue | None = None
    telemetry: TelemetryRuntime | None = None


def create_app(
    settings: Settings | None = None,
    *,
    overrides: ServiceOverrides | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    overrides = overrides or ServiceOverrides()
    logger = configure_logging(settings.log_level)
    engine = overrides.engine or build_engine(settings.database_url)
    storage = overrides.storage or Boto3ObjectStorage(settings)
    progress_store = overrides.progress_store or RedisGateway(settings)
    task_queue = overrides.task_queue or CeleryTaskQueue(settings)
    telemetry = overrides.telemetry or create_telemetry(
        "portrait-style-api", settings.otel_exporter_otlp_endpoint
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.logger = logger
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        app.state.storage = storage
        app.state.progress_store = progress_store
        app.state.task_queue = task_queue
        app.state.telemetry = telemetry

        try:
            if settings.auto_create_schema:
                Base.metadata.create_all(engine)
            if settings.initialize_storage_on_startup and isinstance(storage, Boto3ObjectStorage):
                storage.ensure_private_bucket()
            yield
        finally:
            try:
                await progress_store.close()
            finally:
                try:
                    engine.dispose()
                finally:
                    telemetry.shutdown()

    app = FastAPI(
        title="Portrait Style Transfer API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.expose_api_docs else None,
        openapi_url="/openapi.json" if settings.expose_api_docs else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.logger = logger
    install_error_handlers(app)
    app.include_router(api_router)

    if settings.expose_metrics:

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.add_middleware(RateLimitMiddleware, settings=settings, store=progress_store)
    app.add_middleware(RequestSessionMiddleware, settings=settings)
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    app.add_middleware(RequestMetricsMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-ID"],
        expose_headers=["X-CSRF-Token", "X-Request-ID"],
    )
    app.add_middleware(RequestTracingMiddleware, tracer=telemetry.tracer)
    return app


app = create_app()
