from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from portrait_api.config import Settings
from portrait_api.db import build_engine
from portrait_api.main import ServiceOverrides, create_app
from portrait_api.services.queue import MemoryTaskQueue
from portrait_api.services.redis_gateway import MemoryProgressStore
from portrait_api.services.storage import MemoryObjectStorage
from sqlalchemy import Engine


@dataclass(slots=True)
class ApiHarness:
    client: TestClient
    engine: Engine
    storage: MemoryObjectStorage
    progress: MemoryProgressStore
    queue: MemoryTaskQueue
    csrf: str

    @property
    def unsafe_headers(self) -> dict[str, str]:
        return {"X-CSRF-Token": self.csrf}

    def refresh_session(self) -> None:
        self.client.cookies.clear()
        response = self.client.get("/api/v1/health/live")
        self.csrf = response.headers["X-CSRF-Token"]


@pytest.fixture
def api() -> Iterator[ApiHarness]:
    settings = Settings(
        app_env="test",
        expose_api_docs=True,
        expose_metrics=True,
        database_url="sqlite+pysqlite://",
        require_models_for_readiness=False,
        initialize_storage_on_startup=False,
        auto_create_schema=True,
        cookie_secure_override=False,
        session_secret="test-session-secret-with-sufficient-entropy",
        cors_origins=["http://testserver"],
    )
    engine = build_engine(settings.database_url)
    storage = MemoryObjectStorage()
    progress = MemoryProgressStore()
    queue = MemoryTaskQueue()
    app = create_app(
        settings,
        overrides=ServiceOverrides(
            engine=engine,
            storage=storage,
            progress_store=progress,
            task_queue=queue,
        ),
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        yield ApiHarness(
            client=client,
            engine=engine,
            storage=storage,
            progress=progress,
            queue=queue,
            csrf=response.headers["X-CSRF-Token"],
        )


def png_bytes(
    color: tuple[int, int, int] = (96, 128, 160),
    *,
    size: tuple[int, int] = (96, 128),
) -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def portrait_png() -> bytes:
    return png_bytes()
