from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from PIL import Image
from portrait_api.config import Settings
from portrait_api.db import Base, build_engine, build_session_factory
from portrait_api.models import AlgorithmProfile, AssetKind
from portrait_api.repositories import AssetRepository, JobRepository
from portrait_api.services.storage import MemoryObjectStorage
from portrait_worker.infrastructure import WorkerInfrastructure
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __enter__(self) -> FakePipeline:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> FakePipeline:
        self.commands.append(("set", args, kwargs))
        return self

    def publish(self, *args: Any, **kwargs: Any) -> FakePipeline:
        self.commands.append(("publish", args, kwargs))
        return self

    def execute(self) -> list[object]:
        results: list[object] = []
        for command, args, kwargs in self.commands:
            if command == "set":
                results.append(self.redis.set(*args, **kwargs))
            else:
                self.redis.published.append((str(args[0]), str(args[1])))
                results.append(1)
        return results


@dataclass(slots=True)
class FakeRedis:
    values: dict[str, str] = field(default_factory=dict)
    published: list[tuple[str, str]] = field(default_factory=list)

    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def expire(self, key: str, _seconds: int) -> bool:
        return key in self.values

    def exists(self, key: str) -> int:
        return int(key in self.values)

    def eval(self, _script: str, _keys: int, key: str, token: str) -> int:
        if self.values.get(key) != token:
            return 0
        del self.values[key]
        return 1

    def pipeline(self, *, transaction: bool = True) -> FakePipeline:
        del transaction
        return FakePipeline(self)


@dataclass(slots=True)
class WorkerHarness:
    infrastructure: WorkerInfrastructure
    engine: Engine
    session_factory: sessionmaker[Session]
    storage: MemoryObjectStorage
    redis: FakeRedis

    def create_job(
        self,
        *,
        debug_artifacts: bool = False,
        expires_at: datetime | None = None,
    ) -> tuple[str, str, str]:
        session_id = __import__("uuid").uuid4()
        expires = expires_at or datetime.now(UTC) + timedelta(hours=1)
        input_data = portrait_bytes((75, 110, 145))
        reference_data = portrait_bytes((190, 150, 115))
        with self.session_factory.begin() as db:
            assets = AssetRepository(db)
            input_asset = assets.create(
                session_id=session_id,
                kind=AssetKind.INPUT,
                object_key=f"uploads/input/{session_id}.png",
                mime_type="image/png",
                width=256,
                height=256,
                byte_size=len(input_data),
                sha256="1" * 64,
                metadata={"normalized": True},
                expires_at=expires,
            )
            reference_asset = assets.create(
                session_id=session_id,
                kind=AssetKind.REFERENCE,
                object_key=f"uploads/reference/{session_id}.png",
                mime_type="image/png",
                width=256,
                height=256,
                byte_size=len(reference_data),
                sha256="2" * 64,
                metadata={"normalized": True},
                expires_at=expires,
            )
            job = JobRepository(db).create(
                session_id=session_id,
                input_asset_id=input_asset.id,
                reference_asset_id=reference_asset.id,
                style_id=None,
                algorithm_profile=AlgorithmProfile.PAPER_EXACT,
                settings={
                    "algorithm_profile": "paper_exact",
                    "transfer_strength": 1.0,
                    "residual_strength": 1.0,
                    "global_range_mix": 0.25,
                    "eye_highlights": True,
                    "background_mode": "KEEP",
                    "background_color": None,
                    "dense_alignment": False,
                    "processing_long_edge": 512,
                    "output_format": "PNG",
                    "jpeg_quality": 95,
                    "debug_artifacts": debug_artifacts,
                    "random_seed": 0,
                },
                expires_at=expires,
            )
        self.storage.put_bytes(input_asset.object_key, input_data, "image/png")
        self.storage.put_bytes(reference_asset.object_key, reference_data, "image/png")
        return str(job.id), input_asset.object_key, reference_asset.object_key

    def progress(self, job_id: str) -> dict[str, object] | None:
        raw = self.redis.values.get(f"job:{job_id}:progress")
        return json.loads(raw) if raw else None


@pytest.fixture
def worker(tmp_path: Any) -> Iterator[WorkerHarness]:
    settings = Settings(
        app_env="test",
        database_url="sqlite+pysqlite://",
        model_dir=tmp_path,
        require_models_for_readiness=False,
        allow_heuristic_analyzer=True,
        initialize_storage_on_startup=False,
        cookie_secure_override=False,
        session_secret="worker-tests-only-secret",
    )
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    storage = MemoryObjectStorage()
    fake_redis = FakeRedis()
    infrastructure = WorkerInfrastructure(
        settings=settings,
        engine=engine,
        session_factory=factory,
        storage=storage,
        redis=cast(Redis, fake_redis),
    )
    try:
        yield WorkerHarness(infrastructure, engine, factory, storage, fake_redis)
    finally:
        engine.dispose()


def portrait_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(output, "PNG")
    return output.getvalue()
