from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from portrait_api.config import Settings, get_settings
from portrait_api.db import build_engine, build_session_factory
from portrait_api.services.storage import Boto3ObjectStorage, ObjectStorage
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

_RELEASE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass(slots=True)
class WorkerInfrastructure:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    storage: ObjectStorage
    redis: Redis

    def acquire_job_lock(self, job_id: str) -> str | None:
        token = secrets.token_hex(16)
        acquired = self.redis.set(
            f"job:{job_id}:lock",
            token,
            nx=True,
            ex=self.settings.job_lock_ttl_seconds,
        )
        return token if acquired else None

    def refresh_job_lock(self, job_id: str, token: str) -> bool:
        key = f"job:{job_id}:lock"
        if self.redis.get(key) != token:
            return False
        return bool(self.redis.expire(key, self.settings.job_lock_ttl_seconds))

    def release_job_lock(self, job_id: str, token: str) -> None:
        self.redis.eval(_RELEASE_LOCK, 1, f"job:{job_id}:lock", token)

    def set_progress(self, job_id: str, event: dict[str, Any]) -> None:
        payload = json.dumps(event, separators=(",", ":"), default=str)
        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.set(
                f"job:{job_id}:progress",
                payload,
                ex=self.settings.redis_progress_ttl_seconds,
            )
            pipeline.publish(f"job:{job_id}:events", payload)
            pipeline.execute()

    def cancel_requested(self, job_id: str) -> bool:
        return bool(self.redis.exists(f"job:{job_id}:cancel"))


def build_infrastructure(settings: Settings | None = None) -> WorkerInfrastructure:
    settings = settings or get_settings()
    engine = build_engine(settings.database_url)
    return WorkerInfrastructure(
        settings=settings,
        engine=engine,
        session_factory=build_session_factory(engine),
        storage=Boto3ObjectStorage(settings),
        redis=Redis.from_url(settings.redis_url, decode_responses=True),
    )


@lru_cache(maxsize=1)
def get_infrastructure() -> WorkerInfrastructure:
    return build_infrastructure()
