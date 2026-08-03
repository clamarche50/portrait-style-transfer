from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from fastapi import Request
from portrait_api.config import Settings
from portrait_api.services.queue import TaskQueue
from portrait_api.services.redis_gateway import ProgressStore
from portrait_api.services.storage import ObjectStorage
from sqlalchemy.orm import Session


@dataclass(frozen=True, slots=True)
class Principal:
    session_id: uuid.UUID
    owner_id: uuid.UUID | None = None


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_principal(request: Request) -> Principal:
    return Principal(session_id=request.state.session_id)


def get_settings_from_app(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_storage(request: Request) -> ObjectStorage:
    return cast(ObjectStorage, request.app.state.storage)


def get_progress_store(request: Request) -> ProgressStore:
    return cast(ProgressStore, request.app.state.progress_store)


def get_task_queue(request: Request) -> TaskQueue:
    return cast(TaskQueue, request.app.state.task_queue)
