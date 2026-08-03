from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from portrait_api.config import Settings
from portrait_api.dependencies import (
    get_db,
    get_progress_store,
    get_settings_from_app,
    get_storage,
)
from portrait_api.services.model_validation import verify_required_models
from portrait_api.services.redis_gateway import ProgressStore
from portrait_api.services.storage import ObjectStorage
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    request: Request,
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_storage),
    progress_store: ProgressStore = Depends(get_progress_store),
    settings: Settings = Depends(get_settings_from_app),
) -> JSONResponse:
    checks: dict[str, object] = {}
    ready_state = True
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        request.app.state.logger.warning(
            "readiness_database_failed",
            extra={"request_id": request.state.request_id, "error_type": type(exc).__name__},
        )
        checks["database"] = "unavailable"
        ready_state = False
    try:
        checks["redis"] = "ok" if await progress_store.ping() else "unavailable"
        ready_state = ready_state and checks["redis"] == "ok"
    except Exception as exc:
        request.app.state.logger.warning(
            "readiness_redis_failed",
            extra={"request_id": request.state.request_id, "error_type": type(exc).__name__},
        )
        checks["redis"] = "unavailable"
        ready_state = False
    try:
        checks["object_storage"] = "ok" if storage.ping() else "unavailable"
        ready_state = ready_state and checks["object_storage"] == "ok"
    except Exception as exc:
        request.app.state.logger.warning(
            "readiness_storage_failed",
            extra={"request_id": request.state.request_id, "error_type": type(exc).__name__},
        )
        checks["object_storage"] = "unavailable"
        ready_state = False

    if settings.require_models_for_readiness:
        model_validation = verify_required_models(settings)
        checks["models"] = "ok" if model_validation.valid else model_validation.public_details()
        ready_state = ready_state and model_validation.valid
    else:
        checks["models"] = "not_required"

    return JSONResponse(
        status_code=200 if ready_state else 503,
        content={"status": "ready" if ready_state else "not_ready", "checks": checks},
    )
