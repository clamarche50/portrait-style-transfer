from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from .config import ServiceConfig
from .contracts import EngineFailure, TransferRequestSettings
from .preprocessing import decode_image
from .runtime import ENGINE_ID, EngineRuntime

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger("portrait_ai_engine")
CONFIG = ServiceConfig.from_environment()
RUNTIME = EngineRuntime(CONFIG)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if CONFIG.eager_load:
        await asyncio.to_thread(RUNTIME.ensure_loaded)
    yield


app = FastAPI(
    title="Portrait InstantStyle inference service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


async def _read_bounded(upload: UploadFile) -> bytes:
    payload = await upload.read(CONFIG.max_upload_bytes + 1)
    if len(payload) > CONFIG.max_upload_bytes:
        raise EngineFailure("AI_INVALID_IMAGE", "Image payload is too large")
    return payload


def _error(exc: EngineFailure, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
            }
        },
    )


def _authorize(authorization: str | None) -> None:
    if CONFIG.api_token is None:
        return
    if authorization is None or not hmac.compare_digest(
        authorization,
        f"Bearer {CONFIG.api_token}",
    ):
        raise HTTPException(
            status_code=401, detail="Invalid inference-service credentials"
        )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> Any:
    try:
        return await asyncio.to_thread(RUNTIME.readiness)
    except EngineFailure as exc:
        return _error(exc, 503)


@app.post("/v1/transfer")
async def transfer(
    content: Annotated[UploadFile, File()],
    style: Annotated[UploadFile, File()],
    settings: Annotated[str, Form()],
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    _authorize(authorization)
    try:
        parsed_settings = TransferRequestSettings.model_validate_json(settings)
        content_bytes, style_bytes = await asyncio.gather(
            _read_bounded(content), _read_bounded(style)
        )
        content_image = decode_image(
            content_bytes,
            max_bytes=CONFIG.max_upload_bytes,
            max_pixels=CONFIG.max_decoded_pixels,
        )
        style_image = decode_image(
            style_bytes,
            max_bytes=CONFIG.max_upload_bytes,
            max_pixels=CONFIG.max_decoded_pixels,
        )
        result = await asyncio.to_thread(
            RUNTIME.transfer,
            content=content_image,
            style=style_image,
            settings=parsed_settings,
        )
        diagnostic_json = json.dumps(
            result.diagnostics, separators=(",", ":"), sort_keys=True
        )
        diagnostic_header = (
            base64.urlsafe_b64encode(diagnostic_json.encode()).decode().rstrip("=")
        )
        return Response(
            content=result.image_png,
            media_type="image/png",
            headers={
                "X-Portrait-Engine": ENGINE_ID,
                "X-Portrait-Diagnostics": diagnostic_header,
                "Cache-Control": "no-store",
            },
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=exc.errors(include_url=False)
        ) from exc
    except EngineFailure as exc:
        status = 503 if exc.code.startswith(("AI_CUDA", "AI_MODEL", "AI_GPU")) else 422
        return _error(exc, status)
    except Exception:
        LOGGER.exception("Unhandled InstantStyle inference failure")
        return _error(
            # Deterministic crashes retry with the identical tensors and
            # inputs, so retrying only loops until exhaustion. Surface the
            # failure immediately and let the user fix the input or config.
            EngineFailure(
                "AI_INFERENCE_FAILED", "The AI engine failed safely", retryable=False
            ),
            500,
        )
