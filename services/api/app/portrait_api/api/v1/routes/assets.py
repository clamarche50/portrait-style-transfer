from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from portrait_api.config import Settings
from portrait_api.dependencies import (
    Principal,
    get_db,
    get_principal,
    get_settings_from_app,
    get_storage,
)
from portrait_api.errors import AppError
from portrait_api.metrics import STORAGE_ERRORS, UPLOAD_BYTES, VALIDATION_ERRORS
from portrait_api.models import Asset, AssetKind, StyleExample
from portrait_api.repositories import AssetRepository
from portrait_api.schemas.assets import AssetResponse, DownloadUrlResponse
from portrait_api.security import DownloadTokenSigner
from portrait_api.services.image_validation import ImageNormalizer
from portrait_api.services.storage import ObjectStorage
from portrait_api.time import is_expired
from portrait_api.urls import asset_content_url
from sqlalchemy import func, select
from sqlalchemy.orm import Session

router = APIRouter()

_USER_UPLOAD_KINDS = {AssetKind.INPUT, AssetKind.REFERENCE, AssetKind.STYLE_EXAMPLE}
_PREFIX = {
    AssetKind.INPUT: "uploads/input",
    AssetKind.REFERENCE: "uploads/reference",
    AssetKind.STYLE_EXAMPLE: "styles/examples",
}


def _owned_asset(db: Session, principal: Principal, asset_id: uuid.UUID) -> Asset:
    asset = AssetRepository(db).get_owned(asset_id, principal.session_id)
    if asset is None or is_expired(asset.expires_at):
        raise AppError("ASSET_NOT_FOUND", "The asset was not found.", 404)
    return asset


def _visible_asset(db: Session, principal: Principal, asset_id: uuid.UUID) -> Asset:
    asset = AssetRepository(db).get_visible(asset_id, principal.session_id)
    if asset is None or is_expired(asset.expires_at):
        raise AppError("ASSET_NOT_FOUND", "The asset was not found.", 404)
    return asset


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise AppError(
                "UPLOAD_TOO_LARGE",
                "The uploaded image exceeds the encoded-size limit.",
                413,
                {"max_bytes": max_bytes},
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
async def upload_asset(
    request: Request,
    kind: AssetKind = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings_from_app),
    storage: ObjectStorage = Depends(get_storage),
) -> AssetResponse:
    if kind not in _USER_UPLOAD_KINDS:
        raise AppError("INVALID_ASSET_KIND", "This asset kind cannot be uploaded directly.", 422)
    repository = AssetRepository(db)
    count, recent_bytes = repository.recent_usage(principal.session_id)
    if count >= settings.max_assets_per_session:
        raise AppError("ASSET_QUOTA_EXCEEDED", "The session asset quota has been reached.", 429)

    raw = await _read_limited(file, settings.max_upload_bytes)
    if recent_bytes + len(raw) > settings.max_session_upload_bytes_24h:
        raise AppError(
            "UPLOAD_QUOTA_EXCEEDED",
            "The 24-hour upload quota has been reached.",
            429,
        )
    try:
        normalized = ImageNormalizer(settings).normalize(raw, file.content_type)
    except AppError as exc:
        VALIDATION_ERRORS.labels(exc.code).inc()
        raise
    asset_id = uuid.uuid4()
    object_key = f"{_PREFIX[kind]}/{asset_id}.{normalized.extension}"
    try:
        storage.put_bytes(object_key, normalized.data, normalized.mime_type)
    except Exception as exc:
        STORAGE_ERRORS.labels("upload").inc()
        request.app.state.logger.exception(
            "asset_upload_failed",
            extra={"request_id": request.state.request_id, "error_type": type(exc).__name__},
        )
        raise AppError(
            "STORAGE_UNAVAILABLE", "The upload could not be stored. Try again later.", 503
        ) from exc
    try:
        asset = repository.create(
            asset_id=asset_id,
            session_id=principal.session_id,
            kind=kind,
            object_key=object_key,
            mime_type=normalized.mime_type,
            width=normalized.width,
            height=normalized.height,
            byte_size=len(normalized.data),
            sha256=normalized.sha256,
            metadata={"normalized": True, "source_format": normalized.source_format},
            expires_at=datetime.now(UTC) + timedelta(hours=settings.asset_ttl_hours),
        )
        db.commit()
    except Exception:
        db.rollback()
        try:
            storage.delete(object_key)
        except Exception:
            STORAGE_ERRORS.labels("rollback_delete").inc()
        raise
    UPLOAD_BYTES.observe(len(normalized.data))
    request.app.state.logger.info(
        "asset_uploaded",
        extra={
            "request_id": request.state.request_id,
            "session_hash": request.state.session_hash,
            "asset_id": str(asset.id),
            "kind": kind.value,
            "width": asset.width,
            "height": asset.height,
            "byte_size": asset.byte_size,
        },
    )
    return AssetResponse.from_entity(asset, preview_url=asset_content_url(settings, asset.id))


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(
    asset_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> AssetResponse:
    asset = _owned_asset(db, principal, asset_id)
    return AssetResponse.from_entity(
        asset, preview_url=asset_content_url(request.app.state.settings, asset.id)
    )


@router.get("/{asset_id}/content")
def asset_content(
    asset_id: uuid.UUID,
    request: Request,
    download: bool = False,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
    settings: Settings = Depends(get_settings_from_app),
) -> Response:
    asset = _visible_asset(db, principal, asset_id)
    signer = DownloadTokenSigner(settings)
    if download and not signer.valid(
        request.cookies.get(settings.download_cookie_name),
        asset.id,
        principal.session_id,
    ):
        raise AppError(
            "DOWNLOAD_LINK_INVALID",
            "The download link is invalid or has expired.",
            403,
        )
    try:
        payload = storage.get_bytes(asset.object_key)
    except Exception as exc:
        STORAGE_ERRORS.labels("content_read").inc()
        request.app.state.logger.exception(
            "asset_content_read_failed",
            extra={
                "request_id": request.state.request_id,
                "asset_id": str(asset.id),
                "error_type": type(exc).__name__,
            },
        )
        raise AppError("STORAGE_UNAVAILABLE", "The image is temporarily unavailable.", 503) from exc
    extension = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        asset.mime_type, "bin"
    )
    disposition = "attachment" if download else "inline"
    response = Response(
        content=payload,
        media_type=asset.mime_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{asset.id}.{extension}"',
            "Content-Length": str(len(payload)),
        },
    )
    if download:
        response.delete_cookie(
            settings.download_cookie_name,
            domain=settings.cookie_domain,
            path=signer.cookie_path(asset.id),
            secure=settings.cookie_secure,
            httponly=True,
            samesite="strict",
        )
    return response


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_asset(
    asset_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
) -> Response:
    repository = AssetRepository(db)
    asset = _owned_asset(db, principal, asset_id)
    if repository.is_used_by_active_job(asset.id):
        raise AppError("ASSET_IN_USE", "Cancel the active job before deleting this asset.", 409)
    if db.scalar(select(func.count(StyleExample.id)).where(StyleExample.asset_id == asset.id)):
        raise AppError(
            "ASSET_IN_STYLE",
            "Remove this example from its style collection before deleting the asset.",
            409,
        )
    try:
        storage.delete(asset.object_key)
    except Exception as exc:
        STORAGE_ERRORS.labels("delete").inc()
        request.app.state.logger.exception(
            "asset_delete_failed",
            extra={
                "request_id": request.state.request_id,
                "asset_id": str(asset.id),
                "error_type": type(exc).__name__,
            },
        )
        raise AppError(
            "STORAGE_DELETE_FAILED", "The asset could not be deleted. Try again later.", 503
        ) from exc
    repository.mark_deleted(asset)
    request.app.state.logger.info(
        "asset_deleted",
        extra={
            "request_id": request.state.request_id,
            "session_hash": request.state.session_hash,
            "asset_id": str(asset.id),
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{asset_id}/download-url", response_model=DownloadUrlResponse)
def asset_download_url(
    asset_id: uuid.UUID,
    response: Response,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings_from_app),
) -> DownloadUrlResponse:
    asset = _owned_asset(db, principal, asset_id)
    DownloadTokenSigner(settings).set_cookie(response, asset.id, principal.session_id)
    return DownloadUrlResponse(
        url=asset_content_url(settings, asset.id, download=True),
        expires_in_seconds=settings.signed_url_ttl_seconds,
        expires_at=datetime.now(UTC) + timedelta(seconds=settings.signed_url_ttl_seconds),
    )
