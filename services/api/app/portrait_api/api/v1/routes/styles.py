from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Response, status
from portrait_api.config import Settings
from portrait_api.dependencies import (
    Principal,
    get_db,
    get_principal,
    get_settings_from_app,
    get_storage,
    get_task_queue,
)
from portrait_api.errors import AppError
from portrait_api.metrics import STORAGE_ERRORS
from portrait_api.models import AssetKind, Job, JobStatus, Style, StyleExample
from portrait_api.models.enums import TERMINAL_JOB_STATUSES
from portrait_api.repositories import AssetRepository, StyleRepository
from portrait_api.schemas.common import MessageResponse
from portrait_api.schemas.styles import (
    AddStyleExampleRequest,
    CreateStyleRequest,
    RankedStyleExample,
    RankStyleRequest,
    RankStyleResponse,
    StyleExampleResponse,
    StyleResponse,
    UpdateStyleRequest,
)
from portrait_api.services.image_validation import ImageNormalizer
from portrait_api.services.queue import TaskQueue
from portrait_api.services.ranking import StyleRankingService
from portrait_api.services.storage import ObjectStorage
from portrait_api.urls import asset_content_url
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

router = APIRouter()


def _visible_style(db: Session, principal: Principal, style_id: uuid.UUID) -> Style:
    style = StyleRepository(db).get_visible(style_id, principal.session_id)
    if style is None:
        raise AppError("STYLE_NOT_FOUND", "The style was not found.", 404)
    return style


def _owned_style(db: Session, principal: Principal, style_id: uuid.UUID) -> Style:
    style = StyleRepository(db).get_owned(style_id, principal.session_id)
    if style is None:
        raise AppError("STYLE_NOT_FOUND", "The style was not found.", 404)
    return style


def _style_response(
    style: Style,
    config: Settings | None = None,
) -> StyleResponse:
    preview_url = None
    available = [example for example in style.examples if example.asset.deleted_at is None]
    if config is not None and available:
        preview_url = asset_content_url(config, available[0].asset.id)
    return StyleResponse.from_entity(style, preview_url=preview_url)


def _active_example_uses(db: Session, example: StyleExample) -> int:
    active_statuses = tuple(status for status in JobStatus if status not in TERMINAL_JOB_STATUSES)
    selected_count = int(
        db.scalar(
            select(func.count(Job.id)).where(
                Job.selected_style_example_id == example.id,
                Job.status.in_(active_statuses),
                Job.deleted_at.is_(None),
            )
        )
        or 0
    )
    asset_count = int(
        db.scalar(
            select(func.count(Job.id)).where(
                (Job.input_asset_id == example.asset_id)
                | (Job.reference_asset_id == example.asset_id),
                Job.status.in_(active_statuses),
                Job.deleted_at.is_(None),
            )
        )
        or 0
    )
    return selected_count + asset_count


def _delete_style_example_records(
    *,
    db: Session,
    principal: Principal,
    storage: ObjectStorage,
    example: StyleExample,
) -> bool:
    """Remove one example and its now-unreferenced, session-owned source asset."""
    asset = example.asset
    asset_repository = AssetRepository(db)
    delete_asset = (
        asset.deleted_at is None
        and asset.session_id == principal.session_id
        and not asset_repository.is_style_example_elsewhere(
            asset.id,
            excluding_example_id=example.id,
        )
        and not asset_repository.is_used_by_active_job(asset.id)
    )
    keys = [example.feature_object_key]
    if delete_asset:
        keys.append(asset.object_key)
    for key in keys:
        if not key:
            continue
        try:
            storage.delete(key)
        except Exception as exc:
            STORAGE_ERRORS.labels("style_asset_delete").inc()
            raise AppError(
                "STORAGE_DELETE_FAILED",
                "The style example could not be deleted. Try again later.",
                503,
            ) from exc
    try:
        storage.delete_prefix(f"styles/{example.style_id}/examples/{example.id}/")
    except Exception as exc:
        STORAGE_ERRORS.labels("style_derived_delete").inc()
        raise AppError(
            "STORAGE_DELETE_FAILED",
            "The style example could not be deleted. Try again later.",
            503,
        ) from exc

    # Keep historical jobs valid on databases where FK ON DELETE actions are disabled.
    db.execute(
        update(Job)
        .where(Job.selected_style_example_id == example.id)
        .values(selected_style_example_id=None)
        .execution_options(synchronize_session=False)
    )
    db.delete(example)
    db.flush()
    if delete_asset:
        asset_repository.mark_deleted(asset)
    return delete_asset


@router.get("", response_model=list[StyleResponse])
def list_styles(
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
) -> list[StyleResponse]:
    return [
        _style_response(style, config)
        for style in StyleRepository(db).list_visible(principal.session_id)
    ]


@router.post("", response_model=StyleResponse, status_code=status.HTTP_201_CREATED)
def create_style(
    payload: CreateStyleRequest,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> StyleResponse:
    style = StyleRepository(db).create(
        session_id=principal.session_id,
        name=payload.name,
        description=payload.description,
        rights_confirmed=payload.rights_confirmed,
        is_public=payload.is_public,
    )
    if style.is_public:
        request.app.state.logger.info(
            "style_published",
            extra={
                "request_id": request.state.request_id,
                "session_hash": request.state.session_hash,
                "style_id": str(style.id),
            },
        )
    return StyleResponse.from_entity(style)


@router.get("/{style_id}", response_model=StyleResponse)
def get_style(
    style_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
) -> StyleResponse:
    return _style_response(_visible_style(db, principal, style_id), config)


@router.patch("/{style_id}", response_model=StyleResponse)
def update_style(
    style_id: uuid.UUID,
    payload: UpdateStyleRequest,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> StyleResponse:
    style = _owned_style(db, principal, style_id)
    updates = payload.model_dump(exclude_unset=True)
    next_rights = updates.get("rights_confirmed", style.rights_confirmed)
    next_public = updates.get("is_public", style.is_public)
    if next_public and not next_rights:
        raise AppError("RIGHTS_REQUIRED", "Rights confirmation is required for publication.", 422)
    was_public = style.is_public
    for field, value in updates.items():
        setattr(style, field, value)
    style.updated_at = datetime.now(UTC)
    db.flush()
    if style.is_public != was_public:
        request.app.state.logger.info(
            "style_publication_changed",
            extra={
                "request_id": request.state.request_id,
                "session_hash": request.state.session_hash,
                "style_id": str(style.id),
                "is_public": style.is_public,
            },
        )
    return StyleResponse.from_entity(style)


@router.delete("/{style_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_style(
    style_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
) -> Response:
    style = _owned_style(db, principal, style_id)
    active_statuses = tuple(status for status in JobStatus if status not in TERMINAL_JOB_STATUSES)
    active_jobs = int(
        db.scalar(
            select(func.count(Job.id)).where(
                Job.style_id == style.id,
                Job.status.in_(active_statuses),
                Job.deleted_at.is_(None),
            )
        )
        or 0
    )
    if active_jobs:
        raise AppError("STYLE_IN_USE", "Cancel active jobs before deleting this style.", 409)
    examples = list(style.examples)
    if any(_active_example_uses(db, example) for example in examples):
        raise AppError("STYLE_IN_USE", "Cancel active jobs before deleting this style.", 409)
    deleted_assets = sum(
        _delete_style_example_records(
            db=db,
            principal=principal,
            storage=storage,
            example=example,
        )
        for example in examples
    )
    style.deleted_at = datetime.now(UTC)
    style.is_public = False
    db.flush()
    request.app.state.logger.info(
        "style_deleted",
        extra={
            "request_id": request.state.request_id,
            "session_hash": request.state.session_hash,
            "style_id": str(style.id),
            "deleted_example_assets": deleted_assets,
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{style_id}/examples",
    response_model=StyleExampleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_style_example(
    style_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    config: Settings = Depends(get_settings_from_app),
    storage: ObjectStorage = Depends(get_storage),
    queue: TaskQueue = Depends(get_task_queue),
) -> StyleExampleResponse:
    style = _owned_style(db, principal, style_id)
    if not style.rights_confirmed:
        raise AppError("RIGHTS_REQUIRED", "Confirm image rights before adding examples.", 422)
    asset_repository = AssetRepository(db)
    created_object_key: str | None = None
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise AppError("FILE_REQUIRED", "A portrait image file is required.", 422)
        chunks: list[bytes] = []
        total = 0
        try:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > config.max_upload_bytes:
                    raise AppError(
                        "UPLOAD_TOO_LARGE",
                        "The uploaded image exceeds the encoded-size limit.",
                        413,
                        {"max_bytes": config.max_upload_bytes},
                    )
                chunks.append(chunk)
        finally:
            await upload.close()
        count, recent_bytes = asset_repository.recent_usage(principal.session_id)
        if count >= config.max_assets_per_session:
            raise AppError("ASSET_QUOTA_EXCEEDED", "The session asset quota has been reached.", 429)
        raw = b"".join(chunks)
        if recent_bytes + len(raw) > config.max_session_upload_bytes_24h:
            raise AppError(
                "UPLOAD_QUOTA_EXCEEDED", "The 24-hour upload quota has been reached.", 429
            )
        normalized = ImageNormalizer(config).normalize(raw, upload.content_type)
        asset_id = uuid.uuid4()
        created_object_key = f"styles/examples/{asset_id}.{normalized.extension}"
        try:
            storage.put_bytes(created_object_key, normalized.data, normalized.mime_type)
        except Exception as exc:
            raise AppError(
                "STORAGE_UNAVAILABLE",
                "The style example could not be stored. Try again later.",
                503,
            ) from exc
        asset = asset_repository.create(
            asset_id=asset_id,
            session_id=principal.session_id,
            kind=AssetKind.STYLE_EXAMPLE,
            object_key=created_object_key,
            mime_type=normalized.mime_type,
            width=normalized.width,
            height=normalized.height,
            byte_size=len(normalized.data),
            sha256=normalized.sha256,
            metadata={"normalized": True, "source_format": normalized.source_format},
            expires_at=datetime.now(UTC) + timedelta(hours=config.asset_ttl_hours),
        )
    elif content_type.startswith("application/json"):
        try:
            payload = AddStyleExampleRequest.model_validate(await request.json())
        except (ValidationError, ValueError, TypeError) as exc:
            raise AppError(
                "VALIDATION_ERROR", "The style example request is invalid.", 422
            ) from exc
        requested_asset = asset_repository.get_owned(payload.asset_id, principal.session_id)
        if requested_asset is None or requested_asset.kind != AssetKind.STYLE_EXAMPLE:
            raise AppError("STYLE_ASSET_NOT_FOUND", "The style example asset was not found.", 404)
        asset = requested_asset
    else:
        raise AppError(
            "UNSUPPORTED_MEDIA_TYPE",
            "Use multipart/form-data with a file or application/json with an asset_id.",
            415,
        )
    repository = StyleRepository(db)
    try:
        example = repository.add_example(style, asset)
        key = StyleRankingService(storage).index_example(style.id, example, asset)
        example.feature_object_key = key
        example.quality = {
            "width": asset.width,
            "height": asset.height,
            "indexed": True,
            "full_ingestion": "QUEUED",
        }
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        if created_object_key:
            storage.delete(created_object_key)
        raise AppError("STYLE_EXAMPLE_EXISTS", "This asset is already in the style.", 409) from exc
    except Exception:
        db.rollback()
        if created_object_key:
            storage.delete(created_object_key)
        if "example" in locals():
            storage.delete(f"styles/{style.id}/features/{example.id}.npy")
        raise
    # Commit before publishing so a fast worker can always observe the example.
    db.commit()
    try:
        queue.enqueue_style_index(str(style.id))
    except Exception:
        example.quality = {**(example.quality or {}), "full_ingestion": "QUEUE_FAILED"}
        db.commit()
        request.app.state.logger.exception(
            "style_ingestion_enqueue_failed",
            extra={
                "request_id": request.state.request_id,
                "session_hash": request.state.session_hash,
                "style_id": str(style.id),
                "example_id": str(example.id),
            },
        )
    return StyleExampleResponse.from_entity(example)


@router.delete("/{style_id}/examples/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_style_example(
    style_id: uuid.UUID,
    example_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
) -> Response:
    _owned_style(db, principal, style_id)
    example = StyleRepository(db).get_example_owned(style_id, example_id, principal.session_id)
    if example is None:
        raise AppError("STYLE_EXAMPLE_NOT_FOUND", "The style example was not found.", 404)
    if _active_example_uses(db, example):
        raise AppError("STYLE_EXAMPLE_IN_USE", "The example is in use by an active job.", 409)
    _delete_style_example_records(
        db=db,
        principal=principal,
        storage=storage,
        example=example,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{style_id}/reindex", response_model=MessageResponse, status_code=status.HTTP_202_ACCEPTED
)
def reindex_style(
    style_id: uuid.UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    queue: TaskQueue = Depends(get_task_queue),
) -> MessageResponse:
    style = _owned_style(db, principal, style_id)
    if not style.examples:
        raise AppError("STYLE_EMPTY", "The style has no examples to index.", 422)
    queue.enqueue_style_index(str(style.id))
    return MessageResponse(message="Style reindexing was queued.")


@router.post("/{style_id}/rank", response_model=RankStyleResponse)
def rank_style(
    style_id: uuid.UUID,
    payload: RankStyleRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    storage: ObjectStorage = Depends(get_storage),
) -> RankStyleResponse:
    style = _visible_style(db, principal, style_id)
    input_asset = AssetRepository(db).get_owned(payload.input_asset_id, principal.session_id)
    if input_asset is None or input_asset.kind != AssetKind.INPUT:
        raise AppError("INPUT_ASSET_NOT_FOUND", "The input portrait was not found.", 404)
    if not style.examples:
        raise AppError("STYLE_EMPTY", "The style has no examples.", 422)
    try:
        results = StyleRankingService(storage).rank(
            input_asset,
            style.examples,
            limit=payload.limit,
        )
    except Exception as exc:
        raise AppError(
            "STYLE_RANKING_FAILED",
            "The style examples could not be ranked for this portrait.",
            422,
        ) from exc
    return RankStyleResponse(
        style_id=style.id,
        input_asset_id=input_asset.id,
        results=[
            RankedStyleExample(
                example_id=item.example_id,
                asset_id=item.asset_id,
                score=item.score,
                diagnostics={
                    "metric": "weighted_compatibility_v1",
                    "analysis_mode": "lightweight_image_statistics",
                    "rank": index + 1,
                    "components": {
                        "local_energy_ncc": item.energy_ncc,
                        "pose_similarity": item.pose_similarity,
                        "landmark_shape_similarity": item.landmark_shape_similarity,
                        "photometric_compatibility": item.photometric_compatibility,
                        "mask_quality": item.mask_quality,
                    },
                },
            )
            for index, item in enumerate(results)
        ],
    )
