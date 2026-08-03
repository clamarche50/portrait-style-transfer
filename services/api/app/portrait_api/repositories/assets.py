from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from portrait_api.models import Asset, AssetKind, Job, JobStatus, Style, StyleExample
from portrait_api.models.enums import TERMINAL_JOB_STATUSES
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session


class AssetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        asset_id: uuid.UUID | None = None,
        session_id: uuid.UUID,
        kind: AssetKind,
        object_key: str,
        mime_type: str,
        width: int,
        height: int,
        byte_size: int,
        sha256: str,
        metadata: dict[str, Any],
        expires_at: datetime,
        owner_id: uuid.UUID | None = None,
    ) -> Asset:
        asset = Asset(
            id=asset_id or uuid.uuid4(),
            owner_id=owner_id,
            session_id=session_id,
            kind=kind,
            object_key=object_key,
            mime_type=mime_type,
            width=width,
            height=height,
            byte_size=byte_size,
            sha256=sha256,
            metadata_json=metadata,
            expires_at=expires_at,
            created_at=datetime.now(UTC),
        )
        self.session.add(asset)
        self.session.flush()
        return asset

    def get_owned(self, asset_id: uuid.UUID, session_id: uuid.UUID) -> Asset | None:
        return self.session.scalar(
            select(Asset).where(
                Asset.id == asset_id,
                Asset.session_id == session_id,
                Asset.deleted_at.is_(None),
            )
        )

    def get_visible(self, asset_id: uuid.UUID, session_id: uuid.UUID) -> Asset | None:
        return self.session.scalar(
            select(Asset)
            .outerjoin(StyleExample, StyleExample.asset_id == Asset.id)
            .outerjoin(Style, Style.id == StyleExample.style_id)
            .where(
                Asset.id == asset_id,
                Asset.deleted_at.is_(None),
                or_(
                    Asset.session_id == session_id,
                    (Style.is_public.is_(True) & Style.deleted_at.is_(None)),
                ),
            )
        )

    def recent_usage(self, session_id: uuid.UUID, *, hours: int = 24) -> tuple[int, int]:
        since = datetime.now(UTC) - timedelta(hours=hours)
        count, total = self.session.execute(
            select(func.count(Asset.id), func.coalesce(func.sum(Asset.byte_size), 0)).where(
                Asset.session_id == session_id,
                Asset.deleted_at.is_(None),
                Asset.created_at >= since,
            )
        ).one()
        return int(count), int(total)

    def is_used_by_active_job(self, asset_id: uuid.UUID) -> bool:
        active = tuple(status for status in JobStatus if status not in TERMINAL_JOB_STATUSES)
        return bool(
            self.session.scalar(
                select(func.count(Job.id)).where(
                    or_(Job.input_asset_id == asset_id, Job.reference_asset_id == asset_id),
                    Job.status.in_(active),
                    Job.deleted_at.is_(None),
                )
            )
        )

    def is_used_by_other_job(self, asset_id: uuid.UUID, *, excluding_job_id: uuid.UUID) -> bool:
        return bool(
            self.session.scalar(
                select(func.count(Job.id)).where(
                    Job.id != excluding_job_id,
                    or_(Job.input_asset_id == asset_id, Job.reference_asset_id == asset_id),
                    Job.deleted_at.is_(None),
                )
            )
        )

    def is_style_example(self, asset_id: uuid.UUID) -> bool:
        return bool(
            self.session.scalar(
                select(func.count(StyleExample.id)).where(StyleExample.asset_id == asset_id)
            )
        )

    def is_style_example_elsewhere(
        self,
        asset_id: uuid.UUID,
        *,
        excluding_example_id: uuid.UUID,
    ) -> bool:
        return bool(
            self.session.scalar(
                select(func.count(StyleExample.id)).where(
                    StyleExample.asset_id == asset_id,
                    StyleExample.id != excluding_example_id,
                )
            )
        )

    def mark_deleted(self, asset: Asset, *, when: datetime | None = None) -> None:
        asset.deleted_at = when or datetime.now(UTC)
        self.session.flush()

    def expired_batch(self, *, limit: int = 100) -> list[Asset]:
        now = datetime.now(UTC)
        return list(
            self.session.scalars(
                select(Asset)
                .where(Asset.deleted_at.is_(None), Asset.expires_at <= now)
                .order_by(Asset.expires_at)
                .limit(limit)
            )
        )
