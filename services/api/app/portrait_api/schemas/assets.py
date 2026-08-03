from __future__ import annotations

import uuid
from datetime import datetime

from portrait_api.models import Asset, AssetKind
from portrait_api.schemas.common import ApiModel


class AssetResponse(ApiModel):
    id: uuid.UUID
    kind: AssetKind
    mime_type: str
    width: int
    height: int
    byte_size: int
    sha256: str
    metadata: dict[str, object]
    preview_url: str | None = None
    analysis: dict[str, object] | None = None
    expires_at: datetime
    created_at: datetime

    @classmethod
    def from_entity(cls, asset: Asset, *, preview_url: str | None = None) -> AssetResponse:
        return cls(
            id=asset.id,
            kind=asset.kind,
            mime_type=asset.mime_type,
            width=asset.width,
            height=asset.height,
            byte_size=asset.byte_size,
            sha256=asset.sha256,
            metadata=asset.metadata_json,
            preview_url=preview_url,
            analysis=None,
            expires_at=asset.expires_at,
            created_at=asset.created_at,
        )


class DownloadUrlResponse(ApiModel):
    url: str
    expires_in_seconds: int
    expires_at: datetime
