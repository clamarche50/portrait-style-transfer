from __future__ import annotations

import uuid
from urllib.parse import urlencode

from portrait_api.config import Settings


def asset_content_url(
    settings: Settings,
    asset_id: uuid.UUID,
    *,
    download: bool = False,
) -> str:
    prefix = settings.public_api_base_url.rstrip("/")
    query: dict[str, str] = {}
    if download:
        query["download"] = "true"
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{prefix}/assets/{asset_id}/content{suffix}"
