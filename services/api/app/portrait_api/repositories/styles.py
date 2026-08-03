from __future__ import annotations

import uuid
from datetime import UTC, datetime

from portrait_api.models import Asset, Style, StyleExample
from sqlalchemy import or_, select
from sqlalchemy.orm import Session


class StyleRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_visible(self, session_id: uuid.UUID) -> list[Style]:
        return list(
            self.session.scalars(
                select(Style)
                .where(
                    Style.deleted_at.is_(None),
                    or_(Style.session_id == session_id, Style.is_public.is_(True)),
                )
                .order_by(Style.updated_at.desc())
            ).unique()
        )

    def get_visible(self, style_id: uuid.UUID, session_id: uuid.UUID) -> Style | None:
        return self.session.scalar(
            select(Style).where(
                Style.id == style_id,
                Style.deleted_at.is_(None),
                or_(Style.session_id == session_id, Style.is_public.is_(True)),
            )
        )

    def get_owned(self, style_id: uuid.UUID, session_id: uuid.UUID) -> Style | None:
        return self.session.scalar(
            select(Style).where(
                Style.id == style_id,
                Style.session_id == session_id,
                Style.deleted_at.is_(None),
            )
        )

    def create(
        self,
        *,
        session_id: uuid.UUID,
        name: str,
        description: str,
        rights_confirmed: bool,
        is_public: bool,
    ) -> Style:
        now = datetime.now(UTC)
        style = Style(
            session_id=session_id,
            name=name,
            description=description,
            rights_confirmed=rights_confirmed,
            is_public=is_public,
            created_at=now,
            updated_at=now,
        )
        self.session.add(style)
        self.session.flush()
        return style

    def add_example(self, style: Style, asset: Asset) -> StyleExample:
        example = StyleExample(
            style_id=style.id,
            asset_id=asset.id,
            quality={},
            created_at=datetime.now(UTC),
        )
        self.session.add(example)
        style.updated_at = datetime.now(UTC)
        self.session.flush()
        return example

    def get_example_owned(
        self, style_id: uuid.UUID, example_id: uuid.UUID, session_id: uuid.UUID
    ) -> StyleExample | None:
        return self.session.scalar(
            select(StyleExample)
            .join(Style, Style.id == StyleExample.style_id)
            .where(
                StyleExample.id == example_id,
                StyleExample.style_id == style_id,
                Style.session_id == session_id,
                Style.deleted_at.is_(None),
            )
        )
