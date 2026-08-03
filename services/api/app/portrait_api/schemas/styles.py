from __future__ import annotations

import uuid
from datetime import datetime

from portrait_api.models import Style, StyleExample
from portrait_api.schemas.common import ApiModel
from pydantic import Field, model_validator


class CreateStyleRequest(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    rights_confirmed: bool
    is_public: bool = False

    @model_validator(mode="after")
    def rights_required(self) -> CreateStyleRequest:
        if not self.rights_confirmed:
            raise ValueError("Rights confirmation is required to create a style")
        return self


class UpdateStyleRequest(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    rights_confirmed: bool | None = None
    is_public: bool | None = None

    @model_validator(mode="after")
    def publication_requires_rights(self) -> UpdateStyleRequest:
        if self.is_public is True and self.rights_confirmed is False:
            raise ValueError("A public style must have confirmed rights")
        return self


class StyleExampleResponse(ApiModel):
    id: uuid.UUID
    asset_id: uuid.UUID
    quality: dict[str, object]
    indexed: bool
    created_at: datetime

    @classmethod
    def from_entity(cls, example: StyleExample) -> StyleExampleResponse:
        return cls(
            id=example.id,
            asset_id=example.asset_id,
            quality=example.quality,
            indexed=example.feature_object_key is not None,
            created_at=example.created_at,
        )


class StyleResponse(ApiModel):
    id: uuid.UUID
    name: str
    description: str
    rights_confirmed: bool
    is_public: bool
    examples: list[StyleExampleResponse]
    example_count: int
    preview_url: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entity(cls, style: Style, *, preview_url: str | None = None) -> StyleResponse:
        return cls(
            id=style.id,
            name=style.name,
            description=style.description,
            rights_confirmed=style.rights_confirmed,
            is_public=style.is_public,
            examples=[StyleExampleResponse.from_entity(example) for example in style.examples],
            example_count=len(style.examples),
            preview_url=preview_url,
            created_at=style.created_at,
            updated_at=style.updated_at,
        )


class AddStyleExampleRequest(ApiModel):
    asset_id: uuid.UUID


class RankStyleRequest(ApiModel):
    input_asset_id: uuid.UUID
    limit: int = Field(default=3, ge=1, le=20)


class RankedStyleExample(ApiModel):
    example_id: uuid.UUID
    asset_id: uuid.UUID
    score: float
    diagnostics: dict[str, object]


class RankStyleResponse(ApiModel):
    style_id: uuid.UUID
    input_asset_id: uuid.UUID
    results: list[RankedStyleExample]
