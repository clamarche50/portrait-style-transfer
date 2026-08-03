from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", use_enum_values=True)


class MessageResponse(ApiModel):
    message: str
