from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TransferRequestSettings(BaseModel):
    """AI-native controls accepted by the isolated inference service."""

    model_config = ConfigDict(extra="forbid")

    algorithm_profile: Literal["ai_instantstyle_v1"] = "ai_instantstyle_v1"
    style_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    structure_strength: float = Field(default=0.9, ge=0.0, le=1.0)
    inference_steps: int = Field(default=30, ge=10, le=50)
    random_seed: int = Field(default=0, ge=0, le=2_147_483_647)


@dataclass(frozen=True, slots=True)
class TransferOutput:
    image_png: bytes
    diagnostics: dict[str, Any]


class EngineFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
