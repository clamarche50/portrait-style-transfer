from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Literal

from portrait_api.models import (
    AlgorithmProfile,
    BackgroundMode,
    Job,
    JobStatus,
    OutputFormat,
    ProcessingStage,
)
from portrait_api.schemas.common import ApiModel
from pydantic import Field, model_validator


class TransferSettingsRequest(ApiModel):
    algorithm_profile: AlgorithmProfile = AlgorithmProfile.PAPER_EXACT
    transfer_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    residual_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    global_range_mix: float = Field(default=0.25, ge=0.0, le=1.0)
    eye_highlights: bool = True
    background_mode: BackgroundMode = BackgroundMode.KEEP
    background_color: str | None = None
    dense_alignment: bool = True
    processing_long_edge: int = Field(default=1280, ge=512, le=2048)
    output_format: OutputFormat = OutputFormat.PNG
    jpeg_quality: int = Field(default=95, ge=70, le=100)
    debug_artifacts: bool = False
    random_seed: int = Field(default=0, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def validate_background(self) -> TransferSettingsRequest:
        if self.algorithm_profile != AlgorithmProfile.PAPER_EXACT:
            raise ValueError("Only the paper_exact profile is available through the public API")
        if self.background_mode == BackgroundMode.SOLID:
            if not self.background_color or not re.fullmatch(
                r"#[0-9a-fA-F]{6}", self.background_color
            ):
                raise ValueError("background_color must be #RRGGBB for SOLID mode")
        elif self.background_color is not None:
            raise ValueError("background_color is only valid for SOLID mode")
        return self


class CreateJobRequest(ApiModel):
    input_asset_id: uuid.UUID
    reference_asset_id: uuid.UUID | None = None
    style_id: uuid.UUID | None = None
    settings: TransferSettingsRequest = Field(default_factory=TransferSettingsRequest)

    @model_validator(mode="after")
    def exactly_one_reference(self) -> CreateJobRequest:
        if (self.reference_asset_id is None) == (self.style_id is None):
            raise ValueError("Exactly one of reference_asset_id and style_id is required")
        return self


class JobWarning(ApiModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


def _warnings(value: object) -> list[JobWarning]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[JobWarning] = []
    for item in value:
        if isinstance(item, dict):
            code = str(item.get("code", "PROCESSING_WARNING"))
            message = str(item.get("message", code.replace("_", " ").capitalize()))
            severity = str(item.get("severity", "warning"))
            if severity not in {"info", "warning", "error"}:
                severity = "warning"
            normalized.append(JobWarning(code=code, message=message, severity=severity))
        else:
            raw = str(item)
            code = re.sub(r"[^A-Z0-9]+", "_", raw.upper()).strip("_")
            normalized.append(
                JobWarning(
                    code=code or "PROCESSING_WARNING",
                    message=raw.replace("_", " ").strip().capitalize(),
                )
            )
    return normalized


class JobResponse(ApiModel):
    id: uuid.UUID
    status: JobStatus
    stage: ProcessingStage
    progress: int
    input_asset_id: uuid.UUID
    reference_asset_id: uuid.UUID | None
    style_id: uuid.UUID | None
    selected_style_example_id: uuid.UUID | None
    algorithm_profile: AlgorithmProfile
    settings: dict[str, object]
    corrections: list[dict[str, object]]
    diagnostics_summary: dict[str, object]
    error_code: str | None
    error_message: str | None
    error_message_safe: str | None
    output_asset_id: uuid.UUID | None = None
    output_url: str | None = None
    input_preview_url: str | None = None
    reference_preview_url: str | None = None
    warnings: list[JobWarning] = Field(default_factory=list)
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    expires_at: datetime

    @classmethod
    def from_entity(
        cls,
        job: Job,
        *,
        output_asset_id: uuid.UUID | None = None,
        output_url: str | None = None,
        input_preview_url: str | None = None,
        reference_preview_url: str | None = None,
    ) -> JobResponse:
        diagnostics = job.diagnostics or {}
        public_summary = diagnostics.get("summary", {})
        if not isinstance(public_summary, dict):
            public_summary = {}
        return cls(
            id=job.id,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            input_asset_id=job.input_asset_id,
            reference_asset_id=job.reference_asset_id,
            style_id=job.style_id,
            selected_style_example_id=job.selected_style_example_id,
            algorithm_profile=job.algorithm_profile,
            settings=job.settings,
            corrections=job.corrections or [],
            diagnostics_summary=public_summary,
            error_code=job.error_code,
            error_message=job.error_message_safe,
            error_message_safe=job.error_message_safe,
            output_asset_id=output_asset_id,
            output_url=output_url,
            input_preview_url=input_preview_url,
            reference_preview_url=reference_preview_url,
            warnings=_warnings(public_summary.get("warnings", [])),
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            expires_at=job.expires_at,
        )


Coordinate = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Coordinate, Coordinate]


class MaskCorrection(ApiModel):
    type: Literal["mask"]
    operation: Literal["ADD", "REMOVE"]
    radius: float = Field(ge=0.001, le=0.5)
    points: list[Point] = Field(min_length=1, max_length=4096)


class AlignmentCorrection(ApiModel):
    type: Literal["alignment"]
    input_points: list[Point] = Field(min_length=1, max_length=128)
    reference_points: list[Point] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def matching_points(self) -> AlignmentCorrection:
        if len(self.input_points) != len(self.reference_points):
            raise ValueError("input_points and reference_points must have the same length")
        return self


class GainCopyCorrection(ApiModel):
    type: Literal["gain_copy"]
    source_polygon: list[Point] = Field(min_length=3, max_length=256)
    target_polygon: list[Point] = Field(min_length=3, max_length=256)
    levels: list[int] = Field(default_factory=lambda: list(range(6)), min_length=1, max_length=6)

    @model_validator(mode="after")
    def valid_levels(self) -> GainCopyCorrection:
        if len(set(self.levels)) != len(self.levels) or any(
            level not in range(6) for level in self.levels
        ):
            raise ValueError("levels must contain unique values from 0 through 5")
        return self


class EyeCorrection(ApiModel):
    type: Literal["eye"]
    eye: Literal["LEFT", "RIGHT"]
    pupil_center: Point
    iris_radius: float | None = Field(default=None, ge=0.001, le=0.5)
    highlight_scale: float | None = Field(default=None, gt=0, le=4)
    highlight_rotation_degrees: float | None = Field(default=None, ge=-180, le=180)


class BackgroundCorrection(ApiModel):
    type: Literal["background"]
    mode: BackgroundMode
    color: str | None = None

    @model_validator(mode="after")
    def valid_color(self) -> BackgroundCorrection:
        if self.mode == BackgroundMode.SOLID:
            if not self.color or not re.fullmatch(r"#[0-9a-fA-F]{6}", self.color):
                raise ValueError("color must be #RRGGBB for SOLID mode")
        elif self.color is not None:
            raise ValueError("color is only valid for SOLID mode")
        return self


Correction = Annotated[
    MaskCorrection
    | AlignmentCorrection
    | GainCopyCorrection
    | EyeCorrection
    | BackgroundCorrection,
    Field(discriminator="type"),
]


class CorrectionRequest(ApiModel):
    corrections: list[Correction] = Field(min_length=1, max_length=256)


class DiagnosticArtifactResponse(ApiModel):
    asset_id: uuid.UUID
    kind: str
    download_url: str | None = None


class JobDiagnosticsResponse(ApiModel):
    job_id: uuid.UUID
    diagnostics: dict[str, object]
    artifacts: list[DiagnosticArtifactResponse]


class JobEvent(ApiModel):
    job_id: uuid.UUID
    status: JobStatus
    stage: ProcessingStage
    progress: int
    message: str
    timestamp: datetime
