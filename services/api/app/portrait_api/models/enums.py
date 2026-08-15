from __future__ import annotations

from enum import StrEnum


class AssetKind(StrEnum):
    INPUT = "INPUT"
    REFERENCE = "REFERENCE"
    STYLE_EXAMPLE = "STYLE_EXAMPLE"
    OUTPUT = "OUTPUT"
    DEBUG = "DEBUG"
    EXPORT = "EXPORT"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ProcessingStage(StrEnum):
    VALIDATING = "VALIDATING"
    DECODING = "DECODING"
    FACE_LANDMARKS = "FACE_LANDMARKS"
    SEGMENTATION = "SEGMENTATION"
    QUALITY_ANALYSIS = "QUALITY_ANALYSIS"
    REFERENCE_SELECTION = "REFERENCE_SELECTION"
    AI_GENERATION = "AI_GENERATION"
    BACKGROUND = "BACKGROUND"
    POSTPROCESSING = "POSTPROCESSING"
    UPLOADING_OUTPUT = "UPLOADING_OUTPUT"
    COMPLETED = "COMPLETED"


class AlgorithmProfile(StrEnum):
    AI_INSTANTSTYLE_V1 = "ai_instantstyle_v1"


class ArtifactKind(StrEnum):
    OUTPUT = "OUTPUT"
    INPUT_MASK = "INPUT_MASK"
    REFERENCE_MASK = "REFERENCE_MASK"
    OTHER = "OTHER"


class BackgroundMode(StrEnum):
    KEEP = "KEEP"
    BLUR = "BLUR"
    SOLID = "SOLID"
    REFERENCE = "REFERENCE"


class OutputFormat(StrEnum):
    PNG = "PNG"
    JPEG = "JPEG"


TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.EXPIRED,
}
