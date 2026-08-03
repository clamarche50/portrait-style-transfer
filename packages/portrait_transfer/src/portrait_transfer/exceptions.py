"""Typed failures raised by the portrait-transfer library."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    DECODE_FAILURE = "DECODE_FAILURE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    FACE_FAILURE = "FACE_FAILURE"
    QUALITY_FAILURE = "QUALITY_FAILURE"
    PAIR_INCOMPATIBLE = "PAIR_INCOMPATIBLE"
    MASK_FAILURE = "MASK_FAILURE"
    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    OPTIONAL_DEPENDENCY = "OPTIONAL_DEPENDENCY"
    CANCELLED = "CANCELLED"


class PortraitTransferError(RuntimeError):
    """Base error with a stable machine-readable code and safe context."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


class InputValidationError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.INVALID_INPUT, message, context=context)


class DecodeError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.DECODE_FAILURE, message, context=context)


class ImageTooLargeError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.IMAGE_TOO_LARGE, message, context=context)


class FaceDetectionError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.FACE_FAILURE, message, context=context)


class QualityFailure(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.QUALITY_FAILURE, message, context=context)


class PairCompatibilityError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.PAIR_INCOMPATIBLE, message, context=context)


class MaskFailure(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.MASK_FAILURE, message, context=context)


class AlignmentFailure(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.ALIGNMENT_FAILURE, message, context=context)


class OptionalDependencyError(PortraitTransferError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.OPTIONAL_DEPENDENCY, message, context=context)


class ProcessingCancelled(PortraitTransferError):
    def __init__(self, message: str = "Portrait transfer was cancelled") -> None:
        super().__init__(ErrorCode.CANCELLED, message)
