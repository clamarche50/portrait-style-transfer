from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .contracts import EngineFailure


@dataclass(frozen=True, slots=True)
class QualityReport:
    border_anisotropy_input: float
    border_anisotropy_output: float
    border_anisotropy_limit: float

    def as_dict(self) -> dict[str, float]:
        return {
            "border_anisotropy_input": self.border_anisotropy_input,
            "border_anisotropy_output": self.border_anisotropy_output,
            "border_anisotropy_limit": self.border_anisotropy_limit,
        }


def _border_anisotropy(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    band = max(8, min(64, array.shape[0] // 10))
    border = np.concatenate((array[:band], array[-band:]), axis=0)
    horizontal_change = float(np.mean(np.abs(np.diff(border, axis=1))))
    vertical_change = float(np.mean(np.abs(np.diff(border, axis=0))))
    return horizontal_change / max(vertical_change, 1e-6)


def validate_output(content: Image.Image, output: Image.Image) -> QualityReport:
    if output.size != content.size:
        raise EngineFailure(
            "AI_QUALITY_GUARD_FAILED", "Output dimensions changed unexpectedly"
        )
    array = np.asarray(output)
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise EngineFailure("AI_QUALITY_GUARD_FAILED", "Output pixels are invalid")
    input_ratio = _border_anisotropy(content)
    output_ratio = _border_anisotropy(output)
    limit = max(8.0, input_ratio * 4.0)
    if output_ratio > limit:
        raise EngineFailure(
            "AI_QUALITY_GUARD_FAILED",
            "The generated portrait contained stretched border artifacts",
            retryable=True,
        )
    return QualityReport(input_ratio, output_ratio, limit)
