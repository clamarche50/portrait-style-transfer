"""Reference preprocessing into rankable style metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .image_io import normalize_rgb
from .preflight import PortraitAnalyzer, analyze_portrait
from .selection import build_style_feature
from .types import PortraitAnalysis, StyleFeature


@dataclass(frozen=True)
class IngestedStyle:
    identifier: str
    rgb: NDArray[np.float32]
    analysis: PortraitAnalysis
    feature: StyleFeature


def ingest_style(
    identifier: str, rgb: ArrayLike, analyzer: PortraitAnalyzer | None = None
) -> IngestedStyle:
    image = normalize_rgb(rgb)
    analysis = analyze_portrait(image, analyzer)
    feature = build_style_feature(
        identifier,
        image,
        analysis.masks.head,
        pose=analysis.pose,
        landmarks=analysis.landmarks,
        mask_quality=analysis.quality.mask_confidence,
    )
    return IngestedStyle(identifier, image, analysis, feature)
