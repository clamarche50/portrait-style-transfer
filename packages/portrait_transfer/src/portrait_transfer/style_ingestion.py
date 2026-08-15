"""Reference preprocessing into rankable style metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import PortraitTransferError
from .image_io import normalize_rgb
from .landmarks import canonical_68_landmarks
from .preflight import PortraitAnalyzer, analyze_portrait
from .quality import analyze_quality
from .selection import build_style_feature
from .types import (
    BoundingBox,
    PortraitAnalysis,
    PortraitMasks,
    PoseEstimate,
    StyleFeature,
)


@dataclass(frozen=True)
class IngestedStyle:
    identifier: str
    rgb: NDArray[np.float32]
    analysis: PortraitAnalysis
    feature: StyleFeature
    has_face: bool = True


def _faceless_analysis(image: NDArray[np.float32]) -> PortraitAnalysis:
    """Neutral analysis for style references without a detectable face.

    Paintings, textures and other non-portrait references stay valid style
    sources: the whole image acts as the transfer region and every
    face-derived ranking signal degrades to neutral instead of failing.
    """

    height, width = image.shape[:2]
    image_shape = (height, width, image.shape[2])
    ones = np.ones((height, width), dtype=np.float32)
    zeros = np.zeros((height, width), dtype=np.float32)
    landmarks = canonical_68_landmarks(image_shape)
    face_box = BoundingBox(0.0, 0.0, float(width), float(height))
    masks = PortraitMasks(
        person=ones,
        head=ones,
        face_skin=zeros,
        hair=zeros,
        eyes=(zeros, zeros),
        irises=(zeros, zeros),
        effective_transfer=ones,
        foreground_alpha=ones,
    )
    quality = analyze_quality(
        image, landmarks, face_box, ones, mask_confidence=0.5
    )
    return PortraitAnalysis(
        landmarks=landmarks,
        face_box=face_box,
        pose=PoseEstimate(),
        quality=quality,
        masks=masks,
        warnings=("style_reference_no_face",),
    )


def ingest_style(
    identifier: str, rgb: ArrayLike, analyzer: PortraitAnalyzer | None = None
) -> IngestedStyle:
    image = normalize_rgb(rgb)
    try:
        analysis = analyze_portrait(image, analyzer)
    except PortraitTransferError:
        analysis = _faceless_analysis(image)
        feature = build_style_feature(
            identifier,
            image,
            analysis.masks.head,
            pose=None,
            landmarks=None,
            mask_quality=0.5,
        )
        return IngestedStyle(identifier, image, analysis, feature, has_face=False)
    feature = build_style_feature(
        identifier,
        image,
        analysis.masks.head,
        pose=analysis.pose,
        landmarks=analysis.landmarks,
        mask_quality=analysis.quality.mask_confidence,
    )
    return IngestedStyle(identifier, image, analysis, feature)
