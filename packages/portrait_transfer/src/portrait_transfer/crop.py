"""Canonical crop extraction with reversible coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry.sampling import bilinear_sample, identity_map
from .geometry.transforms import transform_points
from .types import BoundingBox, PortraitAnalysis, PortraitMasks


@dataclass(frozen=True)
class CropContext:
    original_shape: tuple[int, int]
    crop_shape: tuple[int, int]
    source_box: BoundingBox
    original_to_crop: NDArray[np.float64]
    landmarks: NDArray[np.float32]

    def _crop_to_original_map(self) -> NDArray[np.float32]:
        height, width = self.crop_shape
        grid = identity_map((height, width))
        scale_x = width / self.source_box.width
        scale_y = height / self.source_box.height
        mapping = grid.copy()
        mapping[..., 0] = self.source_box.x + grid[..., 0] / scale_x
        mapping[..., 1] = self.source_box.y + grid[..., 1] / scale_y
        return mapping

    def extract(
        self, image: ArrayLike, *, mode: str = "reflect"
    ) -> NDArray[np.float32]:
        return np.asarray(
            bilinear_sample(image, self._crop_to_original_map(), mode=mode),
            dtype=np.float32,
        )

    def extract_mask(self, mask: ArrayLike) -> NDArray[np.float32]:
        return np.clip(
            np.asarray(
                bilinear_sample(mask, self._crop_to_original_map(), mode="constant"),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )

    def extract_masks(self, masks: PortraitMasks) -> PortraitMasks:
        return PortraitMasks(
            person=self.extract_mask(masks.person),
            head=self.extract_mask(masks.head),
            face_skin=self.extract_mask(masks.face_skin),
            hair=self.extract_mask(masks.hair),
            eyes=(self.extract_mask(masks.eyes[0]), self.extract_mask(masks.eyes[1])),
            irises=(
                self.extract_mask(masks.irises[0]),
                self.extract_mask(masks.irises[1]),
            ),
            effective_transfer=self.extract_mask(masks.effective_transfer),
            foreground_alpha=self.extract_mask(masks.foreground_alpha),
        )

    def composite_back(
        self, original: ArrayLike, processed_crop: ArrayLike, alpha: ArrayLike
    ) -> NDArray[np.float32]:
        original_value = np.asarray(original, dtype=np.float32)
        grid = identity_map(self.original_shape)
        flat_points = grid.reshape(-1, 2)
        crop_coordinates = transform_points(flat_points, self.original_to_crop).reshape(
            (*self.original_shape, 2)
        )
        processed = np.asarray(
            bilinear_sample(processed_crop, crop_coordinates, mode="constant"),
            dtype=np.float32,
        )
        matte = np.asarray(
            bilinear_sample(alpha, crop_coordinates, mode="constant"), dtype=np.float32
        )
        matte = np.clip(matte, 0.0, 1.0)[..., None]
        return cast(
            NDArray[np.float32],
            (original_value * (1.0 - matte) + processed * matte).astype(np.float32),
        )


def canonical_crop_side(face: BoundingBox) -> float:
    """Square side of the canonical crop box for a face box."""
    return max(face.width * 1.7, face.height * 1.75)


def create_canonical_crop(
    image: ArrayLike,
    analysis: PortraitAnalysis,
    *,
    processing_long_edge: int = 1280,
    output_shape: tuple[int, int] | None = None,
) -> CropContext:
    source = np.asarray(image)
    height, width = source.shape[:2]
    face = analysis.face_box
    x1 = face.x - 0.35 * face.width
    x2 = face.x2 + 0.35 * face.width
    y1 = face.y - 0.45 * face.height
    y2 = face.y2 + 0.30 * face.height
    side = canonical_crop_side(face)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    box = BoundingBox(center_x - side / 2.0, center_y - side / 2.0, side, side)
    if output_shape is None:
        edge = max(32, min(round(side), int(processing_long_edge)))
        crop_shape = (edge, edge)
    else:
        crop_shape = output_shape
    scale_x = crop_shape[1] / box.width
    scale_y = crop_shape[0] / box.height
    transform = np.asarray(
        [
            [scale_x, 0.0, -box.x * scale_x],
            [0.0, scale_y, -box.y * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    landmarks = transform_points(analysis.landmarks, transform)
    return CropContext((height, width), crop_shape, box, transform, landmarks)


def create_input_canonical_crop(
    image: ArrayLike, analysis: PortraitAnalysis, processing_long_edge: int = 1280
) -> CropContext:
    return create_canonical_crop(
        image, analysis, processing_long_edge=processing_long_edge
    )
