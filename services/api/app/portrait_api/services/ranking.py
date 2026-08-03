from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from portrait_api.models import Asset, StyleExample
from portrait_api.services.storage import ObjectStorage
from portrait_transfer.alignment.anchors import normalized_landmark_shape
from portrait_transfer.selection import rank_style_examples
from portrait_transfer.types import PoseEstimate, StyleFeature


@dataclass(frozen=True, slots=True)
class RankedExample:
    example_id: uuid.UUID
    asset_id: uuid.UUID
    score: float
    energy_ncc: float | None = None
    pose_similarity: float | None = None
    landmark_shape_similarity: float | None = None
    photometric_compatibility: float | None = None
    mask_quality: float | None = None


class StyleRankingService:
    def __init__(self, storage: ObjectStorage) -> None:
        self.storage = storage

    @staticmethod
    def _rgb(data: bytes) -> NDArray[np.float32]:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    @staticmethod
    def _lightweight_feature(identifier: str, image: NDArray[np.float32]) -> StyleFeature:
        from portrait_transfer.selection import build_style_feature

        mask = np.ones(image.shape[:2], dtype=np.float32)
        feature = build_style_feature(identifier, image, mask)
        return StyleFeature(
            identifier=identifier,
            vector=np.asarray(feature.vector, dtype=np.float32).reshape(-1),
            pose=None,
            landmark_shape=None,
            photometric_lab=feature.photometric_lab,
            mask_quality=0.5,
        )

    @classmethod
    def _feature(cls, image: NDArray[np.float32]) -> NDArray[np.float32]:
        return cls._lightweight_feature("api-ranking", image).vector

    @staticmethod
    def _npy(data: bytes) -> NDArray[np.float32]:
        value = np.asarray(np.load(io.BytesIO(data), allow_pickle=False), dtype=np.float32)
        if not value.size or not np.isfinite(value).all():
            raise ValueError("Ranking artifact must contain finite values")
        return value

    @staticmethod
    def _number(value: object, default: float) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            return default
        result = float(value)
        return result if np.isfinite(result) else default

    @classmethod
    def _pose(cls, quality: dict[str, object]) -> PoseEstimate | None:
        value = quality.get("pose")
        if not isinstance(value, dict):
            return None
        return PoseEstimate(
            yaw=cls._number(value.get("yaw"), 0.0),
            pitch=cls._number(value.get("pitch"), 0.0),
            roll=cls._number(value.get("roll"), 0.0),
        )

    @staticmethod
    def _photometric(quality: dict[str, object]) -> NDArray[np.float32] | None:
        value = quality.get("photometric_lab")
        if not isinstance(value, list) or len(value) != 3:
            return None
        result = np.asarray(value, dtype=np.float32)
        return result if np.isfinite(result).all() else None

    def _candidate_feature(self, example: StyleExample) -> StyleFeature:
        vector: NDArray[np.float32]
        if example.feature_object_key:
            vector = self._npy(self.storage.get_bytes(example.feature_object_key)).reshape(-1)
        else:
            vector = self._feature(self._rgb(self.storage.get_bytes(example.asset.object_key)))

        quality = dict(example.quality or {})
        landmark_shape: NDArray[np.float32] | None = None
        if quality.get("full_ingestion") == "COMPLETED":
            prefix = f"styles/{example.style_id}/examples/{example.id}"
            landmarks = self._npy(self.storage.get_bytes(f"{prefix}/landmarks.npy"))
            landmark_shape = normalized_landmark_shape(landmarks)

        return StyleFeature(
            identifier=str(example.id),
            vector=vector,
            pose=self._pose(quality),
            landmark_shape=landmark_shape,
            photometric_lab=self._photometric(quality),
            mask_quality=float(
                np.clip(self._number(quality.get("mask_confidence"), 0.5), 0.0, 1.0)
            ),
        )

    def index_example(self, style_id: uuid.UUID, example: StyleExample, asset: Asset) -> str:
        feature = self._feature(self._rgb(self.storage.get_bytes(asset.object_key)))
        return self.index_vector(style_id, example, feature)

    def index_vector(
        self,
        style_id: uuid.UUID,
        example: StyleExample,
        feature: NDArray[np.float32],
    ) -> str:
        output = io.BytesIO()
        np.save(output, np.asarray(feature, dtype=np.float32).reshape(-1), allow_pickle=False)
        key = f"styles/{style_id}/features/{example.id}.npy"
        self.storage.put_bytes(key, output.getvalue(), "application/octet-stream")
        return key

    def rank(
        self,
        input_asset: Asset,
        examples: list[StyleExample],
        *,
        limit: int,
    ) -> list[RankedExample]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = self._lightweight_feature(
            str(input_asset.id),
            self._rgb(self.storage.get_bytes(input_asset.object_key)),
        )
        return self.rank_compatible(query, examples, limit=limit)

    def rank_compatible(
        self,
        query: StyleFeature,
        examples: list[StyleExample],
        *,
        limit: int,
    ) -> list[RankedExample]:
        """Rank worker-side candidates with all precomputed compatibility signals."""

        features = tuple(self._candidate_feature(example) for example in examples)
        examples_by_id = {str(example.id): example for example in examples}
        results: list[RankedExample] = []
        for ranked in rank_style_examples(query, features, top_k=limit):
            example = examples_by_id[ranked.identifier]
            results.append(
                RankedExample(
                    example_id=example.id,
                    asset_id=example.asset_id,
                    score=ranked.score,
                    energy_ncc=ranked.energy_ncc,
                    pose_similarity=ranked.pose_similarity,
                    landmark_shape_similarity=ranked.landmark_shape_similarity,
                    photometric_compatibility=ranked.photometric_compatibility,
                    mask_quality=ranked.mask_quality,
                )
            )
        return results
