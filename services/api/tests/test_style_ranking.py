from __future__ import annotations

import io
import uuid
from types import SimpleNamespace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from portrait_api.models import StyleExample
from portrait_api.services.ranking import StyleRankingService
from portrait_api.services.storage import MemoryObjectStorage
from portrait_transfer.alignment.anchors import normalized_landmark_shape
from portrait_transfer.types import PoseEstimate, StyleFeature


def _npy(value: NDArray[np.float32]) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=False)
    return output.getvalue()


def _example(
    *,
    style_id: uuid.UUID,
    example_id: uuid.UUID,
    feature_key: str,
    quality: dict[str, object],
) -> StyleExample:
    return cast(
        StyleExample,
        SimpleNamespace(
            id=example_id,
            style_id=style_id,
            asset_id=uuid.uuid4(),
            feature_object_key=feature_key,
            quality=quality,
        ),
    )


def test_worker_ranking_combines_precomputed_compatibility_signals() -> None:
    storage = MemoryObjectStorage()
    service = StyleRankingService(storage)
    style_id = uuid.uuid4()
    energy_only_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    compatible_id = uuid.UUID("00000000-0000-0000-0000-000000000002")

    rng = np.random.default_rng(7)
    query_vector = rng.normal(size=96).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)
    compatible_vector = query_vector + rng.normal(0.0, 0.08, size=96).astype(np.float32)
    compatible_vector /= np.linalg.norm(compatible_vector)

    landmarks = rng.uniform(20.0, 220.0, size=(68, 2)).astype(np.float32)
    incompatible_landmarks = landmarks[::-1].copy()
    head_mask = np.zeros((64, 64), dtype=np.float32)
    head_mask[8:58, 14:50] = 1.0
    incompatible_mask = np.zeros_like(head_mask)
    incompatible_mask[:7, :7] = 1.0

    examples = []
    for example_id, vector, example_landmarks, mask, quality in (
        (
            energy_only_id,
            query_vector,
            incompatible_landmarks,
            incompatible_mask,
            {
                "full_ingestion": "COMPLETED",
                "pose": {"yaw": 90.0, "pitch": 75.0, "roll": 90.0},
                "photometric_lab": [5.0, 100.0, -100.0],
                "mask_confidence": 0.0,
            },
        ),
        (
            compatible_id,
            compatible_vector,
            landmarks,
            head_mask,
            {
                "full_ingestion": "COMPLETED",
                "pose": {"yaw": 4.0, "pitch": -2.0, "roll": 1.0},
                "photometric_lab": [52.0, 3.0, 6.0],
                "mask_confidence": 0.95,
            },
        ),
    ):
        prefix = f"styles/{style_id}/examples/{example_id}"
        feature_key = f"styles/{style_id}/features/{example_id}.npy"
        storage.put_bytes(feature_key, _npy(vector), "application/octet-stream")
        storage.put_bytes(
            f"{prefix}/landmarks.npy",
            _npy(example_landmarks),
            "application/octet-stream",
        )
        storage.put_bytes(
            f"{prefix}/head-mask.npy",
            _npy(mask),
            "application/octet-stream",
        )
        examples.append(
            _example(
                style_id=style_id,
                example_id=example_id,
                feature_key=feature_key,
                quality=quality,
            )
        )

    query = StyleFeature(
        identifier="input",
        vector=query_vector,
        pose=PoseEstimate(yaw=4.0, pitch=-2.0, roll=1.0),
        landmark_shape=normalized_landmark_shape(landmarks),
        photometric_lab=np.asarray([52.0, 3.0, 6.0], dtype=np.float32),
        mask_quality=0.9,
    )
    ranked = service.rank_compatible(query, examples, limit=2)

    assert [item.example_id for item in ranked] == [compatible_id]
    assert ranked[0].energy_ncc is not None
    assert ranked[0].energy_ncc < 1.0
    assert ranked[0].pose_similarity == 1.0
    expected_score = (
        0.65 * ranked[0].energy_ncc
        + 0.15 * ranked[0].pose_similarity
        + 0.10 * ranked[0].landmark_shape_similarity
        + 0.05 * ranked[0].photometric_compatibility
        + 0.05 * ranked[0].mask_quality
    )
    assert np.isclose(ranked[0].score, expected_score)


def test_worker_ranking_uses_neutral_metadata_fallback_and_uuid_tie_break() -> None:
    storage = MemoryObjectStorage()
    service = StyleRankingService(storage)
    style_id = uuid.uuid4()
    vector = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    examples = []
    for suffix in (2, 1):
        example_id = uuid.UUID(f"00000000-0000-0000-0000-{suffix:012d}")
        feature_key = f"styles/{style_id}/features/{example_id}.npy"
        storage.put_bytes(feature_key, _npy(vector), "application/octet-stream")
        examples.append(
            _example(
                style_id=style_id,
                example_id=example_id,
                feature_key=feature_key,
                quality={"full_ingestion": "QUEUED"},
            )
        )

    query = StyleFeature(identifier="input", vector=vector)
    ranked = service.rank_compatible(query, examples, limit=2)

    assert [str(item.example_id) for item in ranked] == sorted(
        str(example.id) for example in examples
    )
    assert all(item.pose_similarity == 0.5 for item in ranked)
    assert all(item.photometric_compatibility == 0.5 for item in ranked)
    assert all(item.mask_quality == 0.5 for item in ranked)
