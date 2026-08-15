from __future__ import annotations

import numpy as np
import pytest
from portrait_transfer.exceptions import FaceDetectionError, MaskFailure
from portrait_transfer.segmentation import build_effective_mask
from portrait_transfer.selection import rank_style_examples
from portrait_transfer.style_ingestion import ingest_style
from portrait_transfer.types import PoseEstimate, StyleFeature


def test_effective_mask_handles_overlap_and_rejects_missing() -> None:
    first = np.zeros((20, 20), dtype=np.float32)
    second = np.zeros_like(first)
    first[2:12, 2:12] = 1.0
    second[7:17, 7:17] = 1.0
    effective = build_effective_mask(first, second, minimum_coverage=0.01)
    assert effective.sum() == 25
    with pytest.raises(MaskFailure):
        build_effective_mask(first, np.zeros_like(first), minimum_coverage=0.01)


def test_identical_style_feature_ranks_first(rng: np.random.Generator) -> None:
    vector = rng.normal(size=64).astype(np.float32)
    vector /= np.linalg.norm(vector)
    query = StyleFeature("query", vector, PoseEstimate())
    candidates = (
        StyleFeature("different", np.roll(vector, 17), PoseEstimate(yaw=10)),
        StyleFeature("identical", vector.copy(), PoseEstimate()),
    )
    ranked = rank_style_examples(query, candidates, top_k=2)
    assert ranked[0].identifier == "identical"
    assert ranked[0].energy_ncc > ranked[1].energy_ncc


class _NoFaceAnalyzer:
    def analyze(self, rgb: np.ndarray) -> object:
        raise FaceDetectionError("MediaPipe detected no face")


def test_ingest_style_accepts_faceless_reference(rng: np.random.Generator) -> None:
    painting = rng.uniform(0.0, 1.0, size=(96, 128, 3)).astype(np.float32)
    ingested = ingest_style("painting", painting, _NoFaceAnalyzer())  # type: ignore[arg-type]
    assert ingested.has_face is False
    assert ingested.feature.pose is None
    assert ingested.feature.landmark_shape is None
    assert ingested.analysis.masks.head.shape == (96, 128)
    assert (ingested.analysis.masks.head == 1.0).all()
    assert "style_reference_no_face" in ingested.analysis.warnings


def test_faceless_reference_survives_ranking(rng: np.random.Generator) -> None:
    painting = rng.uniform(0.0, 1.0, size=(96, 128, 3)).astype(np.float32)
    portrait = rng.uniform(0.2, 0.8, size=(96, 128, 3)).astype(np.float32)
    faceless = ingest_style("painting", painting, _NoFaceAnalyzer())  # type: ignore[arg-type]
    query = ingest_style("input", portrait).feature
    ranked = rank_style_examples(query, [faceless.feature], top_k=1)
    assert len(ranked) == 1
    assert ranked[0].pose_similarity == 0.5
    assert ranked[0].landmark_shape_similarity == 0.5
