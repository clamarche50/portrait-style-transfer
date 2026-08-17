from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from portrait_api.models import ArtifactKind, ProcessingStage

_STAGE_ORDER = list(ProcessingStage)

_EARLIEST_STAGE = {
    "mask": ProcessingStage.SEGMENTATION,
    "alignment": ProcessingStage.AFFINE_ALIGNMENT,
    "gain_copy": ProcessingStage.MULTISCALE_TRANSFER,
    "eye": ProcessingStage.EYE_HIGHLIGHTS,
    "background": ProcessingStage.BACKGROUND,
}

_ARTIFACT_STAGE = {
    ArtifactKind.INPUT_MASK: ProcessingStage.SEGMENTATION,
    ArtifactKind.REFERENCE_MASK: ProcessingStage.SEGMENTATION,
    ArtifactKind.AFFINE_PREVIEW: ProcessingStage.AFFINE_ALIGNMENT,
    ArtifactKind.PIECEWISE_PREVIEW: ProcessingStage.PIECEWISE_ALIGNMENT,
    ArtifactKind.DENSE_PREVIEW: ProcessingStage.DENSE_ALIGNMENT,
    ArtifactKind.ENERGY: ProcessingStage.MULTISCALE_TRANSFER,
    ArtifactKind.GAIN: ProcessingStage.MULTISCALE_TRANSFER,
    ArtifactKind.OUTPUT: ProcessingStage.POSTPROCESSING,
    ArtifactKind.OTHER: ProcessingStage.POSTPROCESSING,
}


@dataclass(frozen=True, slots=True)
class InvalidationPlan:
    earliest_stage: ProcessingStage
    invalidated_stages: tuple[ProcessingStage, ...]
    invalidated_artifact_kinds: frozenset[ArtifactKind]
    correction_hash: str


def build_invalidation_plan(
    new_corrections: list[dict[str, Any]],
    *,
    persisted_corrections: list[dict[str, Any]] | None = None,
) -> InvalidationPlan:
    requested = [_EARLIEST_STAGE[item["type"]] for item in new_corrections]
    earliest = min(requested, key=_STAGE_ORDER.index)
    earliest_index = _STAGE_ORDER.index(earliest)
    invalidated_stages = tuple(_STAGE_ORDER[earliest_index:])
    artifact_kinds = frozenset(
        kind
        for kind, stage in _ARTIFACT_STAGE.items()
        if _STAGE_ORDER.index(stage) >= earliest_index
    )
    canonical = json.dumps(
        persisted_corrections if persisted_corrections is not None else new_corrections,
        sort_keys=True,
        separators=(",", ":"),
    )
    return InvalidationPlan(
        earliest_stage=earliest,
        invalidated_stages=invalidated_stages,
        invalidated_artifact_kinds=artifact_kinds,
        correction_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )
