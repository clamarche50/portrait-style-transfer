"""Alignment diagnostic aggregation."""

from __future__ import annotations

from numpy.typing import ArrayLike

from ..geometry.validity import map_validity
from ..types import AlignmentDiagnostics


def collect_alignment_diagnostics(
    mapping: ArrayLike,
    source_shape: tuple[int, int] | tuple[int, int, int],
    *,
    selected_stage: str,
    anchor_error: float,
    inlier_count: int,
    fallback_reason: str | None = None,
) -> AlignmentDiagnostics:
    report = map_validity(mapping, source_shape)
    return AlignmentDiagnostics(
        selected_stage=selected_stage,
        anchor_error=float(anchor_error),
        inlier_count=int(inlier_count),
        valid_fraction=report.valid_fraction,
        negative_jacobian_fraction=report.negative_jacobian_fraction,
        displacement_p50=report.displacement_p50,
        displacement_p95=report.displacement_p95,
        fallback_reason=fallback_reason,
    )
