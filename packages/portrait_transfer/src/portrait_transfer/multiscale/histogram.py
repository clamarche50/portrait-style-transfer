"""Mask-aware empirical-CDF histogram transfer with unequal sample counts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _match_channel(
    source: NDArray[np.float32],
    reference: NDArray[np.float32],
    source_mask: NDArray[np.bool_],
    reference_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    output = source.copy()
    source_values = source[source_mask]
    reference_values = reference[reference_mask]
    if source_values.size == 0 or reference_values.size == 0:
        return output
    _, inverse, counts = np.unique(
        source_values, return_inverse=True, return_counts=True
    )
    reference_unique, reference_counts = np.unique(reference_values, return_counts=True)
    source_cdf = np.cumsum(counts, dtype=np.float64) / source_values.size
    reference_cdf = (
        np.cumsum(reference_counts, dtype=np.float64) / reference_values.size
    )
    mapped_unique = np.interp(source_cdf, reference_cdf, reference_unique)
    output[source_mask] = mapped_unique[inverse]
    return output


def histogram_match(
    source: ArrayLike,
    reference: ArrayLike,
    mask: ArrayLike,
    reference_mask: ArrayLike | None = None,
) -> NDArray[np.float32]:
    source_value = np.asarray(source, dtype=np.float32)
    reference_value = np.asarray(reference, dtype=np.float32)
    source_mask = np.asarray(mask, dtype=np.float32) > 0.5
    reference_mask_value = (
        source_mask
        if reference_mask is None
        else np.asarray(reference_mask, dtype=np.float32) > 0.5
    )
    if (
        source_value.shape[:2] != source_mask.shape
        or reference_value.shape[:2] != reference_mask_value.shape
    ):
        raise ValueError("mask shapes must match their images")
    if source_value.ndim == 2 and reference_value.ndim == 2:
        return _match_channel(
            source_value, reference_value, source_mask, reference_mask_value
        )
    if (
        source_value.ndim != 3
        or reference_value.ndim != 3
        or source_value.shape[2] != reference_value.shape[2]
    ):
        raise ValueError("source and reference must have compatible channels")
    return np.stack(
        [
            _match_channel(
                source_value[..., channel],
                reference_value[..., channel],
                source_mask,
                reference_mask_value,
            )
            for channel in range(source_value.shape[2])
        ],
        axis=-1,
    ).astype(np.float32)


def rank_transfer_one_dimensional(
    source: ArrayLike, reference: ArrayLike
) -> NDArray[np.float32]:
    source_value = np.asarray(source, dtype=np.float32)
    reference_value = np.asarray(reference, dtype=np.float32)
    mask = np.ones(source_value.shape, dtype=np.float32)
    reference_mask = np.ones(reference_value.shape, dtype=np.float32)
    return histogram_match(source_value, reference_value, mask, reference_mask)


def apply_global_range_mix(
    local: ArrayLike, matched: ArrayLike, mix: float
) -> NDArray[np.float32]:
    if not 0.0 <= float(mix) <= 1.0:
        raise ValueError("mix must be in [0, 1]")
    local_value = np.asarray(local, dtype=np.float32)
    matched_value = np.asarray(matched, dtype=np.float32)
    return ((1.0 - mix) * local_value + mix * matched_value).astype(np.float32)
