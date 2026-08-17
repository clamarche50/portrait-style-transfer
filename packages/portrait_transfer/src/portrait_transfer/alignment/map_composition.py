"""Exact composition of absolute backward maps and residual flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry.sampling import bilinear_sample, identity_map


def compose_backward_maps(
    outer_map: ArrayLike, inner_map: ArrayLike
) -> NDArray[np.float32]:
    """Compose destination->intermediate and intermediate->source maps.

    `outer_map` maps intermediate coordinates to the original source;
    `inner_map` maps final destination coordinates into that intermediate grid.
    """

    outer = np.asarray(outer_map, dtype=np.float32)
    inner = np.asarray(inner_map, dtype=np.float32)
    if outer.ndim != 3 or outer.shape[2] != 2 or inner.ndim != 3 or inner.shape[2] != 2:
        raise ValueError("both maps must be HxWx2")
    return np.asarray(bilinear_sample(outer, inner, mode="border"), dtype=np.float32)


def compose_with_residual(
    base_map: ArrayLike, residual_flow: ArrayLike
) -> NDArray[np.float32]:
    """Implement M_final(x) = sample(M_base, x + residual(x))."""

    base = np.asarray(base_map, dtype=np.float32)
    residual = np.asarray(residual_flow, dtype=np.float32)
    if residual.shape != base.shape:
        raise ValueError("residual_flow must match base_map shape")
    intermediate = identity_map((base.shape[0], base.shape[1])) + residual
    return compose_backward_maps(base, intermediate)


def naive_add_offsets(
    base_map: ArrayLike, residual_flow: ArrayLike
) -> NDArray[np.float32]:
    """Historical approximation retained only as a diagnostic comparator."""

    return np.asarray(base_map, dtype=np.float32) + np.asarray(
        residual_flow, dtype=np.float32
    )
