"""Archived-source constants exposed for diagnostics, never selected by style name."""

from __future__ import annotations

from typing import Final

SOURCE_2014_BASELINE: Final[dict[str, float | int | tuple[float, ...]]] = {
    "dense_scale": 0.25,
    "dense_cell_size": 7,
    "dense_grid_spacing": 1,
    "dense_levels": 4,
    "alpha": 500.0,
    "gamma": 10.0,
    "d": 1_000_000.0,
    "wsize": 3,
    "topwsize": 4,
    "top_iterations": 60,
    "iterations": 30,
    "gain_low": 0.9,
    "gain_high": 2.8,
    "energy_sigmas": (8.0, 16.0, 32.0, 64.0, 128.0),
}


def source_profile_metadata() -> dict[str, float | int | tuple[float, ...]]:
    return dict(SOURCE_2014_BASELINE)
