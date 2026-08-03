from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def textured_rgb() -> np.ndarray:
    height = width = 128
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack(
        (
            ((xx % 23) / 22.0),
            ((yy % 29) / 28.0),
            (((xx + 2 * yy) % 31) / 30.0),
        ),
        axis=-1,
    ).astype(np.float32)
