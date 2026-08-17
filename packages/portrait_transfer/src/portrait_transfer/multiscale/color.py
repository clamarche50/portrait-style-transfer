"""Profile-explicit Lab band policies; style names are never consulted."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import AlgorithmProfile, LegacyColorMode


def should_transfer_band(
    channel_index: int,
    level: int,
    *,
    profile: AlgorithmProfile,
    legacy_mode: LegacyColorMode = LegacyColorMode.ALL_CHANNELS,
    monochrome_style: bool = False,
) -> bool:
    if channel_index not in (0, 1, 2):
        raise ValueError("channel_index must be 0, 1, or 2")
    if profile is AlgorithmProfile.PAPER_EXACT:
        if channel_index == 0:
            return True
        return level >= 3 and not monochrome_style
    if legacy_mode is LegacyColorMode.MONOCHROME_L_ONLY:
        return channel_index == 0
    return True


def transfer_band(
    input_band: ArrayLike, gain: ArrayLike, *, enabled: bool
) -> NDArray[np.float32]:
    source = np.asarray(input_band, dtype=np.float32)
    if not enabled:
        return source.copy()
    return (source * np.asarray(gain, dtype=np.float32)).astype(np.float32)
