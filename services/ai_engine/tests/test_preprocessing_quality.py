from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from services.ai_engine.contracts import EngineFailure
from services.ai_engine.preprocessing import (
    decode_image,
    letterbox,
    restore_from_letterbox,
    restore_from_resize,
    scale_short_side,
)
from services.ai_engine.quality import validate_output


def _png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (50, 80, 120))
    target = io.BytesIO()
    image.save(target, format="PNG")
    return target.getvalue()


def test_letterbox_round_trip_restores_original_dimensions() -> None:
    original = decode_image(_png(1000, 1320), max_bytes=1_000_000, max_pixels=2_000_000)
    square, transform = letterbox(original, 512)
    restored = restore_from_letterbox(square, transform)
    assert square.size == (512, 512)
    assert restored.size == (1000, 1320)


def test_official_short_side_scaling_retains_more_portrait_detail() -> None:
    original = decode_image(_png(1000, 1320), max_bytes=1_000_000, max_pixels=2_000_000)
    scaled, transform = scale_short_side(original, 512, 768)
    restored = restore_from_resize(scaled, transform)
    assert scaled.size == (512, 672)
    assert restored.size == original.size


def test_quality_guard_accepts_unchanged_image() -> None:
    content = Image.fromarray(np.full((256, 192, 3), 127, dtype=np.uint8))
    report = validate_output(content, content.copy())
    assert report.border_anisotropy_output <= report.border_anisotropy_limit


def test_quality_guard_rejects_stretched_border_stripes() -> None:
    rng = np.random.default_rng(7)
    content = Image.fromarray(rng.integers(0, 255, (256, 192, 3), dtype=np.uint8))
    failed = np.asarray(content).copy()
    stripe = np.tile(np.arange(192, dtype=np.uint8), (32, 1))
    failed[:32] = np.repeat(stripe[..., None], 3, axis=2)
    failed[-32:] = np.repeat(stripe[..., None], 3, axis=2)
    with pytest.raises(EngineFailure, match="stretched border"):
        validate_output(content, Image.fromarray(failed))
