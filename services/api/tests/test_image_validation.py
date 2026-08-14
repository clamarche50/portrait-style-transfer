from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from portrait_api.config import Settings
from portrait_api.errors import AppError
from portrait_api.services.image_validation import ImageNormalizer


def _settings(**updates: object) -> Settings:
    return Settings(
        app_env="test",
        require_models_for_readiness=False,
        initialize_storage_on_startup=False,
        max_decoded_pixels=1_000_000,
        **updates,
    )


def test_rejects_mime_mismatch_and_invalid_payload() -> None:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, "PNG")
    with pytest.raises(AppError, match="file contents") as mismatch:
        ImageNormalizer(_settings()).normalize(output.getvalue(), "image/jpeg")
    assert mismatch.value.code == "MIME_MISMATCH"

    with pytest.raises(AppError) as invalid:
        ImageNormalizer(_settings()).normalize(b"not-an-image", "application/octet-stream")
    assert invalid.value.code == "INVALID_IMAGE"


def test_rejects_animated_webp() -> None:
    output = io.BytesIO()
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    frames[0].save(output, "WEBP", save_all=True, append_images=frames[1:], duration=10)
    with pytest.raises(AppError) as animated:
        ImageNormalizer(_settings()).normalize(output.getvalue(), "image/webp")
    assert animated.value.code == "ANIMATED_IMAGE"


def test_rejects_upload_that_expands_past_limit_during_normalization() -> None:
    rng = np.random.default_rng(17)
    pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(output, "JPEG", quality=5)
    payload = output.getvalue()
    assert len(payload) < 1_024

    with pytest.raises(AppError) as expanded:
        ImageNormalizer(_settings(max_upload_bytes=1_024)).normalize(payload, "image/jpeg")
    assert expanded.value.code == "NORMALIZED_UPLOAD_TOO_LARGE"
    assert expanded.value.status_code == 413


def test_normalization_never_mutates_pillow_process_global() -> None:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(output, "PNG")
    process_limit = Image.MAX_IMAGE_PIXELS

    ImageNormalizer(_settings()).normalize(output.getvalue(), "image/png")

    assert Image.MAX_IMAGE_PIXELS == process_limit
