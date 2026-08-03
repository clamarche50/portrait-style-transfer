from __future__ import annotations

import io

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
