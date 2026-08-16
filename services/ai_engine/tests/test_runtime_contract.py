from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from services.ai_engine.config import ServiceConfig
from services.ai_engine.contracts import TransferRequestSettings
from services.ai_engine.runtime import EngineRuntime


def _config() -> ServiceConfig:
    return ServiceConfig(
        model_root=Path("unused"),
        manifest_path=Path("unused/manifest.json"),
        device="cpu",
        dtype="float32",
        verify_models=False,
        eager_load=False,
        backend="stub",
        allow_stub_backend=True,
        api_token=None,
        max_upload_bytes=1024,
        max_decoded_pixels=100_000,
        inference_short_side=256,
        max_long_side=384,
        guidance_scale=5.0,
        style_scale_limit=1.0,
        faceid_scale_limit=1.0,
        controlnet_scale=0.8,
        img2img_base_strength=0.85,
        img2img_structure_weight=0.25,
        palette_blend=0.8,
        identity_repair_below=0.45,
        identity_fail_below=0.30,
        prompt="a portrait of a person, high quality",
        negative_prompt="lowres",
    )


def test_zero_style_strength_is_exact_identity() -> None:
    runtime = EngineRuntime(_config())
    content = Image.new("RGB", (96, 128), (12, 34, 56))
    style = Image.new("RGB", (128, 96), (200, 180, 160))
    result = runtime.transfer(
        content=content,
        style=style,
        settings=TransferRequestSettings(style_strength=0),
    )
    assert result.diagnostics["identity_short_circuit"] is True
    assert result.image_png.startswith(b"\x89PNG")


def test_keyword_roles_are_not_reversed() -> None:
    runtime = EngineRuntime(_config())
    content = Image.new("RGB", (96, 128), (255, 0, 0))
    style = Image.new("RGB", (96, 128), (0, 0, 255))
    result = runtime.transfer(
        content=content,
        style=style,
        settings=TransferRequestSettings(style_strength=1),
    )
    assert result.diagnostics["engine"] == "ai_instantstyle_v1"
    output = Image.open(io.BytesIO(result.image_png)).convert("RGB")
    pixel = output.getpixel((output.width // 2, output.height // 2))
    assert isinstance(pixel, tuple)
    red, _, blue = pixel
    assert output.size == content.size
    assert red > blue


def test_engine_settings_reject_unknown_classical_controls() -> None:
    with pytest.raises(ValidationError):
        TransferRequestSettings.model_validate(
            {"algorithm_profile": "ai_instantstyle_v1", "residual_strength": 1.0}
        )


def test_engine_settings_reject_legacy_dgpst_profile() -> None:
    with pytest.raises(ValidationError):
        TransferRequestSettings.model_validate({"algorithm_profile": "ai_dgpst_v1"})
