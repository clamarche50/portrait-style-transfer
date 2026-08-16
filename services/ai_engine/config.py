from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    model_root: Path
    manifest_path: Path
    device: str
    dtype: str
    verify_models: bool
    eager_load: bool
    backend: str
    allow_stub_backend: bool
    api_token: str | None
    max_upload_bytes: int
    max_decoded_pixels: int
    inference_short_side: int
    max_long_side: int
    guidance_scale: float
    style_scale_limit: float
    faceid_scale_limit: float
    controlnet_scale: float
    img2img_base_strength: float
    img2img_structure_weight: float
    palette_blend: float
    identity_repair_below: float
    identity_fail_below: float
    prompt: str
    negative_prompt: str

    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        model_root = Path(os.getenv("ENGINE_MODEL_ROOT", "/models/instantstyle"))
        dtype = os.getenv("ENGINE_DTYPE", "float16").strip().lower()
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("ENGINE_DTYPE must be float16, bfloat16, or float32")
        short_side = _bounded_int("ENGINE_INFERENCE_SHORT_SIDE", 1024, 512, 1024)
        if short_side % 64:
            raise ValueError("ENGINE_INFERENCE_SHORT_SIDE must be a multiple of 64")
        max_long_side = _bounded_int("ENGINE_MAX_LONG_SIDE", 1280, short_side, 1536)
        if max_long_side % 64:
            raise ValueError("ENGINE_MAX_LONG_SIDE must be a multiple of 64")
        guidance_scale = float(os.getenv("ENGINE_GUIDANCE_SCALE", "5.0"))
        if guidance_scale < 1.0 or guidance_scale > 12.0:
            raise ValueError("ENGINE_GUIDANCE_SCALE must be between 1.0 and 12.0")
        style_scale_limit = float(os.getenv("ENGINE_STYLE_SCALE_LIMIT", "1.0"))
        faceid_scale_limit = float(os.getenv("ENGINE_FACEID_SCALE_LIMIT", "1.0"))
        if not 0.0 < style_scale_limit <= 1.0 or not 0.0 < faceid_scale_limit <= 1.0:
            raise ValueError("ENGINE scale limits must be within (0.0, 1.0]")
        controlnet_scale = float(os.getenv("ENGINE_CONTROLNET_SCALE", "0.35"))
        if not 0.0 <= controlnet_scale <= 1.0:
            raise ValueError("ENGINE_CONTROLNET_SCALE must be within [0.0, 1.0]")
        img2img_base_strength = float(os.getenv("ENGINE_IMG2IMG_BASE_STRENGTH", "0.65"))
        img2img_structure_weight = float(
            os.getenv("ENGINE_IMG2IMG_STRUCTURE_WEIGHT", "0.20")
        )
        if (
            not 0.5 <= img2img_base_strength <= 0.95
            or not 0.0 <= img2img_structure_weight <= 0.45
        ):
            raise ValueError(
                "ENGINE_IMG2IMG_BASE_STRENGTH must be within [0.5, 0.95] and "
                "ENGINE_IMG2IMG_STRUCTURE_WEIGHT within [0.0, 0.45]"
            )
        palette_blend = float(os.getenv("ENGINE_PALETTE_BLEND", "0.8"))
        if not 0.0 <= palette_blend <= 1.0:
            raise ValueError("ENGINE_PALETTE_BLEND must be within [0.0, 1.0]")
        identity_repair_below = float(os.getenv("ENGINE_IDENTITY_REPAIR_BELOW", "0.45"))
        identity_fail_below = float(os.getenv("ENGINE_IDENTITY_FAIL_BELOW", "0.30"))
        if not 0.0 < identity_fail_below < identity_repair_below <= 1.0:
            raise ValueError(
                "ENGINE identity thresholds must satisfy 0 < fail < repair <= 1"
            )
        return cls(
            model_root=model_root,
            manifest_path=Path(
                os.getenv("ENGINE_MANIFEST_PATH", str(model_root / "manifest.json"))
            ),
            device=os.getenv("ENGINE_DEVICE", "cuda").strip().lower(),
            dtype=dtype,
            verify_models=_as_bool("ENGINE_VERIFY_MODELS", True),
            eager_load=_as_bool("ENGINE_EAGER_LOAD", True),
            backend=os.getenv("ENGINE_BACKEND", "instantstyle").strip().lower(),
            allow_stub_backend=_as_bool("ENGINE_ALLOW_STUB_BACKEND", False),
            api_token=os.getenv("AI_ENGINE_API_TOKEN") or None,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
            max_decoded_pixels=int(os.getenv("MAX_DECODED_PIXELS", "8000000")),
            inference_short_side=short_side,
            max_long_side=max_long_side,
            guidance_scale=guidance_scale,
            style_scale_limit=style_scale_limit,
            faceid_scale_limit=faceid_scale_limit,
            controlnet_scale=controlnet_scale,
            img2img_base_strength=img2img_base_strength,
            img2img_structure_weight=img2img_structure_weight,
            palette_blend=palette_blend,
            identity_repair_below=identity_repair_below,
            identity_fail_below=identity_fail_below,
            prompt=os.getenv("ENGINE_PROMPT", "a portrait of a person, high quality"),
            negative_prompt=os.getenv(
                "ENGINE_NEGATIVE_PROMPT",
                "lowres, blurry, deformed, distorted face, bad anatomy, "
                "watermark, signature, text",
            ),
        )
