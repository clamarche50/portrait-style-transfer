from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    model_root: Path
    manifest_path: Path
    upstream_root: Path
    device: str
    dtype: str
    verify_models: bool
    eager_load: bool
    backend: str
    allow_stub_backend: bool
    api_token: str | None
    max_upload_bytes: int
    max_decoded_pixels: int
    inference_size: int
    max_long_side: int

    @classmethod
    def from_environment(cls) -> "ServiceConfig":
        model_root = Path(os.getenv("DGPST_MODEL_ROOT", "/models/dgpst"))
        dtype = os.getenv("DGPST_DTYPE", "float16").strip().lower()
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("DGPST_DTYPE must be float16, bfloat16, or float32")
        inference_size = int(os.getenv("DGPST_INFERENCE_SIZE", "512"))
        if inference_size < 256 or inference_size > 1024 or inference_size % 64:
            raise ValueError(
                "DGPST_INFERENCE_SIZE must be a multiple of 64 from 256 to 1024"
            )
        max_long_side = int(os.getenv("DGPST_MAX_LONG_SIDE", "768"))
        if max_long_side < inference_size or max_long_side > 1024 or max_long_side % 16:
            raise ValueError(
                "DGPST_MAX_LONG_SIDE must be a multiple of 16 between inference size and 1024"
            )
        return cls(
            model_root=model_root,
            manifest_path=Path(
                os.getenv("DGPST_MANIFEST_PATH", str(model_root / "manifest.json"))
            ),
            upstream_root=Path(os.getenv("DGPST_UPSTREAM_ROOT", "/opt/dgpst")),
            device=os.getenv("DGPST_DEVICE", "cuda").strip().lower(),
            dtype=dtype,
            verify_models=_as_bool("DGPST_VERIFY_MODELS", True),
            eager_load=_as_bool("DGPST_EAGER_LOAD", True),
            backend=os.getenv("DGPST_BACKEND", "dgpst").strip().lower(),
            allow_stub_backend=_as_bool("DGPST_ALLOW_STUB_BACKEND", False),
            api_token=os.getenv("AI_ENGINE_API_TOKEN") or None,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
            max_decoded_pixels=int(os.getenv("MAX_DECODED_PIXELS", "8000000")),
            inference_size=inference_size,
            max_long_side=max_long_side,
        )
