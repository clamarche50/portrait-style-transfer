from __future__ import annotations

import gc
import importlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from .config import ServiceConfig
from .contracts import EngineFailure, TransferOutput, TransferRequestSettings
from .manifest import VerifiedManifest, verify_manifest
from .preprocessing import encode_png, restore_from_resize, scale_short_side
from .quality import validate_output

LOGGER = logging.getLogger("portrait_ai_engine")

ENGINE_ID = "ai_instantstyle_v1"


class Backend(Protocol):
    def generate(
        self,
        *,
        content: Image.Image,
        style: Image.Image,
        style_strength: float,
        structure_strength: float,
        inference_steps: int,
        seed: int,
    ) -> tuple[Image.Image, dict[str, Any]]: ...


@dataclass(slots=True)
class _LoadedState:
    backend: Backend
    manifest: VerifiedManifest | None
    loaded_at: float


class StubBackend:
    """Explicit opt-in backend for contract tests; never enabled in production by default."""

    def generate(
        self,
        *,
        content: Image.Image,
        style: Image.Image,
        style_strength: float,
        structure_strength: float,
        inference_steps: int,
        seed: int,
    ) -> tuple[Image.Image, dict[str, Any]]:
        del structure_strength, inference_steps, seed
        style_resized = style.resize(content.size, Image.Resampling.LANCZOS)
        return Image.blend(content, style_resized, style_strength * 0.25), {
            "backend": "stub"
        }


class InstantStyleBackend:
    """SDXL + InstantStyle (style) + FaceID PlusV2 (identity) portrait engine."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.torch = importlib.import_module("torch")
        if config.device != "cuda" or not self.torch.cuda.is_available():
            raise EngineFailure(
                "AI_CUDA_UNAVAILABLE",
                "The portrait engine requires a CUDA GPU; the worker could not access one",
            )
        self.dtype = getattr(self.torch, config.dtype)
        self.pipe, self.style_clip, self.face_clip = self._load_pipeline()
        self.style_clip_processor, self.face_clip_processor = self._load_processors()
        self.faces = self._load_face_analyzer()

    def _load_pipeline(self) -> Any:
        torch = self.torch
        model_root = self.config.model_root

        from diffusers import DDIMScheduler, StableDiffusionXLPipeline
        from transformers import CLIPVisionModelWithProjection

        load_options = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        variant_options = (
            {"variant": "fp16"} if self.config.dtype in {"float16", "bfloat16"} else {}
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(model_root / "sdxl-base"),
            **load_options,
            **variant_options,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # InstantStyle injects style through two attention layers only; the
        # FaceID PlusV2 adapter preserves the subject's identity.
        pipe.load_ip_adapter(
            str(model_root / "ip-adapter"),
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
            image_encoder_folder=None,
        )
        pipe.load_ip_adapter(
            str(model_root / "faceid"),
            subfolder=None,
            weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
            image_encoder_folder=None,
        )
        style_clip = CLIPVisionModelWithProjection.from_pretrained(
            str(model_root / "ip-adapter" / "sdxl_models" / "image_encoder"),
            local_files_only=True,
        ).eval()
        face_clip = CLIPVisionModelWithProjection.from_pretrained(
            str(model_root / "ip-adapter" / "models" / "image_encoder"),
            local_files_only=True,
        ).eval()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        # Keeps the RTX worker inside 12GB of VRAM by staging UNet/text
        # encoder/VAE transfers to the GPU only while they run.
        pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        gc.collect()
        return pipe, style_clip, face_clip

    def _load_processors(self) -> Any:
        from transformers import CLIPImageProcessor

        model_root = self.config.model_root
        style_clip_processor = CLIPImageProcessor.from_pretrained(
            str(model_root / "ip-adapter" / "sdxl_models" / "image_encoder"),
            local_files_only=True,
        )
        face_clip_processor = CLIPImageProcessor.from_pretrained(
            str(model_root / "ip-adapter" / "models" / "image_encoder"),
            local_files_only=True,
        )
        return style_clip_processor, face_clip_processor

    def _load_face_analyzer(self) -> Any:
        from .faceid import FaceIdentityExtractor

        return FaceIdentityExtractor(self.config.model_root / "insightface")

    def _style_embeds(self, style: Image.Image) -> Any:
        torch = self.torch
        inputs = self.style_clip_processor(images=style, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.style_clip(**inputs, output_hidden_states=True).hidden_states[
                -2
            ]
        embeds = torch.cat([torch.zeros_like(hidden), hidden], dim=0).unsqueeze(1)
        return embeds.to(device="cuda", dtype=self.dtype)

    def _faceid_embeds(self, content: Image.Image, face: Any) -> tuple[Any, Any]:
        from .faceid import FaceIdentityExtractor

        torch = self.torch
        embedding = torch.from_numpy(face.embedding).unsqueeze(0)
        id_embeds = (
            torch.cat([torch.zeros_like(embedding), embedding], dim=0)
            .unsqueeze(1)
            .to(device="cuda", dtype=self.dtype)
        )
        face_crop = FaceIdentityExtractor.crop(content, face.bbox)
        inputs = self.face_clip_processor(images=face_crop, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.face_clip(**inputs, output_hidden_states=True).hidden_states[
                -2
            ]
        clip_embeds = (
            torch.cat([torch.zeros_like(hidden), hidden], dim=0)
            .unsqueeze(1)
            .to(device="cuda", dtype=self.dtype)
        )
        return id_embeds, clip_embeds

    def generate(
        self,
        *,
        content: Image.Image,
        style: Image.Image,
        style_strength: float,
        structure_strength: float,
        inference_steps: int,
        seed: int,
    ) -> tuple[Image.Image, dict[str, Any]]:
        torch = self.torch
        face = self.faces.extract(content)
        if face is None:
            raise EngineFailure(
                "AI_FACE_NOT_FOUND",
                "No face could be detected in the portrait",
                retryable=False,
            )
        style_scale = min(style_strength, self.config.style_scale_limit)
        faceid_scale = min(structure_strength, self.config.faceid_scale_limit)
        before_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                style_embeds = self._style_embeds(style)
                id_embeds, clip_embeds = self._faceid_embeds(content, face)
                faceid_projection = (
                    self.pipe.unet.encoder_hid_proj.image_projection_layers[1]
                )
                faceid_projection.clip_embeds = clip_embeds
                faceid_projection.shortcut = False
                # InstantStyle keeps style inside down.block_2 / up.block_0 so
                # identity-critical layers stay untouched.
                self.pipe.set_ip_adapter_scale(
                    [
                        {
                            "down": {"block_2": [0.0, style_scale]},
                            "up": {"block_0": [style_scale, 0.0]},
                        },
                        faceid_scale,
                    ]
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = self.pipe(
                    prompt=self.config.prompt,
                    negative_prompt=self.config.negative_prompt,
                    width=content.width,
                    height=content.height,
                    ip_adapter_image_embeds=[style_embeds, id_embeds],
                    num_inference_steps=inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                    eta=0.0,
                )
            output = result.images[0]
            if output.size != content.size:
                output = output.resize(content.size, Image.Resampling.LANCZOS)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EngineFailure(
                "AI_GPU_OUT_OF_MEMORY",
                "The GPU does not have enough free memory; close GPU-heavy apps and retry",
                retryable=True,
            ) from exc
        finally:
            gc.collect()
        elapsed = time.perf_counter() - started
        peak = max(before_peak, torch.cuda.max_memory_allocated())
        return output, {
            "backend": "instantstyle",
            "inference_seconds": round(elapsed, 3),
            "peak_vram_mib": round(peak / (1024 * 1024), 1),
            "torch_version": str(torch.__version__),
            "face_det_score": round(face.det_score, 3),
        }


class EngineRuntime:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._state: _LoadedState | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self._state is not None:
            return
        with self._load_lock:
            if self._state is not None:
                return
            manifest = None
            if self.config.verify_models:
                manifest = verify_manifest(
                    self.config.manifest_path, self.config.model_root
                )
            if self.config.backend == "stub":
                if not self.config.allow_stub_backend:
                    raise EngineFailure(
                        "AI_STUB_FORBIDDEN",
                        "The stub backend requires ENGINE_ALLOW_STUB_BACKEND=true",
                    )
                backend: Backend = StubBackend()
            elif self.config.backend == "instantstyle":
                backend = InstantStyleBackend(self.config)
            else:
                raise EngineFailure("AI_BACKEND_INVALID", "Unknown AI backend")
            self._state = _LoadedState(backend, manifest, time.time())

    def readiness(self) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._state is not None
        manifest = self._state.manifest
        return {
            "status": "ready",
            "engine": ENGINE_ID,
            "backend": self.config.backend,
            "model_artifacts": manifest.artifact_count if manifest else None,
            "model_bytes": manifest.total_bytes if manifest else None,
            "manifest_sha256": manifest.manifest_sha256 if manifest else None,
            "inference_short_side": self.config.inference_short_side,
            "max_long_side": self.config.max_long_side,
        }

    def transfer(
        self,
        *,
        content: Image.Image,
        style: Image.Image,
        settings: TransferRequestSettings,
    ) -> TransferOutput:
        self.ensure_loaded()
        assert self._state is not None
        manifest_sha256 = (
            self._state.manifest.manifest_sha256
            if self._state.manifest is not None
            else None
        )
        if settings.style_strength == 0:
            report = validate_output(content, content)
            return TransferOutput(
                encode_png(content),
                {
                    "engine": ENGINE_ID,
                    "seed": settings.random_seed,
                    "model_manifest_sha256": manifest_sha256,
                    "identity_short_circuit": True,
                    "quality": report.as_dict(),
                },
            )
        content_model, content_transform = scale_short_side(
            content,
            self.config.inference_short_side,
            self.config.max_long_side,
            stride=64,
        )
        style_model, _ = scale_short_side(
            style,
            self.config.inference_short_side,
            self.config.max_long_side,
            stride=64,
        )
        attempt_settings = list(
            dict.fromkeys(
                [
                    (settings.style_strength, settings.structure_strength),
                    (
                        min(settings.style_strength, 0.6),
                        max(settings.structure_strength, 0.95),
                    ),
                ]
            )
        )
        last_failure: EngineFailure | None = None
        with self._inference_lock:
            for attempt, (style_strength, structure_strength) in enumerate(
                attempt_settings, 1
            ):
                try:
                    generated, backend_diagnostics = self._state.backend.generate(
                        content=content_model,
                        style=style_model,
                        style_strength=style_strength,
                        structure_strength=structure_strength,
                        inference_steps=settings.inference_steps,
                        seed=settings.random_seed,
                    )
                    restored = restore_from_resize(generated, content_transform)
                    report = validate_output(content, restored)
                    diagnostics = {
                        "engine": ENGINE_ID,
                        "seed": settings.random_seed,
                        "model_manifest_sha256": manifest_sha256,
                        "content_model_size": list(content_model.size),
                        "style_model_size": list(style_model.size),
                        "attempt": attempt,
                        "style_strength_requested": settings.style_strength,
                        "style_strength_applied": style_strength,
                        "structure_strength_requested": settings.structure_strength,
                        "structure_strength_applied": structure_strength,
                        "inference_steps": settings.inference_steps,
                        "quality": report.as_dict(),
                        **backend_diagnostics,
                    }
                    return TransferOutput(encode_png(restored), diagnostics)
                except EngineFailure as exc:
                    last_failure = exc
                    if exc.code == "AI_GPU_OUT_OF_MEMORY":
                        # An immediate identical retry cannot lower the tensor footprint.
                        # Let the worker's delayed retry handle transient GPU contention.
                        raise
                    if not exc.retryable:
                        raise
                    if attempt == len(attempt_settings):
                        raise EngineFailure(
                            exc.code, exc.message, retryable=False
                        ) from exc
                    LOGGER.warning(
                        "InstantStyle attempt %d failed quality/resource guard: %s",
                        attempt,
                        exc,
                    )
        assert last_failure is not None
        raise last_failure
