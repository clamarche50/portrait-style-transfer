from __future__ import annotations

import gc
import importlib
import logging
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import ServiceConfig
from .contracts import EngineFailure, TransferOutput, TransferRequestSettings
from .manifest import VerifiedManifest, verify_manifest
from .preprocessing import encode_png, restore_from_resize, scale_short_side
from .quality import validate_output

LOGGER = logging.getLogger("portrait_ai_engine")


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


class DGPSTBackend:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.torch = importlib.import_module("torch")
        if config.device != "cuda" or not self.torch.cuda.is_available():
            raise EngineFailure(
                "AI_CUDA_UNAVAILABLE",
                "DGPST requires a CUDA GPU; the RTX worker could not access one",
            )
        upstream = str(config.upstream_root)
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        self.device = self.torch.device("cuda:0")
        self.dtype = getattr(self.torch, config.dtype)
        self.model = self._load_model()

    def _load_model(self) -> Any:
        torch = self.torch
        model_root = self.config.model_root
        base_path = model_root / "stable-diffusion-v1-5"
        controlnet_path = model_root / "checkpoints" / "CelebA_default"
        image_encoder_path = model_root / "ip_adapter" / "models" / "image_encoder"
        ip_adapter_path = (
            model_root
            / "ip_adapter"
            / "models"
            / "ip-adapter-full-face_sd15.safetensors"
        )
        checkpoint_path = controlnet_path / "latest_checkpoint.pth"

        from diffusers import ControlNetModel
        from ip_adapter.ip_adapter_control import IPAdapter2
        from models.DGPST_model import DGPSTModel, WavePool, WaveUnpool
        from models.networks.dift_sd import MyUNet2DConditionModel
        from models.pipeline_dgpst import DGPSTPipeline
        from src.config import RunConfig
        from src.eunms import Model_Type, Scheduler_Type
        from src.schedulers.ddim_scheduler import MyDDIMScheduler
        from torchvision import transforms

        opt = SimpleNamespace(
            auto_mask=False,
            checkpoints_dir=str(controlnet_path.parent),
            continue_train=False,
            gamma_interpolate=0.75,
            isTrain=False,
            lambda_Cycwarp=0.0,
            lambda_Maskwarp=0.0,
            local_rank=0,
            name=controlnet_path.name,
            num_gpus=1,
            post_process=False,
            pretrained_name=None,
            prompt_content="a photo of a portrait",
            prompt_output="a photo of a portrait",
            prompt_style="a photo of a portrait",
            region_style=False,
            resume_iter="latest",
            structure_strength=0.9,
            inference_steps=30,
            training_stage=2,
            up_ft_index=2,
        )
        model = DGPSTModel(opt)
        load_options = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        variant_options = {"variant": "fp16"} if self.config.dtype == "float16" else {}
        unet = MyUNet2DConditionModel.from_pretrained(
            str(base_path), subfolder="unet", **load_options, **variant_options
        )
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path), **load_options
        )
        pipe = DGPSTPipeline.from_pretrained(
            str(base_path),
            controlnet=controlnet,
            unet=unet,
            safety_checker=None,
            requires_safety_checker=False,
            **load_options,
            **variant_options,
        )
        pipe.scheduler = MyDDIMScheduler.from_config(pipe.scheduler.config)
        pipe.cfg = RunConfig(
            model_type=Model_Type.SD15,
            num_inference_steps=30,
            num_inversion_steps=10,
            num_renoise_steps=1,
            scheduler_type=Scheduler_Type.DDIM,
            perform_noise_correction=False,
            seed=0,
        )
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        model.ip_model = IPAdapter2(
            pipe, str(image_encoder_path), str(ip_adapter_path), self.device
        )
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
        )
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = [
            name
            for name in incompatible.unexpected_keys
            if not name.startswith("model.")
        ]
        if unexpected:
            LOGGER.info("Ignored %d training-only checkpoint keys", len(unexpected))
        del state
        gc.collect()

        model.to_tensor = transforms.Compose([transforms.ToTensor()])
        model.wav = WavePool(3)
        model.wavunpool = WaveUnpool(3)
        model.wav4 = WavePool(4)
        model.wavunpool4 = WaveUnpool(4)
        model.to(device=self.device, dtype=self.dtype)
        model.requires_grad_(False)
        model.eval()
        torch.cuda.empty_cache()
        return model

    def _tensor(self, image: Image.Image) -> Any:
        torch = self.torch
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )

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
        self.model.opt.gamma_interpolate = style_strength
        self.model.opt.structure_strength = structure_strength
        self.model.opt.inference_steps = inference_steps
        self.model.ip_model.pipe.cfg.seed = seed
        before_peak = torch.cuda.max_memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ),
            ):
                prediction, _ = self.model.generate(
                    real_B=self._tensor(style),
                    real_A=self._tensor(content),
                    mask=None,
                    mask_ref=None,
                    seed=seed,
                )
            if (
                prediction.ndim != 4
                or prediction.shape[0] < 1
                or prediction.shape[1] != 3
                or tuple(prediction.shape[-2:]) != (content.height, content.width)
                or not bool(torch.isfinite(prediction).all().item())
            ):
                raise EngineFailure(
                    "AI_QUALITY_GUARD_FAILED",
                    "The AI engine produced an invalid image tensor",
                    retryable=False,
                )
            output = (
                prediction[0]
                .detach()
                .float()
                .clamp(-1.0, 1.0)
                .add(1.0)
                .mul(127.5)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .round()
                .astype(np.uint8)
            )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EngineFailure(
                "AI_GPU_OUT_OF_MEMORY",
                "The GPU does not have enough free memory for DGPST; close GPU-heavy apps and retry",
                retryable=True,
            ) from exc
        finally:
            gc.collect()
        elapsed = time.perf_counter() - started
        peak = max(before_peak, torch.cuda.max_memory_allocated(self.device))
        return Image.fromarray(output), {
            "backend": "dgpst",
            "inference_seconds": round(elapsed, 3),
            "peak_vram_mib": round(peak / (1024 * 1024), 1),
            "torch_version": str(torch.__version__),
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
                        "The stub backend requires DGPST_ALLOW_STUB_BACKEND=true",
                    )
                backend: Backend = StubBackend()
            elif self.config.backend == "dgpst":
                backend = DGPSTBackend(self.config)
            else:
                raise EngineFailure("AI_BACKEND_INVALID", "Unknown AI backend")
            self._state = _LoadedState(backend, manifest, time.time())

    def readiness(self) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._state is not None
        manifest = self._state.manifest
        return {
            "status": "ready",
            "engine": "ai_dgpst_v1",
            "backend": self.config.backend,
            "model_artifacts": manifest.artifact_count if manifest else None,
            "model_bytes": manifest.total_bytes if manifest else None,
            "manifest_sha256": manifest.manifest_sha256 if manifest else None,
            "inference_size": self.config.inference_size,
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
                    "engine": "ai_dgpst_v1",
                    "seed": settings.random_seed,
                    "model_manifest_sha256": manifest_sha256,
                    "identity_short_circuit": True,
                    "quality": report.as_dict(),
                },
            )
        content_model, content_transform = scale_short_side(
            content, self.config.inference_size, self.config.max_long_side
        )
        style_model, _ = scale_short_side(
            style, self.config.inference_size, self.config.max_long_side
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
                        "engine": "ai_dgpst_v1",
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
                        "DGPST attempt %d failed quality/resource guard: %s",
                        attempt,
                        exc,
                    )
        assert last_failure is not None
        raise last_failure
