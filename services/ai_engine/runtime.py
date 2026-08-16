from __future__ import annotations

import gc
import importlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import ServiceConfig
from .contracts import EngineFailure, TransferOutput, TransferRequestSettings
from .landmarks import FaceKeypointDetector, render_keypoints
from .manifest import VerifiedManifest, verify_manifest
from .preprocessing import encode_png, restore_from_resize, scale_short_side
from .quality import validate_output

LOGGER = logging.getLogger("portrait_ai_engine")

ENGINE_ID = "ai_instantstyle_v1"


def denoise_strength_for(
    structure_strength: float, base: float, weight: float
) -> float:
    """Map the user's structure control to the img2img denoise strength.

    A high structure value keeps more of the source photograph: with the
    default base of 0.65 and weight 0.2, structure 1.0 repaints 45% of the
    timesteps and structure 0.0 repaints 65%. The source image stays the
    diffusion init in every case, which is what keeps the subject's identity
    anchored while the style adapters repaint texture.
    """
    strength = base - weight * structure_strength
    return max(0.45, min(0.9, strength))


def cosine_similarity(
    left: np.typing.NDArray[np.float32], right: np.typing.NDArray[np.float32]
) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _rgb_to_lab(rgb: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    """Convert an HxWx3 float RGB array (0..1) to Lab with a D65 white."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (
        linear
        @ np.asarray(
            [
                [0.4124564, 0.2126729, 0.0193339],
                [0.3575761, 0.7151522, 0.1191920],
                [0.1804375, 0.0721750, 0.9503041],
            ],
            dtype=np.float32,
        ).T
    )
    xyz = xyz / np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _lab_to_rgb(lab: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.float32]:
    """Invert _rgb_to_lab back to an HxWx3 float RGB array (0..1)."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > 0.206893, f**3, (f - 16.0 / 116.0) / 7.787)
    xyz = xyz * np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    linear = (
        xyz
        @ np.asarray(
            [
                [3.2404542, -0.9692660, 0.0556434],
                [-1.5371385, 1.8760108, -0.2040259],
                [-0.4985314, 0.0415560, 1.0572252],
            ],
            dtype=np.float32,
        ).T
    )
    rgb = np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )
    return np.clip(rgb, 0.0, 1.0)


def palette_transfer(
    output: Image.Image, reference: Image.Image, blend: float
) -> Image.Image:
    """Match the output's Lab statistics to the reference palette.

    Diffusion keeps the source photograph's color mood; a Reinhard-style
    mean/variance transfer in Lab space pulls the finished portrait toward
    the reference art's palette without touching geometry, so the identity
    guard's work is preserved. blend=1.0 is a full match; blend=0.0 returns
    the output unchanged.
    """
    if blend <= 0.0:
        return output
    source = np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0
    target = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    lab_source = _rgb_to_lab(source)
    lab_target = _rgb_to_lab(target)
    source_mean = lab_source.mean(axis=(0, 1))
    source_std = lab_source.std(axis=(0, 1)) + 1e-6
    target_mean = lab_target.mean(axis=(0, 1))
    target_std = lab_target.std(axis=(0, 1))
    matched = (lab_source - source_mean) * (target_std / source_std) + target_mean
    matched_rgb = _lab_to_rgb(matched) * 255.0
    blended = source * 255.0 * (1.0 - blend) + matched_rgb * blend
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), "RGB")


def face_anchor_image(
    output: Image.Image,
    source: Image.Image,
    source_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Composite the source face region back onto a styled output.

    Stylized passes can drift the face. Pasting the source face back under a
    feathered elliptical mask gives the final low-denoise harmonizing pass a
    strong identity anchor while the surrounding canvas keeps its style. When
    the painted face has moved (target_bbox), the pasted crop is resized to
    the target region so the anchor lands where the output face actually sits.
    """

    from PIL import ImageDraw, ImageFilter

    def expand(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        margin_x = (x2 - x1) * 0.35
        margin_y = (y2 - y1) * 0.45
        return (
            int(x1 - margin_x),
            int(y1 - margin_y),
            int(x2 + margin_x),
            int(y2 + margin_y),
        )

    width, height = output.size
    source_left, source_top, source_right, source_bottom = expand(source_bbox)
    source_crop = source.crop(
        (
            max(0, source_left),
            max(0, source_top),
            min(width, source_right),
            min(height, source_bottom),
        )
    )
    target_left, target_top, target_right, target_bottom = expand(
        target_bbox or source_bbox
    )
    left = max(0, target_left)
    top = max(0, target_top)
    right = min(width, target_right)
    bottom = min(height, target_bottom)
    if right - left < 8 or bottom - top < 8:
        return output.copy()
    pasted = output.copy()
    pasted.paste(
        source_crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS),
        (left, top),
    )
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((left, top, right, bottom), fill=255)
    radius = max(1.0, (right - left + bottom - top) / 16.0)
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(pasted, output, mask)


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
    """SDXL portrait engine: img2img init + InstantID ControlNet keypoints +
    InstantStyle (style) + FaceID PlusV2 (identity) adapters.

    The source portrait seeds the diffusion latents (img2img), the InstantID
    ControlNet locks the facial pose through keypoints, and the two IP
    adapters carry style and identity. A post-inference identity check
    compares the output face against the source face embedding and runs one
    repair pass with stronger identity conditioning when the match is weak.
    """

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
        self.landmarks = self._load_landmark_detector()

    def _load_pipeline(self) -> Any:
        from inspect import signature

        torch = self.torch
        model_root = self.config.model_root

        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        from transformers import CLIPVisionModelWithProjection

        load_options = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        variant_options = (
            {"variant": "fp16"} if self.config.dtype in {"float16", "bfloat16"} else {}
        )
        try:
            # The InstantID ControlNet is stored as a locally converted fp16
            # checkpoint (see scripts/provision_instantstyle_models.py); the
            # upstream fp32 file would double the mapped-memory footprint of
            # the container cgroup on this WSL2 host.
            controlnet = ControlNetModel.from_pretrained(
                str(model_root / "instantid" / "ControlNetModel"),
                local_files_only=True,
                torch_dtype=self.dtype,
                variant="fp16",
                use_safetensors=True,
            )
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                str(model_root / "sdxl-base"),
                controlnet=controlnet,
                **load_options,
                **variant_options,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            # InstantStyle injects style through two attention layers only; the
            # FaceID PlusV2 adapter preserves the subject's identity. Both
            # adapters must be registered in a single call: loading them
            # sequentially rebuilds the projection container from the last call
            # only, leaving the FaceID layer missing.
            pipe.load_ip_adapter(
                [str(model_root / "ip-adapter"), str(model_root / "faceid")],
                subfolder=["sdxl_models", None],
                weight_name=[
                    "ip-adapter_sdxl.safetensors",
                    "ip-adapter-faceid-plusv2_sdxl.bin",
                ],
                image_encoder_folder=None,
            )
            supported = set(signature(pipe.__call__).parameters)
            if (
                not {"control_image", "ip_adapter_image_embeds", "strength"}
                <= supported
            ):
                raise EngineFailure(
                    "AI_ADAPTER_LOAD_FAILED",
                    "The pinned diffusers build lacks img2img/IP-Adapter support",
                )
        except EngineFailure:
            raise
        except Exception as exc:
            raise EngineFailure(
                "AI_MODEL_LOAD_FAILED",
                "The style and identity model set could not be loaded",
            ) from exc
        projection = getattr(pipe.unet, "encoder_hid_proj", None)
        layers = (
            getattr(projection, "image_projection_layers", None) if projection else None
        )
        if not layers or len(layers) != 2:
            raise EngineFailure(
                "AI_ADAPTER_LOAD_FAILED",
                "The style and identity adapters could not be attached to the model",
            )
        # fp16 keeps both encoders inside the container memory budget; they
        # are moved to the GPU per-module by enable_model_cpu_offload anyway.
        style_clip = CLIPVisionModelWithProjection.from_pretrained(
            str(model_root / "ip-adapter" / "sdxl_models" / "image_encoder"),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).eval()
        face_clip = CLIPVisionModelWithProjection.from_pretrained(
            str(model_root / "ip-adapter" / "models" / "image_encoder"),
            local_files_only=True,
            torch_dtype=self.dtype,
        ).eval()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        # Keeps the RTX worker inside 12GB of VRAM by staging UNet/text
        # encoder/VAE/ControlNet transfers to the GPU only while they run.
        pipe.enable_model_cpu_offload()
        # Safetensors weights are memory-mapped lazily; the first inference
        # would otherwise fault every weight page into the container cgroup
        # in one burst, outpacing reclaim and tripping the OOM kill. Fault
        # them in gradually here, during warmup, where reclaim keeps up.
        for module in (
            pipe.unet,
            pipe.text_encoder,
            pipe.text_encoder_2,
            pipe.vae,
            pipe.controlnet,
            style_clip,
            face_clip,
        ):
            for parameter in module.parameters():
                parameter.data.sum()
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

    def _load_landmark_detector(self) -> Any:
        return FaceKeypointDetector(self.config.model_root / "insightface")

    def _set_adapter_scales(self, style_scale: float, faceid_scale: float) -> None:
        # InstantStyle injects style through up.block_0 only (the paper's
        # style-only site). The layout site (down.block_2) is deliberately
        # skipped: references that contain a figure would otherwise carry its
        # identity content into the source face. The middle attention layer
        # gets the scale while its neighbours stay disabled.
        self.pipe.set_ip_adapter_scale(
            [
                {
                    "up": {"block_0": [0.0, style_scale, 0.0]},
                },
                faceid_scale,
            ]
        )

    def _identity_similarity(self, image: Image.Image, source: Any) -> float:
        detected = self.faces.extract(image)
        if detected is None:
            return 0.0
        return cosine_similarity(source.embedding, detected.embedding)

    def _sanitized_style(self, style: Image.Image) -> tuple[Image.Image, bool]:
        """Blur any face detected in the reference before embedding.

        References often contain a figure; its facial identity competes with
        the source subject through the style adapter. Blurring the face region
        keeps the palette and brushwork in the style signal while removing the
        identity content.
        """

        from PIL import ImageFilter

        reference_face = self.faces.extract(style)
        if reference_face is None:
            return style, False
        width, height = style.size
        x1, y1, x2, y2 = reference_face.bbox
        margin_x = (x2 - x1) * 0.5
        margin_y = (y2 - y1) * 0.6
        left = max(0, int(x1 - margin_x))
        top = max(0, int(y1 - margin_y))
        right = min(width, int(x2 + margin_x))
        bottom = min(height, int(y2 + margin_y))
        if right - left < 8 or bottom - top < 8:
            return style, False
        region = style.crop((left, top, right, bottom))
        radius = max(2.0, (right - left + bottom - top) / 12.0)
        blurred = region.filter(ImageFilter.GaussianBlur(radius))
        sanitized = style.copy()
        sanitized.paste(blurred, (left, top))
        return sanitized, True

    def _style_embeds(self, style: Image.Image) -> Any:
        torch = self.torch
        inputs = self.style_clip_processor(images=style, return_tensors="pt")
        with torch.inference_mode():
            # The InstantStyle adapter projection consumes the bigG encoder's
            # pooled 1280-d embeddings, not its token hidden states.
            pooled = self.style_clip(**inputs).image_embeds
        embeds = torch.cat([torch.zeros_like(pooled), pooled], dim=0)
        # The pipeline input check requires 3D/4D embeds; the projection
        # flattens this token axis anyway.
        return embeds.unsqueeze(1).to(device="cuda", dtype=self.dtype)

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
        keypoints = self.landmarks.extract(content)
        style_scale = min(style_strength, self.config.style_scale_limit)
        faceid_scale = min(structure_strength, self.config.faceid_scale_limit)
        denoise_strength = denoise_strength_for(
            structure_strength,
            self.config.img2img_base_strength,
            self.config.img2img_structure_weight,
        )
        controlnet_scale = (
            self.config.controlnet_scale if keypoints is not None else 0.0
        )
        control_image = (
            render_keypoints(content.width, content.height, keypoints)
            if keypoints is not None
            else Image.new("RGB", content.size, (0, 0, 0))
        )
        before_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        best_output: Image.Image | None = None
        best_similarity = -1.0
        attempt_details: list[dict[str, Any]] = []
        style_for_embeds, style_face_masked = self._sanitized_style(style)
        try:
            with torch.inference_mode():
                style_embeds = self._style_embeds(style_for_embeds)
                id_embeds, clip_embeds = self._faceid_embeds(content, face)
                projection = getattr(self.pipe.unet, "encoder_hid_proj", None)
                layers = (
                    getattr(projection, "image_projection_layers", [])
                    if projection
                    else []
                )
                if len(layers) != 2:
                    raise EngineFailure(
                        "AI_ADAPTER_LOAD_FAILED",
                        "The style and identity adapters are not attached to the model",
                    )
                faceid_projection = layers[1]
                faceid_projection.clip_embeds = clip_embeds
                faceid_projection.shortcut = False
                passes: list[dict[str, Any]] = [
                    {
                        "style_scale": style_scale,
                        "faceid_scale": faceid_scale,
                        "denoise_strength": denoise_strength,
                        "controlnet_scale": controlnet_scale,
                        "init_image": content,
                    }
                ]
                pass_index = 0
                while pass_index < len(passes):
                    attempt_params = passes[pass_index]
                    self._set_adapter_scales(
                        attempt_params["style_scale"], attempt_params["faceid_scale"]
                    )
                    generator = torch.Generator(device="cpu").manual_seed(
                        seed + pass_index
                    )
                    result = self.pipe(
                        prompt=self.config.prompt,
                        negative_prompt=self.config.negative_prompt,
                        image=attempt_params["init_image"],
                        control_image=control_image,
                        controlnet_conditioning_scale=attempt_params[
                            "controlnet_scale"
                        ],
                        width=content.width,
                        height=content.height,
                        strength=attempt_params["denoise_strength"],
                        ip_adapter_image_embeds=[style_embeds, id_embeds],
                        num_inference_steps=inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                        eta=0.0,
                    )
                    output = result.images[0]
                    if output.size != content.size:
                        output = output.resize(content.size, Image.Resampling.LANCZOS)
                    similarity = self._identity_similarity(output, face)
                    attempt_details.append(
                        {
                            "attempt": pass_index + 1,
                            "identity_similarity": round(similarity, 3),
                            "style_scale": attempt_params["style_scale"],
                            "faceid_scale": attempt_params["faceid_scale"],
                            "denoise_strength": attempt_params["denoise_strength"],
                            "controlnet_scale": attempt_params["controlnet_scale"],
                        }
                    )
                    if similarity > best_similarity:
                        best_output = output
                        best_similarity = similarity
                    if similarity >= self.config.identity_repair_below:
                        break
                    if pass_index == 0:
                        # Stylized anchor: composite the source face back onto
                        # the styled output under a feathered mask, aligned to
                        # wherever the painted face drifted, then harmonize
                        # with a light repaint that keeps the brushwork ON the
                        # face. A photographic backstop is only used when this
                        # stylized attempt still misses the hard threshold.
                        output_face = self.faces.extract(best_output)
                        target_bbox = (
                            output_face.bbox if output_face is not None else face.bbox
                        )
                        passes.append(
                            {
                                "style_scale": max(style_scale * 0.8, 0.3),
                                "faceid_scale": self.config.faceid_scale_limit,
                                "denoise_strength": 0.33,
                                "controlnet_scale": min(controlnet_scale * 1.1, 1.0),
                                "init_image": face_anchor_image(
                                    best_output, content, face.bbox, target_bbox
                                ),
                            }
                        )
                    elif pass_index == 1 and best_output is not None:
                        if similarity >= self.config.identity_fail_below:
                            # The stylized anchor is acceptable: stop here so
                            # the photographic backstop cannot replace it.
                            break
                        # Identity backstop: the hard threshold is still
                        # missed, so repaint the pasted face as shallowly as
                        # possible and let identity win over style.
                        output_face = self.faces.extract(best_output)
                        target_bbox = (
                            output_face.bbox if output_face is not None else face.bbox
                        )
                        passes.append(
                            {
                                "style_scale": max(style_scale * 0.2, 0.1),
                                "faceid_scale": self.config.faceid_scale_limit,
                                "denoise_strength": 0.25,
                                "controlnet_scale": min(controlnet_scale * 1.2, 1.0),
                                "init_image": face_anchor_image(
                                    best_output, content, face.bbox, target_bbox
                                ),
                            }
                        )
                    pass_index += 1
                if (
                    best_output is None
                    or best_similarity < self.config.identity_fail_below
                ):
                    LOGGER.warning(
                        "identity guard: best=%.3f fail_below=%.3f attempts=%s "
                        "keypoints=%s denoise=%.3f style=%.3f faceid=%.3f",
                        best_similarity,
                        self.config.identity_fail_below,
                        attempt_details,
                        keypoints is not None,
                        denoise_strength,
                        style_scale,
                        faceid_scale,
                    )
                    if best_output is not None:
                        import base64 as _b64
                        import io as _io

                        _buffer = _io.BytesIO()
                        best_output.save(_buffer, format="PNG")
                        LOGGER.warning(
                            "identity guard best_attempt_png_b64=%s",
                            _b64.b64encode(_buffer.getvalue()).decode(),
                        )
                    raise EngineFailure(
                        "AI_IDENTITY_GUARD_FAILED",
                        "The generated portrait did not preserve the source face",
                        retryable=False,
                    )
                assert best_output is not None
                # Palette match: shift the finished portrait toward the
                # reference art's Lab statistics. The guard has already
                # anchored the face, so lower the blend (or drop the step)
                # if the color shift erodes identity.
                palette_blend = self.config.palette_blend
                matched_output = palette_transfer(best_output, style, palette_blend)
                matched_similarity = self._identity_similarity(matched_output, face)
                if matched_similarity < self.config.identity_repair_below:
                    reduced_blend = palette_blend * 0.5
                    reduced_output = palette_transfer(best_output, style, reduced_blend)
                    reduced_similarity = self._identity_similarity(reduced_output, face)
                    if reduced_similarity >= self.config.identity_repair_below:
                        best_output = reduced_output
                        best_similarity = reduced_similarity
                        palette_blend = reduced_blend
                    else:
                        palette_blend = 0.0
                else:
                    best_output = matched_output
                    best_similarity = matched_similarity
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise EngineFailure(
                "AI_GPU_OUT_OF_MEMORY",
                "The GPU does not have enough free memory; close GPU-heavy apps and retry",
                retryable=True,
            ) from exc
        finally:
            # Release the caching allocator's pooled blocks: on WSL2 the
            # reserved GPU segments are charged to the container cgroup, and
            # holding them between requests trips the container memory limit.
            torch.cuda.empty_cache()
            gc.collect()
        elapsed = time.perf_counter() - started
        peak = max(before_peak, torch.cuda.max_memory_allocated())
        return best_output, {
            "backend": "instantstyle_img2img_controlnet",
            "inference_seconds": round(elapsed, 3),
            "peak_vram_mib": round(peak / (1024 * 1024), 1),
            "torch_version": str(torch.__version__),
            "face_det_score": round(face.det_score, 3),
            "keypoints_detected": keypoints is not None,
            "style_face_masked": style_face_masked,
            "denoise_strength": round(denoise_strength, 3),
            "controlnet_scale": round(controlnet_scale, 3),
            "identity_similarity": round(best_similarity, 3),
            "palette_blend": round(palette_blend, 3),
            "identity_attempts": len(attempt_details),
            "identity_repair_below": self.config.identity_repair_below,
            "identity_fail_below": self.config.identity_fail_below,
            "identity_attempts_detail": attempt_details,
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
