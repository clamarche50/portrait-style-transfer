from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .contracts import EngineFailure


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    left: int
    top: int
    width: int
    height: int
    canvas_size: int


@dataclass(frozen=True, slots=True)
class ResizeTransform:
    original_width: int
    original_height: int
    model_width: int
    model_height: int


def decode_image(data: bytes, *, max_bytes: int, max_pixels: int) -> Image.Image:
    if not data or len(data) > max_bytes:
        raise EngineFailure("AI_INVALID_IMAGE", "Image payload is empty or too large")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"}:
                raise EngineFailure(
                    "AI_INVALID_IMAGE", "Use a JPEG, PNG, or WebP portrait"
                )
            width, height = opened.size
            if width < 64 or height < 64 or width * height > max_pixels:
                raise EngineFailure(
                    "AI_INVALID_IMAGE", "Portrait dimensions are outside safe limits"
                )
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except EngineFailure:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise EngineFailure(
            "AI_INVALID_IMAGE", "The portrait could not be decoded"
        ) from exc
    return image


def letterbox(image: Image.Image, size: int) -> tuple[Image.Image, LetterboxTransform]:
    original_width, original_height = image.size
    scale = min(size / original_width, size / original_height)
    width = max(1, min(size, round(original_width * scale)))
    height = max(1, min(size, round(original_height * scale)))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left = (size - width) // 2
    top = (size - height) // 2
    right = size - width - left
    bottom = size - height - top
    array = np.asarray(resized, dtype=np.uint8)
    pad_mode: Literal["reflect", "edge"] = (
        "reflect" if width > 1 and height > 1 else "edge"
    )
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode=pad_mode)
    transform = LetterboxTransform(
        original_width=original_width,
        original_height=original_height,
        left=left,
        top=top,
        width=width,
        height=height,
        canvas_size=size,
    )
    return Image.fromarray(padded), transform


def restore_from_letterbox(
    image: Image.Image, transform: LetterboxTransform
) -> Image.Image:
    if image.size != (transform.canvas_size, transform.canvas_size):
        image = image.resize(
            (transform.canvas_size, transform.canvas_size), Image.Resampling.LANCZOS
        )
    crop = image.crop(
        (
            transform.left,
            transform.top,
            transform.left + transform.width,
            transform.top + transform.height,
        )
    )
    return crop.resize(
        (transform.original_width, transform.original_height), Image.Resampling.LANCZOS
    )


def scale_short_side(
    image: Image.Image, short_side: int, max_long_side: int, *, stride: int = 16
) -> tuple[Image.Image, ResizeTransform]:
    """Scale the short side for inference while capping the long side for VRAM."""

    original_width, original_height = image.size
    scale = min(
        short_side / min(original_width, original_height),
        max_long_side / max(original_width, original_height),
    )
    width = max(stride, int(round((original_width * scale) / stride)) * stride)
    height = max(stride, int(round((original_height * scale) / stride)) * stride)
    width = min(width, max_long_side)
    height = min(height, max_long_side)
    transform = ResizeTransform(
        original_width=original_width,
        original_height=original_height,
        model_width=width,
        model_height=height,
    )
    return image.resize((width, height), Image.Resampling.LANCZOS), transform


def restore_from_resize(image: Image.Image, transform: ResizeTransform) -> Image.Image:
    if image.size != (transform.model_width, transform.model_height):
        image = image.resize(
            (transform.model_width, transform.model_height), Image.Resampling.LANCZOS
        )
    return image.resize(
        (transform.original_width, transform.original_height), Image.Resampling.LANCZOS
    )


def encode_png(image: Image.Image) -> bytes:
    target = io.BytesIO()
    image.convert("RGB").save(target, format="PNG", optimize=False, compress_level=6)
    return target.getvalue()
