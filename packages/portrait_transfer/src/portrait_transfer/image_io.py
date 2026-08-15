"""Validated, metadata-free image decoding and deterministic encoding."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.ndimage import zoom

from .config import ImageLimits
from .exceptions import DecodeError, ImageTooLargeError, InputValidationError


@dataclass(frozen=True)
class DecodedImage:
    rgb: NDArray[np.float32]
    sha256: str
    mime_type: str
    width: int
    height: int
    source_format: str


def normalize_rgb(
    image: ArrayLike, *, alpha_background: float = 0.5
) -> NDArray[np.float32]:
    """Return finite HxWx3 sRGB values in [0, 1]."""

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise InputValidationError(
            "Expected an HxW grayscale, RGB, or RGBA image", shape=array.shape
        )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)

    original_dtype = array.dtype
    array = array.astype(np.float32, copy=False)
    if np.issubdtype(original_dtype, np.integer):
        maximum = float(np.iinfo(original_dtype).max)
        array = array / maximum
    elif array.size and float(np.nanmax(array)) > 1.5:
        array = array / 255.0

    if array.shape[2] == 4:
        alpha = np.clip(array[..., 3:4], 0.0, 1.0)
        array = array[..., :3] * alpha + float(alpha_background) * (1.0 - alpha)
    else:
        array = array[..., :3]

    if not np.isfinite(array).all():
        raise InputValidationError("Image contains NaN or infinite values")
    if min(array.shape[:2]) < 2:
        raise InputValidationError("Image dimensions must both be at least two pixels")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _mime_for_format(source_format: str) -> str:
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[source_format]


def decode_image(payload: bytes, limits: ImageLimits | None = None) -> DecodedImage:
    """Decode an image after header-level allocation checks and EXIF orientation."""

    limits = limits or ImageLimits()
    if not payload:
        raise DecodeError("Image payload is empty")
    if len(payload) > limits.max_encoded_bytes:
        raise ImageTooLargeError(
            "Encoded image exceeds the configured limit",
            encoded_bytes=len(payload),
            limit=limits.max_encoded_bytes,
        )

    try:
        with Image.open(BytesIO(payload)) as probe:
            source_format = str(probe.format or "").upper()
            width, height = probe.size
            if source_format not in limits.allowed_formats:
                raise DecodeError(
                    "Unsupported image format", source_format=source_format
                )
            if (
                width * height > limits.max_decoded_pixels
                or max(width, height) > limits.max_original_long_edge
            ):
                raise ImageTooLargeError(
                    "Decoded dimensions exceed configured limits",
                    width=width,
                    height=height,
                )
            probe.verify()

        with Image.open(BytesIO(payload)) as opened:
            decoded = ImageOps.exif_transpose(opened)
            decoded.load()
            if "A" in decoded.getbands():
                rgba = np.asarray(decoded.convert("RGBA"))
                rgb = normalize_rgb(rgba)
            else:
                rgb = normalize_rgb(np.asarray(decoded.convert("RGB")))
    except ImageTooLargeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DecodeError(
            "Image decoder rejected the payload", detail=str(exc)
        ) from exc

    return DecodedImage(
        rgb=rgb,
        sha256=sha256(payload).hexdigest(),
        mime_type=_mime_for_format(source_format),
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        source_format=source_format,
    )


def _uint8_rgb(image: ArrayLike) -> NDArray[np.uint8]:
    return cast(
        NDArray[np.uint8], np.rint(normalize_rgb(image) * 255.0).astype(np.uint8)
    )


def encode_png(image: ArrayLike) -> bytes:
    """Encode deterministic RGB PNG bytes without metadata."""

    buffer = BytesIO()
    Image.fromarray(_uint8_rgb(image), mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def encode_jpeg(image: ArrayLike, quality: int = 95) -> bytes:
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    buffer = BytesIO()
    Image.fromarray(_uint8_rgb(image), mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()


def resize_long_edge(
    image: ArrayLike, long_edge: int, *, order: int = 1
) -> NDArray[np.float32]:
    array = np.asarray(image, dtype=np.float32)
    if long_edge < 2:
        raise ValueError("long_edge must be at least two")
    scale = min(1.0, float(long_edge) / max(array.shape[:2]))
    if scale == 1.0:
        return array.copy()
    factors = (scale, scale) if array.ndim == 2 else (scale, scale, 1.0)
    return cast(
        NDArray[np.float32],
        zoom(array, factors, order=order, mode="reflect", prefilter=order > 1).astype(
            np.float32
        ),
    )


def strip_metadata(image: ArrayLike) -> NDArray[np.float32]:
    """Arrays carry no EXIF/IPTC metadata; return a normalized detached copy."""

    return normalize_rgb(image).copy()
