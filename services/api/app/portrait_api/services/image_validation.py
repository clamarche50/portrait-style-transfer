from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import ClassVar, cast

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from portrait_api.config import Settings
from portrait_api.errors import AppError


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    data: bytes
    mime_type: str
    extension: str
    width: int
    height: int
    sha256: str
    source_format: str


class ImageNormalizer:
    _formats: ClassVar[dict[str, tuple[str, str]]] = {
        "JPEG": ("image/jpeg", "jpg"),
        "PNG": ("image/png", "png"),
        "WEBP": ("image/webp", "webp"),
    }
    _claimed_types: ClassVar[frozenset[str]] = frozenset(
        {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def normalize(self, data: bytes, claimed_content_type: str | None) -> NormalizedImage:
        if not data:
            raise AppError("EMPTY_UPLOAD", "The uploaded file is empty.", 400)
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(
                "UPLOAD_TOO_LARGE",
                "The uploaded image exceeds the encoded-size limit.",
                413,
                {"max_bytes": self.settings.max_upload_bytes},
            )
        if claimed_content_type and claimed_content_type.lower() not in self._claimed_types:
            raise AppError("UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, and WebP are accepted.", 415)

        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = self.settings.max_decoded_pixels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                self._verify(data)
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    source_format = str(source.format or "").upper()
                    if source_format not in self._formats:
                        raise AppError(
                            "UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, and WebP are accepted.", 415
                        )
                    if getattr(source, "n_frames", 1) != 1:
                        raise AppError("ANIMATED_IMAGE", "Animated images are not supported.", 422)
                    expected_mime, extension = self._formats[source_format]
                    if (
                        claimed_content_type
                        and claimed_content_type != "application/octet-stream"
                        and claimed_content_type.lower() != expected_mime
                    ):
                        raise AppError(
                            "MIME_MISMATCH",
                            "The file contents do not match the declared media type.",
                            415,
                        )
                    image = self._to_srgb(source)
                    image = ImageOps.exif_transpose(image)
                    image = self._flatten_alpha(image)
                    width, height = image.size
                    if width * height > self.settings.max_decoded_pixels:
                        raise AppError(
                            "IMAGE_TOO_LARGE",
                            "The decoded image exceeds the pixel limit.",
                            413,
                            {"max_pixels": self.settings.max_decoded_pixels},
                        )
                    if max(width, height) > self.settings.max_original_long_edge:
                        raise AppError(
                            "IMAGE_DIMENSION_TOO_LARGE",
                            "The image dimensions exceed the supported limit.",
                            413,
                            {"max_long_edge": self.settings.max_original_long_edge},
                        )
                    normalized = self._encode(image, source_format)
        except AppError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise AppError("DECOMPRESSION_BOMB", "The image dimensions are unsafe.", 413) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise AppError("INVALID_IMAGE", "The uploaded file is not a valid image.", 422) from exc
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

        return NormalizedImage(
            data=normalized,
            mime_type=expected_mime,
            extension=extension,
            width=width,
            height=height,
            sha256=hashlib.sha256(normalized).hexdigest(),
            source_format=source_format,
        )

    @staticmethod
    def _verify(data: bytes) -> None:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()

    @staticmethod
    def _to_srgb(source: Image.Image) -> Image.Image:
        icc_profile = source.info.get("icc_profile")
        if not icc_profile:
            return source.copy()
        try:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            output_profile = ImageCms.createProfile("sRGB")
            return cast(
                Image.Image,
                ImageCms.profileToProfile(source, input_profile, output_profile, outputMode="RGB"),
            )
        except (OSError, ValueError, ImageCms.PyCMSError):
            return source.copy()

    @staticmethod
    def _flatten_alpha(image: Image.Image) -> Image.Image:
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")

    @staticmethod
    def _encode(image: Image.Image, source_format: str) -> bytes:
        output = io.BytesIO()
        if source_format == "JPEG":
            image.save(output, "JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
        elif source_format == "WEBP":
            image.save(output, "WEBP", quality=95, method=6, exact=True)
        else:
            image.save(output, "PNG", compress_level=6, optimize=False)
        return output.getvalue()
