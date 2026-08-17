from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from portrait_transfer.color.lab import lab_to_rgb, lab_to_rgb_gamut_safe, rgb_to_lab
from portrait_transfer.color.legacy_color import legacy_lab_to_rgb, legacy_rgb_to_lab
from portrait_transfer.config import ImageLimits
from portrait_transfer.exceptions import ImageTooLargeError
from portrait_transfer.image_io import decode_image, encode_jpeg, encode_png


def test_standard_rgb_lab_round_trip_is_bounded(rng: np.random.Generator) -> None:
    rgb = rng.random((31, 29, 3), dtype=np.float32)
    restored = lab_to_rgb(rgb_to_lab(rgb), clip=True)
    assert np.max(np.abs(restored - rgb)) < 2e-5


def test_legacy_rgb_lab_round_trip_is_bounded(rng: np.random.Generator) -> None:
    rgb = rng.random((17, 19, 3), dtype=np.float32)
    restored = legacy_lab_to_rgb(legacy_rgb_to_lab(rgb))
    assert np.max(np.abs(restored - rgb)) < 2e-4


def test_gamut_safe_conversion_is_finite_and_ranged() -> None:
    lab = np.asarray([[[2.0, 4.0, -3.0], [np.nan, np.inf, -np.inf]]], dtype=np.float32)
    rgb = lab_to_rgb_gamut_safe(lab)
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_exif_orientation_applied_and_metadata_stripped() -> None:
    array = np.zeros((7, 11, 3), dtype=np.uint8)
    array[:, :3, 0] = 255
    source = Image.fromarray(array, mode="RGB")
    exif = source.getexif()
    exif[274] = 6
    payload_buffer = BytesIO()
    source.save(payload_buffer, format="JPEG", exif=exif, comment=b"private")
    decoded = decode_image(payload_buffer.getvalue())
    assert decoded.rgb.shape[:2] == (11, 7)
    clean_payload = encode_png(decoded.rgb)
    with Image.open(BytesIO(clean_payload)) as clean:
        assert len(clean.getexif()) == 0
        assert "comment" not in clean.info


def test_oversized_dimensions_rejected_before_array_decode() -> None:
    image = Image.new("RGB", (20, 20), color=(1, 2, 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    limits = ImageLimits(max_decoded_pixels=100, max_original_long_edge=100)
    with pytest.raises(ImageTooLargeError):
        decode_image(buffer.getvalue(), limits)


def test_png_is_byte_stable_and_jpeg_is_metadata_free(textured_rgb: np.ndarray) -> None:
    first = encode_png(textured_rgb)
    second = encode_png(textured_rgb.copy())
    assert first == second
    jpeg = encode_jpeg(textured_rgb, quality=91)
    with Image.open(BytesIO(jpeg)) as decoded:
        assert decoded.format == "JPEG"
        assert len(decoded.getexif()) == 0
