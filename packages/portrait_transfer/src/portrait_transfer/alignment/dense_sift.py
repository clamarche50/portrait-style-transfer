"""Dense descriptor protocol and dependency-optional extractors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, sobel

from ..config import DenseSettings
from ..exceptions import OptionalDependencyError


def grayscale(rgb: ArrayLike) -> NDArray[np.float32]:
    value = np.asarray(rgb, dtype=np.float32)
    if value.ndim == 2:
        return value
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError("expected grayscale or RGB image")
    return cast(
        NDArray[np.float32],
        (
            0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]
        ).astype(np.float32),
    )


def local_contrast_normalize(
    image: ArrayLike, sigma: float = 2.0, epsilon: float = 1e-3
) -> NDArray[np.float32]:
    value = grayscale(image)
    mean = cast(
        NDArray[np.float32], gaussian_filter(value, sigma=sigma, mode="reflect")
    )
    variance = cast(
        NDArray[np.float32],
        gaussian_filter((value - mean) ** 2, sigma=sigma, mode="reflect"),
    )
    return cast(
        NDArray[np.float32],
        ((value - mean) / np.sqrt(variance + epsilon**2)).astype(np.float32),
    )


def dense_descriptor(
    image: ArrayLike, *, sigma: float = 1.0, orientation_bins: int = 8
) -> NDArray[np.float32]:
    """Small clean-room dense gradient descriptor suitable for CPU fallback.

    It is intentionally not claimed as bit-compatible with the archived MEX
    descriptor. The optional Kornia adapter supplies a denser SIFT-style
    research backend when that separately installed dependency is available.
    """

    normalized = local_contrast_normalize(image)
    gx = cast(NDArray[np.float32], sobel(normalized, axis=1, mode="reflect") / 8.0)
    gy = cast(NDArray[np.float32], sobel(normalized, axis=0, mode="reflect") / 8.0)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    channels: list[NDArray[np.float32]] = [normalized, gx, gy, magnitude]
    for index in range(orientation_bins):
        center = -np.pi + (2.0 * np.pi * index / orientation_bins)
        response = np.maximum(np.cos(angle - center), 0.0) ** 3 * magnitude
        channels.append(
            cast(
                NDArray[np.float32],
                gaussian_filter(response, sigma=sigma, mode="reflect"),
            )
        )
    descriptor = np.stack(channels, axis=-1).astype(np.float32)
    norm = np.sqrt(np.sum(descriptor * descriptor, axis=-1, keepdims=True) + 1e-6)
    return cast(NDArray[np.float32], descriptor / norm)


class DenseCorrespondenceBackend(Protocol):
    refine: Callable[..., Any]


@dataclass(frozen=True)
class KorniaDenseDescriptorExtractor:
    patch_size: int = 16
    num_spatial_bins: int = 4
    num_ang_bins: int = 8

    def extract(self, image: ArrayLike) -> NDArray[np.float32]:
        try:
            import kornia
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "Kornia dense descriptors require the 'gpu' package extra",
                extra="gpu",
            ) from exc
        gray = grayscale(image)
        tensor = torch.from_numpy(gray).float()[None, None]
        descriptor = kornia.feature.DenseSIFTDescriptor(
            patch_size=self.patch_size,
            num_spatial_bins=self.num_spatial_bins,
            num_ang_bins=self.num_ang_bins,
        )(tensor)
        return cast(
            NDArray[np.float32],
            descriptor.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32),
        )


def refine_with_dense_sift(
    *,
    input_crop: ArrayLike,
    reference_rgb: ArrayLike,
    initial_backward_map: ArrayLike,
    input_mask: ArrayLike,
    reference_mask: ArrayLike,
    backend: DenseCorrespondenceBackend,
    settings: DenseSettings | None = None,
) -> Any:
    return backend.refine(
        input_crop=input_crop,
        reference_rgb=reference_rgb,
        initial_backward_map=initial_backward_map,
        input_mask=input_mask,
        reference_mask=reference_mask,
        settings=settings or DenseSettings(),
    )
