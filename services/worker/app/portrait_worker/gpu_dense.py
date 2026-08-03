from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from portrait_api.config import Settings
from portrait_transfer.alignment.flow_optimization import (
    CpuDenseCorrespondence,
    DenseRefinementResult,
    NoOpDenseCorrespondence,
)
from portrait_transfer.alignment.map_composition import compose_with_residual
from portrait_transfer.config import DenseSettings
from portrait_transfer.exceptions import OptionalDependencyError
from portrait_transfer.geometry.sampling import warp
from portrait_transfer.geometry.validity import map_validity
from portrait_transfer.types import AlignmentDiagnostics


@dataclass(slots=True)
class KorniaGpuDenseCorrespondence:
    """Quarter-scale DenseSIFT residual-flow optimization on an explicit CUDA device."""

    device: str = "cuda"
    maximum_long_edge: int = 384

    def _libraries(self) -> tuple[Any, Any]:
        try:
            import kornia
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "The GPU dense backend requires portrait-transfer[gpu]"
            ) from exc
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise OptionalDependencyError(
                "CUDA was requested but no CUDA runtime/device is available",
                device=self.device,
            )
        return torch, kornia

    @staticmethod
    def _grid(torch: Any, mapping: Any, source_shape: tuple[int, int]) -> Any:
        source_height, source_width = source_shape
        x = 2.0 * mapping[:, 0] / max(source_width - 1, 1) - 1.0
        y = 2.0 * mapping[:, 1] / max(source_height - 1, 1) - 1.0
        return torch.stack((x, y), dim=-1)

    def refine(
        self,
        *,
        input_crop: ArrayLike,
        reference_rgb: ArrayLike,
        initial_backward_map: ArrayLike,
        input_mask: ArrayLike,
        reference_mask: ArrayLike,
        settings: DenseSettings | None = None,
    ) -> DenseRefinementResult:
        settings = settings or DenseSettings()
        torch, kornia = self._libraries()
        functional = torch.nn.functional
        target_np = np.asarray(input_crop, dtype=np.float32)
        reference_np = np.asarray(reference_rgb, dtype=np.float32)
        initial_np = np.asarray(initial_backward_map, dtype=np.float32)
        input_mask_np = np.asarray(input_mask, dtype=np.float32)
        reference_mask_np = np.asarray(reference_mask, dtype=np.float32)
        if (
            target_np.ndim != 3
            or reference_np.ndim != 3
            or initial_np.shape != (*target_np.shape[:2], 2)
            or input_mask_np.shape != target_np.shape[:2]
            or reference_mask_np.shape != reference_np.shape[:2]
        ):
            raise ValueError("GPU dense inputs have incompatible shapes")

        full_height, full_width = target_np.shape[:2]
        scale = min(0.25, self.maximum_long_edge / max(full_height, full_width))
        height = max(32, round(full_height * scale))
        width = max(32, round(full_width * scale))
        scale_x = width / full_width
        scale_y = height / full_height

        def tensor(value: NDArray[np.float32]) -> Any:
            return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)

        target = tensor(target_np).permute(2, 0, 1)[None]
        reference = tensor(reference_np).permute(2, 0, 1)[None]
        target = functional.interpolate(target, size=(height, width), mode="bilinear")
        reference = functional.interpolate(reference, size=(height, width), mode="bilinear")
        target_gray = 0.2126 * target[:, :1] + 0.7152 * target[:, 1:2] + 0.0722 * target[:, 2:3]
        reference_gray = (
            0.2126 * reference[:, :1] + 0.7152 * reference[:, 1:2] + 0.0722 * reference[:, 2:3]
        )
        descriptor = kornia.feature.DenseSIFTDescriptor(
            patch_size=16,
            num_spatial_bins=4,
            num_ang_bins=8,
        ).to(self.device)
        target_features = descriptor(target_gray)
        reference_features = descriptor(reference_gray)
        if target_features.shape[-2:] != (height, width):
            target_features = functional.interpolate(
                target_features, size=(height, width), mode="bilinear"
            )
        if reference_features.shape[-2:] != (height, width):
            reference_features = functional.interpolate(
                reference_features, size=(height, width), mode="bilinear"
            )

        initial = tensor(initial_np).permute(2, 0, 1)[None]
        initial = functional.interpolate(initial, size=(height, width), mode="bilinear")
        initial[:, 0] *= scale_x
        initial[:, 1] *= scale_y
        target_support = tensor(input_mask_np)[None, None]
        target_support = functional.interpolate(
            target_support, size=(height, width), mode="bilinear"
        )
        reference_support = tensor(reference_mask_np)[None, None]
        reference_support = functional.interpolate(
            reference_support, size=(height, width), mode="bilinear"
        )

        residual = torch.zeros((1, 2, height, width), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([residual], lr=0.35)
        maximum = settings.max_displacement * max(scale_x, scale_y)

        def data_loss(flow: Any) -> Any:
            sampling = initial + flow
            grid = self._grid(torch, sampling, (height, width))
            warped_features = functional.grid_sample(
                reference_features, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            warped_mask = functional.grid_sample(
                reference_support, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            weights = (target_support * warped_mask).clamp(0.0, 1.0)
            difference = torch.sqrt((target_features - warped_features).pow(2) + 1e-6).mean(
                1, keepdim=True
            )
            return (difference * weights).sum() / weights.sum().clamp_min(1e-6)

        with torch.no_grad():
            before = float(data_loss(residual).detach().cpu())
        for _ in range(max(1, int(sum(settings.iterations)))):
            optimizer.zero_grad(set_to_none=True)
            data_term = data_loss(residual)
            smooth_x = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).pow(2).mean()
            smooth_y = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).pow(2).mean()
            magnitude = residual.pow(2).mean()
            loss = (
                data_term
                + settings.smoothness * (smooth_x + smooth_y)
                + settings.magnitude * magnitude
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True).clamp_min(1e-6)
                residual.mul_(torch.minimum(torch.ones_like(norm), maximum / norm))
        with torch.no_grad():
            after = float(data_loss(residual).detach().cpu())
            full_residual = functional.interpolate(
                residual, size=(full_height, full_width), mode="bilinear"
            )[0]
            full_residual[0] /= scale_x
            full_residual[1] /= scale_y
            residual_np = full_residual.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        candidate = compose_with_residual(initial_np, residual_np)
        reference_shape = (reference_np.shape[0], reference_np.shape[1])
        report = map_validity(candidate, reference_shape)
        improved = after <= before + settings.min_loss_improvement
        valid = bool(
            improved
            and report.valid_fraction >= settings.min_valid_fraction
            and report.negative_jacobian_fraction <= settings.max_negative_jacobian_fraction
        )
        selected = candidate if valid else initial_np
        selected_report = map_validity(selected, reference_shape)
        diagnostics = AlignmentDiagnostics(
            selected_stage="dense" if valid else "line",
            anchor_error=0.0,
            inlier_count=0,
            valid_fraction=selected_report.valid_fraction,
            negative_jacobian_fraction=selected_report.negative_jacobian_fraction,
            displacement_p50=selected_report.displacement_p50,
            displacement_p95=selected_report.displacement_p95,
            descriptor_loss_before=before,
            descriptor_loss_after=after,
            fallback_reason=None if valid else "gpu_dense_validation_failed",
            metadata={
                "backend": "kornia_cuda",
                "device": self.device,
                "optimization_height": height,
                "optimization_width": width,
            },
        )
        # Materialize once here so device failures happen before the pipeline marks progress.
        warp(reference_np, selected, mode="border")
        return DenseRefinementResult(selected, residual_np, valid, diagnostics)


def build_dense_backend(
    settings: Settings, *, enabled: bool
) -> CpuDenseCorrespondence | NoOpDenseCorrespondence | KorniaGpuDenseCorrespondence:
    if not enabled:
        return NoOpDenseCorrespondence()
    if settings.enable_gpu:
        if not settings.dense_alignment_device.lower().startswith("cuda"):
            raise OptionalDependencyError("GPU processing requires DENSE_ALIGNMENT_DEVICE=cuda")
        return KorniaGpuDenseCorrespondence(settings.dense_alignment_device)
    return CpuDenseCorrespondence()


__all__ = ["KorniaGpuDenseCorrespondence", "build_dense_backend"]
