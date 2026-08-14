"""Apply the small, audited inference-only compatibility patch to pinned DGPST.

The Docker build checks out one exact upstream commit before invoking this file.
Every replacement is asserted so an upstream drift fails the image build instead
of silently producing a different engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(path: Path, old: str, new: str, *, count: int | None = None) -> None:
    source = path.read_text(encoding="utf-8")
    actual = source.count(old)
    expected = count if count is not None else 1
    if actual != expected:
        raise RuntimeError(
            f"Expected {expected} occurrences in {path} but found {actual}: {old!r}"
        )
    path.write_text(source.replace(old, new), encoding="utf-8")


def patch(root: Path) -> None:
    network_package = root / "models" / "networks" / "__init__.py"
    replace(network_package, "import util\n", "")

    adapter = root / "ip_adapter" / "ip_adapter_control.py"
    replace(adapter, "dtype=torch.float32", "dtype=self.pipe.unet.dtype", count=10)

    pipeline = root / "models" / "pipeline_dgpst.py"
    replace(
        pipeline,
        "self.vae.encode(image_sd.cuda())",
        "self.vae.encode(image_sd.to(device=device, dtype=self.vae.dtype))",
    )
    replace(
        pipeline,
        ".to(dtype=torch.float32)\n\n            extra_step_kwargs",
        ".to(dtype=self.unet.dtype)\n\n            extra_step_kwargs",
    )

    model = root / "models" / "DGPST_model.py"
    replace(model, "import lpips\n", "")
    replace(
        model,
        "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor\n",
        "",
    )
    replace(model, "from models.networks.drawingmodel import Generator\n", "")
    replace(model, "import util\n", "")
    replace(
        model,
        "parser.add_argument('--gamma_interpolate', default=1, type=int)",
        "parser.add_argument('--gamma_interpolate', default=0.75, type=float)",
    )
    replace(
        model,
        "controlnet_conditioning_scale=0.9",
        "controlnet_conditioning_scale=self.opt.structure_strength",
        count=8,
    )
    replace(
        model,
        "num_inference_steps=30",
        "num_inference_steps=self.opt.inference_steps",
        count=5,
    )
    replace(
        model,
        "mask=None, seed=None)\n            fea_inputA",
        "mask=None, seed=seed)\n            fea_inputA",
    )
    replace(
        model,
        "mask=None, seed=None)\n        B,C,H,W",
        "mask=None, seed=seed)\n        B,C,H,W",
    )
    replace(
        model,
        "mask=None, seed=None)\n        fea_inputA",
        "mask=None, seed=seed)\n        fea_inputA",
    )

    dift = root / "models" / "networks" / "dift_sd.py"
    replace(dift, "import matplotlib.pyplot as plt\n", "")
    replace(
        dift,
        "from diffusers.models.unet_2d_condition import UNet2DConditionModel",
        "from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel",
    )

    replace(
        pipeline,
        ".latent_dist.sample() * self.vae.config.scaling_factor",
        ".latent_dist.sample(generator=generator) * self.vae.config.scaling_factor",
    )
    replace(
        pipeline,
        "noise = torch.randn_like(latents).to(device)",
        "noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    patch(args.root.resolve())


if __name__ == "__main__":
    main()
