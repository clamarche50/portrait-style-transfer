# DGPST runtime models

This directory is a local, read-only runtime mount for the AI engine. Large
weights and Hugging Face cache metadata are ignored by Git and excluded from
Docker build contexts. Only this file and `manifest.json` are versioned.

Provision and verify the exact pinned artifact set:

```sh
make models-dgpst
python scripts/provision_dgpst_models.py --verify-only
```

The Make target supplies the pinned `huggingface-hub` and `gdown` provisioning
dependencies through `uv`; the offline verification command uses only the
Python standard library.

The manifest paths are relative to this directory. The Stable Diffusion and
IP-Adapter entries pin immutable Hugging Face revisions. The DGPST authors
publish their checkpoint through a mutable Google Drive folder, so its local
SHA-256 and byte length are the authoritative pin.

`latest_checkpoint.pth` is a pickle-capable PyTorch container. Never load an
unverified copy, and use `torch.load(..., weights_only=True)` where the upstream
structure permits it. The sibling safetensors files do not execute pickle.

The DGPST checkpoint has no separate weight license in the upstream repository.
Do not redistribute it or claim commercial-use rights without human review.
