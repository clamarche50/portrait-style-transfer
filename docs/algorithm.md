# Algorithm

## Production profile

`AI_DGPST_V1` is the only public transfer profile. It integrates the official
implementation of Wang et al., *Domain Generalizable Portrait Style Transfer*
(ICCV 2025), pinned to commit
`aada535bde5b87f9ece9a4af1c0628a93f46a342`. The service applies a small,
asserted inference-only compatibility patch during the Docker build; every
replacement must match the pinned source exactly or the build fails.

The former `PAPER_EXACT` and `SOURCE_2014_COMPAT` numerical pipelines remain in
the repository for historical tests and private comparisons. They are not
accepted by the public job API and are not silent fallbacks for AI failure.

## Model composition

The inference graph combines:

- Stable Diffusion v1.5 as the local generative backbone;
- the official DGPST `CelebA_default` ControlNet and learned checkpoint;
- the SD1.5 full-face IP-Adapter and its CLIP vision encoder;
- DGPST's semantic/feature correspondence and wavelet-based networks.

The content upload is the structure image. The reference upload is the texture
and appearance image. No hosted inference provider, face-recognition API, or
runtime Hugging Face request is used. The service starts in offline mode and
loads only `/models/dgpst` after manifest verification.

## Preprocessing and restoration

Both payloads must decode as JPEG, PNG, or WebP and pass encoded-byte, decoded-
pixel, and minimum-dimension limits. EXIF orientation is applied and input is
converted to RGB.

Each image follows the official DGPST `scale_shortside` policy: its short side
is scaled toward 512 pixels without changing aspect ratio. Dimensions are
rounded to model-compatible multiples of 16, and the long side is capped at 768
pixels by default for the local 12 GiB GPU. There is no square padding or crop.
The content transform is retained and the generated result is resized back to
the original content dimensions. `DGPST_MAX_LONG_SIDE` is an explicit
quality/VRAM control and must remain a multiple of 16 from 512 through 1024.

A post-inference guard rejects:

- changed output dimensions;
- non-RGB or non-finite pixels; and
- a large increase in border anisotropy associated with stretched-edge artifacts.

The upstream quality fallback based on automatic semantic masks is deliberately
not enabled. Its referenced ADE20K SegFormer does not provide the required
19-class face topology, while a known matching face parser is restricted to
non-commercial research/education. A mask mode must not ship until a correctly
licensed 19-class parser is pinned, validated, and added to the manifest.

## User controls

- `style_strength` (`0..1`, default `0.75`) controls interpolation toward reference appearance.
- `structure_strength` (`0..1`, default `0.90`) controls DGPST ControlNet conditioning.
- `inference_steps` (`10..50`, default `30`) trades latency for denoising refinement.
- `random_seed` (`0..2^31-1`) selects the diffusion noise sequence.

The same inputs, model artifacts, settings, and seed are intended to be
repeatable on the same runtime, but CUDA kernels and dependency changes can
still produce small differences. Diagnostics record the engine, controls,
elapsed time, seed, manifest identity, and peak allocated GPU memory.

## Safety and failure behavior

The service eagerly loads and verifies models before becoming ready. Missing,
truncated, or modified files fail closed. CUDA unavailability, insufficient GPU
memory, invalid model state, invalid input, and output-quality failures use
specific error codes. Production never substitutes a test stub or the legacy
classical engine after an AI error.

`latest_checkpoint.pth` is verified before loading and is opened with
`torch.load(..., weights_only=True, mmap=True)`. Other learned weights use
safetensors. Model files are read-only inside the container.

## Known limitations

DGPST is generative. It can change facial details, apparent age, expression,
hair, glasses, jewelry, background, lighting, or perceived identity. Structure
conditioning reduces drift but does not prove identity preservation. It can
also reproduce bias and unsafe tendencies inherited from its training data and
Stable Diffusion v1.5.

The Stable Diffusion safety checker is disabled in the DGPST pipeline, and this
repository does not yet include an equivalent local input/output moderation
model. Public release is blocked on explicit acceptable-use/reporting controls
and a reviewed local moderation strategy; private-network isolation is not
content-safety enforcement.

Outputs must be presented as AI-edited/generated portraits. Do not use them as
identity evidence, without the depicted person's permission, for deceptive
impersonation, or to infer factual attributes. Real-quality validation requires
rights-cleared, diverse portrait pairs; synthetic and stub tests establish only
software behavior.

## Provenance

`models/dgpst/manifest.json` is the machine-readable authority for the exact 19
runtime artifacts. Hugging Face files pin immutable commit revisions. The DGPST
authors publish their checkpoint through a mutable Google Drive folder, so each
downloaded artifact is pinned by byte length and SHA-256. See
`THIRD_PARTY_NOTICES.md` and `docs/licensing-review.md` before redistribution.
