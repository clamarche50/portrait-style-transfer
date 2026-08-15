# Algorithm

## Production profile

`ai_instantstyle_v1` is the only public transfer profile. It combines three
pinned, license-reviewed model sets into one diffusion pass:

- Stable Diffusion XL 1.0 (fp16) as the local generative backbone;
- InstantStyle (IP-Adapter SDXL weights) for reference appearance, injected
  only through the two style-sensitive attention blocks;
- IP-Adapter FaceID PlusV2 (SDXL) plus InsightFace embeddings to preserve the
  subject's identity.

The content upload is the structure/identity image. The reference upload is
the style and appearance image. No hosted inference provider, face-recognition
API, or runtime Hugging Face request is used. The service starts in offline
mode and loads only `/models/instantstyle` after manifest verification.

The deterministic analysis layer in `packages/portrait_transfer` performs
preflight, MediaPipe face analysis, segmentation, quality scoring, and
reference style ranking. It no longer renders pixels; the classical 2014
multiscale transfer pipeline has been removed.

## Preprocessing and restoration

Both payloads must decode as JPEG, PNG, or WebP and pass encoded-byte,
decoded-pixel, and minimum-dimension limits. EXIF orientation is applied and
input is converted to RGB.

Each image is scaled so its short side targets 1024 pixels (the SDXL native
resolution) without changing aspect ratio. Dimensions are rounded to
model-compatible multiples of 64, and the long side is capped at 1280 pixels
by default for the local 12 GiB GPU. There is no square padding or crop. The
content transform is retained and the generated result is resized back to the
original content dimensions. `ENGINE_MAX_LONG_SIDE` is an explicit
quality/VRAM control and must remain a multiple of 64 from 512 through 1536.

Inference fits the 12 GiB card through sequential CPU offload of the text
encoders and adapters, VAE slicing/tiling, and CPU-side CLIP vision encoders.

A post-inference guard rejects:

- changed output dimensions;
- non-RGB or non-finite pixels; and
- a large increase in border anisotropy associated with stretched-edge artifacts.

## User controls

- `style_strength` (`0..1`, default `0.75`) scales the InstantStyle attention injection.
- `structure_strength` (`0..1`, default `0.90`) scales the FaceID identity adapter.
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
specific error codes. Production never substitutes a test stub after an AI
error.

Safetensors artifacts are used for the SDXL and IP-Adapter weights. The FaceID
PlusV2 checkpoint is a legacy PyTorch file and is opened with restricted
`weights_only` semantics. Model files are read-only inside the container.

## Known limitations

Diffusion is generative. It can change facial details, apparent age,
expression, hair, glasses, jewelry, background, lighting, or perceived
identity. The FaceID adapter reduces drift but does not prove identity
preservation. The engine can also reproduce bias and unsafe tendencies
inherited from the Stable Diffusion XL training data.

The Stable Diffusion safety checker is disabled in this pipeline, and this
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

`models/instantstyle/manifest.json` is the machine-readable authority for the
runtime artifacts: the SDXL base tree, the InstantStyle/IP-Adapter weights and
CLIP image encoder, the FaceID PlusV2 checkpoint, and the InsightFace
buffalo_l ONNX pack. Hugging Face artifacts pin immutable commit revisions;
each downloaded file is additionally pinned by byte length and SHA-256. See
`THIRD_PARTY_NOTICES.md` before redistribution.
