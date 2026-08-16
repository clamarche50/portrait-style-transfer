# Algorithm

## Production profile

`ai_instantstyle_v1` is the only public transfer profile. It combines four
pinned, license-reviewed model sets into one diffusion pass:

- Stable Diffusion XL 1.0 (fp16) as the local generative backbone;
- InstantStyle (IP-Adapter SDXL weights) for reference appearance, injected
  only through the style-sensitive `up.block_0` attention site;
- IP-Adapter FaceID PlusV2 (SDXL) plus InsightFace buffalo_l embeddings to
  preserve the subject's identity;
- the InstantID facial-keypoint ControlNet, which locks the facial pose
  during repainting.

The content upload is the structure/identity image and also seeds the
latents: generation is image-to-image with the source portrait as the init,
so the subject's composition and identity are anchored by the input instead
of being regenerated from noise. The reference upload supplies style and
appearance through the style adapter only. No hosted inference provider,
face-recognition API, or runtime Hugging Face request is used. The service
starts in offline mode and loads only `/models/instantstyle` after manifest
verification.

The deterministic analysis layer in `packages/portrait_transfer` performs
preflight, MediaPipe face analysis, segmentation, quality scoring, and
reference style ranking. It no longer renders pixels; the classical 2014
multiscale transfer pipeline has been removed.

### Identity guard

After each pass the engine detects the output face with buffalo_l and
compares its ArcFace embedding against the source face embedding. When the
cosine similarity drops below `ENGINE_IDENTITY_REPAIR_BELOW` (default 0.45),
a stylized anchor pass runs: the source face region is composited back onto
the styled output under a feathered elliptical mask (aligned to wherever the
painted face drifted), then harmonized with a light repaint at reduced denoise
(`0.33`) and a strong style scale (`0.8 × style`) so the brushwork survives on
the pasted face. If that pass still misses the threshold but stays above
`ENGINE_IDENTITY_FAIL_BELOW` (default 0.30), it is accepted; otherwise one
photographic backstop pass repaints the pasted face as shallowly as possible
and lets identity win over style. If the best attempt still falls below
`ENGINE_IDENTITY_FAIL_BELOW`, the engine fails the request with
`AI_IDENTITY_GUARD_FAILED` instead of returning an unrecognizable subject.

After the guard passes, a Lab-space Reinhard-style palette match shifts the
finished portrait's color statistics toward the reference art
(`ENGINE_PALETTE_BLEND`, default `0.8`). The blend is halved or dropped if
re-measuring identity shows the color shift eroding the face below the repair
threshold, so the palette step never trades away the guard's work.

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
encoders, ControlNet, and adapters, VAE slicing/tiling, and CPU-side CLIP
vision encoders. The InstantID ControlNet is stored as a locally converted
fp16 checkpoint: the upstream fp32 file would double the mapped-memory
footprint charged to the WSL2 container cgroup.

A post-inference guard rejects:

- changed output dimensions;
- non-RGB or non-finite pixels; and
- a large increase in border anisotropy associated with stretched-edge artifacts.

## User controls

- `style_strength` (`0..1`, default `0.75`) scales the InstantStyle attention injection.
- `structure_strength` (`0..1`, default `0.90`) scales the source-anchoring effect:
  higher values repaint fewer denoising timesteps (img2img denoise strength
  `0.65 - 0.20 × structure_strength`, clamped to `[0.45, 0.9]`). The FaceID
  identity adapter runs at its full limit (`ENGINE_FACEID_SCALE_LIMIT`, default
  `1.0`) in every pass, and the InstantID ControlNet runs at
  `ENGINE_CONTROLNET_SCALE` (default `0.35`).
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

Safetensors artifacts are used for the SDXL, IP-Adapter, and converted
ControlNet weights. The FaceID PlusV2 checkpoint is a legacy PyTorch file and
is opened with restricted `weights_only` semantics. Model files are read-only
inside the container.

## Known limitations

Diffusion is generative. It can change facial details, apparent age,
expression, hair, glasses, jewelry, background, lighting, or perceived
identity. The img2img init, keypoint ControlNet, FaceID adapter, and identity
guard all reduce drift, but none of them proves identity preservation. The
identity guard compares ArcFace embeddings of the same person across stylized
renderings; it rejects clearly wrong subjects but can also false-fail on
unusual faces or heavy stylization. The engine can also reproduce bias and
unsafe tendencies inherited from the Stable Diffusion XL training data.

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
CLIP image encoder, the FaceID PlusV2 checkpoint, the locally fp16-converted
InstantID ControlNet, and the InsightFace buffalo_l and antelopev2 ONNX
packs. Hugging Face artifacts pin immutable commit revisions; each downloaded
file is additionally pinned by byte length and SHA-256. See
`THIRD_PARTY_NOTICES.md` before redistribution.
