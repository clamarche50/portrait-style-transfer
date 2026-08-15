# Third-party notices

This repository is a clean-room implementation informed by published research
and public library APIs. Project lockfiles are the authoritative inventory for
the web, API, worker, and classical package. The AI image currently pins every
direct Python requirement plus its PyTorch/CUDA pair, but does not yet carry a
fully resolved hash lock. Its release inventory is therefore the SBOM generated
from the built image; release automation must reconcile that SBOM with this
notice before distribution.

## Research reference

The algorithm is based on:

YiChang Shih, Sylvain Paris, Connelly Barnes, William T. Freeman, and Frédo Durand. “Style Transfer for Headshot Portraits.” *ACM Transactions on Graphics* 33(4), SIGGRAPH 2014.

The citation is attribution, not a software-license grant. The uploaded paper and serialized authors' archive are not included in this repository or its images. The archive has no clear top-level license and includes components with restrictive notices; see `docs/licensing-review.md`.

The active AI engine combines Stable Diffusion XL 1.0 with InstantStyle (Xiaoxiao Wu et al., “InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation”, 2024) and the IP-Adapter FaceID PlusV2 identity adapter (Hu Ye et al., “IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models”, 2023). Inference runs entirely through pinned diffusers/transformers libraries against locally mounted weights; the engine copies no upstream inference source code.

## Principal runtime libraries

The application is expected to include or interact with the following projects. Their own licenses and notices govern their code:

- FastAPI, Pydantic, SQLAlchemy, Alembic, Celery, boto3, NumPy, SciPy, scikit-image, Pillow, Prometheus client, and related Python dependencies.
- OpenCV, PyTorch, Torchvision, Diffusers, Transformers, Accelerate, PEFT, safetensors, InsightFace, ONNX Runtime, and MediaPipe Tasks.
- Next.js, React, vinext, Vite, Tailwind CSS, and frontend dependencies.
- PostgreSQL, Redis, MinIO, Caddy, Prometheus, and Grafana container images.

Common upstream licenses include Apache-2.0, BSD, MIT, PostgreSQL, and AGPL licenses. MinIO server and Grafana distributions require particular attention for network deployment and redistribution. Do not rely on this summary in place of the exact notices shipped with locked versions.

## Model artifacts

The Face Landmarker and selfie multiclass segmentation artifacts listed in `models/manifest.json` are downloaded separately from official MediaPipe storage.

`models/instantstyle/manifest.json` records the complete AI runtime set:

- Stable Diffusion XL 1.0 base (fp16) from `stabilityai/stable-diffusion-xl-base-1.0`, governed by Stability AI's OpenRAIL++-M community license. Its model card documents prohibited/misuse scenarios, bias, and the need for safety controls.
- The InstantStyle IP-Adapter SDXL weights and SDXL CLIP image encoder from `h94/IP-Adapter`, whose repository/model card states Apache-2.0.
- The FaceID PlusV2 SDXL checkpoint from `h94/IP-Adapter-FaceID`, whose repository states Apache-2.0 for code; weight terms follow that card.
- The InsightFace buffalo_l ONNX pack from `public-data/insightface`. InsightFace pretrained models are offered for non-commercial research purposes; commercial use requires separate review.

Model artifacts are not covered by this repository's Apache-2.0 license. They are ignored by Git, excluded from build contexts, mounted locally, and must not be bundled with a release until their exact terms and notices have been reviewed. The FaceID `.bin` checkpoint is pickle-capable; the runtime verifies its SHA-256 and loads it with restricted `weights_only` semantics before accepting it.

## Deliberately excluded software

The project does not redistribute or compile the uploaded SIFT Flow MEX/C++, MATLAB source, iris helper code, or photographer/example data. An externally provided research-only SIFT Flow adapter, if ever enabled, must remain disabled by default and requires independent licensing approval.

## Release checklist

Before a release:

1. Generate an SBOM from all Python, npm, and container lock data.
2. Collect all required license texts and attribution notices.
3. Review SDXL OpenRAIL++-M obligations, InsightFace non-commercial model terms, IP-Adapter/MediaPipe terms, and container-image licenses.
4. Run the model-binary tracking guard.
5. Obtain human legal review before making a commercial-distribution claim.
6. Resolve every high or critical dependency/image advisory, or record a
   time-bounded, owner-approved exception with exploitability and mitigation.

The manual `production-ai-image` workflow builds the real CUDA image, runs a
high/critical Trivy scan, and emits a CycloneDX SBOM. CPU contract CI uses a
stub image and is not a substitute for that release evidence. A successful
`pip check` confirms only that installed requirements are mutually compatible;
it does not clear known vulnerabilities.
