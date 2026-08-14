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

The active AI engine is based on Xinbo Wang, Wenju Xu, Qing Zhang, and Wei-Shi Zheng, “Domain Generalizable Portrait Style Transfer” (ICCV 2025, arXiv:2507.04243). Its official source is `https://github.com/wangxb29/DGPST`, integrated at commit `aada535bde5b87f9ece9a4af1c0628a93f46a342`. The source repository states the MIT License. The Docker build retrieves that exact commit and applies a small asserted inference-compatibility patch; it does not copy from the uploaded 2014 archive.

## Principal runtime libraries

The application is expected to include or interact with the following projects. Their own licenses and notices govern their code:

- FastAPI, Pydantic, SQLAlchemy, Alembic, Celery, boto3, NumPy, SciPy, scikit-image, Pillow, Prometheus client, and related Python dependencies.
- OpenCV, PyTorch, Torchvision, Diffusers, Transformers, Accelerate, safetensors, Kornia, and MediaPipe Tasks.
- Next.js, React, vinext, Vite, Tailwind CSS, and frontend dependencies.
- PostgreSQL, Redis, MinIO, Caddy, Prometheus, and Grafana container images.

Common upstream licenses include Apache-2.0, BSD, MIT, PostgreSQL, and AGPL licenses. MinIO server and Grafana distributions require particular attention for network deployment and redistribution. Do not rely on this summary in place of the exact notices shipped with locked versions.

## Model artifacts

The Face Landmarker and selfie multiclass segmentation artifacts listed in `models/manifest.json` are downloaded separately from official MediaPipe storage.

`models/dgpst/manifest.json` records the complete AI runtime set:

- Stable Diffusion v1.5 from `stable-diffusion-v1-5/stable-diffusion-v1-5` at revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, governed by CreativeML OpenRAIL-M. Its model card documents prohibited/misuse scenarios, bias, imperfect faces, and the need for safety controls.
- IP-Adapter full-face SD1.5 and its CLIP image encoder from `h94/IP-Adapter` at revision `018e402774aeeddd60609b4ecdb7e298259dc729`, whose repository/model card states Apache-2.0.
- The DGPST `CelebA_default` checkpoint from the Google Drive folder linked by the official DGPST README. The DGPST repository does not state a separate license for pretrained weights. Redistribution and commercial-use status are therefore `REVIEW_REQUIRED`, even though the source code is MIT-licensed.

Model artifacts are not covered by this repository's Apache-2.0 license. They are ignored by Git, excluded from build contexts, mounted locally, and must not be bundled with a release until their exact terms and notices have been reviewed. The PyTorch `.pth` checkpoint is pickle-capable; the runtime verifies its SHA-256 and uses `weights_only=True` before accepting it.

## Deliberately excluded software

The project does not redistribute or compile the uploaded SIFT Flow MEX/C++, MATLAB source, iris helper code, or photographer/example data. An externally provided research-only SIFT Flow adapter, if ever enabled, must remain disabled by default and requires independent licensing approval.

## Release checklist

Before a release:

1. Generate an SBOM from all Python, npm, and container lock data.
2. Collect all required license texts and attribution notices.
3. Review DGPST checkpoint rights, OpenRAIL-M obligations, IP-Adapter/MediaPipe terms, and container-image licenses.
4. Run the copied-source compliance check.
5. Obtain human legal review before making a commercial-distribution claim.
6. Resolve every high or critical dependency/image advisory, or record a
   time-bounded, owner-approved exception with exploitability and mitigation.

The manual `production-ai-image` workflow builds the real CUDA image, runs a
high/critical Trivy scan, and emits a CycloneDX SBOM. CPU contract CI uses a
stub image and is not a substitute for that release evidence. A successful
`pip check` confirms only that installed requirements are mutually compatible;
it does not clear known vulnerabilities.
