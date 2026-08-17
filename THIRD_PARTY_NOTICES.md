# Third-party notices

This repository is a clean-room implementation informed by published research and public library APIs. The dependency lockfiles are the authoritative version inventory; release automation must generate an SBOM and reconcile it with this notice before distribution.

## Research reference

The algorithm is based on:

YiChang Shih, Sylvain Paris, Connelly Barnes, William T. Freeman, and Frédo Durand. “Style Transfer for Headshot Portraits.” *ACM Transactions on Graphics* 33(4), SIGGRAPH 2014.

The citation is attribution, not a software-license grant. The uploaded paper and serialized authors' archive are not included in this repository or its images. The archive has no clear top-level license and includes components with restrictive notices; see `docs/licensing-review.md`.

## Principal runtime libraries

The application is expected to include or interact with the following projects. Their own licenses and notices govern their code:

- FastAPI, Pydantic, SQLAlchemy, Alembic, Celery, boto3, NumPy, SciPy, scikit-image, Pillow, Prometheus client, and related Python dependencies.
- OpenCV, PyTorch, Torchvision, Kornia, and MediaPipe Tasks.
- Next.js, React, vinext, Vite, Tailwind CSS, and frontend dependencies.
- PostgreSQL, Redis, MinIO, Caddy, Prometheus, and Grafana container images.

Common upstream licenses include Apache-2.0, BSD, MIT, PostgreSQL, and AGPL licenses. MinIO server and Grafana distributions require particular attention for network deployment and redistribution. Do not rely on this summary in place of the exact notices shipped with locked versions.

## Model artifacts

The Face Landmarker and selfie multiclass segmentation artifacts listed in `models/manifest.json` are downloaded separately from official MediaPipe storage. Model artifacts are not covered by this repository's Apache-2.0 license. Review the MediaPipe model terms and any model-card restrictions before redistributing weights.

## Deliberately excluded software

The project does not redistribute or compile the uploaded SIFT Flow MEX/C++, MATLAB source, iris helper code, or photographer/example data. An externally provided research-only SIFT Flow adapter, if ever enabled, must remain disabled by default and requires independent licensing approval.

## Release checklist

Before a release:

1. Generate an SBOM from all Python, npm, and container lock data.
2. Collect all required license texts and attribution notices.
3. Review model terms and container-image licenses.
4. Run the copied-source compliance check.
5. Obtain human legal review before making a commercial-distribution claim.
