# Portrait Style Transfer

Portrait Style Transfer is a privacy-conscious web application that uses one headshot as structure/content and a second portrait as the visual-style reference. The production profile, `AI_DGPST_V1`, runs the ICCV 2025 Domain Generalizable Portrait Style Transfer model locally on an isolated GPU service. Stable Diffusion v1.5 supplies the generative backbone, while the DGPST ControlNet and full-face IP-Adapter condition structure and reference appearance.

The former clean-room implementation of Shih et al., *Style Transfer for Headshot Portraits* (SIGGRAPH 2014), remains in `packages/portrait_transfer` for tests and historical comparison; it is no longer the public transfer engine. AI output is synthesized and can change identity details despite structure conditioning. Treat every result as an edited/generated image, not a factual photograph.

> Licensing boundary: the uploaded paper and serialized MATLAB/C++ archive are local reference material. They are ignored by Git, excluded from container build contexts, and are not redistributed by this project. See [the licensing review](docs/licensing-review.md).

## Repository map

- `app/`, `public/`, and the root `package.json`: Next.js/vinext web application.
- `packages/portrait_transfer/`: independently tested Python image-processing package.
- `services/api/`: FastAPI API, persistence, and object-storage integration.
- `services/worker/`: Celery workers and retention tasks.
- `services/ai_engine/`: internal-only DGPST GPU inference API.
- `docs/`: architecture, algorithm, operations, privacy, and source-migration records.
- `scripts/`: model management, source auditing, cleanup, and benchmarking tools.
- `infra/`: Dockerfiles, Caddy, Prometheus, and Grafana configuration.

The root-level web layout is retained because this repository is also configured for OpenAI Sites/Cloudflare hosting. The containerized deployment remains the reference deployment for API-backed image processing.

## Prerequisites

- Node.js 22.22.2 or newer in the Node 22 line, and npm (the checked-in frontend currently uses `package-lock.json`).
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- GNU Make through WSL/Git Bash/Linux, or run the documented underlying commands directly.
- Docker Engine with Compose v2 for the complete local stack.
- At least 16 GB system RAM; 32 GB is recommended for local model loading.
- NVIDIA Container Toolkit and a CUDA GPU with about 12 GB VRAM. The reference workstation uses an RTX 5070 with Blackwell-compatible PyTorch CUDA 12.8 wheels.
- At least 25 GB of free Docker disk for the 8.3 GiB model tree, multi-gigabyte
  CUDA image, and transient build/cache layers.

No model is downloaded during a web request or by `make bootstrap`.

## Local setup

```sh
cp .env.example .env
make bootstrap
make models
make verify-models
make migrate
make dev
```

The Caddy endpoint is `https://localhost` by default and uses Caddy's local CA. Direct development endpoints are `http://localhost:3000` for the web app and `http://localhost:8000/api/v1/health/live` for the API. Required model absence intentionally makes readiness fail.

For web-only work:

```sh
npm ci
npm run dev
```

## Reference audit (local only)

Set `REFERENCE_ARCHIVE_FILE` to the uploaded serialized archive and keep `REFERENCE_SOURCE_DIR` under the ignored `reference/original-matlab/` path:

```sh
make extract-reference
make audit-reference
```

The parser rejects absolute paths, traversal, drive-prefixed paths, duplicate members, links, and NUL bytes. It writes only into the resolved extraction root. It never compiles or executes extracted code. The PDF is read manually; it is not copied into tracked files.

## Useful commands

```text
make bootstrap                  install locked dependencies
make models                     explicitly download all verified model artifacts
make models-dgpst               download only DGPST/SD1.5/IP-Adapter artifacts
make verify-models              hash-check every local runtime model offline
make lint                       run Python and web lint checks
make typecheck                  run Python and TypeScript checks
make test                       run required synthetic/mocked suites
make test-integration           run service integration tests
make test-web                   run web unit tests
make test-e2e                   run mocked-processing browser tests
make build                      create production package/web builds
make docker-build               build web/API/worker and GPU AI images
make smoke                      start Compose and check readiness
make benchmark                  run an explicitly configured benchmark command
make purge-expired              purge expired private assets
```

Standard CPU CI uses `compose.ci.yml`, which explicitly replaces DGPST with a
small deterministic contract stub and removes the GPU reservation. It proves
API/worker/sidecar integration only. Never use that overlay for deployment or
interpret it as real-model quality validation.

The full target list and required environment for targets needing private fixtures are documented in the `Makefile` and [validation protocol](docs/validation-protocol.md).

## Models

`models/manifest.json` pins the MediaPipe analysis assets. `models/dgpst/manifest.json` records all 19 DGPST, Stable Diffusion v1.5, and IP-Adapter runtime files by exact byte length and SHA-256. Hugging Face artifacts use immutable commit revisions; the mutable Google Drive checkpoint is pinned by its downloaded bytes. For air-gapped deployment, provision both trees and verify without network access:

```sh
python scripts/download_models.py --manifest models/manifest.json --output-dir models --offline
python scripts/provision_dgpst_models.py --verify-only
```

Model binaries are ignored, excluded from Docker build contexts, mounted read-only at runtime, and never downloaded in a request. Review [the model notices](THIRD_PARTY_NOTICES.md) before any redistribution; the DGPST checkpoint has no separately stated weight license.

## Tests and builds

Required CI uses synthetic images and mocked model adapters; no real face is committed. A separate opt-in suite reads rights-cleared private fixtures from ignored `tests/fixtures/private/`:

```sh
make test
make test-real-models
make build
make docker-build
```

Do not interpret a mocked-CV smoke test as validation of real model quality. Benchmark numbers are recorded only when the actual models and hardware were available.

## Privacy and limitations

Uploads are private, metadata is stripped, filenames are replaced, downloads use owner-authorized API routes, and the default retention period is 24 hours. Users can delete jobs and assets sooner. Uploads are not used for training, and the application stores neither face-recognition embeddings nor demographic labels. Inference stays inside the private Docker backend and makes no third-party model API call.

The engine is optimized for a single, clearly visible headshot and a portrait style reference. Diffusion is stochastic and may change facial details, accessories, text, background, lighting, or perceived identity. Seeded requests improve repeatability but do not make the output evidentiary or guarantee identity preservation. The upstream Stable Diffusion safety checker is disabled; public release requires a reviewed local moderation strategy and abuse controls. See [privacy](docs/privacy.md), [algorithm details](docs/algorithm.md), and [troubleshooting](docs/troubleshooting.md).

## Deployment

The Compose stack runs web, API, a light CPU Celery orchestrator, the internal `ai-engine` GPU sidecar, PostgreSQL, Redis, MinIO, and Caddy. The AI service has no published port and loads only the read-only, checksum-verified model mount. The current hosted layout uses Vercel for the Next.js frontend and a Cloudflare Tunnel to the local API; neither MinIO nor the inference sidecar is exposed. See [deployment.md](docs/deployment.md).

## Research attribution

Xinbo Wang, Wenju Xu, Qing Zhang, and Wei-Shi Zheng. “Domain Generalizable Portrait Style Transfer.” ICCV 2025, arXiv:2507.04243. The runtime integrates the official DGPST implementation at pinned commit `aada535bde5b87f9ece9a4af1c0628a93f46a342`; see [algorithm details](docs/algorithm.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

YiChang Shih, Sylvain Paris, Connelly Barnes, William T. Freeman, and Frédo Durand. “Style Transfer for Headshot Portraits.” *ACM Transactions on Graphics* 33(4), SIGGRAPH 2014.

The uploaded 2014 paper/archive remains a behavioral reference under the clean-room policy described in [source-audit.md](docs/source-audit.md) and [porting-map.md](docs/porting-map.md). The new AI engine does not claim bitwise or perceptual equivalence to either paper's published figures.

## Screenshots

No fabricated screenshot or real portrait is checked in. Release maintainers may add screenshots made from consented, rights-cleared inputs after privacy and publication review; record provenance alongside the assets.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Report security problems privately through the repository host's private vulnerability-reporting feature.
