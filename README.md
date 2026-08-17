# Portrait Style Transfer

Portrait Style Transfer is a privacy-conscious web application for transferring the photographic appearance of a reference headshot to an input portrait while preserving the input subject's geometry. Its classical, portrait-specific pipeline is based on Shih et al., *Style Transfer for Headshot Portraits* (SIGGRAPH 2014). It is not neural style transfer, diffusion, face recognition, or image generation.

The production profile, `PAPER_EXACT`, combines landmark alignment, Beier-Neely line morphing, optional clean-room dense SIFT refinement, full-resolution masked Laplacian stacks, local-energy gain transfer, and reference-residual transfer. `SOURCE_2014_COMPAT` exists only for private regression work and is never a public product control.

> Licensing boundary: the uploaded paper and serialized MATLAB/C++ archive are local reference material. They are ignored by Git, excluded from container build contexts, and are not redistributed by this project. See [the licensing review](docs/licensing-review.md).

## Repository map

- `app/`, `public/`, and the root `package.json`: Next.js/vinext web application.
- `packages/portrait_transfer/`: independently tested Python image-processing package.
- `services/api/`: FastAPI API, persistence, and object-storage integration.
- `services/worker/`: Celery workers and retention tasks.
- `docs/`: architecture, algorithm, operations, privacy, and source-migration records.
- `scripts/`: model management, source auditing, cleanup, and benchmarking tools.
- `infra/`: Dockerfiles, Caddy, Prometheus, and Grafana configuration.

The root-level web layout is retained because this repository is also configured for OpenAI Sites/Cloudflare hosting. The containerized deployment remains the reference deployment for API-backed image processing.

## Prerequisites

- Node.js 22.22.2 or newer in the Node 22 line, and npm (the checked-in frontend currently uses `package-lock.json`).
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- GNU Make through WSL/Git Bash/Linux, or run the documented underlying commands directly.
- Docker Engine with Compose v2 for the complete local stack.
- At least 4 GB RAM for infrastructure plus worker memory; 8 GB is recommended.
- Optional NVIDIA Container Toolkit and a compatible CUDA GPU for the `gpu` profile.

No model is downloaded during a web request or by `make bootstrap`.

## Local setup

```sh
cp .env.example .env
make bootstrap
make models
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
make models                     explicitly download verified model artifacts
make lint                       run Python and web lint checks
make typecheck                  run Python and TypeScript checks
make test                       run required synthetic/mocked suites
make test-integration           run service integration tests
make test-web                   run web unit tests
make test-e2e                   run mocked-processing browser tests
make build                      create production package/web builds
make docker-build               build CPU container images
make smoke                      start Compose and check readiness
make benchmark                  run an explicitly configured benchmark command
make purge-expired              purge expired private assets
```

The full target list and required environment for targets needing private fixtures are documented in the `Makefile` and [validation protocol](docs/validation-protocol.md).

## Models

`models/manifest.json` pins the required MediaPipe assets to official download locations. `scripts/download_models.py` uses atomic replacement and validates SHA-256 whenever the manifest provides one. For air-gapped deployment, place files at their expected paths and verify them directly:

```sh
python scripts/download_models.py --manifest models/manifest.json --output-dir models --offline
```

Model binaries are ignored and are not part of releases. Review their upstream terms before redistribution.

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

Uploads are private, metadata is stripped, filenames are replaced, downloads use short-lived signed URLs, and the default retention period is 24 hours. Users can delete jobs and assets sooner. Uploads are not used for training, and the application stores neither face-recognition embeddings nor demographic labels.

Version 1 accepts one near-frontal, sufficiently large, in-focus headshot. It warns or rejects multiple faces, profiles, extreme pose, severe occlusion, blur, and poor input/reference compatibility. It does not change pose, expression, facial shape, perspective, focal length, or synthesize/remove hard shadows. See [privacy](docs/privacy.md), [algorithm details](docs/algorithm.md), and [troubleshooting](docs/troubleshooting.md).

## Deployment

The Compose stack runs web, API, CPU worker, PostgreSQL, Redis, MinIO, and Caddy. GPU and observability are explicit profiles/overlays. Production deployments must replace example secrets, use externally managed TLS and private storage, apply database migrations once, provide model files before readiness, and configure backups and lifecycle policies. See [deployment.md](docs/deployment.md).

## Research attribution

YiChang Shih, Sylvain Paris, Connelly Barnes, William T. Freeman, and Frédo Durand. “Style Transfer for Headshot Portraits.” *ACM Transactions on Graphics* 33(4), SIGGRAPH 2014.

The paper explains the method; the uploaded authors' archive was used only as a behavioral reference under the clean-room policy described in [source-audit.md](docs/source-audit.md) and [porting-map.md](docs/porting-map.md). No claim of exact published-result parity is made: the original prepared mattes, backgrounds, eye layers, candidate files, evaluation data, and a legally usable original SIFT Flow runtime are unavailable.

## Screenshots

No fabricated screenshot or real portrait is checked in. Release maintainers may add screenshots made from consented, rights-cleared inputs after privacy and publication review; record provenance alongside the assets.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Report security problems privately through the repository host's private vulnerability-reporting feature.
