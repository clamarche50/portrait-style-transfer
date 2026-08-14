# Deployment

## Local GPU Compose

The production transfer engine requires an NVIDIA GPU. On Windows, Docker
Desktop must use its Linux/WSL2 engine and expose the NVIDIA container runtime.
The reference host is a Ryzen 7 7800X3D, 32 GB RAM, and RTX 5070 with 12 GB VRAM.

1. Copy `.env.example` to `.env` and replace every development credential.
2. Install frontend/service dependencies with `make bootstrap`.
3. Provision both model sets with `make models`.
4. Verify all model bytes offline with `make verify-models`.
5. Apply the database migration once with `make migrate`.
6. Start with `docker compose up --build -d --wait --wait-timeout 900`.

The DGPST provisioning target uses pinned `huggingface-hub` and `gdown` tools.
It downloads no file during an API request or image build. For an air-gapped
host, copy the ignored model directory through an approved channel and run:

```sh
python scripts/download_models.py --manifest models/manifest.json --output-dir models --offline
python scripts/provision_dgpst_models.py --verify-only
```

A clean AI image build still needs the pinned Python base image, GitHub source,
PyTorch index, and Python package index. Build and scan the image on an approved
connected builder, export it through the organization's artifact process, then
load that exact image on the air-gapped host; copying model bytes alone is not
an offline image-build procedure.

Before first build, confirm the GPU runtime independently:

```sh
docker info
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The second command is an explicit, ephemeral GPU diagnostic. It downloads an
image if absent and must be run only where that is authorized.

## Runtime topology

The default stack includes `ai-engine`; it is not an optional profile. The
service:

- builds PyTorch 2.10.0 and torchvision 0.25.0 from the official CUDA 12.8 index;
- checks out DGPST commit `aada535bde5b87f9ece9a4af1c0628a93f46a342`;
- applies an asserted inference compatibility patch at image-build time;
- mounts `./models/dgpst` at `/models/dgpst` read-only;
- has no published port and joins only the internal `backend` network;
- verifies the 19-file manifest before `/health/ready` succeeds; and
- runs one Uvicorn process so only one model copy occupies the GPU.

Inference follows official short-side 512 scaling. `DGPST_MAX_LONG_SIDE=768`
caps portrait resolution for the reference RTX 5070; raise it only after a real
VRAM/latency benchmark, in multiples of 16, up to the service maximum of 1024.

The CPU Celery worker calls `http://ai-engine:8010/v1/transfer` with a bounded
timeout and optionally a bearer token. It remains responsible for private
object reads/writes, leases, cancellation state, retry policy, and terminal job
updates. `worker-gpu` is legacy dense-alignment tooling and is not needed for
the AI path. Keep `WORKER_TASK_TIME_LIMIT_SECONDS` above
`AI_ENGINE_REQUEST_TIMEOUT_SECONDS`; the defaults are 900 and 600 seconds so
object I/O and final state persistence still have a bounded grace window.
The internal sidecar retains the public API's 15 MiB encoded-image and 8 MP
decoded-pixel limits. The API rejects normalized representations that expand
past 15 MiB before they reach the worker or sidecar.

Default local limits are 10 GB RAM, 2 GB temporary memory, and 1 GB shared
memory for `ai-engine`, plus a 5 GB cap for each Celery worker. The 5 GB worker
cap is required by the public 8 MP image limit: full-resolution face analysis
can peak above 4 GB before the sidecar downsizes inference to 512/768 pixels.
Docker Desktop itself currently needs at least about 15 GB allocated to run the
full stack. Close competing GPU workloads if model
load reports insufficient VRAM; do not terminate unrelated services
automatically. Reserve at least 25 GB of Docker disk for the 8.3 GiB model tree,
the multi-gigabyte CUDA image, and transient build/cache layers.

## Vercel frontend and Cloudflare Tunnel

The hosted frontend uses `next.config.ts` to rewrite same-origin `/api/v1/*`
requests to the deployment's `API_ORIGIN`. Configure Vercel with the public
HTTPS hostname assigned to the Cloudflare Tunnel; do not put a trailing
`/api/v1` on `API_ORIGIN`.

The local stack adds `compose.tunnel.yml`. Cloudflared reads its token from
`.cloudflare-tunnel-token`, which is ignored by Git, and connects outbound to
Cloudflare. Configure the tunnel ingress to the Compose API service (port 8000),
not to a loopback-only MinIO, worker, database, or `ai-engine` endpoint.

Recommended runtime origin settings:

```text
APP_BASE_URL=https://portrait-style-transfer.vercel.app
PUBLIC_API_BASE_URL=/api/v1
CORS_ORIGINS=https://portrait-style-transfer.vercel.app
COOKIE_SECURE_OVERRIDE=true
AI_ENGINE_URL=http://ai-engine:8010
AI_ENGINE_API_TOKEN=<at-least-32-random-characters>
```

Production startup requires the same strong `AI_ENGINE_API_TOKEN` in the worker
and sidecar environments. Never put it in `NEXT_PUBLIC_*`, Vercel client bundles,
logs, or the tunnel configuration.

Start the API-backed hosted layout with:

```sh
docker compose -f compose.yml -f compose.tunnel.yml up --build -d --wait --wait-timeout 900 \
  postgres redis minio minio-init api ai-engine worker-cpu cloudflared
```

Vercel and Cloudflare are edge/proxy layers only. PostgreSQL, Redis, object
storage, job workers, models, and GPU inference remain on the private backend.

## Health, readiness, and rollout

- `/health/live` proves only that a process event loop is alive.
- API readiness checks database, Redis, object storage, and MediaPipe assets.
- AI readiness checks the CUDA runtime, pinned source/model initialization, and the complete DGPST manifest.
- `worker-cpu` starts only after `ai-engine` is healthy.

The first AI startup can take several minutes because it hashes roughly 8 GB
and loads the model. Compose grants a 600-second health start period. A checksum
failure is not recoverable by retry; replace the artifact through the explicit
provisioning workflow. Roll out API, worker, AI service, model manifest, and
frontend schema together when the public settings contract changes.

## Production checklist

- Replace passwords, session secrets, tunnel credentials, and the optional AI token through a secret manager.
- Use exact HTTPS origins, secure cookies, CSRF protection, and a strict CORS allowlist.
- Use managed PostgreSQL/Redis/S3 where possible and keep every data service private.
- Restrict the workers' non-published `egress` network at the host/firewall to
  approved managed S3 and OTLP destinations; `ai-engine` remains backend-only.
- Require S3 encryption and blocked public access; local MinIO's development `none` setting is not a production baseline.
- Provision and verify model artifacts before rolling traffic. Disable all runtime model downloads.
- Run migrations once as a release job before new API/worker instances.
- Keep one GPU request in flight until measured capacity proves higher concurrency safe.
- Apply platform memory, temporary-storage, GPU, and wall-clock limits.
- Scan Python/npm/container dependencies and reconcile an SBOM with model and source notices.
  Treat every unresolved high or critical finding as a release blocker unless a
  time-bounded, owner-approved exception documents exploitability and mitigation.
  `pip check` proves dependency consistency only; it is not a vulnerability audit.
- Complete human license review for DGPST weights and the OpenRAIL model before distribution or commercial claims.
- Treat public release as blocked until a reviewed local moderation strategy,
  acceptable-use policy, and abuse-reporting process cover the disabled Stable
  Diffusion safety checker.
- Do not enable automatic face masks until a correctly licensed, validated
  19-class face parser is pinned in the model manifest.
- Do not market output as identity-preserving, factual, or equivalent to published paper figures.

## Observability and retention

Optional OTLP tracing remains disabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is
blank. Telemetry may include static task/route names, status, stage duration,
model-manifest digest, and aggregate GPU memory. It must exclude image bytes,
embeddings, filenames, object keys, URLs, cookies, request bodies, and raw error
text.

Run the maintenance queue continuously or schedule `make purge-expired`.
Deletion removes object bytes before metadata is soft-deleted; operators must
also document backup expiration. Back up PostgreSQL and required object metadata
with encryption and access logs. Redis must never be the only durable job record.
