# Architecture

## System shape

```text
Browser
  | HTTPS, JSON, multipart, SSE
  v
Vercel Next.js frontend -- same-origin /api/v1 rewrite
  |
  v
Cloudflare Tunnel -- only the API origin is published
  |
  v
FastAPI /api/v1
  |-- PostgreSQL (metadata and durable job state)
  |-- Redis (Celery broker, progress, locks)
  `-- private S3/MinIO (input, reference, output bytes)
          ^
          |
    CPU Celery worker -- HTTP/multipart --> ai-engine:8010
                                            |
                                            `-- RTX GPU + read-only InstantStyle models
```

Local all-in-one deployments can put Caddy and the root Next.js server in front
of the same API. Vercel and Cloudflare replace only that public edge. They do not
replace the stateful API, worker, database, broker, object store, or AI engine.

## Deployment units

- `web`: root Next.js production server for local Compose.
- `api`: validation, authorization, persistence, private content routes, and queue submission.
- `worker-cpu`: one-job-at-a-time Celery orchestrator and the single local maintenance scheduler.
- `ai-engine`: internal FastAPI service that owns the InstantStyle model and GPU. It has no host port.
- `postgres`, `redis`, `minio`: durable metadata, coordination, and private object bytes.
- `cloudflared`: outbound tunnel client attached to the API's frontend network only.
- `caddy`: local HTTPS reverse proxy.
- `prometheus`, `grafana`: optional local observability overlay.

The CPU worker stays small and performs storage/job orchestration. A single
long-lived AI process loads the multi-gigabyte model once, verifies every
artifact before readiness, and serializes inference on one GPU. Model binaries
are bind-mounted read-only at `/models/instantstyle`; they are never baked into the
source image or downloaded while handling a request.

## Job lifecycle

Statuses are `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCEL_REQUESTED`,
`CANCELLED`, and `EXPIRED`.

1. The API validates ownership and stores the selected `ai_instantstyle_v1` settings.
2. The worker acquires the Redis job lease and downloads the private content and style objects.
3. It sends both images and AI-native settings to `POST http://ai-engine:8010/v1/transfer`.
4. The sidecar scales each image's short side toward 1024, preserves aspect
   ratio, rounds to multiples of 64, and caps the long side at 1280 by default.
5. InstantStyle generates one result using style, identity, step, and seed controls.
6. The sidecar restores the content dimensions and rejects invalid pixels, changed dimensions, or stretched-border artifacts.
7. The worker encodes the requested format, uploads output, commits success transactionally, and publishes the terminal event.

Cancellation remains cooperative around the inference request; an in-flight GPU
kernel cannot be interrupted safely. Infrastructure failures can retry, while
model, CUDA, validation, and quality failures return stable public error codes.

## Data model and storage

- `users`: optional accounts; anonymous sessions remain supported.
- `assets`: private input, reference, style example, output, debug, and export metadata.
- `styles` and `style_examples`: owner-scoped collections and rights affirmation.
- `jobs`: source selection, profile, AI settings, state, diagnostics, and expiry.
- `job_artifacts`: links jobs to private outputs and diagnostic artifacts.

PostgreSQL never stores image binaries. Logical object prefixes remain
`uploads/input/`, `uploads/reference/`, `styles/examples/`, `jobs/debug/`,
`outputs/`, and `exports/`. Content is served through owner-authorized API routes;
MinIO is not exposed through the tunnel.

## Trust boundaries

- Browser input is untrusted and decoder/size limited in both API and sidecar.
- Session cookies, CSRF tokens, tunnel credentials, and an optional internal AI bearer token are secrets.
- Redis is coordination infrastructure, never an authorization source.
- The AI engine is reachable only on the Compose `backend` network.
- Workers have a separate, non-published `egress` network for an explicitly
  configured managed S3 or OTLP endpoint. The AI engine is not attached to it.
- The API and queue workers receive individual MediaPipe file mounts, not the
  InstantStyle directory; only `ai-engine` can read the generative weights.
- The AI engine runs with a read-only root filesystem, dropped capabilities, no-new-privileges, and offline Hugging Face flags.
- Workers read only object keys already authorized and recorded on a job.
- The model manifest uses safe relative paths, exact lengths, and SHA-256 values; readiness fails closed on any drift.
- The local 2014 research workspace remains untrusted, ignored, and excluded from every image.

## Observability

The API and worker retain privacy-safe JSON logs, metrics, and optional OTLP
traces. AI diagnostics may include engine/profile identifiers, elapsed inference
time, seed, settings, model-manifest digest, and peak allocated CUDA memory. They
must not include pixels, prompts derived from private text, filenames, object
keys, signed URLs, cookies, raw session IDs, or image embeddings.
