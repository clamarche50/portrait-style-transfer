# Architecture

## System shape

```text
Browser
  │ HTTPS, JSON, multipart, SSE
  ▼
Caddy ──► root Next.js/vinext web application
  │
  └─────► FastAPI /api/v1
             ├── PostgreSQL (metadata and durable job state)
             ├── Redis (Celery broker, progress, locks)
             └── private S3/MinIO (all image bytes and artifacts)
                         ▲
                         │
                    Celery worker
                         └── portrait_transfer package (CPU; optional CUDA dense alignment)
```

The web package is at repository root because the cloned starter is also configured for OpenAI Sites/Cloudflare hosting. Python services retain the service/package separation from the implementation specification. A Sites deployment can host the UI, but it does not replace the stateful API, worker, PostgreSQL, Redis, or private object storage.

## Deployment units

- `web`: root Next.js/vinext production server.
- `api`: short-running HTTP validation, authorization, persistence, signed URLs, and queue submission.
- `worker-cpu`: one-job-at-a-time CPU queue consumer.
- `worker-gpu`: opt-in queue consumer for CUDA dense correspondence.
- `postgres`, `redis`, `minio`: durable metadata, coordination, and private bytes.
- `caddy`: HTTPS and security headers.
- `prometheus`, `grafana`: optional local observability overlay.

Image processing never runs inside an HTTP request. PostgreSQL never stores image binaries. The API records object keys, dimensions, byte counts, hashes, expiry, settings, diagnostics, and state transitions; only owner-authorized signed URLs expose objects.

## Job lifecycle

Statuses are `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCEL_REQUESTED`, `CANCELLED`, and `EXPIRED`. Processing stages are:

1. validating and decoding;
2. face landmarks, segmentation, and quality analysis;
3. optional style-reference selection;
4. affine, line-morph, and dense alignment;
5. multiscale transfer;
6. eye highlights and background;
7. postprocessing and output upload;
8. completion.

The worker acquires an idempotency lock, transitions state transactionally, publishes progress after each stage, and checks cancellation between stages. It uploads output before committing success. Temporary files live in a unique job directory and are removed in `finally`. Only transient storage/database failures retry.

## Data model

- `users`: optional accounts; anonymous sessions remain supported.
- `assets`: private input, reference, style example, output, debug, and export metadata.
- `styles` and `style_examples`: owner-scoped collections, rights affirmation, quality/features.
- `jobs`: source selection, profile, settings, corrections, status, stage, progress, diagnostics, expiry.
- `job_artifacts`: links jobs to private outputs and diagnostic artifacts.

Authorization accepts an authenticated owner or a cryptographically protected anonymous session. Ownership checks happen before metadata reads, mutation, cancellation, corrections, deletion, or URL signing.

## Storage layout

Logical prefixes are `uploads/input/`, `uploads/reference/`, `styles/examples/`, `jobs/debug/`, `outputs/`, and `exports/`. Objects are private. Debug data inherits the job expiry. Upload normalization replaces client filenames and removes metadata before durable storage.

## Cache and correction invalidation

Cache keys include input/reference hashes, algorithm version, profile, settings, model versions, and correction hash. A mask edit invalidates mask-aware pyramids and all later stages; alignment invalidates maps and later stages; gain edits invalidate transfer onward; eye/background edits invalidate only their stage and final composition.

## Trust boundaries

- Browser input is untrusted and decoder-limited.
- Signed URLs are secrets and never logged.
- Redis is private coordination infrastructure, not an authorization source.
- Workers read only owner-authorized object keys recorded on the job.
- The local research workspace is untrusted and ignored; parser/audit tools never execute it.
- Model files are provisioned before startup and checked by readiness.

## Observability

JSON logs carry request/job identifiers, a hashed session identifier, stage, duration, worker, algorithm version, dense backend, dimensions, and safe outcome code. Metrics include queue/state counts, stage latency, upload sizes, worker memory, alignment fallback, validation codes, storage errors, and deletion lag. Optional OTLP/HTTP traces cover FastAPI requests and Celery task execution. They use only declared route templates or allowlisted static task names plus method/status/state; raw URLs and route values, UUIDs, query strings, headers/cookies/bodies, pixels, filenames, emails, object keys, task payloads/results, exception text, and signed URLs are excluded. A blank `OTEL_EXPORTER_OTLP_ENDPOINT` leaves tracing disabled.
