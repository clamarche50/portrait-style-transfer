# Deployment

## Local Compose

1. Copy `.env.example` to `.env` and replace development credentials.
2. Run `make bootstrap`.
3. Run `make models`; readiness intentionally fails without required files.
4. Run `make migrate`.
5. Run `make dev`, or `docker compose up --build -d`.

Direct ports bind to loopback except Caddy. Caddy serves `https://localhost` with its internal CA. Trust that CA only on a development machine; use managed public certificates in production.

The default stack includes root-level web, FastAPI, one CPU worker, PostgreSQL, Redis, MinIO, and Caddy. Start optional services with:

```sh
docker compose --profile gpu up -d worker-gpu
docker compose -f compose.yml -f compose.observability.yml --profile observability up -d
```

GPU startup requires NVIDIA Container Toolkit plus a locked worker dependency set containing a compatible CUDA PyTorch build. `gpus: all` does not install a host driver.

For local Compose, the single CPU worker also runs Celery Beat and owns the hourly maintenance schedule. Production operators that scale CPU workers must run exactly one dedicated scheduler instead of enabling embedded Beat on every replica.

## Production checklist

- Replace every example password and secret with an external secret manager.
- Set exact HTTPS origins, secure-cookie flags, and a strict CORS allowlist.
- Use managed PostgreSQL/Redis/S3 where possible; keep them on private networks.
- Set `S3_SERVER_SIDE_ENCRYPTION=AES256` (or `aws:kms` with `S3_KMS_KEY_ID`) for production S3, plus version/lifecycle policy, blocked public access, and least-privilege credentials. Local MinIO defaults to `none` because the development stack does not provision KES; production settings reject that value.
- Provision models from `manifest.json` before rolling traffic; do not give request containers general download permission.
- Run migrations once as a release job before new API/worker instances.
- Run CPU and GPU queues separately, one memory-intensive job per worker process initially.
- Apply platform-enforced CPU, memory, temporary-storage, and wall-clock limits. Compose `mem_limit` is a local baseline, not a universal orchestrator policy.
- Use an external/publicly trusted TLS issuer. Preserve CSP, HSTS, `nosniff`, and no-referrer headers.
- Export logs/traces without pixels, object keys, signed URLs, emails, filenames, or raw session IDs.
- Configure backup/restore tests for metadata and object lifecycle. Deletion obligations must include backups according to published policy.
- Run image/SBOM vulnerability scans and license review for every release.

## Readiness and rollout

Liveness only proves the process loop. Readiness checks database, Redis, object storage/bucket, and every required model. Workers should perform an equivalent startup check before consuming queues. Roll out API and worker versions together when cache keys, job schemas, or algorithm versions change.

For zero-downtime schema changes, use expand/migrate/contract: add compatible columns, deploy dual-compatible code, backfill, then remove old fields in a later release.

## Optional OTLP tracing

Tracing is off by default. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to a reachable OTLP/HTTP collector base URL, such as `http://collector:4318`, or to its full `/v1/traces` endpoint. The API and workers append `/v1/traces` when the suffix is absent. An empty or whitespace-only value leaves the manual tracer inert: it starts no exporter thread, performs no telemetry network requests, and creates no worker task spans.

The API exports one server span per request using only the HTTP method, declared route template, and response status. Workers export one consumer span per allowlisted Celery task using only its static task name and terminal state. Raw/request URLs, route values (including UUIDs), query strings, headers, cookies, request bodies, filenames, image/object keys, task IDs/arguments/results, exception text, and signed URLs are never added to spans. The dependency-free exporter batches OTLP JSON off the request/task path and attempts a bounded flush during FastAPI lifespan shutdown and Celery process/worker shutdown. Protect the collector with TLS and collector-side access controls in production.

## Object retention

Run the maintenance queue continuously or schedule `make purge-expired`. The purge deletes object bytes before soft-deleting metadata, retries failure, and reports deletion lag. Do not treat an S3 lifecycle rule alone as proof that metadata and derived artifacts were removed.

## OpenAI Sites/Cloudflare UI option

`.openai/hosting.json` can host the root UI. D1/R2 are not substitutes for the specified service data plane unless the API/storage implementation is deliberately redesigned and revalidated. Configure the hosted UI's API origin and CORS without exposing MinIO or worker infrastructure.

## Backup and disaster recovery

Back up PostgreSQL and required object metadata with encryption and access logs. Redis task state is reconstructable from PostgreSQL and should not be the only durable job record. Document recovery-point and recovery-time objectives before production; none are promised by this repository.
