# Troubleshooting

## Readiness is unhealthy

Inspect both `GET /api/v1/health/ready` and the internal AI service's
`GET /health/ready`. API readiness covers PostgreSQL, Redis, object storage,
buckets, and MediaPipe assets. AI readiness covers the complete DGPST manifest,
CUDA availability, and model initialization. Readiness never downloads weights.

Run the offline checks first:

```sh
make verify-models
docker compose config --quiet
docker compose run --rm ai-engine python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## DGPST model provisioning fails

The tree contains 19 pinned files and occupies about 8.3 GiB. Confirm sufficient
disk space and access to the official Hugging Face repositories and the Google
Drive folder linked by DGPST. Partial downloads never replace a valid artifact.
If a source changes bytes, do not weaken verification: review provenance and
licensing, then update the manifest deliberately.

Air-gapped deployments must pre-provision the files and run
`python scripts/provision_dgpst_models.py --verify-only` before starting Compose.

## AI service cannot see the GPU

On the host, check `nvidia-smi`. In Docker Desktop, enable the WSL 2 backend and
NVIDIA GPU support, then run the container diagnostic above. The AI image pins a
CUDA 12.8-compatible PyTorch build for the target RTX 5070. A healthy host driver
does not by itself prove that the Docker runtime has GPU access.

## AI service exits or is killed while loading

The default sidecar limit is 10 GiB system memory and the GPU has 12 GiB VRAM.
Close other GPU-heavy applications and inspect `docker compose logs ai-engine`.
If Docker Desktop has less than the required system-memory allocation, increase
it before changing the service limit. Do not enable more than one Uvicorn worker;
each process would load another model copy.

## AI request times out

Inspect `docker compose logs ai-engine worker-cpu` and confirm
`AI_ENGINE_URL=http://ai-engine:8010`. Cold initialization is covered by the
600-second readiness start period; inference uses
`AI_ENGINE_REQUEST_TIMEOUT_SECONDS` (600 seconds by default). Increase the
request timeout only after confirming that the GPU is active and the process is
making progress, and keep `WORKER_TASK_TIME_LIMIT_SECONDS` safely above it.

## Job remains queued

The `worker-cpu` queue worker orchestrates the internal GPU sidecar. Check that
it consumes `portrait-cpu`, Redis URLs match, `ai-engine` is healthy, concurrency
quota permits work, and the job has not expired or been cancelled. The optional
`worker-gpu` profile is legacy dense-alignment tooling and does not run DGPST.

## Job fails portrait validation

Use the safe error code and compatibility report. One near-frontal face is
required. Improve facial resolution, focus, lighting, crop, or occlusion, or use
a reference with a closer pose and expression. The application rejects unsafe
inputs instead of presenting an unreliable result.

## Identity, hair, glasses, or background changed

DGPST is generative; structure conditioning reduces but does not eliminate
identity drift. Increase structure strength, reduce style strength, and use a
reference with similar pose, framing, hair silhouette, and eyewear. Review the
full image before use. Background modes are applied after generation, but a
subject boundary can still be imperfect.

## Repeated runs differ

Record the same input/reference bytes, model-manifest hashes, profile,
`random_seed`, strengths, inference steps, container revision, GPU/PyTorch
runtime, and output settings. A seed improves repeatability but does not promise
byte-identical results across different GPU or dependency versions.

## Reference extraction refuses a path

This is a security feature of the legacy research-workspace importer. It rejects
absolute paths, traversal, Windows device names, duplicates, symlink traversal,
and existing destinations unless overwrite is explicit. Never compile or run
the extracted 2014 archive.

## Compose cannot bind ports or trust TLS

Change host-side ports in a local override. The AI sidecar intentionally has no
host port. `tls internal` uses Caddy's local CA; trust it for development or use
the documented loopback HTTP endpoint. Production must use a publicly trusted
issuer.
