# Validation protocol

Required CI uses only synthetic, abstract, or programmatically generated images and mocked model adapters. Rights-cleared real portraits live under ignored `tests/fixtures/private/` and run only through explicit local commands, never in CI.

## Required numeric properties

Test the retained analysis-layer invariants: masked-Gaussian constant preservation and background isolation; Laplacian stack reconstruction; non-negative energy; band counts and energy ordering for the paper and source sigma profiles; Lab round trip; finite gamut conversion; deterministic encoding; valid segment topology; and style-ranking determinism.

Hypothesis should cover shapes, mask support, finite values, boundary behavior, and degenerate segments. Tolerances must be stated by dtype/backend, not silently relaxed.

## Engine contract tests

The AI engine contract suite runs without a GPU: request/settings validation (including rejection of retired profiles), stub-backend gating, preprocessing scale/restore invariants, and output-quality guards. The stub backend is only reachable behind `ENGINE_ALLOW_STUB_BACKEND=true` and must never satisfy deployment readiness.

## Integration and browser tests

With isolated PostgreSQL, Redis, and MinIO, test upload, normalization, job creation, SSE progress, successful mocked processing, deterministic validation failure, cancellation/temp cleanup, signed URL authorization, deletion, expiry purge, style creation, ingestion/indexing, and ranking.

Web tests cover form validation, settings serialization, progress/reconnect state, errors, ranking, and delete confirmation. Browser flow is upload input/reference → start → observe progress → inspect result/diagnostics → download → delete. Required CI uses mocked processing; an optional local suite exercises real models.

## Private evaluation set

Record consent/provenance outside Git and stratify clean, typical, and difficult cases. Do not include celebrities, paper figures, famous-photographer collections, or uncontrolled scraped data.

Run ablations across `style_strength`, `structure_strength`, `inference_steps`, and seed repeatability; compare background modes KEEP/REMOVE/SOLID/BLUR on the same pair.

## Measurements

- inference latency and peak GPU VRAM;
- face detection/embedding scores reported by the engine;
- output guard outcomes (dimension, finiteness, border anisotropy);
- stage and total job latency;
- peak CPU memory of the analysis layer.

Do not use a face-recognition identity metric without separate license and privacy approval.

Human reviewers score 1–5 for subject preservation, style similarity, facial naturalness, hair, eyes, boundary, background, and overall preference.

## Performance reporting

The engineering targets are GPU inference p50 under 120 s and peak VRAM under 12 GiB at the default 1024/1280 limits on the RTX 5070. They are provisional. Every report must identify GPU, RAM/VRAM, OS, container/runtime, model-manifest digest, dependency hashes, image size, settings, warmups, sample count, and p50/p95.

## Release gate

Run lint, formatting check, mypy, package/engine unit tests, integration, web unit, browser E2E, production build, OpenAPI drift, CPU image build, Compose readiness/smoke, model-binary tracking guard, secret/dependency scan, and vulnerability scan. Clearly mark unavailable GPU and private real-model suites; never report them as passing when skipped.
