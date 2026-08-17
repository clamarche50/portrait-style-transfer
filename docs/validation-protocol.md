# Validation protocol

Required CI uses only synthetic, abstract, or programmatically generated images and mocked model adapters. Rights-cleared real portraits live under ignored `tests/fixtures/private/` and run only through `make test-real-models`.

## Required numeric properties

Test masked-Gaussian constant preservation and background isolation; stack reconstruction; non-negative energy; identity gain; pre-smoothing clamp; strength-zero identities; chroma-band preservation; residual-zero behavior; identity backward warp; exact composition; effective-mask failure; Lab round trip; finite gamut conversion; deterministic encoding; scalar/vectorized Beier-Neely parity; valid segment topology; exact paper/source band counts, energy ordering, and gain-smoothing differences; and absence of style-name production branching.

Hypothesis should cover shapes, mask support, finite values, boundary behavior, and degenerate segments. Tolerances must be stated by dtype/backend, not silently relaxed.

## Integration and browser tests

With isolated PostgreSQL, Redis, and MinIO, test upload, normalization, job creation, SSE progress, successful mocked processing, deterministic validation failure, cancellation/temp cleanup, signed URL authorization, deletion, expiry purge, style creation, ingestion/indexing, and ranking.

Web tests cover form validation, settings serialization, progress/reconnect state, errors, ranking, and delete confirmation. Browser flow is upload input/reference → start → observe progress → inspect result/diagnostics → download → delete. Required CI uses mocked processing; an optional local suite exercises real models.

## Private evaluation set

Record consent/provenance outside Git and stratify clean, typical, and difficult cases. Do not include celebrities, paper figures, famous-photographer collections, or uncontrolled scraped data.

Run ablations for affine only; affine plus Beier-Neely; full dense; masks off; unclamped/unsmoothed gain; one/two/five/six scales; source post-warp versus paper pre-warp energy; residual off; range mix off; and eye highlights off.

## Measurements

- landmark reprojection error;
- descriptor alignment loss and improvement;
- valid-map fraction, displacement percentiles, negative-Jacobian fraction;
- input/reference mask overlap and local-energy correlation;
- structural similarity outside style-sensitive regions;
- noise amplification;
- stage and total latency;
- peak CPU memory and optional GPU VRAM.

Do not use a face-recognition identity metric without separate license and privacy approval.

Human reviewers score 1–5 for subject preservation, style similarity, facial naturalness, hair, eyes, boundary, background, and overall preference. Reviewers must know which cases used fallbacks or manual correction.

## Performance reporting

The engineering targets are CPU p50 under 12 s/p95 under 30 s/peak under 2 GB and GPU p50 under 6 s/p95 under 15 s/VRAM under 4 GB for a 1024–1280 crop. They are provisional. Every report must identify CPU/GPU, RAM/VRAM, OS, container/runtime, model and dependency hashes, crop size, dense settings, debug setting, warmups, sample count, and p50/p95. `scripts/benchmark_pipeline.py` records a command digest and timings without persisting private fixture paths, and can ingest per-stage JSON emitted by a pipeline CLI.

## Release gate

Run lint, formatting check, mypy, unit/property, legacy primitive, integration, web unit, browser E2E, production build, OpenAPI drift, CPU image build, Compose readiness/smoke, copied-source guard, secret/dependency scan, and vulnerability scan. Clearly mark unavailable GPU and private real-model suites; never report them as passing when skipped.
