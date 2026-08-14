# Security policy

## Reporting a vulnerability

Use the repository host's private vulnerability-reporting feature. Do not open a public issue containing an exploit, private image, signed URL, secret, user identifier, object key, or infrastructure address. Include affected revision, impact, minimal reproduction, and any suggested mitigation. Maintainers should acknowledge a report promptly and coordinate disclosure after a fix is available.

There is currently no published security-support SLA or dedicated security email. Do not invent or infer one from contributor metadata.

## Supported versions

Security fixes target the current default branch until versioned releases are published. Operators are responsible for rebuilding pinned dependencies and container images after advisories.

## Security invariants

- Accept only decoder-validated JPEG, PNG, and WebP; SVG and active content are rejected.
- Enforce encoded-byte and decoded-pixel limits before expensive allocation.
- Replace filenames, correct orientation, normalize sRGB, and strip all metadata.
- Keep buckets private and issue short-lived, owner-authorized signed URLs.
- Never log pixels, signed URLs, filenames, email addresses, or raw session IDs.
- Enforce ownership, upload/job quotas, rate limits, processing timeouts, and memory limits.
- Use strict CORS, CSRF protection for cookie-authenticated mutations, secure cookies, CSP, HSTS, `nosniff`, and `no-referrer` in production.
- Run containers as non-root with a read-only root filesystem where practical.
- Keep the DGPST service on the private Compose network; never publish port 8010
  or route a public tunnel directly to it.
- Verify every model artifact against the pinned manifest before loading it.
  Load the legacy PyTorch checkpoint with restricted `weights_only` semantics;
  never deserialize an unverified replacement.
- Keep Hugging Face clients offline at inference time and never send portrait
  pixels to a hosted model API without a separate, explicit product decision.
- Store no face-recognition embeddings and perform no identity or demographic inference.
- Purge expired assets and audit failed deletions.

Example credentials in `.env.example` are intentionally invalid for production. Production startup must fail for missing or unsafe required configuration.

## Reference-material boundary

Never attach the uploaded archive or paper to a vulnerability report. The ignored reference workspace is untrusted input: extract it only with `scripts/extract_reference_archive.py`; never compile or execute its contents. CI rejects MATLAB, MEX, and copied archive material in shipping paths.
