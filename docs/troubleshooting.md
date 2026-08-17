# Troubleshooting

## Readiness is unhealthy

Inspect `GET /api/v1/health/ready`. The common intentional failure is missing model files. Run `make models` or place verified offline files named by `models/manifest.json` under the mounted model directory. Also verify PostgreSQL, Redis, MinIO health, bucket creation, and service credentials. Readiness does not download models.

## Model download fails

Confirm outbound HTTPS and official URL access. Partial files are temporary and never replace a valid model. If an upstream URL/version changed, update the manifest only after reviewing the official model page, terms, filename, and recorded digest. Air-gapped users should run `download_models.py --offline` after provisioning.

## Reference extraction refuses a path

This is a security feature. The parser rejects absolute paths, `..`, backslashes, Windows device names, duplicates, symlink traversal, and existing destinations unless `--overwrite` is explicit. Extract only to ignored `reference/original-matlab/`; never weaken checks to accommodate an unexpected archive.

## Job remains queued

Check that a worker consumes `portrait-cpu` or `portrait-gpu`, Redis URLs match, concurrency quota permits work, and the job has not expired/cancelled. GPU jobs fall back to CPU routing only according to configured policy; starting a GPU profile without a usable CUDA runtime is not sufficient.

## Job fails validation

Use the safe error code and compatibility report. One near-frontal face is required. Improve facial resolution, focus, lighting, crop, occlusion, or choose a reference with closer pose/expression. The application intentionally rejects profiles, multiple faces, severe blur, and poor overlap rather than returning a misleading success.

## Mask or hair boundary artifacts

Use the non-destructive add/remove mask brush, then rerun. Mask edits invalidate mask-aware pyramids and downstream stages. Wispy hair remains a known difficult case; lowering transfer strength may be preferable to overconfident matting.

## Alignment or doubled features

Inspect affine, line-morph, dense preview, valid fraction, displacement, and Jacobian diagnostics. Add paired control points or disable dense alignment. The fallback chain must be visible; never add morph and dense offsets directly.

## Noise or harsh texture

Lower transfer strength, dynamic-range mix, or residual strength; select a cleaner compatible reference. High gains can amplify input noise even though gain is clamped and smoothed.

## Eye result is implausible

Disable one/both highlights or correct pupil/iris centers. Automatic transfer intentionally switches off for closed, occluded, blurred, off-axis, oversized, or out-of-iris catchlights.

## Reference background contains face fragments

Do not accept the output. Inspect reference alpha and inpainting diagnostics or select keep/blur/solid mode. Facial reference pixels must never be warped into background fill.

## Compose cannot bind ports or trust TLS

Change host-side ports in local overrides. `tls internal` uses Caddy's local CA; either trust it for development or use direct loopback HTTP endpoints. Production must use a publicly trusted issuer.

## Deterministic hash differs

Confirm identical lockfiles, model hashes, algorithm profile/version, seed, CPU/GPU backend, image encoder parameters, and architecture. Byte-stable PNG expectations apply to locked synthetic CI; compare decoded pixels and diagnostics separately when investigating cross-platform codec differences.

## Vercel shows "The request could not be completed"

The hosted frontend reaches the API only through the Cloudflare Tunnel
(`compose.tunnel.yml`). When the local Docker stack restarts without that
file, the `cloudflared` container is left stopped and every request from the
deployed site fails at the network layer.

Check and recover:

```powershell
docker ps -a --format "{{.Names}}: {{.Status}}" | Select-String cloudflared
# If it is Exited, bring the tunnel back up:
docker compose -f compose.yml -f compose.tunnel.yml up -d cloudflared
docker logs --tail 20 portrait-style-transfer-cloudflared-1  # expect "Registered tunnel connection"
```

The tunnel secret lives in `.cloudflare-tunnel-token` (gitignored). If the
repository folder was moved, restore that file from its previous location
before recreating the container, or the secret mount fails.

Cloudflare bot protection can return error 1010 to non-browser HTTP clients
such as scripts; real browsers are unaffected, so verify the tunnel with a
browser request to `https://api-portrait.cyberchords.app/api/v1/health/ready`
rather than a script.
