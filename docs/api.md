# API contract

All application routes are below `/api/v1`. JSON errors use:

```json
{"error":{"code":"POSE_TOO_LARGE","message":"The portrait is too far from a frontal pose.","details":{},"request_id":"uuid"}}
```

Safe details may contain actionable numeric thresholds. Responses never contain stack traces, filesystem paths, internal model state, raw object keys, or secrets.

## Health

- `GET /health/live`: process liveness only.
- `GET /health/ready`: PostgreSQL, Redis, object storage, required bucket, and required model availability. Missing models produce non-ready status; they are never downloaded here.

## Assets

- `POST /assets/upload`: multipart fields `kind` and `file`; accepts decoder-validated JPEG/PNG/WebP. Returns owner-scoped normalized asset metadata. Face preflight analysis is generated asynchronously when a transfer job or style-example ingestion starts.
- `GET /assets/{asset_id}`: authorized metadata.
- `DELETE /assets/{asset_id}`: idempotent owner deletion/cascade policy.
- `POST /assets/{asset_id}/download-url`: short-lived signed URL.

Client filename is not retained. Upload limits default to 15 MB encoded, 8 MP decoded, and 8000 pixels on the original long edge. The 8 MP cap keeps source/output face-analysis masks within the worker's bounded memory budget.

## Jobs

- `POST /jobs`: exactly one of `reference_asset_id` and `style_id` is required.
- `GET /jobs/{job_id}`: state, stage, progress, safe diagnostics summary, and selected style example.
- `GET /jobs/{job_id}/events`: owner-authorized SSE progress stream with keepalive and terminal event.
- `POST /jobs/{job_id}/cancel`: transitions active jobs to cancel-requested.
- `DELETE /jobs/{job_id}`: cancels if needed, deletes owned objects, and soft-deletes metadata.
- `POST /jobs/{job_id}/download-url`: signed output URL for a successful, unexpired job.
- `GET /jobs/{job_id}/diagnostics`: signed/serialized private diagnostics.
- `POST /jobs/{job_id}/corrections`: AI jobs accept background corrections only. Classical mask/alignment/gain/eye corrections are rejected instead of being silently ignored. A rerun always executes the full AI pipeline.
- `POST /jobs/{job_id}/rerun`: enqueue from the earliest invalidated cached stage.

Representative settings:

```json
{
  "input_asset_id": "uuid",
  "reference_asset_id": "uuid",
  "style_id": null,
  "settings": {
    "algorithm_profile": "ai_dgpst_v1",
    "style_strength": 0.75,
    "structure_strength": 0.9,
    "inference_steps": 30,
    "random_seed": 0,
    "background_mode": "KEEP",
    "background_color": null,
    "output_format": "PNG",
    "jpeg_quality": 95
  }
}
```

`ai_dgpst_v1` is the only public profile. The server rejects the retired
classical fields and public attempts to select `paper_exact` or
`source_2014_compat`; there is no silent fallback from AI inference to either
legacy engine.

## Styles

- `GET /styles`, `POST /styles`, `GET/PATCH/DELETE /styles/{style_id}`.
- `POST /styles/{style_id}/examples`, `DELETE /styles/{style_id}/examples/{example_id}`.
- `POST /styles/{style_id}/reindex`: rebuild the private face-analysis and ranking feature set used to select a style example.
- `POST /styles/{style_id}/rank`: accepts an input asset and returns top-three IDs plus score components and compatibility warnings.

Creation requires `rights_confirmed=true`. Collections default private. Publication is a privileged, audited change.

## Sessions, CSRF, and quotas

Anonymous clients receive a signed secure session. Cookie-authenticated mutating routes require CSRF protection. CORS is an explicit allowlist. Rate, upload, and concurrent-job quotas are enforced by both session/user and IP. All reads and writes verify ownership before disclosing existence-sensitive details.

## Progress semantics

Progress is monotonic within an attempt. SSE may reconnect using the current durable state; it is not the source of truth. Terminal status comes from PostgreSQL. Cancellation is cooperative between deterministic stages.
