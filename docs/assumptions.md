# Assumptions and decisions

These assumptions record the active InstantStyle AI implementation and the
retained deterministic analysis layer.

1. **Single near-frontal face.** Defaults reject absolute yaw above 25 degrees,
   pitch above 20 degrees, or roll above 20 degrees and require roughly 150
   pixels between eyes. Thresholds are configuration, not demographic judgments.
2. **Segmentation model.** MediaPipe selfie multiclass confidence is the initial
   matte; GrabCut and deterministic image processing refine it. Classes absent
   from a model version are derived from landmarks/confidence, not invented.
3. **Reference scale.** Analysis preprocessing establishes a canonical face
   scale before fixed-sigma energy features while maps retain original
   coordinates.
4. **GPU inference is authoritative.** `ai_instantstyle_v1` requires the
   internal CUDA sidecar. The CPU worker owns queue and storage orchestration;
   it does not run the diffusion model and has no fallback engine.
5. **Target hardware.** Local deployment targets the RTX 5070 12 GiB with CUDA
   12.8-compatible PyTorch wheels. GPU validation is separate from CPU CI.
6. **Background default.** Input background is retained unless selected otherwise.
7. **Retention default.** Assets and jobs expire after 24 hours; operators may
   shorten this. Deletion failures are retried and audited.
8. **Anonymous ownership.** The local MVP uses a signed, secure session cookie.
   Authentication remains pluggable and anonymous styles are session-private.
9. **Single public profile.** `ai_instantstyle_v1` is the only profile accepted
   by the public create-job schema. The package-level `paper_exact` /
   `source_2014_compat` values select style-ranking sigma profiles only.
10. **Portrait analysis.** Uploads are decoded, normalized, and validated before
    transfer. Worker validation repeats security- and pairing-sensitive checks.
11. **Style ingestion.** Example ingestion uses asynchronous worker state rather
    than blocking the add-example request.
12. **Web layout.** Vercel hosts only the Next.js UI. The API, worker, stores,
    and GPU sidecar remain on the operator-controlled backend host.
13. **Package managers.** The web lock is npm's `package-lock.json`; Python uses uv.
14. **Model provisioning.** Request handlers never download weights. Operators
    run `make models` or mount verified offline artifacts; readiness fails
    otherwise. InstantStyle files are read-only and excluded from image build
    contexts.
15. **Model checksums.** Manifests record local SHA-256 digests from pinned
    official artifacts. A byte change fails closed and requires model,
    provenance, and license review.
16. **No seeded face.** The repository does not ship a real-face demo style. The
    seed flow accepts only a user-supplied, rights-cleared example.
17. **Corrections.** Public corrections are background-only; any rerun executes
    the full AI pipeline from validation.
18. **Performance.** GPU latency and memory are measurements on named hardware,
    not guarantees. One server process avoids duplicate model residency.
