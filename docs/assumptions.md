# Assumptions and decisions

These assumptions record both the active AI implementation and the retained
clean-room legacy implementation.

1. **Single near-frontal face.** Defaults reject absolute yaw above 25 degrees,
   pitch above 20 degrees, or roll above 20 degrees and require roughly 150
   pixels between eyes. Thresholds are configuration, not demographic judgments.
2. **Segmentation model.** MediaPipe selfie multiclass confidence is the initial
   matte; GrabCut and deterministic image processing refine it. Classes absent
   from a model version are derived from landmarks/confidence, not invented.
3. **Reference scale.** Legacy preprocessing establishes a canonical face scale
   before fixed-sigma energy features while maps retain original coordinates.
4. **Dense correspondence.** The retained Kornia dense-SIFT and robust-flow path
   is a clean-room approximation; it cannot reproduce bundled SIFT Flow exactly.
5. **GPU inference is authoritative.** `AI_DGPST_V1` requires the internal CUDA
   sidecar. The CPU worker owns queue and storage orchestration; it does not run
   the diffusion model or fall back silently to the classical engine.
6. **Target hardware.** Local deployment targets the RTX 5070 12 GiB with CUDA
   12.8-compatible PyTorch wheels. GPU validation is separate from CPU CI.
7. **Background default.** Input background is retained unless selected otherwise.
8. **Retention default.** Assets and jobs expire after 24 hours; operators may
   shorten this. Deletion failures are retried and audited.
9. **Anonymous ownership.** The local MVP uses a signed, secure session cookie.
   Authentication remains pluggable and anonymous styles are session-private.
10. **Legacy profile authorization.** `paper_exact` and `source_2014_compat` are
    not accepted by the public create-job schema. Any future internal comparison
    path must remain explicit and separate.
11. **Portrait analysis.** Uploads are decoded, normalized, and validated before
    transfer. Worker validation repeats security- and pairing-sensitive checks.
12. **Style ingestion.** Example ingestion uses asynchronous worker state rather
    than blocking the add-example request.
13. **Web layout.** Vercel hosts only the Next.js UI. The API, worker, stores, and
    GPU sidecar remain on the operator-controlled backend host.
14. **Package managers.** The web lock is npm's `package-lock.json`; Python uses uv.
15. **Model provisioning.** Request handlers never download weights. Operators
    run `make models` or mount verified offline artifacts; readiness fails
    otherwise. DGPST files are read-only and excluded from image build contexts.
16. **Model checksums.** Manifests record local SHA-256 digests from pinned
    official artifacts. A byte change fails closed and requires model,
    provenance, and license review. DGPST checkpoint weight rights remain
    `REVIEW_REQUIRED`.
17. **No seeded face.** The repository does not ship a real-face demo style. The
    seed flow accepts only a user-supplied, rights-cleared example.
18. **Histogram masks.** The legacy input and reference paths use separate valid
    masks and empirical CDF interpolation supports different sample counts.
19. **Triangle test wording.** The legacy roadmap's triangle test is a generic
    geometry test; triangulated warping is not substituted for Beier-Neely.
20. **Performance.** GPU latency and memory are measurements on named hardware,
    not guarantees. One server process avoids duplicate model residency.
