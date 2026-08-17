# Assumptions and decisions

These assumptions make underspecified roadmap requirements executable without changing the research method.

1. **Single near-frontal face.** Defaults reject absolute yaw above 25°, pitch above 20°, or roll above 20° and require roughly 150 pixels between eyes. Thresholds are configuration, not demographic judgments.
2. **Segmentation model.** MediaPipe selfie multiclass confidence is the initial matte; GrabCut and deterministic image processing refine it. Classes unavailable from a model version are derived from landmarks/confidence, not invented labels.
3. **Reference scale.** Reference preprocessing establishes a canonical face scale before fixed-sigma energy features; maps still address original reference coordinates. This avoids applying nominal pixel neighborhoods at incomparable face scales.
4. **Dense correspondence.** Kornia dense SIFT plus continuous robust flow is a clean-room approximation. It cannot reproduce the bundled discrete SIFT Flow runtime exactly.
5. **CPU is authoritative.** Every required pipeline stage runs on CPU. Dense alignment may use reduced descriptor resolution/iterations or a validated fallback under latency pressure; multiscale transfer is never skipped.
6. **GPU is opt-in.** GPU build/runtime validation requires NVIDIA hardware and compatible locked PyTorch wheels and is reported separately from CPU CI.
7. **Background default.** Input background is retained unless the user selects another mode.
8. **Retention default.** Assets/jobs expire after 24 hours; operators may shorten this. Deletion failures are retried and audited.
9. **Anonymous ownership.** Local MVP uses a signed, secure session cookie. Authentication remains pluggable; anonymous styles are private to that session.
10. **Source profile authorization.** With `ENABLE_SOURCE_COMPAT_PROFILE=false` it is unavailable. Enabling it still requires an internal/admin execution path; arbitrary public request bodies cannot select it.
11. **Portrait analysis.** Uploads are decoded, normalized, and validated immediately. Face landmarks, masks, pose, quality signals, and the overlay-ready preflight result are produced by the worker when a transfer job or style-example ingestion begins; job validation repeats security- and pairing-sensitive checks.
12. **Style ingestion.** Example ingestion uses asynchronous worker state rather than blocking the add-example request; transfer jobs and ingestion tasks share job-state primitives where practical.
13. **Web layout.** The cloned repository uses a root Next.js/vinext package with OpenAI Sites metadata rather than `apps/web`. Python service/package paths follow the roadmap. Sites can host the UI only; the full processing stack remains externally deployed.
14. **Package managers.** The existing web lock is npm's `package-lock.json`; Python uses uv. Switching to pnpm is deferred until it can be done atomically without invalidating the cloned hosting scaffold.
15. **Model provisioning.** Bootstrap and request handlers do not download weights. Operators run `make models` or mount verified offline artifacts; readiness fails otherwise.
16. **Model checksums.** Upstream did not publish SHA-256 values alongside these URLs, so the manifest records digests computed from the pinned official artifacts on 2026-08-03. Any upstream byte change fails closed and requires explicit model/version review.
17. **No seeded face.** The repository cannot include a useful real-face demo style. The seed script creates a private collection and optionally ingests a user-supplied rights-cleared example.
18. **Histogram masks.** Input and reference provide separate valid masks; empirical CDF interpolation supports different sample counts.
19. **Triangle test wording.** The roadmap's triangle-vertex test is treated as a generic geometric sampling/transform test. Triangulated piecewise warping is not substituted for Beier-Neely.
20. **Performance.** CPU/GPU latency and memory figures are targets measured on named hardware, never guaranteed properties.
