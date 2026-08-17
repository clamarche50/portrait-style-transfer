# Clean-room porting map

“Exact” below means the documented equation is reimplemented and numerically tested; it does not mean source was copied or published-result parity was achieved.

| Original/reference | Production path | Paper/source relationship | Status |
|---|---|---|---|
| `style_transfer.m` | `packages/portrait_transfer/src/portrait_transfer/pipeline.py` | Stage orchestration with corrected profile separation | Replaced cleanly |
| `morph.m` | `alignment/beier_neely.py` | Same line equations and `a=10,b=1,p=1`; chunked/vectorized | Exact equation |
| `face.con` | `alignment/legacy_66_connections.py`, `alignment/anchors.py` | Documented topology mapped to stable MediaPipe groups | Replaced detector, preserved curves |
| `sift_flow.m` | `alignment/dense_sift.py`, `flow_optimization.py` | Dense SIFT concept; new continuous optimizer | Approximation |
| naive morph/flow sum | `alignment/map_composition.py` | Correct absolute-map composition | Corrected |
| `warpImage.m` | `geometry/sampling.py` | Backward bilinear sampling with the archive's 0.6 out-of-bounds fill | Exact primitive |
| `thresh_v.m` | `geometry/validity.py` | Bounds/support validity plus modern diagnostics | Extended |
| affine stage in paper | `alignment/similarity.py` | Explicit robust eyes/mouth/nose transform | Added from paper |
| `laplacian_pyramid.m` | `multiscale/laplacian.py` | Source five-band stack is the default; paper six-band stack selectable | Exact per profile |
| `gaussian_pyramid.m` | `multiscale/masked_gaussian.py` | Normalized mask-aware separable Gaussian | Corrected/extended |
| paper energy equations | `multiscale/energy.py` | Default computes energy after warping (archive behavior); paper order selectable | Exact per profile |
| paper robust gain | `multiscale/gain.py` | sqrt, clamp, and `3*2^l` smoothing | Exact equation plus guards |
| `sum_pyramid.m` | `multiscale/reconstruction.py` | Sum details and residual | Exact primitive |
| `HistTransferOneD.m` | `multiscale/histogram.py` | Masked empirical CDF, unequal counts | Replaced/extended |
| `RGB2Lab.m`, `Lab2RGB.m` | `color/lab.py`, `color/legacy_color.py` | Default uses the archive's non-gamma Lab; standards-compliant sRGB/D65 Lab selectable | Exact per profile |
| `eye_transfer.m` | `eyes/highlight_transfer.py` | Pupil/iris placement with confidence gating | Reimplemented |
| prepared eye layers | `eyes/extraction.py`, `eyes/inpainting.py` | Automatic arbitrary-reference extraction/removal | Added |
| bundled iris helpers | MediaPipe iris anchors plus `eyes/*` | No bundled helper code | Replaced |
| `skin.m` | `segmentation.py` | MediaPipe confidence, GrabCut, morphology, feather | Omitted threshold heuristic |
| precomputed foreground/masks | `segmentation.py`, `crop.py` | Automatic ingestion | Added |
| precomputed background | `background.py` | Segmentation and inpainting | Added |
| `run_flickr.m`, `candidates.mat` | `selection.py`, style API/worker | Local energy NCC plus compatibility ranking | Implemented selector |
| `run_mit.m` | private validation manifest and opt-in tests | Rights-cleared evaluation only | Replaced |
| `local_match.m` | none | Empty source stub | Omitted |
| bundled MEX/C++ | none | Unclear/restrictive licensing | Deliberately excluded |
| paper manual corrections | `alignment/correction_constraints.py`, gain/mask/eye correction workflow | Non-destructive constraints and cache invalidation | Added |

Production modules cite relevant paper equations and this map, not source-line copies. Changes to profile semantics require documentation, ablation, and numeric regression tests.
