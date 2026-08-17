# Audit of the uploaded 2014 source archive

## Scope and method

The uploaded `clamarche50-temp-8a5edab282632443.txt` is a text serialization containing a directory listing and delimited file bodies. It was inspected as reference material. The production repository does not contain the serialization or extracted source. `scripts/extract_reference_archive.py` can reproduce a local, path-safe extraction under the ignored `reference/original-matlab/`; `scripts/audit_reference_source.py` inventories hashes and checks the shipping boundary without executing code.

The uploaded `2014_portrait.pdf` is the corresponding SIGGRAPH 2014 paper. It is cited but not redistributed.

## Archive contents observed

- `code/code/style_transfer.m`: top-level orchestration over already prepared assets.
- `morph.m` and `face.con`: 66-landmark Beier-Neely segment morph.
- `sift_flow.m`: quarter-resolution dense SIFT Flow wrapper.
- `libs/SIFTflow/`: MATLAB wrappers and bundled MEX/C++ implementations.
- `libs/image_pyramids/laplacian_pyramid.m`: non-downsampled masked stack.
- `RGB2Lab.m` and `Lab2RGB.m`: legacy color conversion.
- `eye_transfer.m` and `libs/iris/`: prepared catchlight application and circular iris helpers.
- `HistTransferOneD.m`: rank-based one-dimensional transfer.
- `skin.m`, `warpImage.m`, and `thresh_v.m`: skin heuristic, backward sampling, and flow validity.
- `run_flickr.m` and `run_mit.m`: batch runners.
- Empty `local_match.m` and experimental/unused helpers such as `quinx.m`.

The archive is not an application. It contains no upload normalization, model provisioning, automatic general-purpose preprocessing, API, UI, database, object storage, queue, authentication, retention, security controls, or deployment.

## Precomputed dependencies

The top-level path expects external images, extracted foregrounds, masks, backgrounds, landmarks, prepared left/right eye foreground and alpha layers, and `candidates.mat` retrieval scores. The shown code does not create all of these inputs. Consequently, production requires automated landmarking, segmentation/matte refinement, background extraction/inpainting, eye-highlight extraction, style feature indexing, and candidate ranking.

## Behavior and constants

`style_transfer.m` morphs then applies SIFT Flow, approximately adds morph and residual offsets, replaces the background before Lab transfer, clamps gains to `0.9..2.8`, warps the reference residual, selects color-channel behavior from hardcoded style names, optionally applies prepared eye layers, and writes JPEG.

`morph.m` adds four boundary segments, evaluates the weighted line transform per pixel, and uses `a=10`, `b=1`, `p=1`.

`sift_flow.m` works at 25% scale with dense SIFT `cellsize=7`, `gridspacing=1`, and records legacy parameters `alpha=500`, `gamma=10`, `d=1000000`, four levels, windows 3/4, and iteration counts 60/30.

The stack helper is mask-normalized and non-downsampled. Its six-level call yields five details plus a residual at blur scales 4, 8, 16, 32, and 64. Eye transfer expects prepared layers, detects target iris/pupil, uses `Lab L > 60` as a bright-pixel baseline, interpolates removed pixels, then resizes and alpha-composites the prepared highlight. `run_flickr.m` sorts precomputed scores; it does not derive retrieval features.

## Paper/source discrepancies

- The paper specifies an eye/mouth affine stage; the visible top-level source does not.
- Source offsets are approximately added; production composes absolute maps.
- Paper decomposition has six details from sigma 2 plus residual; source call has five details from sigma 4.
- Paper reference energy is calculated before warp; source effectively calculates it after warping detail.
- Paper gain is clamped and Gaussian-smoothed; source clamps only.
- Energy neighborhoods, mask use, and chroma-band handling differ.
- Source behavior depends on style names and prepared assets; production uses metadata and automatic ingestion.
- Production uses standards-compliant sRGB/D65 Lab; legacy conversion is isolated to parity analysis.

## What cannot be validated

Exact end-to-end parity cannot be established without the original data, prepared masks/backgrounds/eye layers, candidate files, and a legally usable SIFT Flow runtime. Synthetic primitive comparisons can validate equations and expose profile differences, but they cannot substantiate parity with published figures.

## Module-by-module audit verdicts (default profile `source_2014_compat`)

| Python module | MATLAB ground truth | Verdict |
|---|---|---|
| `alignment/beier_neely.py` | `morph.m` | Faithful. `a=10`, `b=1`, `p=1`; weight `(len^p/(a+d))^b`; distance branch for `u<0`/`u>1`; four boundary segments; displacement zeroed outside canvas. |
| `alignment/legacy_66_connections.py` | `face.con` | Faithful. Exactly the 61 zero-based pairs, closed eye loops 36–41/42–47. |
| `alignment/dense_sift.py` + `flow_optimization.py` | `sift_flow.m` + SIFTflow MEX | Documented clean-room equivalent, not bit-compatible with the bundled MEX. Legacy parameters (`alpha=500`, `gamma=10`, `d=1e6`, `cellsize=7`, 0.25 scale, 4 levels, windows 3/4, iterations 60/30) are recorded in diagnostics metadata. Validity-gated with affine/line fallback. |
| `geometry/sampling.py` (warp) | `warpImage.m` | Faithful after audit fix: bilinear sampling, validity mask, and the archive's `0.6` out-of-bounds fill at every reference-image/band/residual warp site. |
| `multiscale/laplacian.py` + `masked_gaussian.py` | `laplacian_pyramid.m` | Faithful. Five bands at sigma 4–64 plus residual, mask-normalized convolution (divide by blurred mask), bands and residual masked. |
| `multiscale/energy.py` + `gain.py` | local match loop in `style_transfer.m` | Faithful. Energy sigmas `8,16,32,64,128` computed after warping the reference band; `gain=sqrt(e_ex/(e_in+1e-4))` clamped to `[0.9, 2.8]`; no gain smoothing; coarsest level is the warped example residual. |
| `multiscale/reconstruction.py` | `sum_pyramid.m` | Faithful. Plain sum of bands plus residual. |
| `multiscale/histogram.py` | `HistTransferOneD.m` | Equivalent rank/sort transfer; gated by `global_range_mix` (0 reproduces the archive's `hist_transfer=false`; product default 0.25). |
| `color/legacy_color.py` | `RGB2Lab.m`/`Lab2RGB.m` | Faithful. Ruzon matrices, D65 `0.950456`/`1.088754`, thresholds `0.008856`/`0.206893`, `903.3` branch, no gamma correction. |
| `eyes/*` | `eye_transfer.m` + `libs/iris` | Semantic parity plus documented extensions. The archive expects prepared catchlight layers; the port auto-extracts them, removes existing catchlights before placement, and gates on confidence. Iris geometry comes from landmarks/segmentation instead of Daugman `thresh`. Manual fixes available via CorrectionStudio eye operations. |
| `background.py` | background lines in `style_transfer.m` | `REFERENCE` mode composites `mask*out + (1-mask)*bg_ex` like the archive; `bg_ex` is derived automatically (nearest-fill + smoothing) because the archive ships it precomputed. KEEP/BLUR/SOLID are product modes. |
| `segmentation.py` | mask usage in `style_transfer.m` | Extension. The archive takes `mask_in`/`mask_ex` as inputs; the port derives feathered masks from landmarks/analysis and intersects input and warped-reference masks, with coverage guard. |
| `pipeline.py` stage order | `style_transfer.m` | Matches: correspondence → local (multiscale) match → eye highlights → background composite. |

Edge cases from the paper's limitations section (glasses, beards, varying lighting) are handled by the effective-mask intersection, dense-flow robustness with fallbacks, CorrectionStudio manual strokes, and the Lab gain + `residual_strength`/`global_range_mix` controls.
