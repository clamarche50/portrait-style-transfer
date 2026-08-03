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
