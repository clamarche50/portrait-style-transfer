# Algorithm

## Profiles and coordinate convention

`PAPER_EXACT` is the production default. It implements the paper-specified multiscale equations. `SOURCE_2014_COMPAT` is gated, development-only behavior for understanding the uploaded 2014 archive; it is not presented as a user style or exact published-result reproduction.

Every warp is an absolute backward map:

```text
M[y, x] = source coordinate sampled to produce destination pixel (x, y)
```

Offsets exist only at library boundaries. Reference image fields are derived from original reference coordinates rather than repeatedly resampling intermediate images.

## Preflight and canonical crop

Decoder validation corrects EXIF orientation, converts to sRGB, handles alpha against a neutral background, strips metadata, and enforces encoded/decoded limits. MediaPipe Face Landmarker must find exactly one near-frontal face with both irises and major features in frame. Quality analysis measures facial resolution, blur, exposure, noise, crop truncation, matte confidence, and an occlusion proxy.

The input head crop includes hair, ears, chin, and upper shoulders when available, with 35% horizontal, 45% top, and 30% bottom margins. It is reflect-padded when necessary and processed at a configurable long edge, normally 1280. Transforms among original input, canonical crop, and reference coordinates are retained for final composition.

MediaPipe multiclass segmentation is refined with GrabCut, morphology, connected-component selection, and a soft distance/edge feather. The effective transfer support is input head alpha multiplied by the warped reference head alpha. Unsupported or low-confidence areas preserve the input.

## Correspondence

### Similarity/partial affine

Eye centers, mouth center, and optionally nose base define a RANSAC transform from reference to input. Reflections, excessive scale/rotation, too few inliers, and high normalized anchor error invalidate the transform.

### Beier-Neely line morph

Named landmark curves create directed segment pairs for jaw, brows, nose, eyes, lips, optional forehead/hairline, and the crop boundary. For destination point `X` and destination segment `P'Q'`:

```text
u = dot(X-P', Q'-P') / ||Q'-P'||²
v = dot(X-P', perpendicular(Q'-P')) / ||Q'-P'||
X_i = P + u(Q-P) + v perpendicular(Q-P)/||Q-P||
weight_i = (length(P'Q')^p / (a + segment_distance))^b
M_line(X) = sum(weight_i X_i) / sum(weight_i)
```

Defaults are `a=10`, `b=1`, `p=1`. CPU evaluation is chunked; a scalar implementation serves as the numeric oracle. Zero-length segments, invalid sampling, and foldovers are guarded, with affine fallback.

### Dense descriptor refinement

An ephemeral aligned preview is locally contrast-normalized and optimized at configured coarse-to-fine scales. The CPU path uses a deterministic clean-room dense gradient descriptor and continuous robust residual flow; the optional CUDA path uses Kornia DenseSIFT descriptors. Both paths validate improvement, bounds, and Jacobians. This is a clean-room approximation of the paper's SIFT Flow stage, not the archive's bundled discrete MEX runtime.

The final map composes rather than adds offsets:

```text
M_final(x) = sample(M_affine_line, x + f(x))
```

Validation considers descriptor improvement, valid fraction, low-resolution consistency, displacement percentiles, negative Jacobians, mask overlap, and landmark-edge alignment. Fallback is dense → affine plus line morph → affine → incompatible-pair rejection.

## Lab and full-resolution stack

Production uses standards-compliant sRGB/D65 CIE Lab in float32. L is internally normalized to `[0,1]`; chroma ranges are explicit in code. Gaussian operations use symmetric boundaries and normalized mask support:

```text
masked_gaussian(X,M,sigma) = G_sigma(X*M) / max(G_sigma(M), 1e-6)
```

No spatial downsampling occurs. With `B_s` denoting a masked Gaussian blur:

```text
L0 = I-B2       L3 = B8-B16
L1 = B2-B4      L4 = B16-B32
L2 = B4-B8      L5 = B32-B64
R  = B64
```

The six bands plus residual reconstruct the supported input within numeric tolerance.

## Energy and robust gain

For band `l=0..5`, input and reference energy use sigma `2^(l+1)`. Reference energy is computed in reference coordinates and then warped:

```text
S_l(I) = masked_gaussian(L_l(I)^2, M_input, 2^(l+1))
S_l(E) = masked_gaussian(L_l(E)^2, M_reference, 2^(l+1))
g_l = sqrt(warp(S_l(E), M_final) / (S_l(I) + 1e-4))
g_l = masked_gaussian(clip(g_l, 0.9, 2.8), M_effective, 3*2^l)
g_effective = exp(strength * log(max(g_l, 1e-6)))
```

Alignment-confidence protection blends gain toward one, and explicit correction regions can lock, constrain, or copy gain. These protections are switchable extensions around the paper result.

L transfers all six detail bands. Lab a/b preserve input bands 0–2 and transfer bands 3–5. Production behavior never switches on a photographer/style name. A monochrome flag is derived during ingestion and neutralizes chroma intentionally.

The residual is:

```text
R_out = (1-residual_strength) R_input
      + residual_strength warp(R_reference, M_final)
```

After reconstruction, masked empirical-CDF histogram matching can be mixed with the local result; default mix is 0.25. The result is gamut-compressed, confidence-blended toward input, composited through soft alpha, and restored to original resolution.

## Eyes, backgrounds, and ranking

Style ingestion finds suitable iris-local highlights using `k=3` Lab clustering plus component, luminance, compactness, saturation, and confidence tests. Target highlights are conservatively removed/inpainted; source catchlights are pupil-aligned, iris-scaled, optionally rotated, clipped, and alpha-composited in linear light. Low-confidence cases disable automatically.

Background modes keep or blur input, use a solid color, or use an inpainted/crop-matched reference background. Facial reference pixels are never treated as background.

Style examples store six `32×32` L-energy features plus pose, shape, mask, blur, color, edge, and crop metrics. Severe mismatches are filtered. Ranking weights energy NCC 0.65, pose 0.15, landmark shape 0.10, photometric compatibility 0.05, and mask quality 0.05; it returns the top three with components.

## Source-compatible differences

The private profile merges the finest two bands into a five-band stack beginning at sigma 4, computes energy after warping with effective sigmas `[8,16,32,64,128]`, clamps without paper gain smoothing, and can use a legacy color-range utility. These differences are diagnostic, documented, and prohibited as silent production defaults.
