# portrait-transfer

This package is a clean-room implementation of the image-processing ideas in
*Style Transfer for Headshot Portraits*. It does not contain the authors'
MATLAB/C++ source, their datasets, photographer images, model weights, or SIFT
Flow binaries. Heavy landmark, segmentation, and descriptor models are injected
through runtime protocols; deterministic CPU fallbacks keep the numerical core
and synthetic tests self-contained.

Production analysis is available as `MediaPipePortraitAnalyzer`. Install the
`vision` extra, provision the Face Landmarker task bundle and multiclass selfie
segmenter TFLite file yourself, and pass both local paths to the constructor.
The adapter never downloads model weights and rejects missing or URL-based
paths. It requires exactly one detected 478-point face; the synthetic analyzer
used by `create_default_runtime()` remains an explicit test/development tool.

Correction reruns use private, validated checkpoints returned in
`TransferResult.resume_artifacts`. Persist that complete mapping losslessly and
pass it back as `RuntimeContext.resume_artifacts`, together with
`runtime.corrections["resume_from_stage"]` set to `"multiscale"`, `"eyes"`, or
`"background"`. Stage-specific signatures bind the images, model/backend
identity, relevant settings, and upstream corrections. Integrity, shape,
finite-range, mask-coverage, and correspondence checks fail closed to a full
run. These checkpoint arrays are not user-facing debug artifacts.

`PAPER_EXACT` is the production profile. `SOURCE_2014_COMPAT` is an explicit,
development-only profile for studying numerical differences in the archived
2014 source.
