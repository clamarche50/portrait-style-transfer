# portrait-transfer

This package provides the deterministic analysis layer of the portrait style
transfer pipeline: image preflight, MediaPipe-based face analysis, head
segmentation, quality scoring, reference style ranking, and style ingestion.
The pixel-level transfer itself is performed by the internal InstantStyle GPU
engine (`services/ai_engine`); this package never runs diffusion inference.

Heavy landmark and segmentation models are injected through runtime protocols;
deterministic CPU fallbacks keep the numerical core and synthetic tests
self-contained.

Production analysis is available as `MediaPipePortraitAnalyzer`. Install the
`vision` extra, provision the Face Landmarker task bundle and multiclass selfie
segmenter TFLite file yourself, and pass both local paths to the constructor.
The adapter never downloads model weights and rejects missing or URL-based
paths. It requires exactly one detected 478-point face; the synthetic analyzer
used by `create_default_runtime()` remains an explicit test/development tool.

`AlgorithmProfile` selects the multiscale sigma profile used when ranking a
reference style's energy distribution (`PAPER_EXACT` follows the paper;
`SOURCE_2014_COMPAT` is a development-only variant for studying numerical
differences in the archived 2014 source). It no longer switches inference
engines: every job runs the single `ai_instantstyle_v1` AI profile.
