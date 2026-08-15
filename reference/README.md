# Local reference workspace

This directory is a legal and technical boundary, not a vendored-source directory.

The uploaded serialized archive may be kept locally under `original-matlab/` for personal reference only. That path and any generated manifests are ignored by Git and excluded from Docker contexts. Never move MATLAB, MEX, C++, paper, portrait, background, eye-layer, or candidate files into a tracked path.

Treat any archived material as untrusted input and never compile or execute it. The classical engine that was ported from this research has been removed; no shipping code copies from the archive.

Only `.gitkeep` is tracked under `manifests/`; locally generated JSON reports may contain hashes or details of unredistributable material and remain ignored.
