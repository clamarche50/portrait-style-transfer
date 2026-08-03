# Local reference workspace

This directory is a legal and technical boundary, not a vendored-source directory.

The uploaded serialized archive may be extracted locally to `original-matlab/` with `scripts/extract_reference_archive.py`. That path and generated manifests are ignored by Git and excluded from Docker contexts. Never move extracted MATLAB, MEX, C++, paper, portrait, background, eye-layer, or candidate files into a tracked path.

The extractor treats the serialization as untrusted input, validates every member path, rejects duplicates and link-like paths, uses exclusive file creation, and never executes the result. The audit script records relative paths, sizes, and hashes without compiling source.

Only `.gitkeep` is tracked under `manifests/`; locally generated JSON reports may contain hashes or details of unredistributable material and remain ignored.
