# Licensing review

## Finding

The uploaded archive has no clear top-level software license. Bundled material includes restrictive language: some visualization helpers limit commercial incorporation, while a MEX wrapper states an all-rights-reserved position. Absence of a license is not permission to copy, redistribute, compile into a product, or make commercial-use claims.

The paper itself and any example portraits remain copyrighted works. Research publication does not grant permission to redistribute figures, photographer collections, celebrity portraits, or the authors' data.

## Enforced boundary

1. The uploaded serialization, PDF, extracted files, candidate data, images, and prepared assets remain in ignored local paths.
2. `.dockerignore` excludes all reference paths and known archive filenames.
3. Production code is written from the paper's algorithms and permissively licensed public APIs, not copied line by line.
4. Bundled SIFT Flow MEX/C++ and MATLAB wrappers are neither compiled nor shipped.
5. CI rejects tracked MATLAB/MEX files, forbidden reference paths, and byte-identical files when a local audit source is available.
6. Private source-profile comparisons are disabled publicly and do not imply redistribution rights.
7. Models download separately from official sources and remain outside the repository license.

## Repository license

Original repository code and documentation are offered under Apache License 2.0. `LICENSE` explicitly excludes the paper, archive, model artifacts, uploads, and other third-party material. Dependency and container licenses remain independently applicable; `THIRD_PARTY_NOTICES.md` is a release checklist, not legal advice or a complete SBOM.

## Required human review

Before public or commercial distribution, a qualified reviewer must verify clean-room provenance, dependency/model/container obligations, any patents relevant to implementation, product naming/marketing, dataset rights, and all shipped notices. Until then, documentation must not claim legal clearance or exact equivalence to the authors' results.
