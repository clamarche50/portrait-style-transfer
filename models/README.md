# Runtime models

Model binaries are never committed or baked into application images. The two
manifests serve different runtime components:

- `manifest.json` pins the MediaPipe face-analysis artifacts mounted into the API
  and worker containers.
- `dgpst/manifest.json` pins the 19 DGPST, Stable Diffusion v1.5, and IP-Adapter
  artifacts mounted read-only into the internal GPU inference service.

Provision and verify both sets explicitly:

```sh
make models
make verify-models
```

For an offline deployment, place the expected files at their manifest-relative
paths, then run:

```sh
python scripts/download_models.py --manifest models/manifest.json --output-dir models --offline
python scripts/provision_dgpst_models.py --verify-only
```

Downloads never occur during request handling or readiness checks. Every file
must match its declared byte length and SHA-256 digest. The complete DGPST tree
is about 8.3 GiB; allow additional disk space for container layers and model
loading. See [`dgpst/README.md`](dgpst/README.md) and
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before redistributing
any weights.
