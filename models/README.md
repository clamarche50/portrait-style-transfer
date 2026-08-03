# Runtime models

Model binaries are not committed or baked into the source tree. `manifest.json` lists the official sources and expected filenames used by the MediaPipe Tasks adapters.

Download and verify explicitly:

```sh
python scripts/download_models.py --manifest models/manifest.json --output-dir models
```

For an offline deployment, pre-provision each expected file and run with `--offline`. Both pinned artifacts carry locally verified SHA-256 values in the manifest; downloads and pre-provisioned files must match before use.

API readiness fails if a required model is missing or does not match a declared checksum. Downloads never occur during request handling.
