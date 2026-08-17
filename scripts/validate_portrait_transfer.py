#!/usr/bin/env python3
"""Run a real-image portrait transfer job over HTTP and download the result."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from smoke_job import Client

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _upload(client: Client, kind: str, path: Path) -> dict:
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        raise SystemExit(f"unsupported image type: {path}")
    boundary = f"portrait-validate-{uuid.uuid4().hex}"
    chunks = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="kind"\r\n\r\n{kind}\r\n'.encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    payload, _ = client.request(
        "POST",
        "/assets/upload",
        body=b"".join(chunks),
        content_type=f"multipart/form-data; boundary={boundary}",
        expected=(201,),
    )
    return json.loads(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/validate-output.png")
    )
    parser.add_argument("--profile", default="source_2014_compat")
    parser.add_argument(
        "--background", default="KEEP", choices=["KEEP", "BLUR", "SOLID", "REFERENCE"]
    )
    parser.add_argument("--no-eye-highlights", action="store_true")
    parser.add_argument("--dense-alignment", action="store_true")
    parser.add_argument("--long-edge", type=int, default=1024)
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Upscale both images by this factor before upload (for small test inputs).",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser.parse_args()


def _maybe_upscale(path: Path, factor: float) -> Path:
    if factor <= 1.0:
        return path
    from PIL import Image

    with Image.open(path) as image:
        resized = image.resize(
            (round(image.width * factor), round(image.height * factor)),
            Image.LANCZOS,
        )
        target = Path("outputs") / f"{path.stem}-upscaled{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        resized.save(target, quality=95)
    return target


def main() -> int:
    args = parse_args()
    client = Client(args.url)
    ready = client.json("GET", "/health/ready")
    if str(ready.get("status", "")).lower() not in {"ready", "ok", "healthy"}:
        raise SystemExit(f"API is not ready: {ready}")

    input_asset = _upload(client, "INPUT", _maybe_upscale(args.input, args.upscale))
    reference_asset = _upload(
        client, "REFERENCE", _maybe_upscale(args.reference, args.upscale)
    )
    job = None
    try:
        job = client.json(
            "POST",
            "/jobs",
            {
                "input_asset_id": input_asset["id"],
                "reference_asset_id": reference_asset["id"],
                "settings": {
                    "algorithm_profile": args.profile,
                    "background_mode": args.background,
                    "eye_highlights": not args.no_eye_highlights,
                    "dense_alignment": args.dense_alignment,
                    "processing_long_edge": args.long_edge,
                    "debug_artifacts": True,
                    "output_format": "PNG",
                },
            },
            expected=(202,),
        )
        deadline = time.monotonic() + args.timeout
        while job["status"] in {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"job {job['id']} did not finish within {args.timeout:g}s"
                )
            time.sleep(2)
            job = client.json("GET", f"/jobs/{job['id']}")
        if job["status"] != "SUCCEEDED":
            print(json.dumps(job, indent=2), file=sys.stderr)
            raise SystemExit(
                f"job failed: {job.get('error_code')} {job.get('error_message_safe')}"
            )

        output, _ = client.request("GET", str(job["output_url"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
        diagnostics = job.get("diagnostics") or {}
        print(
            f"job succeeded: id={job['id']} profile={job.get('algorithm_profile')} "
            f"output={args.output} bytes={len(output)}"
        )
        if diagnostics:
            print(json.dumps(diagnostics, indent=2))
        return 0
    finally:
        if job is not None:
            try:
                client.request("DELETE", f"/jobs/{job['id']}", expected=(204, 404))
            except Exception as cleanup_error:
                print(f"warning: cleanup failed: {cleanup_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
