#!/usr/bin/env python3
"""Run a real portrait-style-transfer job and keep the artifacts for review.

Unlike smoke_job.py (which exercises cleanup), this script uploads a real
source portrait and style reference, submits a production job, polls it to
completion, and saves the generated output plus the job diagnostics into an
output directory. The job and assets are left in place so the result can be
inspected through the UI as well.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from smoke_job import Client, _fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("build/validation"))
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--style-strength", type=float, default=0.55)
    parser.add_argument("--structure-strength", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = Client(args.url)
    ready = client.json("GET", "/health/ready")
    if str(ready.get("status", "")).lower() not in {"ready", "ok", "healthy"}:
        raise RuntimeError(f"API is not ready: {ready}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    input_payload, input_name, input_type = _fixture(args.input, 0)
    reference_payload, reference_name, reference_type = _fixture(args.reference, 1)
    print(f"uploading input={input_name} ({len(input_payload)} bytes)")
    input_asset = client.upload(
        "INPUT", input_payload, filename=input_name, content_type=input_type
    )
    print(f"uploading reference={reference_name} ({len(reference_payload)} bytes)")
    reference_asset = client.upload(
        "REFERENCE",
        reference_payload,
        filename=reference_name,
        content_type=reference_type,
    )

    job: dict[str, Any] = client.json(
        "POST",
        "/jobs",
        {
            "input_asset_id": input_asset["id"],
            "reference_asset_id": reference_asset["id"],
            "settings": {
                "algorithm_profile": "ai_instantstyle_v1",
                "style_strength": args.style_strength,
                "structure_strength": args.structure_strength,
                "inference_steps": args.steps,
                "random_seed": args.seed,
                "background_mode": "KEEP",
                "output_format": "PNG",
            },
        },
        expected=(202,),
    )
    print(f"job created: {job['id']}")

    deadline = time.monotonic() + args.timeout
    while job["status"] in {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"job {job['id']} did not finish within {args.timeout:g}s"
            )
        time.sleep(2)
        job = client.json("GET", f"/jobs/{job['id']}")

    tag = f"-{args.tag}" if args.tag else ""
    record_path = out_dir / f"job-{job['id']}{tag}.json"
    record_path.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
    print(f"job record written: {record_path}")

    if job["status"] != "SUCCEEDED":
        print(
            f"job FAILED: {job.get('error_code')} {job.get('error_message_safe')}",
            file=sys.stderr,
        )
        return 1

    output_url = str(job.get("output_url") or "")
    if not output_url:
        raise RuntimeError("successful job did not expose an output URL")
    output, _ = client.request("GET", output_url)
    suffix = ".png" if output.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
    image_path = out_dir / f"output-{job['id']}{tag}{suffix}"
    image_path.write_bytes(output)
    print(f"output written: {image_path} ({len(output)} bytes)")
    diagnostics = job.get("diagnostics") or {}
    print("diagnostics:")
    print(json.dumps(diagnostics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
