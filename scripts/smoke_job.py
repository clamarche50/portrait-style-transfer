#!/usr/bin/env python3
"""Exercise upload, asynchronous processing, download, and deletion over HTTP."""

from __future__ import annotations

import argparse
import binascii
import http.cookiejar
import json
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from typing import Any


def _png(width: int, height: int, variant: int) -> bytes:
    """Create a textured synthetic RGB image without committing a face fixture."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            checker = 34 if ((x // 12 + y // 12 + variant) % 2) else 0
            dx = (x - width / 2) / width
            dy = (y - height / 2) / height
            oval = max(0.0, 1.0 - 3.2 * dx * dx - 2.3 * dy * dy)
            rows.extend(
                (
                    int(max(0, min(255, 55 + checker + 135 * oval))),
                    int(max(0, min(255, 65 + checker + (105 + variant * 8) * oval))),
                    int(max(0, min(255, 78 + checker + (82 + variant * 12) * oval))),
                )
            )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + chunk(b"IEND", b"")
    )


class Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )

    def _csrf(self) -> str | None:
        return next(
            (cookie.value for cookie in self.cookies if cookie.name == "pst_csrf"), None
        )

    def request(
        self,
        method: str,
        path_or_url: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> tuple[bytes, dict[str, str]]:
        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
        elif path_or_url.startswith("/api/"):
            origin = urllib.parse.urlsplit(self.base_url)
            url = f"{origin.scheme}://{origin.netloc}{path_or_url}"
        else:
            url = f"{self.base_url}{path_or_url}"
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        token = self._csrf()
        if method not in {"GET", "HEAD", "OPTIONS"} and token:
            headers["X-CSRF-Token"] = token
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=30) as response:
                payload = response.read()
                status = response.status
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            response_headers = {
                key.lower(): value for key, value in exc.headers.items()
            }
        if status not in expected:
            raise RuntimeError(
                f"{method} {url} returned HTTP {status}: {payload[:500]!r}"
            )
        return payload, response_headers

    def json(
        self,
        method: str,
        path: str,
        value: object | None = None,
        *,
        expected: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        body = None if value is None else json.dumps(value).encode()
        payload, _ = self.request(
            method,
            path,
            body=body,
            content_type="application/json" if body is not None else None,
            expected=expected,
        )
        return json.loads(payload) if payload else {}

    def upload(self, kind: str, payload: bytes) -> dict[str, Any]:
        boundary = f"portrait-smoke-{uuid.uuid4().hex}"
        chunks = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="kind"\r\n\r\n{kind}\r\n'.encode(),
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="synthetic.png"\r\nContent-Type: image/png\r\n\r\n'
            ).encode(),
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        body, _ = self.request(
            "POST",
            "/assets/upload",
            body=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
            expected=(201,),
        )
        return json.loads(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1")
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = Client(args.url)
    ready = client.json("GET", "/health/ready")
    if str(ready.get("status", "")).lower() not in {"ready", "ok", "healthy"}:
        raise RuntimeError(f"API is not ready: {ready}")

    input_asset: dict[str, Any] | None = None
    reference_asset: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    deleted = False
    try:
        input_asset = client.upload("INPUT", _png(512, 512, 0))
        reference_asset = client.upload("REFERENCE", _png(512, 512, 1))
        job = client.json(
            "POST",
            "/jobs",
            {
                "input_asset_id": input_asset["id"],
                "reference_asset_id": reference_asset["id"],
                "settings": {
                    "algorithm_profile": "paper_exact",
                    "dense_alignment": False,
                    "processing_long_edge": 512,
                    "output_format": "PNG",
                },
            },
            expected=(202,),
        )
        deadline = time.monotonic() + args.timeout
        while job["status"] in {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job['id']} did not finish within {args.timeout:g}s"
                )
            time.sleep(1)
            job = client.json("GET", f"/jobs/{job['id']}")
        if job["status"] != "SUCCEEDED":
            raise RuntimeError(
                f"job failed: {job.get('error_code')} {job.get('error_message_safe')}"
            )

        output_url = str(job.get("output_url") or "")
        if not output_url:
            raise RuntimeError("successful job did not expose an output URL")
        output, _ = client.request("GET", output_url)
        if not output.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("output endpoint did not return a PNG")

        download = client.json("POST", f"/jobs/{job['id']}/download-url", {})
        downloaded, headers = client.request("GET", str(download["url"]))
        if downloaded != output or "attachment" not in headers.get(
            "content-disposition", ""
        ):
            raise RuntimeError(
                "download endpoint did not return the generated file as an attachment"
            )

        client.request("DELETE", f"/jobs/{job['id']}", expected=(204,))
        deleted = True
        client.request("GET", f"/jobs/{job['id']}", expected=(404,))
        print(
            f"smoke job succeeded: id={job['id']} bytes={len(output)} "
            f"profile={job.get('algorithm_profile')}"
        )
        return 0
    finally:
        if job is not None and not deleted:
            try:
                client.request("DELETE", f"/jobs/{job['id']}", expected=(204, 404))
            except Exception as cleanup_error:
                print(
                    f"warning: smoke-job cleanup failed: {cleanup_error}",
                    file=sys.stderr,
                )
        elif job is None:
            for asset in (input_asset, reference_asset):
                if asset:
                    try:
                        client.request(
                            "DELETE", f"/assets/{asset['id']}", expected=(204, 404)
                        )
                    except Exception as cleanup_error:
                        print(
                            f"warning: smoke-asset cleanup failed: {cleanup_error}",
                            file=sys.stderr,
                        )


if __name__ == "__main__":
    raise SystemExit(main())
