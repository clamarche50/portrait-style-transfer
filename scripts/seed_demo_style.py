#!/usr/bin/env python3
"""Create a private starter style through the public API.

No portrait is bundled. Supplying --example uploads a user-provided,
rights-cleared image and attaches it to the new style.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ApiClient:
    def __init__(self, cookie_file: Path | None) -> None:
        self.cookie_file = cookie_file
        self.cookies = http.cookiejar.MozillaCookieJar(
            str(cookie_file) if cookie_file else None
        )
        if cookie_file and cookie_file.is_file():
            self.cookies.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookies)
        )
        self.csrf_token = next(
            (cookie.value for cookie in self.cookies if "csrf" in cookie.name.lower()),
            None,
        )

    def request_json(
        self, url: str, payload: bytes, content_type: str
    ) -> dict[str, Any]:
        headers = {"Content-Type": content_type, "Accept": "application/json"}
        if self.csrf_token:
            headers["X-CSRF-Token"] = self.csrf_token
        request = urllib.request.Request(
            url, data=payload, headers=headers, method="POST"
        )
        with self.opener.open(request, timeout=60) as response:
            self.csrf_token = response.headers.get("X-CSRF-Token", self.csrf_token)
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError(f"API returned a non-object response for {url}")
        return body

    def save(self) -> None:
        if not self.cookie_file:
            return
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        self.cookies.save(ignore_discard=True, ignore_expires=True)
        os.chmod(self.cookie_file, 0o600)


def multipart_image(path: Path) -> tuple[bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("example must be JPEG, PNG, or WebP")
    boundary = f"portrait-{uuid.uuid4().hex}"
    chunks = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="kind"\r\n\r\nSTYLE_EXAMPLE\r\n'.encode(),
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="upload{path.suffix.lower()}"\r\nContent-Type: {mime_type}\r\n\r\n'
        ).encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def identifier(response: dict[str, Any], kind: str) -> str:
    value = response.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} response did not include an id")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--name", default="Private starter style")
    parser.add_argument("--description", default="User-owned reference collection")
    parser.add_argument("--example", type=Path)
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="optional private cookie jar; protect it like a credential and never commit it",
    )
    parser.add_argument(
        "--public", action="store_true", help="explicitly publish the seeded style"
    )
    parser.add_argument(
        "--confirm-rights",
        action="store_true",
        help="required affirmation that supplied references may be used",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_rights:
        print("error: --confirm-rights is required", file=sys.stderr)
        return 2
    base = args.api_url.rstrip("/")
    client = ApiClient(args.cookie_file)
    try:
        style_payload = json.dumps(
            {
                "name": args.name,
                "description": args.description,
                "rights_confirmed": True,
                "is_public": args.public,
            }
        ).encode()
        style = client.request_json(f"{base}/styles", style_payload, "application/json")
        style_id = identifier(style, "style")
        visibility = "public" if args.public else "private"
        print(f"created {visibility} style {style_id}")
        if args.example:
            body, content_type = multipart_image(args.example)
            asset = client.request_json(f"{base}/assets/upload", body, content_type)
            asset_id = identifier(asset, "asset")
            example = client.request_json(
                f"{base}/styles/{style_id}/examples",
                json.dumps({"asset_id": asset_id}).encode(),
                "application/json",
            )
            print(
                f"attached example {identifier(example, 'style example')}; ingestion may continue asynchronously"
            )
        client.save()
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
