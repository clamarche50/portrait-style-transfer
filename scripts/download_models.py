#!/usr/bin/env python3
"""Download model artifacts from the checked-in manifest using atomic writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlparse


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_MODEL_HOSTS = {"storage.googleapis.com"}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or PurePath(value).name != value:
        raise ValueError(f"invalid model filename: {value!r}")
    if (
        value in {".", ".."}
        or "\x00" in value
        or any(separator in value for separator in ("/", "\\"))
    ):
        raise ValueError(f"unsafe model filename: {value!r}")
    return value


def verify(path: Path, expected: str | None) -> str:
    if path.stat().st_size == 0:
        raise ValueError(f"model file is empty: {path.name}")
    actual = digest_file(path)
    if expected is not None and actual != expected:
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {expected}, received {actual}"
        )
    return actual


def download(url: str, destination: Path, expected: str | None, timeout: float) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_MODEL_HOSTS:
        raise ValueError(f"model URL must use an approved official HTTPS host: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "portrait-style-transfer-model-fetch/1"}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            tempfile.NamedTemporaryFile(
                "wb", dir=destination.parent, delete=False
            ) as handle,
        ):
            temporary_path = Path(handle.name)
            final_host = urlparse(response.geturl()).hostname
            if final_host not in OFFICIAL_MODEL_HOSTS:
                raise ValueError(
                    f"model download redirected to an unapproved host: {final_host}"
                )
            while block := response.read(1024 * 1024):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"downloaded model is empty: {destination.name}")
        actual = verify(temporary_path, expected)
        os.replace(temporary_path, destination)
        temporary_path = None
        return actual
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify pre-provisioned files without network",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing valid file"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline and args.force:
        print("error: --offline and --force cannot be combined", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        models = manifest["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("manifest must contain a non-empty models list")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: invalid model manifest: {exc}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for entry in models:
        try:
            filename = safe_filename(entry.get("filename"))
            expected = entry.get("sha256")
            if expected is not None:
                if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(
                    expected.lower()
                ):
                    raise ValueError(f"invalid SHA-256 for {filename}")
                expected = expected.lower()
            destination = args.output_dir / filename
            required = bool(entry.get("required", False))
            if destination.is_file() and not args.force:
                actual = verify(destination, expected)
                print(f"verified {filename} sha256={actual}")
                continue
            if args.offline:
                if required:
                    raise FileNotFoundError(
                        f"required offline model is missing: {destination}"
                    )
                print(f"optional model absent: {filename}")
                continue
            url = entry.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError(f"model has no download URL: {filename}")
            actual = download(url, destination, expected, args.timeout)
            checksum_note = (
                "manifest checksum verified"
                if expected
                else "checksum recorded; manifest has no upstream digest"
            )
            print(f"downloaded {filename} sha256={actual} ({checksum_note})")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            failures.append(str(exc))

    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
