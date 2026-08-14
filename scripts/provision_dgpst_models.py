#!/usr/bin/env python3
"""Provision and verify the exact local DGPST inference artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


DEFAULT_MANIFEST = Path("models/dgpst/manifest.json")
DEFAULT_MODEL_ROOT = Path("models/dgpst")
GOOGLE_DRIVE_HOST = "drive.google.com"
HUGGING_FACE_HOST = "huggingface.co"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact path must be a non-empty string")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"unsafe artifact path: {raw!r}")
    return path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest artifacts must be a non-empty list")

    seen: set[PurePosixPath] = set()
    normalized: list[dict[str, Any]] = []
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise ValueError("each artifact must be an object")
        artifact = dict(raw)
        relative = safe_relative_path(artifact.get("path"))
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        sha256 = artifact.get("sha256")
        byte_size = artifact.get("bytes")
        source = artifact.get("source")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"invalid SHA-256 for {relative}")
        try:
            int(sha256, 16)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 for {relative}") from exc
        if not isinstance(byte_size, int) or byte_size <= 0:
            raise ValueError(f"invalid byte size for {relative}")
        if not isinstance(source, str) or urlparse(source).scheme != "https":
            raise ValueError(f"invalid HTTPS source for {relative}")
        artifact["_relative"] = relative
        normalized.append(artifact)
    return normalized


def verify_artifacts(model_root: Path, artifacts: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    resolved_root = model_root.resolve()
    for artifact in artifacts:
        relative = artifact["_relative"]
        path = (model_root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(resolved_root):
            failures.append(f"unsafe resolved path: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != artifact["bytes"]:
            failures.append(
                f"size mismatch: {relative} expected={artifact['bytes']} actual={actual_size}"
            )
            continue
        actual_digest = file_digest(path)
        if actual_digest != artifact["sha256"]:
            failures.append(
                f"checksum mismatch: {relative} expected={artifact['sha256']} "
                f"actual={actual_digest}"
            )
            continue
        print(f"verified {relative} bytes={actual_size} sha256={actual_digest}")
    return failures


def download_hugging_face(
    model_root: Path,
    artifacts: list[dict[str, Any]],
    *,
    token: str | None,
    force: bool,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face download requires `huggingface-hub`; install the "
            "provisioning dependencies first."
        ) from exc

    for artifact in artifacts:
        source = str(artifact["source"])
        parsed = urlparse(source)
        if parsed.hostname != HUGGING_FACE_HOST:
            continue
        repo_id = parsed.path.strip("/")
        revision = artifact.get("revision")
        source_path = artifact.get("source_path")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"Hugging Face revision must be a commit SHA: {source}")
        if not isinstance(source_path, str):
            raise ValueError(f"missing source_path for {artifact['path']}")
        relative = artifact["_relative"]
        local_prefix = relative.parts[0]
        destination = (
            model_root / local_prefix / Path(*PurePosixPath(source_path).parts)
        )
        if destination.is_file() and not force:
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=source_path,
            revision=revision,
            local_dir=model_root / local_prefix,
            token=token,
            force_download=force,
        )


def download_google_drive(
    model_root: Path, artifacts: list[dict[str, Any]], *, force: bool
) -> None:
    records = [
        artifact
        for artifact in artifacts
        if urlparse(str(artifact["source"])).hostname == GOOGLE_DRIVE_HOST
    ]
    if not records:
        return
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "DGPST checkpoint download requires `gdown`; install the provisioning "
            "dependencies first."
        ) from exc

    missing_or_forced = force or any(
        not (model_root / Path(*record["_relative"].parts)).is_file()
        for record in records
    )
    if not missing_or_forced:
        return

    seen_ids: set[str] = set()
    for record in records:
        safe_relative_path(record.get("source_path"))
        source_id = record.get("source_id")
        if (
            not isinstance(source_id, str)
            or len(source_id) < 10
            or not all(
                character.isalnum() or character in "_-" for character in source_id
            )
            or source_id in seen_ids
        ):
            raise ValueError(
                f"invalid or duplicate Google Drive source_id: {source_id!r}"
            )
        seen_ids.add(source_id)

    # Never let a mutable third-party folder choose local paths. Each exact
    # Drive file ID is downloaded into an isolated same-filesystem directory,
    # verified, then atomically moved to its manifest-owned destination.
    model_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dgpst-drive-", dir=model_root
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        for index, record in enumerate(records):
            source_path = safe_relative_path(record["source_path"])
            destination = model_root / Path(*record["_relative"].parts)
            if destination.is_file() and not force:
                continue
            temporary_path = temporary_root / f"artifact-{index}.download"
            downloaded = gdown.download(
                id=record["source_id"],
                output=str(temporary_path),
                quiet=False,
                use_cookies=False,
                resume=False,
            )
            if downloaded is None or not temporary_path.is_file():
                raise RuntimeError(f"gdown failed to download {source_path}")
            actual_size = temporary_path.stat().st_size
            actual_hash = file_digest(temporary_path)
            if actual_size != record["bytes"] or actual_hash != record["sha256"]:
                raise RuntimeError(
                    f"download verification failed for {source_path}: "
                    f"bytes={actual_size} sha256={actual_hash}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download", action="store_true", help="download missing artifacts"
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no network access (the default)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download existing artifacts"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="optional Hugging Face token; prefer the HF_TOKEN environment variable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.force and not args.download:
        print("error: --force requires --download", file=sys.stderr)
        return 2
    try:
        artifacts = load_manifest(args.manifest)
        if args.download:
            import os

            token = args.hf_token or os.environ.get("HF_TOKEN")
            download_hugging_face(
                args.model_root, artifacts, token=token, force=args.force
            )
            download_google_drive(args.model_root, artifacts, force=args.force)
        failures = verify_artifacts(args.model_root, artifacts)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1
    print(f"verified {len(artifacts)} DGPST runtime artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
