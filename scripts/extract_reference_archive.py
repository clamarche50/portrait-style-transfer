#!/usr/bin/env python3
"""Safely extract the uploaded text-serialized research archive.

The format contains repeated ASCII marker blocks of the form::

    ================================================
    FILE: relative/path
    ================================================
    <verbatim bytes>

The script never imports, compiles, or executes extracted content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


MARKER = re.compile(rb"(?m)^={16,}\r?\nFILE:[ \t]*([^\r\n]+)\r?\n={16,}\r?\n")
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class ArchiveError(ValueError):
    """Raised when the serialized archive violates the safe format."""


def _safe_member(raw_name: bytes) -> PurePosixPath:
    try:
        name = raw_name.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ArchiveError("member name is not valid UTF-8") from exc
    if not name or "\x00" in name or "\\" in name:
        raise ArchiveError(f"unsafe member name: {name!r}")
    if name.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", name):
        raise ArchiveError(f"absolute or drive-prefixed member: {name!r}")
    member = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in member.parts):
        raise ArchiveError(f"member contains an unsafe path component: {name!r}")
    for part in member.parts:
        if part.endswith((" ", ".")) or ":" in part:
            raise ArchiveError(f"member is unsafe on Windows: {name!r}")
        stem = part.split(".", 1)[0].upper()
        if stem in WINDOWS_RESERVED:
            raise ArchiveError(f"member uses a reserved device name: {name!r}")
    return member


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def extract(
    source: Path,
    output: Path,
    manifest_path: Path,
    *,
    overwrite: bool,
    max_member_bytes: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    if not source.is_file():
        raise ArchiveError(f"serialized archive does not exist: {source}")
    source_size = source.stat().st_size
    if source_size > max_total_bytes + 16 * 1024 * 1024:
        raise ArchiveError("serialized archive exceeds the configured input byte limit")
    serialized = source.read_bytes()
    matches = list(MARKER.finditer(serialized))
    if not matches:
        raise ArchiveError("no FILE marker blocks were found")

    if output.absolute().is_symlink():
        raise ArchiveError(f"output root must not be a symlink: {output}")
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ArchiveError(f"output root must not be a symlink: {root}")

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for index, marker in enumerate(matches):
        member = _safe_member(marker.group(1))
        relative = member.as_posix()
        if relative in seen:
            raise ArchiveError(f"duplicate member: {relative}")
        seen.add(relative)

        content_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(serialized)
        )
        content = serialized[marker.end() : content_end]
        if len(content) > max_member_bytes:
            raise ArchiveError(f"member exceeds byte limit: {relative}")
        total_bytes += len(content)
        if total_bytes > max_total_bytes:
            raise ArchiveError("archive exceeds total extraction byte limit")

        destination = root.joinpath(*member.parts)
        resolved_parent = destination.parent.resolve()
        try:
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise ArchiveError(f"member escapes output root: {relative}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        for parent in destination.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise ArchiveError(f"member traverses a symlink: {relative}")
        if destination.is_symlink():
            raise ArchiveError(f"member destination is a symlink: {relative}")
        mode = "wb" if overwrite else "xb"
        with destination.open(mode) as handle:
            handle.write(content)
        entries.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_filename": source.name,
        "source_bytes": len(serialized),
        "source_sha256": hashlib.sha256(serialized).hexdigest(),
        "ignored_header_bytes": matches[0].start(),
        "output_root": output.as_posix(),
        "member_count": len(entries),
        "extracted_bytes": total_bytes,
        "members": entries,
        "execution_policy": "content was parsed and written but never executed",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.environ.get("REFERENCE_ARCHIVE_FILE", ""),
        help="path to the uploaded text serialization",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("REFERENCE_SOURCE_DIR", "reference/original-matlab"),
        type=Path,
        help="ignored extraction directory",
    )
    parser.add_argument(
        "--manifest",
        default=Path("reference/manifests/extraction-manifest.json"),
        type=Path,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-member-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source:
        print(
            "error: provide --source or set REFERENCE_ARCHIVE_FILE; no reference material was extracted",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = extract(
            Path(args.source),
            args.output,
            args.manifest,
            overwrite=args.overwrite,
            max_member_bytes=args.max_member_bytes,
            max_total_bytes=args.max_total_bytes,
        )
    except (ArchiveError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"extracted {manifest['member_count']} members ({manifest['extracted_bytes']} bytes); "
        f"manifest: {args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
