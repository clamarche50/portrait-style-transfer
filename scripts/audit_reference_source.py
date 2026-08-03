#!/usr/bin/env python3
"""Inventory local research source and reject copied files in production paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "code/code/style_transfer.m",
    "code/code/morph.m",
    "code/code/face.con",
    "code/code/sift_flow.m",
    "code/code/eye_transfer.m",
    "code/code/HistTransferOneD.m",
    "code/code/RGB2Lab.m",
    "code/code/Lab2RGB.m",
    "code/code/skin.m",
    "code/code/warpImage.m",
    "code/code/thresh_v.m",
    "code/code/local_match.m",
}
FORBIDDEN_TRACKED_PREFIXES = (
    "reference/original-matlab/",
    "reference/original-paper/",
    "reference/uploads/",
)
FORBIDDEN_TRACKED_MARKERS = (
    "clamarche50-temp-",
    "2014_portrait.pdf",
    "/libs/siftflow/",
    "/mexdensesift/",
    "/mexdiscreteflow/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def tracked_paths(root: Path) -> Iterable[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return [root / name for name in names if name]


def audit(
    source_dir: Path | None, production_root: Path | None
) -> tuple[dict[str, Any], list[str]]:
    root = source_dir.resolve() if source_dir is not None else None
    files = (
        sorted(path for path in root.rglob("*") if path.is_file())
        if root is not None
        else []
    )
    entries: list[dict[str, Any]] = []
    source_hashes: dict[str, list[str]] = {}
    restrictive_notice_files = 0
    for path in files:
        if root is None:
            raise RuntimeError(
                "internal audit error: source root is absent while files are present"
            )
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
        )
        source_hashes.setdefault(digest, []).append(relative)
        if path.stat().st_size <= 2 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "all rights reserved" in text or "commercial product" in text:
                restrictive_notice_files += 1

    found = {entry["path"] for entry in entries}
    violations: list[str] = []
    copied: list[dict[str, str]] = []
    forbidden_tracked: list[str] = []
    if production_root is not None:
        prod = production_root.resolve()
        for tracked in tracked_paths(prod):
            if not tracked.is_file():
                continue
            relative = tracked.relative_to(prod).as_posix()
            lowered = relative.lower()
            if relative.startswith(FORBIDDEN_TRACKED_PREFIXES) or any(
                marker in lowered for marker in FORBIDDEN_TRACKED_MARKERS
            ):
                forbidden_tracked.append(relative)
                violations.append(f"reference material is tracked: {relative}")
                continue
            if tracked.suffix.lower() == ".m" or tracked.name.lower().endswith(
                (".mex", ".mexa64", ".mexw64", ".mexmaci64")
            ):
                violations.append(
                    f"MATLAB/MEX file appears in tracked production paths: {relative}"
                )
            digest = sha256_file(tracked)
            if digest in source_hashes:
                for source_relative in source_hashes[digest]:
                    copied.append({"tracked": relative, "source": source_relative})
                violations.append(
                    f"tracked file is byte-identical to reference source: {relative}"
                )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_root": source_dir.as_posix() if source_dir is not None else None,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "expected_files_present": sorted(EXPECTED & found),
        "expected_files_missing": sorted(EXPECTED - found),
        "restrictive_notice_file_count": restrictive_notice_files,
        "files": entries,
        "copied_source_matches": copied,
        "forbidden_tracked_paths": forbidden_tracked,
        "violations": violations,
        "execution_policy": "files were read and hashed but never compiled or executed",
    }
    return manifest, violations


def markdown_report(manifest: dict[str, Any]) -> str:
    present = (
        "\n".join(f"- `{path}`" for path in manifest["expected_files_present"])
        or "- None"
    )
    missing = (
        "\n".join(f"- `{path}`" for path in manifest["expected_files_missing"])
        or "- None"
    )
    violations = "\n".join(f"- {item}" for item in manifest["violations"]) or "- None"
    return f"""# Local reference audit report

This generated report is intentionally ignored by Git. The audit read and hashed files but did not compile or execute them.

- Files: {manifest["file_count"]}
- Bytes: {manifest["total_bytes"]}
- Files containing a restrictive-notice indicator: {manifest["restrictive_notice_file_count"]}

## Expected files present

{present}

## Expected files missing

{missing}

## Shipping-boundary violations

{violations}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            os.environ.get("REFERENCE_SOURCE_DIR", "reference/original-matlab")
        ),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("reference/manifests/source-audit.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reference/manifests/source-audit.md")
    )
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_dir.is_dir():
        if args.allow_missing and args.production_root:
            manifest, violations = audit(None, args.production_root)
            manifest["note"] = (
                "reference source absent; only forbidden tracked paths/extensions were checked"
            )
        else:
            print(
                f"error: reference source directory does not exist: {args.source_dir}",
                file=sys.stderr,
            )
            return 2
    else:
        manifest, violations = audit(args.source_dir, args.production_root)
    atomic_text(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_text(args.report, markdown_report(manifest))
    print(
        f"audited {manifest['file_count']} reference files; violations: {len(violations)}"
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
