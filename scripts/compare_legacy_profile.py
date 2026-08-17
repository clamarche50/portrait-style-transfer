#!/usr/bin/env python3
"""Compare numeric artifacts emitted by paper-exact and source-compatible runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        arrays: dict[str, np.ndarray] = {}
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            if child.suffix.lower() not in {".npy", ".npz", *IMAGE_SUFFIXES}:
                continue
            for key, value in load_arrays(child).items():
                arrays[f"{child.relative_to(path).as_posix()}::{key}"] = value
        if not arrays:
            raise ValueError(f"no comparable artifacts found under {path}")
        return arrays
    if path.suffix.lower() == ".npy":
        return {"array": np.asarray(np.load(path, allow_pickle=False))}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in sorted(archive.files)}
    if path.suffix.lower() in IMAGE_SUFFIXES:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to compare image artifacts") from exc
        with Image.open(path) as image:
            return {"image": np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0}
    raise ValueError(f"unsupported artifact type: {path}")


def metrics(paper: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    if paper.shape != source.shape:
        return {
            "comparable": False,
            "paper_shape": list(paper.shape),
            "source_shape": list(source.shape),
        }
    left = paper.astype(np.float64, copy=False)
    right = source.astype(np.float64, copy=False)
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return {
            "comparable": False,
            "reason": "no jointly finite values",
            "shape": list(left.shape),
        }
    difference = left[finite] - right[finite]
    left_values = left[finite]
    right_values = right[finite]
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    correlation = (
        float(np.sum(left_centered * right_centered) / denominator)
        if denominator
        else float(np.array_equal(left_values, right_values))
    )
    return {
        "comparable": True,
        "shape": list(left.shape),
        "finite_fraction": float(finite.mean()),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_signed_error": float(np.mean(difference)),
        "pearson_correlation": correlation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        required=True,
        help="PAPER_EXACT .npy/.npz/image or artifact directory",
    )
    parser.add_argument(
        "--source", required=True, help="SOURCE_2014_COMPAT artifact with matching keys"
    )
    parser.add_argument(
        "--output", default="-", help="JSON report path or '-' for stdout"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.paper or not args.source:
        print("error: both non-empty artifact paths are required", file=sys.stderr)
        return 2
    try:
        paper = load_arrays(Path(args.paper))
        source = load_arrays(Path(args.source))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    keys = sorted(set(paper) | set(source))
    comparisons: dict[str, Any] = {}
    for key in keys:
        if key not in paper or key not in source:
            comparisons[key] = {
                "comparable": False,
                "reason": "artifact missing from one profile",
            }
        else:
            comparisons[key] = metrics(paper[key], source[key])
    report = {
        "schema_version": 1,
        "paper_profile": "paper_exact",
        "source_profile": "source_2014_compat",
        "comparisons": comparisons,
        "note": "Differences are expected and do not establish parity with published results.",
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output == "-" or not args.output:
        sys.stdout.write(payload)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8", newline="\n")
        print(f"wrote profile comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
