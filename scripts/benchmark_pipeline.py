#!/usr/bin/env python3
"""Run an explicit pipeline command repeatedly and write reproducible timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_stage_timings(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        timings = value.get("stage_timings_ms") if isinstance(value, dict) else None
        if isinstance(timings, dict) and all(
            isinstance(item, (int, float)) for item in timings.values()
        ):
            return {str(key): float(item) for key, item in timings.items()}
    return {}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        help="quoted executable and arguments; no shell expansion",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.command.strip():
        print(
            "error: --command must be an explicit non-empty pipeline command",
            file=sys.stderr,
        )
        return 2
    if args.runs < 1 or args.warmup < 0:
        print("error: runs must be positive and warmup non-negative", file=sys.stderr)
        return 2
    command = shlex.split(args.command, posix=os.name != "nt")
    if not command:
        print("error: command parsing produced no executable", file=sys.stderr)
        return 2
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    measurements: list[dict[str, Any]] = []
    total = args.warmup + args.runs
    for index in range(total):
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"error: benchmark command failed to execute: {exc}", file=sys.stderr)
            return 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.returncode != 0:
            print(
                f"error: benchmark command exited {result.returncode}", file=sys.stderr
            )
            if result.stderr:
                print(result.stderr[-2000:], file=sys.stderr)
            return result.returncode or 1
        if index >= args.warmup:
            measurements.append(
                {
                    "elapsed_ms": elapsed_ms,
                    "stage_timings_ms": parse_stage_timings(result.stdout),
                }
            )
    elapsed = [float(item["elapsed_ms"]) for item in measurements]
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "command_executable": Path(command[0]).name,
        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        "runs": args.runs,
        "warmup_runs": args.warmup,
        "random_seed_environment": environment.get("PYTHONHASHSEED"),
        "elapsed_ms": {
            "minimum": min(elapsed),
            "mean": statistics.fmean(elapsed),
            "median": statistics.median(elapsed),
            "p95": percentile(elapsed, 0.95),
            "maximum": max(elapsed),
        },
        "measurements": measurements,
        "note": "Hardware and actual model availability must accompany published benchmark claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {args.runs} measured runs to {args.output}; median={statistics.median(elapsed):.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
