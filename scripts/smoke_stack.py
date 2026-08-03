#!/usr/bin/env python3
"""Poll an HTTP readiness endpoint and fail unless it reports ready."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/api/v1/health/ready")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                state = (
                    str(payload.get("status", "")).lower()
                    if isinstance(payload, dict)
                    else ""
                )
                if response.status == 200 and state in {"ready", "ok", "healthy"}:
                    print(f"ready: {args.url}")
                    return 0
                last_error = f"HTTP {response.status}, status={state!r}"
        except (
            OSError,
            ValueError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            last_error = str(exc)
        time.sleep(args.interval)
    print(f"error: readiness timed out: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
