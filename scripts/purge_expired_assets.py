#!/usr/bin/env python3
"""Run one bounded expired-asset cleanup batch through worker services."""

from __future__ import annotations

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 10_000:
        print("error: --batch-size must be between 1 and 10000", file=sys.stderr)
        return 2
    try:
        from portrait_worker.cleanup import purge_expired_records
        from portrait_worker.infrastructure import build_infrastructure

        infrastructure = build_infrastructure()
        try:
            result = purge_expired_records(infrastructure, batch_size=args.batch_size)
        finally:
            infrastructure.redis.close()
            infrastructure.engine.dispose()
    except ImportError as exc:
        print(
            f"error: install the worker project before running cleanup: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # Backend exception classes depend on configured drivers.
        print(f"error: cleanup failed with {type(exc).__name__}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("deletion_failures", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
