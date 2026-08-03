#!/usr/bin/env sh
set -eu

OPENAPI_URL="${OPENAPI_URL:-http://localhost:8000/openapi.json}"
OUTPUT="${OPENAPI_TYPES_OUTPUT:-lib/api/schema.d.ts}"
GENERATOR="${OPENAPI_TYPES_BIN:-./node_modules/.bin/openapi-typescript}"

if [ ! -x "$GENERATOR" ]; then
  echo "error: locked openapi-typescript executable not found at $GENERATOR" >&2
  exit 2
fi

TMP_SCHEMA="$(mktemp)"
trap 'rm -f "$TMP_SCHEMA"' EXIT HUP INT TERM

python - "$OPENAPI_URL" "$TMP_SCHEMA" <<'PY'
import pathlib
import sys
import urllib.request

url, destination = sys.argv[1:]
with urllib.request.urlopen(url, timeout=30) as response:
    body = response.read()
if not body.strip().startswith(b"{"):
    raise SystemExit("error: OpenAPI endpoint did not return JSON")
pathlib.Path(destination).write_bytes(body)
PY

mkdir -p "$(dirname "$OUTPUT")"
"$GENERATOR" "$TMP_SCHEMA" --output "$OUTPUT"
echo "generated $OUTPUT from $OPENAPI_URL"
