from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EngineFailure


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    artifact_count: int
    total_bytes: int
    manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: Path, model_root: Path) -> VerifiedManifest:
    try:
        raw = path.read_bytes()
        payload: dict[str, Any] = json.loads(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise EngineFailure("AI_MODEL_MANIFEST_INVALID", f"Cannot read {path}") from exc
    artifacts = payload.get("artifacts")
    if (
        payload.get("schema_version") != 1
        or not isinstance(artifacts, list)
        or not artifacts
    ):
        raise EngineFailure(
            "AI_MODEL_MANIFEST_INVALID", "Model manifest has an invalid schema"
        )
    root = model_root.resolve()
    total_bytes = 0
    for item in artifacts:
        if not isinstance(item, dict):
            raise EngineFailure(
                "AI_MODEL_MANIFEST_INVALID", "Invalid model artifact entry"
            )
        relative = Path(str(item.get("path", "")))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise EngineFailure(
                "AI_MODEL_MANIFEST_INVALID", "Artifact path escapes model root"
            ) from exc
        expected_bytes = int(item.get("bytes", -1))
        expected_hash = str(item.get("sha256", "")).lower()
        if not candidate.is_file() or candidate.stat().st_size != expected_bytes:
            raise EngineFailure(
                "AI_MODEL_MISSING", f"Missing or truncated model artifact: {relative}"
            )
        actual_hash = _sha256(candidate)
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise EngineFailure(
                "AI_MODEL_CHECKSUM_FAILED", f"Checksum failed: {relative}"
            )
        total_bytes += expected_bytes
    return VerifiedManifest(
        artifact_count=len(artifacts),
        total_bytes=total_bytes,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )
