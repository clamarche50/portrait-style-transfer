from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from portrait_api.config import Settings


@dataclass(frozen=True, slots=True)
class ModelValidationResult:
    valid: bool
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    manifest_error: str | None = None

    def public_details(self) -> dict[str, object]:
        details: dict[str, object] = {"status": "invalid"}
        if self.missing:
            details["missing"] = list(self.missing)
        if self.mismatched:
            details["checksum_mismatch"] = list(self.mismatched)
        if self.untracked:
            details["not_in_manifest"] = list(self.untracked)
        if self.manifest_error:
            details["manifest"] = self.manifest_error
        return details


@lru_cache(maxsize=16)
def _sha256(path_text: str, modified_ns: int, byte_size: int) -> str:
    del modified_ns, byte_size
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_models(settings: Settings) -> ModelValidationResult:
    manifest_path = settings.model_dir / settings.model_manifest_file
    try:
        payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError("models must be a list")
    except FileNotFoundError:
        return ModelValidationResult(valid=False, manifest_error="missing")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return ModelValidationResult(valid=False, manifest_error="invalid")

    expected: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        filename = record.get("filename")
        checksum = record.get("sha256")
        if (
            isinstance(filename, str)
            and isinstance(checksum, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", checksum)
        ):
            expected[filename] = checksum.lower()

    missing: list[str] = []
    mismatched: list[str] = []
    untracked: list[str] = []
    for path in settings.required_model_paths:
        if not path.is_file():
            missing.append(path.name)
            continue
        checksum = expected.get(path.name)
        if checksum is None:
            untracked.append(path.name)
            continue
        stat = path.stat()
        actual = _sha256(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if actual != checksum:
            mismatched.append(path.name)
    return ModelValidationResult(
        valid=not (missing or mismatched or untracked),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
        untracked=tuple(untracked),
    )


__all__ = ["ModelValidationResult", "verify_required_models"]
