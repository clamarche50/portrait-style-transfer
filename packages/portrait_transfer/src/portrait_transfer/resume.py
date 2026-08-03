"""Validated, private checkpoints for correction reruns.

Checkpoint arrays are deliberately separate from user-facing debug artifacts.
They carry stage-specific input signatures and integrity digests; malformed,
stale, or incomplete bundles are rejected and the caller performs a full run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .config import TransferSettings
from .crop import CropContext
from .geometry.validity import map_validity
from .types import (
    AlignmentDiagnostics,
    BoundingBox,
    CompatibilityReport,
    FloatArray,
    PortraitMasks,
    QualityReport,
    RuntimeContext,
)

CHECKPOINT_VERSION: Final[int] = 1
RESUME_FROM_STAGE_KEY: Final[str] = "resume_from_stage"
RESUME_MULTISCALE: Final[str] = "multiscale"
RESUME_EYES: Final[str] = "eyes"
RESUME_BACKGROUND: Final[str] = "background"
RESUME_STAGES: Final[tuple[str, ...]] = (
    RESUME_MULTISCALE,
    RESUME_EYES,
    RESUME_BACKGROUND,
)

_SCHEMA = "resume.schema"
_SOURCE_BOX = "resume.input_source_box"
_INPUT_CROP = "resume.input_crop"
_REFERENCE_CROP = "resume.reference_crop"
_INPUT_HEAD = "resume.input_head"
_INPUT_FOREGROUND = "resume.input_foreground_alpha"
_INPUT_IRIS_LEFT = "resume.input_iris_left_base"
_INPUT_IRIS_RIGHT = "resume.input_iris_right_base"
_REFERENCE_HEAD = "resume.reference_head"
_REFERENCE_FOREGROUND = "resume.reference_foreground_alpha"
_REFERENCE_IRIS_LEFT = "resume.reference_iris_left"
_REFERENCE_IRIS_RIGHT = "resume.reference_iris_right"
_MAPPING = "resume.correspondence_mapping"
_PRE_EYE = "resume.pre_eye_rgb"
_POST_EYE = "resume.post_eye_rgb"
_DIAGNOSTICS = "resume.diagnostics_json"


def _signature_key(stage: str) -> str:
    return f"resume.signature.{stage}"


def _integrity_key(stage: str) -> str:
    return f"resume.integrity.{stage}"


_BASE_KEYS: Final[tuple[str, ...]] = (
    _SCHEMA,
    _SOURCE_BOX,
    _INPUT_CROP,
    _REFERENCE_CROP,
    _INPUT_FOREGROUND,
    _INPUT_HEAD,
    _INPUT_IRIS_LEFT,
    _INPUT_IRIS_RIGHT,
    _REFERENCE_HEAD,
    _REFERENCE_FOREGROUND,
    _REFERENCE_IRIS_LEFT,
    _REFERENCE_IRIS_RIGHT,
    _MAPPING,
    _DIAGNOSTICS,
)
_STAGE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    RESUME_MULTISCALE: (
        *_BASE_KEYS,
        _signature_key(RESUME_MULTISCALE),
    ),
    RESUME_EYES: (
        *_BASE_KEYS,
        _PRE_EYE,
        _signature_key(RESUME_EYES),
    ),
    RESUME_BACKGROUND: (
        *_BASE_KEYS,
        _PRE_EYE,
        _POST_EYE,
        _signature_key(RESUME_BACKGROUND),
    ),
}


@dataclass(frozen=True)
class ResumeState:
    stage: str
    input_context: CropContext
    input_crop: FloatArray
    reference_crop: FloatArray
    input_masks: PortraitMasks
    reference_masks: PortraitMasks
    mapping: FloatArray
    pre_eye_rgb: FloatArray | None
    post_eye_rgb: FloatArray | None
    input_quality: QualityReport
    reference_quality: QualityReport
    compatibility: CompatibilityReport
    alignment: AlignmentDiagnostics
    upstream_warnings: tuple[str, ...]
    warnings_after_eyes: tuple[str, ...]


def requested_resume_stage(runtime: RuntimeContext) -> str | None:
    value = runtime.corrections.get(RESUME_FROM_STAGE_KEY)
    if value is None:
        return None
    return str(value).strip().lower()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"object_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _operation_payload(runtime: RuntimeContext, stage: str) -> list[Any]:
    raw = runtime.corrections.get("operations", ())
    operations = raw if isinstance(raw, Sequence) else ()
    selected: list[Any] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            selected.append(_jsonable(operation))
            continue
        operation_type = operation.get("type")
        if stage == RESUME_MULTISCALE and operation_type in (
            "gain_constraint",
            "eye_center",
        ):
            continue
        if stage == RESUME_EYES and operation_type == "eye_center":
            continue
        selected.append(_jsonable(operation))
    return selected


def _correction_payload(runtime: RuntimeContext, stage: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"operations": _operation_payload(runtime, stage)}
    for key, value in sorted(runtime.corrections.items()):
        name = str(key)
        if name in ("operations", RESUME_FROM_STAGE_KEY):
            continue
        if stage == RESUME_MULTISCALE and (
            name == "monochrome_style" or name.startswith("gain_")
        ):
            continue
        payload[name] = _jsonable(value)
    return payload


def _settings_payload(settings: TransferSettings, stage: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "processing_long_edge": settings.processing_long_edge,
        "dense_alignment": settings.dense_alignment,
        "preflight": _jsonable(settings.preflight),
        "beier_neely": _jsonable(settings.beier_neely),
        "dense": _jsonable(settings.dense),
        "random_seed": settings.random_seed,
    }
    if stage in (RESUME_EYES, RESUME_BACKGROUND):
        payload.update(
            {
                "algorithm_profile": settings.algorithm_profile.value,
                "transfer_strength": settings.transfer_strength,
                "residual_strength": settings.residual_strength,
                "global_range_mix": settings.global_range_mix,
                "legacy_color_mode": settings.legacy_color_mode.value,
                "gain": _jsonable(settings.gain),
            }
        )
    if stage == RESUME_BACKGROUND:
        payload["eye_highlights"] = settings.eye_highlights
    return payload


def _update_array_digest(digest: Any, name: str, value: NDArray[Any]) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _stage_signature(
    stage: str,
    input_image: FloatArray,
    reference_image: FloatArray,
    settings: TransferSettings,
    runtime: RuntimeContext,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(f"portrait-transfer-resume-v{CHECKPOINT_VERSION}:{stage}".encode())
    _update_array_digest(digest, "input", np.asarray(input_image, dtype=np.float32))
    _update_array_digest(
        digest, "reference", np.asarray(reference_image, dtype=np.float32)
    )
    analyzer = runtime.analyzer
    analyzer_identity = getattr(
        analyzer,
        "cache_identity",
        f"{type(analyzer).__module__}.{type(analyzer).__qualname__}",
    )
    dense_backend = runtime.dense_backend
    dense_identity = getattr(
        dense_backend,
        "cache_identity",
        f"{type(dense_backend).__module__}.{type(dense_backend).__qualname__}",
    )
    payload: dict[str, Any] = {
        "analyzer": _jsonable(analyzer_identity),
        "dense_backend": _jsonable(dense_identity),
        "settings": _settings_payload(settings, stage),
        "corrections": _correction_payload(runtime, stage),
    }
    if stage == RESUME_BACKGROUND:
        payload["eye_assets"] = _jsonable(runtime.eye_assets)
    digest.update(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return digest.digest()


def _bytes_array(value: bytes) -> FloatArray:
    return np.frombuffer(value, dtype=np.uint8).astype(np.float32)


def _decode_bytes(value: NDArray[Any], *, maximum: int = 131_072) -> bytes:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or len(array) > maximum or not np.isfinite(array).all():
        raise ValueError("checkpoint byte array is malformed")
    rounded = np.rint(array)
    if np.any(array != rounded) or np.any(rounded < 0) or np.any(rounded > 255):
        raise ValueError("checkpoint byte array is outside uint8 range")
    return rounded.astype(np.uint8).tobytes()


def _checkpoint_digest(
    artifacts: Mapping[str, NDArray[Any]], keys: Sequence[str]
) -> bytes:
    digest = hashlib.sha256()
    digest.update(f"portrait-transfer-checkpoint-v{CHECKPOINT_VERSION}".encode())
    for key in keys:
        if key not in artifacts:
            raise KeyError(key)
        _update_array_digest(digest, key, np.asarray(artifacts[key], dtype=np.float32))
    return digest.digest()


def _diagnostic_payload(
    input_quality: QualityReport,
    reference_quality: QualityReport,
    compatibility: CompatibilityReport,
    alignment: AlignmentDiagnostics,
    upstream_warnings: Sequence[str],
    warnings_after_eyes: Sequence[str],
) -> bytes:
    payload = {
        "version": CHECKPOINT_VERSION,
        "input_quality": _jsonable(input_quality),
        "reference_quality": _jsonable(reference_quality),
        "compatibility": _jsonable(compatibility),
        "alignment": _jsonable(alignment),
        "upstream_warnings": list(upstream_warnings),
        "warnings_after_eyes": list(warnings_after_eyes),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_resume_artifacts(
    *,
    input_image: FloatArray,
    reference_image: FloatArray,
    settings: TransferSettings,
    runtime: RuntimeContext,
    input_context: CropContext,
    input_crop: FloatArray,
    reference_crop: FloatArray,
    input_masks: PortraitMasks,
    input_base_irises: tuple[FloatArray, FloatArray],
    reference_masks: PortraitMasks,
    mapping: FloatArray,
    pre_eye_rgb: FloatArray,
    post_eye_rgb: FloatArray,
    input_quality: QualityReport,
    reference_quality: QualityReport,
    compatibility: CompatibilityReport,
    alignment: AlignmentDiagnostics,
    upstream_warnings: Sequence[str],
    warnings_after_eyes: Sequence[str],
) -> dict[str, FloatArray]:
    """Create the complete private checkpoint bundle returned with a result."""

    box = input_context.source_box
    artifacts: dict[str, FloatArray] = {
        _SCHEMA: np.asarray([CHECKPOINT_VERSION], dtype=np.float32),
        _SOURCE_BOX: _bytes_array(
            np.asarray(
                [box.x, box.y, box.width, box.height], dtype=np.float64
            ).tobytes()
        ),
        _INPUT_CROP: np.asarray(input_crop, dtype=np.float32).copy(),
        _REFERENCE_CROP: np.asarray(reference_crop, dtype=np.float32).copy(),
        _INPUT_HEAD: np.asarray(input_masks.head, dtype=np.float32).copy(),
        _INPUT_FOREGROUND: np.asarray(
            input_masks.foreground_alpha, dtype=np.float32
        ).copy(),
        _INPUT_IRIS_LEFT: np.asarray(input_base_irises[0], dtype=np.float32).copy(),
        _INPUT_IRIS_RIGHT: np.asarray(input_base_irises[1], dtype=np.float32).copy(),
        _REFERENCE_HEAD: np.asarray(reference_masks.head, dtype=np.float32).copy(),
        _REFERENCE_FOREGROUND: np.asarray(
            reference_masks.foreground_alpha, dtype=np.float32
        ).copy(),
        _REFERENCE_IRIS_LEFT: np.asarray(
            reference_masks.irises[0], dtype=np.float32
        ).copy(),
        _REFERENCE_IRIS_RIGHT: np.asarray(
            reference_masks.irises[1], dtype=np.float32
        ).copy(),
        _MAPPING: np.asarray(mapping, dtype=np.float32).copy(),
        _PRE_EYE: np.asarray(pre_eye_rgb, dtype=np.float32).copy(),
        _POST_EYE: np.asarray(post_eye_rgb, dtype=np.float32).copy(),
        _DIAGNOSTICS: _bytes_array(
            _diagnostic_payload(
                input_quality,
                reference_quality,
                compatibility,
                alignment,
                upstream_warnings,
                warnings_after_eyes,
            )
        ),
    }
    for stage in RESUME_STAGES:
        artifacts[_signature_key(stage)] = _bytes_array(
            _stage_signature(stage, input_image, reference_image, settings, runtime)
        )
    for stage, keys in _STAGE_KEYS.items():
        artifacts[_integrity_key(stage)] = _bytes_array(
            _checkpoint_digest(artifacts, keys)
        )
    return artifacts


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _diagnostics_from_array(
    value: NDArray[Any],
) -> tuple[
    QualityReport,
    QualityReport,
    CompatibilityReport,
    AlignmentDiagnostics,
    tuple[str, ...],
    tuple[str, ...],
]:
    payload = json.loads(_decode_bytes(value).decode("utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != CHECKPOINT_VERSION
        or not _finite_numbers(payload)
    ):
        raise ValueError("checkpoint diagnostics are invalid")

    def quality(name: str) -> QualityReport:
        data = dict(payload[name])
        data["warnings"] = tuple(str(item) for item in data.get("warnings", ()))
        return QualityReport(**data)

    compatibility_data = dict(payload["compatibility"])
    compatibility_data["warnings"] = tuple(
        str(item) for item in compatibility_data.get("warnings", ())
    )
    alignment_data = dict(payload["alignment"])
    alignment_data["warnings"] = tuple(
        str(item) for item in alignment_data.get("warnings", ())
    )
    upstream_warnings = tuple(
        str(item) for item in payload.get("upstream_warnings", ())
    )
    warnings_after_eyes = tuple(
        str(item) for item in payload.get("warnings_after_eyes", ())
    )
    return (
        quality("input_quality"),
        quality("reference_quality"),
        CompatibilityReport(**compatibility_data),
        AlignmentDiagnostics(**alignment_data),
        upstream_warnings,
        warnings_after_eyes,
    )


def _float_artifact(artifacts: Mapping[str, NDArray[Any]], name: str) -> FloatArray:
    value = np.asarray(artifacts[name], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"checkpoint {name} is non-finite")
    return value.copy()


def _unit_array(value: FloatArray, shape: tuple[int, ...], name: str) -> FloatArray:
    if value.shape != shape or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError(f"checkpoint {name} has an invalid shape or range")
    return value


def load_resume_state(
    *,
    input_image: FloatArray,
    reference_image: FloatArray,
    settings: TransferSettings,
    runtime: RuntimeContext,
) -> ResumeState | None:
    """Load the requested stage only when every integrity check succeeds."""

    stage = requested_resume_stage(runtime)
    if stage not in RESUME_STAGES:
        return None
    artifacts = runtime.resume_artifacts
    try:
        keys = _STAGE_KEYS[stage]
        normalized = {
            key: _float_artifact(artifacts, key)
            for key in (*keys, _integrity_key(stage))
        }
        stored_integrity = _decode_bytes(normalized[_integrity_key(stage)], maximum=32)
        calculated_integrity = _checkpoint_digest(normalized, keys)
        if not hmac.compare_digest(stored_integrity, calculated_integrity):
            raise ValueError("checkpoint integrity mismatch")
        stored_signature = _decode_bytes(normalized[_signature_key(stage)], maximum=32)
        expected_signature = _stage_signature(
            stage, input_image, reference_image, settings, runtime
        )
        if not hmac.compare_digest(stored_signature, expected_signature):
            raise ValueError("checkpoint input signature mismatch")

        schema = normalized[_SCHEMA]
        if schema.shape != (1,) or schema[0] != CHECKPOINT_VERSION:
            raise ValueError("checkpoint schema mismatch")
        input_crop = normalized[_INPUT_CROP]
        reference_crop = normalized[_REFERENCE_CROP]
        if (
            input_crop.ndim != 3
            or input_crop.shape[2] != 3
            or reference_crop.shape != input_crop.shape
            or np.any(input_crop < 0.0)
            or np.any(input_crop > 1.0)
            or np.any(reference_crop < 0.0)
            or np.any(reference_crop > 1.0)
        ):
            raise ValueError("checkpoint canonical RGB crops are invalid")
        crop_shape: tuple[int, int] = (
            int(input_crop.shape[0]),
            int(input_crop.shape[1]),
        )
        if min(crop_shape) < 32 or max(crop_shape) > settings.processing_long_edge:
            raise ValueError("checkpoint canonical crop dimensions are invalid")

        source_bytes = _decode_bytes(normalized[_SOURCE_BOX], maximum=32)
        if len(source_bytes) != 32:
            raise ValueError("checkpoint source box encoding is invalid")
        source = np.frombuffer(source_bytes, dtype=np.float64)
        if source.shape != (4,) or source[2] < 1e-3 or source[3] < 1e-3:
            raise ValueError("checkpoint source box is invalid")
        original_shape: tuple[int, int] = (
            int(input_image.shape[0]),
            int(input_image.shape[1]),
        )
        original_height, original_width = original_shape
        if (
            abs(float(source[0])) > 4.0 * original_width
            or abs(float(source[1])) > 4.0 * original_height
            or float(source[2]) > 4.0 * max(original_width, 1)
            or float(source[3]) > 4.0 * max(original_height, 1)
        ):
            raise ValueError("checkpoint source box is outside safe bounds")
        box = BoundingBox(*(float(item) for item in source))
        scale_x = crop_shape[1] / box.width
        scale_y = crop_shape[0] / box.height
        transform = np.asarray(
            (
                (scale_x, 0.0, -box.x * scale_x),
                (0.0, scale_y, -box.y * scale_y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        if not np.isfinite(transform).all():
            raise ValueError("checkpoint crop transform is invalid")
        context = CropContext(
            original_shape,
            crop_shape,
            box,
            transform,
            np.zeros((68, 2), dtype=np.float32),
        )
        input_foreground = _unit_array(
            normalized[_INPUT_FOREGROUND], crop_shape, _INPUT_FOREGROUND
        )
        if float(np.mean(input_foreground > 0.05)) < 0.001:
            raise ValueError("checkpoint foreground mask has no usable coverage")
        zeros = np.zeros(crop_shape, dtype=np.float32)
        input_head = _unit_array(normalized[_INPUT_HEAD], crop_shape, _INPUT_HEAD)
        reference_head = _unit_array(
            normalized[_REFERENCE_HEAD], crop_shape, _REFERENCE_HEAD
        )
        reference_foreground = _unit_array(
            normalized[_REFERENCE_FOREGROUND], crop_shape, _REFERENCE_FOREGROUND
        )
        input_irises = (
            _unit_array(normalized[_INPUT_IRIS_LEFT], crop_shape, _INPUT_IRIS_LEFT),
            _unit_array(normalized[_INPUT_IRIS_RIGHT], crop_shape, _INPUT_IRIS_RIGHT),
        )
        reference_irises = (
            _unit_array(
                normalized[_REFERENCE_IRIS_LEFT], crop_shape, _REFERENCE_IRIS_LEFT
            ),
            _unit_array(
                normalized[_REFERENCE_IRIS_RIGHT], crop_shape, _REFERENCE_IRIS_RIGHT
            ),
        )
        if (
            float(np.mean(input_head > 0.05)) < 0.001
            or float(np.mean(reference_head > 0.05)) < 0.001
            or any(np.count_nonzero(mask > 0.05) == 0 for mask in input_irises)
            or any(np.count_nonzero(mask > 0.05) == 0 for mask in reference_irises)
        ):
            raise ValueError("checkpoint portrait masks have no usable coverage")
        mapping = normalized[_MAPPING]
        if mapping.shape != (*crop_shape, 2):
            raise ValueError("checkpoint correspondence map shape is invalid")
        reference_shape: tuple[int, int, int] = (
            int(reference_crop.shape[0]),
            int(reference_crop.shape[1]),
            int(reference_crop.shape[2]),
        )
        validity = map_validity(mapping, reference_shape)
        if validity.valid_fraction < 0.70 or validity.negative_jacobian_fraction > 0.05:
            raise ValueError("checkpoint correspondence map is invalid")
        pre_eye: FloatArray | None = None
        post_eye: FloatArray | None = None

        if stage in (RESUME_EYES, RESUME_BACKGROUND):
            pre_eye = _unit_array(normalized[_PRE_EYE], input_crop.shape, _PRE_EYE)
        if stage == RESUME_BACKGROUND:
            post_eye = _unit_array(normalized[_POST_EYE], input_crop.shape, _POST_EYE)

        diagnostics = _diagnostics_from_array(normalized[_DIAGNOSTICS])
        input_masks = PortraitMasks(
            person=input_foreground.copy(),
            head=input_head,
            face_skin=zeros.copy(),
            hair=zeros.copy(),
            eyes=(zeros.copy(), zeros.copy()),
            irises=input_irises,
            effective_transfer=input_head.copy(),
            foreground_alpha=input_foreground,
        )
        reference_masks = PortraitMasks(
            person=reference_foreground.copy(),
            head=reference_head,
            face_skin=zeros.copy(),
            hair=zeros.copy(),
            eyes=(zeros.copy(), zeros.copy()),
            irises=reference_irises,
            effective_transfer=reference_head.copy(),
            foreground_alpha=reference_foreground,
        )
        return ResumeState(
            stage=stage,
            input_context=context,
            input_crop=input_crop,
            reference_crop=reference_crop,
            input_masks=input_masks,
            reference_masks=reference_masks,
            mapping=mapping,
            pre_eye_rgb=pre_eye,
            post_eye_rgb=post_eye,
            input_quality=diagnostics[0],
            reference_quality=diagnostics[1],
            compatibility=diagnostics[2],
            alignment=diagnostics[3],
            upstream_warnings=diagnostics[4],
            warnings_after_eyes=diagnostics[5],
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


__all__ = [
    "CHECKPOINT_VERSION",
    "RESUME_BACKGROUND",
    "RESUME_EYES",
    "RESUME_FROM_STAGE_KEY",
    "RESUME_MULTISCALE",
    "RESUME_STAGES",
    "ResumeState",
    "build_resume_artifacts",
    "load_resume_state",
    "requested_resume_stage",
]
