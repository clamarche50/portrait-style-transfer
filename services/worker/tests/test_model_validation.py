from __future__ import annotations

import hashlib
import json

from portrait_api.config import Settings
from portrait_api.services.model_validation import verify_required_models


def test_model_manifest_checksums_are_enforced(tmp_path) -> None:
    face = b"synthetic-face-model"
    segmenter = b"synthetic-segmenter-model"
    (tmp_path / "face.task").write_bytes(face)
    (tmp_path / "segmenter.tflite").write_bytes(segmenter)
    manifest = {
        "models": [
            {"filename": "face.task", "sha256": hashlib.sha256(face).hexdigest()},
            {
                "filename": "segmenter.tflite",
                "sha256": hashlib.sha256(segmenter).hexdigest(),
            },
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    settings = Settings(
        app_env="test",
        model_dir=tmp_path,
        face_landmarker_model="face.task",
        image_segmenter_model="segmenter.tflite",
    )
    assert verify_required_models(settings).valid

    (tmp_path / "segmenter.tflite").write_bytes(b"tampered")
    result = verify_required_models(settings)
    assert not result.valid
    assert result.mismatched == ("segmenter.tflite",)
