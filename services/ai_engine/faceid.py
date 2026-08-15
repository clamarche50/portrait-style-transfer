"""Identity extraction for the FaceID PlusV2 adapter (insightface buffalo_l on CPU)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_CROP_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class FaceIdentity:
    embedding: np.typing.NDArray[np.float32]  # normalized 512-d ArcFace embedding
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    det_score: float


class FaceIdentityExtractor:
    """Detects the dominant face and produces its identity embedding.

    The ONNX detectors run on the CPU execution provider, which keeps the
    GPU fully dedicated to diffusion inference and avoids CUDA EP issues on
    consumer drivers.
    """

    DET_SIZE = (640, 640)
    CROP_RATIO = _CROP_RATIO

    def __init__(self, model_root: Path) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_root),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=self.DET_SIZE)

    def extract(self, image: Image.Image) -> FaceIdentity | None:
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        faces = self._app.get(bgr)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda candidate: candidate.det_score
            * (candidate.bbox[2] - candidate.bbox[0])
            * (candidate.bbox[3] - candidate.bbox[1]),
        )
        if getattr(face, "normed_embedding", None) is None:
            return None
        return FaceIdentity(
            embedding=np.asarray(face.normed_embedding, dtype=np.float32),
            bbox=(
                float(face.bbox[0]),
                float(face.bbox[1]),
                float(face.bbox[2]),
                float(face.bbox[3]),
            ),
            det_score=float(face.det_score),
        )

    @staticmethod
    def crop(
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        *,
        crop_ratio: float = _CROP_RATIO,
    ) -> Image.Image:
        """Square-crop the detected face with margin for CLIP embedding."""

        width, height = image.size
        x1, y1, x2, y2 = bbox
        side = max(x2 - x1, y2 - y1) * crop_ratio
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = max(0, int(round(center_x - side / 2.0)))
        top = max(0, int(round(center_y - side / 2.0)))
        right = min(width, int(round(center_x + side / 2.0)))
        bottom = min(height, int(round(center_y + side / 2.0)))
        if right - left < 8 or bottom - top < 8:
            return image.copy()
        return image.crop((left, top, right, bottom)).convert("RGB")
