"""Facial keypoint extraction for the InstantID ControlNet.

The antelopev2 scrfd detector runs on the CPU execution provider and supplies
the five facial keypoints (eyes, nose, mouth corners) the InstantID ControlNet
conditions on. It complements the buffalo_l identity extractor in
``faceid.py``, which produces the FaceID adapter embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# The InstantID reference renderer draws this skeleton on a black canvas:
# each eye and mouth corner connects to the nose, then a colored dot marks
# every keypoint. Colors are brightened by 0.6, sticks are 4 px wide, and
# keypoint dots use a 10 px radius.
_LIMB_SEQUENCE = ((0, 2), (1, 2), (3, 2), (4, 2))
_KEYPOINT_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
)
_RENDER_SCALE = 0.6
_STICK_WIDTH = 4
_KEYPOINT_RADIUS = 10


@dataclass(frozen=True, slots=True)
class FaceKeypoints:
    """Five 2D keypoints: right eye, left eye, nose, right mouth, left mouth."""

    points: tuple[tuple[float, float], ...]
    det_score: float


class FaceKeypointDetector:
    """Detects the dominant face and returns its five 2D keypoints."""

    DET_SIZE = (640, 640)

    def __init__(self, model_root: Path) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="antelopev2",
            root=str(model_root),
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=self.DET_SIZE)

    def extract(self, image: Image.Image) -> FaceKeypoints | None:
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
        keypoints = getattr(face, "kps", None)
        if keypoints is None or np.asarray(keypoints).shape != (5, 2):
            return None
        points = tuple(
            (float(x), float(y)) for x, y in np.asarray(keypoints, dtype=np.float32)
        )
        return FaceKeypoints(points=points, det_score=float(face.det_score))


def render_keypoints(width: int, height: int, keypoints: FaceKeypoints) -> Image.Image:
    """Render the InstantID-style keypoint canvas for ControlNet conditioning."""

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index_a, index_b in _LIMB_SEQUENCE:
        color = _KEYPOINT_COLORS[index_a]
        draw.line(
            (keypoints.points[index_a], keypoints.points[index_b]),
            fill=color,
            width=_STICK_WIDTH,
        )
    for index, point in enumerate(keypoints.points):
        x, y = point
        draw.ellipse(
            (
                x - _KEYPOINT_RADIUS,
                y - _KEYPOINT_RADIUS,
                x + _KEYPOINT_RADIUS,
                y + _KEYPOINT_RADIUS,
            ),
            fill=_KEYPOINT_COLORS[index],
        )
    array = np.asarray(canvas, dtype=np.float32) * _RENDER_SCALE
    return Image.fromarray(array.astype(np.uint8))
