"""Production MediaPipe portrait analysis with local, explicit model assets.

This module is intentionally lazy: importing :mod:`portrait_transfer` does not
import MediaPipe or OpenCV.  Constructing :class:`MediaPipePortraitAnalyzer`
requires two caller-provided local model files and never downloads weights.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Any, Protocol, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .alignment.anchors import (
    MEDIAPIPE_LEFT_EYE_CONTOUR,
    MEDIAPIPE_LEFT_IRIS,
    MEDIAPIPE_REQUIRED_IN_FRAME,
    MEDIAPIPE_RIGHT_EYE_CONTOUR,
    MEDIAPIPE_RIGHT_IRIS,
    mediapipe_to_reduced_68,
)
from .exceptions import (
    FaceDetectionError,
    InputValidationError,
    MaskFailure,
    OptionalDependencyError,
)
from .image_io import normalize_rgb
from .quality import analyze_quality
from .segmentation import refine_head_mask
from .types import BoundingBox, PortraitAnalysis, PortraitMasks, PoseEstimate

SELFIE_MULTICLASS_LABELS = (
    "background",
    "hair",
    "body_skin",
    "face_skin",
    "clothes",
    "others",
)

_AUTO_OPENCV = object()
_LABEL_ALIASES = {
    "background": "background",
    "hair": "hair",
    "bodyskin": "body_skin",
    "skin": "body_skin",
    "faceskin": "face_skin",
    "face": "face_skin",
    "clothes": "clothes",
    "clothing": "clothes",
    "other": "others",
    "others": "others",
    "accessories": "others",
    "otheraccessories": "others",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FaceLandmarker(Protocol):
    def detect(self, image: Any) -> Any: ...


class _ImageSegmenter(Protocol):
    @property
    def labels(self) -> Sequence[str]: ...

    def segment(self, image: Any) -> Any: ...


def _require_local_model(path: str | Path, purpose: str) -> Path:
    raw = str(path)
    if not raw.strip() or "://" in raw:
        raise InputValidationError(
            f"{purpose} must be an explicit local model file", model_path=raw
        )
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise InputValidationError(
            f"{purpose} model file does not exist", model_path=raw
        ) from exc
    if not resolved.is_file():
        raise InputValidationError(
            f"{purpose} model path is not a file", model_path=str(resolved)
        )
    return resolved


def _mesh_array(face_landmarks: Any) -> NDArray[np.float32]:
    if isinstance(face_landmarks, np.ndarray):
        points = np.asarray(face_landmarks, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] >= 2:
            points = points[:, :2]
    else:
        try:
            points = np.asarray(
                [(float(point.x), float(point.y)) for point in face_landmarks],
                dtype=np.float32,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise FaceDetectionError(
                "MediaPipe returned malformed face landmarks"
            ) from exc
    if points.shape != (478, 2):
        raise FaceDetectionError(
            "MediaPipe Face Landmarker must return exactly 478 landmarks",
            landmark_count=int(points.shape[0]) if points.ndim else 0,
        )
    if not np.isfinite(points).all():
        raise FaceDetectionError("MediaPipe returned non-finite face landmarks")
    return points


def _pixel_mesh(
    normalized_mesh: NDArray[np.float32],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> NDArray[np.float32]:
    height, width = image_shape[:2]
    pixels = np.empty_like(normalized_mesh)
    pixels[:, 0] = np.clip(normalized_mesh[:, 0], 0.0, 1.0) * max(width - 1, 1)
    pixels[:, 1] = np.clip(normalized_mesh[:, 1], 0.0, 1.0) * max(height - 1, 1)
    return pixels.astype(np.float32)


def _ensure_features_in_frame(normalized_mesh: NDArray[np.float32]) -> None:
    required = normalized_mesh[np.asarray(MEDIAPIPE_REQUIRED_IN_FRAME)]
    if np.any(required < 0.0) or np.any(required > 1.0):
        raise FaceDetectionError(
            "Both irises and all major facial features must be inside the frame"
        )


def face_bounding_box(
    normalized_mesh: ArrayLike,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> BoundingBox:
    """Return a clamped facial-mesh bounding box in image pixels."""

    mesh = np.asarray(normalized_mesh, dtype=np.float32)
    if mesh.shape != (478, 2) or not np.isfinite(mesh).all():
        raise ValueError("normalized_mesh must contain exactly 478 finite x/y points")
    pixels = _pixel_mesh(mesh, image_shape)[:468]
    x1, y1 = np.min(pixels, axis=0)
    x2, y2 = np.max(pixels, axis=0)
    box = BoundingBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    if box.width < 2.0 or box.height < 2.0:
        raise FaceDetectionError("Detected face has no usable spatial extent")
    return box.clamp(image_shape)


def pose_from_facial_transform(
    transformation_matrix: ArrayLike,
    normalized_mesh: ArrayLike,
) -> PoseEstimate:
    """Extract yaw/pitch from MediaPipe's transform and roll from the eye line."""

    matrix = np.asarray(transformation_matrix, dtype=np.float64)
    if matrix.size != 16 or not np.isfinite(matrix).all():
        raise FaceDetectionError(
            "MediaPipe returned an invalid facial transformation matrix"
        )
    matrix = matrix.reshape(4, 4)
    rotation = matrix[:3, :3]
    try:
        u, _, vt = np.linalg.svd(rotation)
    except np.linalg.LinAlgError as exc:
        raise FaceDetectionError(
            "MediaPipe facial transformation could not be decomposed"
        ) from exc
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = np.degrees(
        np.arctan2(
            -rotation[2, 0],
            np.hypot(rotation[2, 1], rotation[2, 2]),
        )
    )

    mesh = np.asarray(normalized_mesh, dtype=np.float64)
    if mesh.shape != (478, 2):
        raise ValueError("normalized_mesh must contain exactly 478 x/y points")
    right_eye = mesh[np.asarray(MEDIAPIPE_RIGHT_EYE_CONTOUR)].mean(axis=0)
    left_eye = mesh[np.asarray(MEDIAPIPE_LEFT_EYE_CONTOUR)].mean(axis=0)
    eye_line = left_eye - right_eye
    if np.linalg.norm(eye_line) < 1e-8:
        raise FaceDetectionError("MediaPipe eye landmarks have no spatial extent")
    roll = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
    return PoseEstimate(float(yaw), float(pitch), float(roll))


def _polygon_mask(
    shape: tuple[int, int],
    points: NDArray[np.float32],
    *,
    feather_sigma: float = 0.75,
) -> NDArray[np.float32]:
    height, width = shape
    canvas = Image.new("L", (width, height), 0)
    coordinates = [
        (
            float(np.clip(point[0], 0, width - 1)),
            float(np.clip(point[1], 0, height - 1)),
        )
        for point in points
    ]
    ImageDraw.Draw(canvas).polygon(coordinates, fill=255)
    mask = np.asarray(canvas, dtype=np.float32) / 255.0
    if feather_sigma > 0:
        mask = gaussian_filter(mask, sigma=feather_sigma, mode="nearest")
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _resize_confidence(
    confidence: ArrayLike, shape: tuple[int, int]
) -> NDArray[np.float32]:
    value = np.asarray(confidence, dtype=np.float32).squeeze()
    if value.ndim != 2 or not np.isfinite(value).all():
        raise MaskFailure("MediaPipe returned a malformed segmentation mask")
    if value.shape != shape:
        pil_mask = Image.fromarray(value, mode="F")
        value = np.asarray(
            pil_mask.resize((shape[1], shape[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _canonical_label(label: str) -> str | None:
    collapsed = "".join(character for character in label.lower() if character.isalnum())
    return _LABEL_ALIASES.get(collapsed)


def _segmentation_confidences(
    result: Any,
    labels: Sequence[str],
    shape: tuple[int, int],
) -> dict[str, NDArray[np.float32]]:
    raw_masks = tuple(getattr(result, "confidence_masks", ()) or ())
    if not raw_masks:
        raise MaskFailure(
            "MediaPipe Image Segmenter returned no confidence masks; "
            "use the multiclass selfie model"
        )
    if len(labels) != len(raw_masks):
        raise MaskFailure(
            "MediaPipe segmentation label count does not match its confidence masks",
            label_count=len(labels),
            mask_count=len(raw_masks),
        )

    confidences: dict[str, NDArray[np.float32]] = {}
    for label, raw_mask in zip(labels, raw_masks):
        canonical = _canonical_label(str(label))
        if canonical is None:
            continue
        if canonical in confidences:
            raise MaskFailure(
                "MediaPipe segmentation labels contain duplicate semantic classes",
                semantic_class=canonical,
            )
        data = raw_mask.numpy_view() if hasattr(raw_mask, "numpy_view") else raw_mask
        confidences[canonical] = _resize_confidence(data, shape)

    required = {"background", "hair", "face_skin"}
    missing = sorted(required.difference(confidences))
    if missing:
        raise MaskFailure(
            "The configured segmenter is not the required multiclass selfie model",
            missing_classes=missing,
        )
    return confidences


def _expanded_face_region(
    shape: tuple[int, int], face_box: BoundingBox
) -> NDArray[np.float32]:
    height, width = shape
    x1 = int(np.floor(face_box.x - 0.45 * face_box.width))
    x2 = int(np.ceil(face_box.x2 + 0.45 * face_box.width))
    y1 = int(np.floor(face_box.y - 0.65 * face_box.height))
    y2 = int(np.ceil(face_box.y2 + 0.30 * face_box.height))
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    region = np.zeros(shape, dtype=np.float32)
    region[y1:y2, x1:x2] = 1.0
    return cast(
        NDArray[np.float32],
        gaussian_filter(region, sigma=1.25, mode="nearest").astype(np.float32),
    )


def _upper_neck_region(
    shape: tuple[int, int], face_box: BoundingBox
) -> NDArray[np.float32]:
    height, width = shape
    x1 = int(np.floor(face_box.center[0] - 0.30 * face_box.width))
    x2 = int(np.ceil(face_box.center[0] + 0.30 * face_box.width))
    y1 = int(np.floor(face_box.y + 0.78 * face_box.height))
    y2 = int(np.ceil(face_box.y2 + 0.25 * face_box.height))
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    region = np.zeros(shape, dtype=np.float32)
    region[y1:y2, x1:x2] = 1.0
    return cast(
        NDArray[np.float32],
        gaussian_filter(region, sigma=1.0, mode="nearest").astype(np.float32),
    )


def refine_with_grabcut(
    rgb: ArrayLike,
    initial_head_confidence: ArrayLike,
    face_box: BoundingBox,
    *,
    cv2_module: Any | None,
) -> tuple[NDArray[np.float32], bool]:
    """Refine a neural head mask, falling back only within this mask stage."""

    image = normalize_rgb(rgb)
    confidence = np.clip(
        np.asarray(initial_head_confidence, dtype=np.float32), 0.0, 1.0
    )
    if confidence.shape != image.shape[:2] or not np.isfinite(confidence).all():
        raise ValueError("initial_head_confidence must match the RGB image")

    def fallback() -> NDArray[np.float32]:
        return refine_head_mask(
            confidence, face_box, threshold=0.28, feather_pixels=3.0
        )

    if cv2_module is None:
        return fallback(), False

    cv2 = cv2_module
    cv2_error: type[BaseException] = getattr(cv2, "error", RuntimeError)
    try:
        confidence_shape = (confidence.shape[0], confidence.shape[1])
        region = _expanded_face_region(confidence_shape, face_box) > 0.01
        grabcut_mask = np.full(confidence.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[region] = cv2.GC_PR_BGD
        grabcut_mask[confidence >= 0.16] = cv2.GC_PR_FGD
        grabcut_mask[confidence >= 0.68] = cv2.GC_FGD

        # A confidently detected facial interior is a valid segmentation seed;
        # it is not a substitute face detector.
        yy, xx = np.meshgrid(
            np.arange(confidence.shape[0]),
            np.arange(confidence.shape[1]),
            indexing="ij",
        )
        inner_face = (
            (xx - face_box.center[0]) / max(0.34 * face_box.width, 1.0)
        ) ** 2 + (
            (yy - face_box.center[1]) / max(0.42 * face_box.height, 1.0)
        ) ** 2 <= 1.0
        grabcut_mask[inner_face] = cv2.GC_FGD

        if not np.any(grabcut_mask == cv2.GC_BGD):
            grabcut_mask[[0, -1], :] = cv2.GC_BGD
            grabcut_mask[:, [0, -1]] = cv2.GC_BGD
        if not np.any(grabcut_mask == cv2.GC_FGD):
            return fallback(), False

        bgr = np.ascontiguousarray(np.rint(image[..., ::-1] * 255.0).astype(np.uint8))
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            bgr,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        foreground = np.logical_or(
            grabcut_mask == cv2.GC_FGD, grabcut_mask == cv2.GC_PR_FGD
        ).astype(np.float32)
        if float(np.mean(foreground)) < 1e-4:
            return fallback(), False
        combined = 0.78 * foreground + 0.22 * confidence
        refined = refine_head_mask(
            combined, face_box, threshold=0.32, feather_pixels=3.0
        )
        return refined, True
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        cv2_error,
    ):
        return fallback(), False


def _mask_confidence(
    confidences: dict[str, NDArray[np.float32]], head: NDArray[np.float32]
) -> float:
    stacked = np.stack(tuple(confidences.values()), axis=0)
    certainty = np.max(stacked, axis=0)
    selected = head > 0.10
    return float(np.mean(certainty[selected])) if np.any(selected) else 0.0


def _crop_truncation(
    normalized_mesh: NDArray[np.float32], head: NDArray[np.float32]
) -> float:
    near_edge = np.logical_or(
        np.any(normalized_mesh[:468] < 0.01, axis=1),
        np.any(normalized_mesh[:468] > 0.99, axis=1),
    )
    landmark_fraction = float(np.mean(near_edge))
    border_mass = float(
        head[0, :].sum() + head[-1, :].sum() + head[:, 0].sum() + head[:, -1].sum()
    )
    border_fraction = border_mass / max(float(head.sum()), 1e-6)
    return float(np.clip(max(landmark_fraction, 4.0 * border_fraction), 0.0, 1.0))


def _landmark_occlusion(face_landmarks: Any) -> float:
    values: list[float] = []
    for point in face_landmarks:
        candidates = [getattr(point, name, None) for name in ("visibility", "presence")]
        finite = [
            float(value)
            for value in candidates
            if value is not None and np.isfinite(value) and float(value) > 0.0
        ]
        if finite:
            values.append(min(finite))
    if not values:
        return 0.0
    return float(np.clip(1.0 - np.mean(values), 0.0, 1.0))


class MediaPipePortraitAnalyzer:
    """Face Landmarker + multiclass Image Segmenter production adapter.

    Use the constructor in production.  It validates both model files before
    importing MediaPipe.  ``from_components`` exists for model-free unit tests
    and for wrappers around already-created MediaPipe task objects.
    """

    _labels: tuple[str, ...] | None

    def __init__(
        self,
        face_landmarker_model_path: str | Path,
        segmenter_model_path: str | Path,
        *,
        min_face_detection_confidence: float = 0.60,
        min_face_presence_confidence: float = 0.60,
        segmentation_labels: Sequence[str] | None = None,
    ) -> None:
        face_model = _require_local_model(
            face_landmarker_model_path, "MediaPipe Face Landmarker"
        )
        segmentation_model = _require_local_model(
            segmenter_model_path, "MediaPipe Image Segmenter"
        )
        for name, value in (
            ("min_face_detection_confidence", min_face_detection_confidence),
            ("min_face_presence_confidence", min_face_presence_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise InputValidationError(f"{name} must be between zero and one")
        face_model_digest = _file_sha256(face_model)
        segmentation_model_digest = _file_sha256(segmentation_model)

        try:
            mp = import_module("mediapipe")
        except (ImportError, OSError) as exc:
            raise OptionalDependencyError(
                "MediaPipe is required for production portrait analysis; "
                "install portrait-transfer[vision]"
            ) from exc

        face_landmarker: Any | None = None
        try:
            face_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                # A limit of two lets us reject multi-face images instead of
                # silently accepting the first face.
                num_faces=2,
                min_face_detection_confidence=min_face_detection_confidence,
                min_face_presence_confidence=min_face_presence_confidence,
                min_tracking_confidence=min_face_presence_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                face_options
            )
            segmentation_options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(segmentation_model)
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
                segmentation_options
            )
        except Exception as exc:
            if face_landmarker is not None:
                with suppress(Exception):
                    face_landmarker.close()
            raise OptionalDependencyError(
                "MediaPipe could not initialize the supplied local vision models",
                error_type=type(exc).__name__,
            ) from exc

        self._face_landmarker = face_landmarker
        self._segmenter = segmenter
        self._labels = tuple(
            segmentation_labels
            if segmentation_labels is not None
            else tuple(getattr(segmenter, "labels", ()) or ())
        )
        self.cache_identity = (
            "mediapipe-portrait-v1:"
            f"face={face_model_digest}:"
            f"segmenter={segmentation_model_digest}:"
            f"detection={min_face_detection_confidence:.8g}:"
            f"presence={min_face_presence_confidence:.8g}:"
            f"labels={','.join(str(label) for label in self._labels)}"
        )
        self._image_factory: Callable[[NDArray[np.uint8]], Any] = lambda data: mp.Image(
            image_format=mp.ImageFormat.SRGB, data=data
        )
        self._cv2_module: Any = _AUTO_OPENCV
        self._owns_components = True
        self._lock = Lock()

    @classmethod
    def from_components(
        cls,
        face_landmarker: _FaceLandmarker,
        segmenter: _ImageSegmenter,
        *,
        segmentation_labels: Sequence[str] | None = None,
        image_factory: Callable[[NDArray[np.uint8]], Any] | None = None,
        cv2_module: Any | None = None,
    ) -> Self:
        """Build around injected runners without importing optional packages."""

        instance = cls.__new__(cls)
        instance._face_landmarker = face_landmarker
        instance._segmenter = segmenter
        instance._labels = tuple(
            segmentation_labels
            if segmentation_labels is not None
            else tuple(getattr(segmenter, "labels", ()) or ())
        )
        effective_labels = instance._labels
        face_identity = getattr(
            face_landmarker,
            "cache_identity",
            f"{type(face_landmarker).__module__}.{type(face_landmarker).__qualname__}",
        )
        segmenter_identity = getattr(
            segmenter,
            "cache_identity",
            f"{type(segmenter).__module__}.{type(segmenter).__qualname__}",
        )
        instance.cache_identity = (
            "mediapipe-injected-v1:"
            f"face={face_identity}:segmenter={segmenter_identity}:"
            f"labels={','.join(str(label) for label in effective_labels)}"
        )
        instance._image_factory = image_factory or (lambda data: data)
        instance._cv2_module = cv2_module
        instance._owns_components = False
        instance._lock = Lock()
        return instance

    def _opencv(self) -> Any | None:
        if self._cv2_module is _AUTO_OPENCV:
            try:
                self._cv2_module = import_module("cv2")
            except (ImportError, OSError):
                self._cv2_module = None
        return self._cv2_module

    def close(self) -> None:
        """Release native MediaPipe task resources owned by this analyzer."""

        if not self._owns_components:
            return
        for component in (self._segmenter, self._face_landmarker):
            with suppress(Exception):
                component.close()
        self._owns_components = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def analyze(self, rgb: NDArray[np.float32]) -> PortraitAnalysis:
        image = normalize_rgb(rgb)
        image_shape = (image.shape[0], image.shape[1])
        uint8_rgb = np.ascontiguousarray(np.rint(image * 255.0).astype(np.uint8))
        media_image = self._image_factory(uint8_rgb)
        try:
            with self._lock:
                face_result = self._face_landmarker.detect(media_image)
        except Exception as exc:
            raise FaceDetectionError(
                "MediaPipe Face Landmarker inference failed",
                error_type=type(exc).__name__,
            ) from exc

        face_sets = tuple(getattr(face_result, "face_landmarks", ()) or ())
        if len(face_sets) == 0:
            raise FaceDetectionError("MediaPipe detected no face")
        if len(face_sets) != 1:
            raise FaceDetectionError(
                "Exactly one face is required", detected_faces=len(face_sets)
            )
        face_landmarks = face_sets[0]
        normalized_mesh = _mesh_array(face_landmarks)
        _ensure_features_in_frame(normalized_mesh)

        transformations = tuple(
            getattr(face_result, "facial_transformation_matrixes", ()) or ()
        )
        if len(transformations) != 1:
            raise FaceDetectionError(
                "MediaPipe did not return one facial transformation matrix",
                transformation_count=len(transformations),
            )

        face_box = face_bounding_box(normalized_mesh, image_shape)
        reduced = mediapipe_to_reduced_68(normalized_mesh, image_shape)
        full_pixels = _pixel_mesh(normalized_mesh, image_shape)
        pose = pose_from_facial_transform(transformations[0], normalized_mesh)

        try:
            with self._lock:
                segmentation_result = self._segmenter.segment(media_image)
        except Exception as exc:
            raise MaskFailure(
                "MediaPipe Image Segmenter inference failed",
                error_type=type(exc).__name__,
            ) from exc

        labels = self._labels
        if labels is None:
            labels = tuple(getattr(self._segmenter, "labels", ()) or ())
        confidences = _segmentation_confidences(
            segmentation_result, labels, image_shape
        )

        left_eye = _polygon_mask(
            image_shape,
            full_pixels[np.asarray(MEDIAPIPE_LEFT_EYE_CONTOUR)],
        )
        right_eye = _polygon_mask(
            image_shape,
            full_pixels[np.asarray(MEDIAPIPE_RIGHT_EYE_CONTOUR)],
        )
        left_iris = _polygon_mask(
            image_shape,
            full_pixels[np.asarray(MEDIAPIPE_LEFT_IRIS)],
            feather_sigma=0.55,
        )
        right_iris = _polygon_mask(
            image_shape,
            full_pixels[np.asarray(MEDIAPIPE_RIGHT_IRIS)],
            feather_sigma=0.55,
        )

        zeros = np.zeros(image_shape, dtype=np.float32)
        body_skin = confidences.get("body_skin", zeros)
        head_seed = np.maximum(confidences["hair"], confidences["face_skin"])
        head_seed = np.maximum(
            head_seed,
            cast(
                NDArray[np.float32],
                body_skin * _upper_neck_region(image_shape, face_box),
            ),
        )
        head_seed = np.maximum.reduce(
            (head_seed, left_eye, right_eye, left_iris, right_iris)
        )
        head_seed *= _expanded_face_region(image_shape, face_box)
        head, used_grabcut = refine_with_grabcut(
            image, head_seed, face_box, cv2_module=self._opencv()
        )
        head = np.maximum.reduce((head, left_eye, right_eye, left_iris, right_iris))
        head = np.clip(head, 0.0, 1.0).astype(np.float32)

        person_confidence = 1.0 - confidences["background"]
        person = refine_head_mask(
            person_confidence,
            face_box,
            threshold=0.25,
            feather_pixels=4.0,
        )
        person = np.maximum(person, head).astype(np.float32)
        face_skin = np.minimum(
            gaussian_filter(confidences["face_skin"], sigma=0.65, mode="nearest"),
            head,
        ).astype(np.float32)
        hair = np.minimum(
            gaussian_filter(confidences["hair"], sigma=0.65, mode="nearest"),
            head,
        ).astype(np.float32)

        masks = PortraitMasks(
            person=person,
            head=head,
            face_skin=face_skin,
            hair=hair,
            eyes=(left_eye, right_eye),
            irises=(left_iris, right_iris),
            effective_transfer=head.copy(),
            foreground_alpha=person.copy(),
        )
        mask_confidence = _mask_confidence(confidences, head)
        accessories = confidences.get("others", zeros)
        head_selection = head > 0.25
        accessory_occlusion = (
            float(np.mean(accessories[head_selection]))
            if np.any(head_selection)
            else 0.0
        )
        occlusion = max(_landmark_occlusion(face_landmarks), accessory_occlusion)
        truncation = _crop_truncation(normalized_mesh, head)
        quality = analyze_quality(
            image,
            reduced,
            face_box,
            head,
            mask_confidence=mask_confidence,
            crop_truncation=truncation,
            occlusion_proxy=occlusion,
        )
        warnings: list[str] = []
        if not used_grabcut:
            warnings.append("grabcut_refinement_fallback")
        if truncation > 0.10:
            warnings.append("head_near_frame_edge")
        return PortraitAnalysis(
            landmarks=reduced,
            face_box=face_box,
            pose=pose,
            quality=quality,
            masks=masks,
            warnings=tuple(warnings),
            full_landmarks=full_pixels,
        )
