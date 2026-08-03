from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from portrait_transfer import (
    MediaPipePortraitAnalyzer,
    create_default_runtime,
    mediapipe_vision,
)
from portrait_transfer.alignment.anchors import (
    MEDIAPIPE_LEFT_EYE_CONTOUR,
    MEDIAPIPE_LEFT_IRIS,
    MEDIAPIPE_RIGHT_EYE_CONTOUR,
    MEDIAPIPE_RIGHT_IRIS,
    MEDIAPIPE_TO_REDUCED_68,
    mediapipe_to_reduced_68,
)
from portrait_transfer.exceptions import (
    FaceDetectionError,
    InputValidationError,
    MaskFailure,
    OptionalDependencyError,
)
from portrait_transfer.mediapipe_vision import (
    SELFIE_MULTICLASS_LABELS,
    pose_from_facial_transform,
    refine_with_grabcut,
)
from portrait_transfer.types import BoundingBox


class _Mask:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def numpy_view(self) -> np.ndarray:
        return self.value


class _FaceRunner:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self.faces = faces
        self.calls = 0

    def detect(self, image: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        matrices = [np.eye(4, dtype=np.float32) for _ in self.faces]
        return SimpleNamespace(
            face_landmarks=self.faces,
            facial_transformation_matrixes=matrices,
        )


class _Segmenter:
    labels = SELFIE_MULTICLASS_LABELS

    def __init__(self, masks: tuple[np.ndarray, ...]) -> None:
        self.masks = masks
        self.calls = 0

    def segment(self, image: np.ndarray) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(confidence_masks=[_Mask(mask) for mask in self.masks])


def _mesh() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 478, endpoint=False)
    mesh = np.column_stack(
        (0.5 + 0.25 * np.cos(angles), 0.48 + 0.34 * np.sin(angles))
    ).astype(np.float32)

    def set_ellipse(indices: tuple[int, ...], center: tuple[float, float], radius):
        local_angles = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
        for index, angle in zip(indices, local_angles):
            mesh[index] = (
                center[0] + radius[0] * np.cos(angle),
                center[1] + radius[1] * np.sin(angle),
            )

    set_ellipse(MEDIAPIPE_RIGHT_EYE_CONTOUR, (0.39, 0.43), (0.055, 0.025))
    set_ellipse(MEDIAPIPE_LEFT_EYE_CONTOUR, (0.61, 0.43), (0.055, 0.025))
    set_ellipse(MEDIAPIPE_RIGHT_IRIS, (0.39, 0.43), (0.014, 0.014))
    set_ellipse(MEDIAPIPE_LEFT_IRIS, (0.61, 0.43), (0.014, 0.014))
    return mesh


def _face_points(mesh: np.ndarray) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=float(point[0]),
            y=float(point[1]),
            visibility=0.98,
            presence=0.97,
        )
        for point in mesh
    ]


def _segmentation_masks(shape: tuple[int, int] = (32, 24)) -> tuple[np.ndarray, ...]:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(0.0, 1.0, shape[1]),
        indexing="ij",
    )
    face = np.exp(-(((xx - 0.5) / 0.22) ** 2 + ((yy - 0.50) / 0.30) ** 2))
    hair = np.exp(-(((xx - 0.5) / 0.30) ** 2 + ((yy - 0.28) / 0.20) ** 2))
    body_skin = 0.5 * np.exp(-(((xx - 0.5) / 0.20) ** 2 + ((yy - 0.78) / 0.18) ** 2))
    clothes = 0.8 * (yy > 0.76) * np.exp(-(((xx - 0.5) / 0.45) ** 2))
    others = np.zeros(shape, dtype=np.float32)
    foreground = np.clip(np.maximum.reduce((face, hair, body_skin, clothes)), 0, 1)
    background = 1.0 - foreground
    return tuple(
        np.asarray(mask, dtype=np.float32)
        for mask in (background, hair, body_skin, face, clothes, others)
    )


def test_mediapipe_mapping_has_stable_68_topology() -> None:
    mesh = _mesh()
    mapped = mediapipe_to_reduced_68(mesh, (101, 201, 3))
    assert len(MEDIAPIPE_TO_REDUCED_68) == 68
    assert mapped.shape == (68, 2)
    expected = mesh[np.asarray(MEDIAPIPE_TO_REDUCED_68)] * np.asarray((200, 100))
    assert np.allclose(mapped, expected)


def test_mapping_rejects_non_478_mesh() -> None:
    with pytest.raises(ValueError, match="exactly 478"):
        mediapipe_to_reduced_68(np.zeros((468, 2)), (100, 100))


def test_pose_uses_transform_for_yaw_and_eye_line_for_roll() -> None:
    angle = np.radians(15.0)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (np.cos(angle), 0.0, np.sin(angle)),
        (0.0, 1.0, 0.0),
        (-np.sin(angle), 0.0, np.cos(angle)),
    )
    mesh = _mesh()
    mesh[np.asarray(MEDIAPIPE_LEFT_EYE_CONTOUR), 1] += 0.04
    pose = pose_from_facial_transform(matrix, mesh)
    assert pose.yaw == pytest.approx(15.0, abs=0.01)
    assert pose.pitch == pytest.approx(0.0, abs=0.01)
    assert pose.roll > 5.0


def test_analyzer_builds_real_masks_from_injected_task_results() -> None:
    face_runner = _FaceRunner([_face_points(_mesh())])
    segmenter = _Segmenter(_segmentation_masks())
    analyzer = MediaPipePortraitAnalyzer.from_components(
        face_runner,
        segmenter,
        cv2_module=None,
    )
    yy, xx = np.meshgrid(np.arange(96), np.arange(80), indexing="ij")
    image = np.stack(
        ((xx % 17) / 16.0, (yy % 19) / 18.0, ((xx + yy) % 23) / 22.0),
        axis=-1,
    ).astype(np.float32)

    analysis = analyzer.analyze(image)

    assert face_runner.calls == segmenter.calls == 1
    assert analysis.landmarks.shape == (68, 2)
    assert analysis.full_landmarks is not None
    assert analysis.full_landmarks.shape == (478, 2)
    assert analysis.face_box.height > image.shape[0] * 0.5
    assert analysis.masks.head.shape == image.shape[:2]
    assert analysis.masks.person.mean() > analysis.masks.head.mean()
    assert analysis.masks.hair.max() > 0.5
    assert analysis.masks.face_skin.max() > 0.5
    assert all(mask.max() > 0.1 for mask in analysis.masks.eyes)
    assert all(mask.max() > 0.1 for mask in analysis.masks.irises)
    assert analysis.quality.mask_confidence > 0.5
    assert "grabcut_refinement_fallback" in analysis.warnings


@pytest.mark.parametrize("face_count", [0, 2])
def test_analyzer_requires_exactly_one_face(face_count: int) -> None:
    faces = [_face_points(_mesh()) for _ in range(face_count)]
    segmenter = _Segmenter(_segmentation_masks())
    analyzer = MediaPipePortraitAnalyzer.from_components(
        _FaceRunner(faces),
        segmenter,
        cv2_module=None,
    )
    expected = "no face" if face_count == 0 else "Exactly one face"
    with pytest.raises(FaceDetectionError, match=expected):
        analyzer.analyze(np.zeros((64, 64, 3), dtype=np.float32))
    assert segmenter.calls == 0


def test_wrong_segmenter_labels_fail_instead_of_fabricating_masks() -> None:
    segmenter = _Segmenter(_segmentation_masks())
    analyzer = MediaPipePortraitAnalyzer.from_components(
        _FaceRunner([_face_points(_mesh())]),
        segmenter,
        segmentation_labels=("background", "person", "a", "b", "c", "d"),
        cv2_module=None,
    )
    with pytest.raises(MaskFailure, match="multiclass selfie model"):
        analyzer.analyze(np.zeros((64, 64, 3), dtype=np.float32))


def test_missing_or_remote_models_fail_before_optional_import(tmp_path) -> None:
    missing = tmp_path / "face.task"
    with pytest.raises(InputValidationError, match="does not exist"):
        MediaPipePortraitAnalyzer(missing, tmp_path / "segmenter.tflite")
    with pytest.raises(InputValidationError, match="explicit local"):
        MediaPipePortraitAnalyzer(
            "https://example.invalid/face.task",
            "https://example.invalid/segmenter.tflite",
        )


def test_missing_mediapipe_dependency_has_an_actionable_error(
    tmp_path, monkeypatch
) -> None:
    face_model = tmp_path / "face.task"
    segmenter_model = tmp_path / "segmenter.tflite"
    face_model.write_bytes(b"local face model fixture")
    segmenter_model.write_bytes(b"local segmenter fixture")

    def missing_dependency(name: str):
        assert name == "mediapipe"
        raise ImportError("model-free test")

    monkeypatch.setattr(mediapipe_vision, "import_module", missing_dependency)
    with pytest.raises(OptionalDependencyError, match="install portrait-transfer"):
        MediaPipePortraitAnalyzer(face_model, segmenter_model)


def test_grabcut_path_is_used_when_available() -> None:
    class FakeCv2:
        GC_BGD = 0
        GC_FGD = 1
        GC_PR_BGD = 2
        GC_PR_FGD = 3
        GC_INIT_WITH_MASK = 1

        def __init__(self) -> None:
            self.called = False

        def grabCut(self, image, mask, rect, bg, fg, iterations, mode) -> None:
            self.called = True
            assert image.dtype == np.uint8
            assert rect is None
            assert iterations == 5
            assert mode == self.GC_INIT_WITH_MASK

    cv2 = FakeCv2()
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    confidence = (((xx - 32.0) / 16.0) ** 2 + ((yy - 30.0) / 22.0) ** 2 <= 1.0).astype(
        np.float32
    )
    image = np.repeat(confidence[..., None], 3, axis=2)
    refined, used = refine_with_grabcut(
        image,
        confidence,
        BoundingBox(18.0, 12.0, 28.0, 38.0),
        cv2_module=cv2,
    )
    assert cv2.called
    assert used
    assert refined.shape == confidence.shape
    assert refined.max() > 0.9


def test_default_runtime_accepts_an_injected_analyzer() -> None:
    analyzer = object()
    runtime = create_default_runtime(enable_cpu_dense=False, analyzer=analyzer)
    assert runtime.analyzer is analyzer


def test_injected_analyzer_cache_identity_tracks_component_config() -> None:
    first = MediaPipePortraitAnalyzer.from_components(
        _FaceRunner([_face_points(_mesh())]),
        _Segmenter(_segmentation_masks()),
        cv2_module=None,
    )
    second = MediaPipePortraitAnalyzer.from_components(
        _FaceRunner([_face_points(_mesh())]),
        _Segmenter(_segmentation_masks()),
        cv2_module=None,
    )
    changed = MediaPipePortraitAnalyzer.from_components(
        _FaceRunner([_face_points(_mesh())]),
        _Segmenter(_segmentation_masks()),
        segmentation_labels=("background", "hair", "skin"),
        cv2_module=None,
    )
    assert first.cache_identity == second.cache_identity
    assert changed.cache_identity != first.cache_identity
