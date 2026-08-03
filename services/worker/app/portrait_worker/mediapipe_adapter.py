from __future__ import annotations

from portrait_api.config import Settings
from portrait_api.services.model_validation import verify_required_models
from portrait_transfer import MediaPipePortraitAnalyzer
from portrait_transfer.exceptions import OptionalDependencyError
from portrait_transfer.preflight import HeuristicPortraitAnalyzer


def build_portrait_analyzer(
    settings: Settings,
) -> MediaPipePortraitAnalyzer | HeuristicPortraitAnalyzer:
    """Build the core production adapter, allowing a test-only explicit fallback."""

    validation = verify_required_models(settings)
    if validation.valid:
        face_model, segmentation_model = settings.required_model_paths
        return MediaPipePortraitAnalyzer(face_model, segmentation_model)
    if settings.allow_heuristic_analyzer and settings.app_env != "production":
        return HeuristicPortraitAnalyzer()
    raise OptionalDependencyError(
        "Required portrait analysis models are unavailable or invalid",
        **validation.public_details(),
    )


__all__ = ["build_portrait_analyzer"]
