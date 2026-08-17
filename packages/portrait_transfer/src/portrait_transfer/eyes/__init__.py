"""Automatic, confidence-gated eye highlight processing."""

from .extraction import extract_highlight_asset
from .highlight_transfer import transfer_eye_highlights
from .inpainting import remove_existing_highlights

__all__ = [
    "extract_highlight_asset",
    "remove_existing_highlights",
    "transfer_eye_highlights",
]
