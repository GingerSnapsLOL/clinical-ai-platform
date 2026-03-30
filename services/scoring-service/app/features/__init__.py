"""Shared clinical feature extraction (single pipeline for all targets)."""

from app.features.builder import ExtractedFeatures, extract_features

__all__ = ["ExtractedFeatures", "extract_features"]
