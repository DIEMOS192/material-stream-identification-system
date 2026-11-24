"""Runtime inference components for material classification."""

from .pipeline import MaterialPrediction, SVMFrameClassifier

__all__ = ["MaterialPrediction", "SVMFrameClassifier"]
