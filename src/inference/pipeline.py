"""Single-frame inference pipeline for the deployed v1 SVM model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from preprocessing.feature_extraction import FeatureExtractor


@dataclass(frozen=True)
class MaterialPrediction:
    """The classification result for one BGR camera frame."""

    label: str | None
    confidence: float
    rejected: bool


class SVMFrameClassifier:
    """Load v1 SVM artifacts once and classify individual camera frames."""

    def __init__(
        self,
        scaler: Any,
        normalizer: Any,
        model: Any,
        rejection_threshold: float,
        inverse_label_map: dict[int, str],
    ) -> None:
        self._scaler = scaler
        self._normalizer = normalizer
        self._model = model
        self._rejection_threshold = rejection_threshold
        self._inverse_label_map = inverse_label_map
        self._feature_extractor = FeatureExtractor()

    @classmethod
    def from_model_directory(cls, model_directory: Path) -> "SVMFrameClassifier":
        """Create a classifier from the v1 serialized SVM artifacts."""

        scaler = joblib.load(model_directory / "svm_scaler.joblib")
        normalizer = joblib.load(model_directory / "svm_normalizer.joblib")
        model = joblib.load(model_directory / "svm_model.joblib")
        rejection_threshold = float(joblib.load(model_directory / "svm_rejection_threshold.joblib"))
        label_map = joblib.load(model_directory / "svm_label_map.joblib")
        inverse_label_map = {value: key for key, value in label_map.items()}
        return cls(scaler, normalizer, model, rejection_threshold, inverse_label_map)

    def predict(self, frame: np.ndarray) -> MaterialPrediction:
        """Classify one BGR frame using the v1 feature and rejection semantics."""

        processed = cv2.resize(frame, (64, 64))
        histogram = self._feature_extractor.extract_color_histogram(processed)
        hog_features = self._feature_extractor.extract_hog(processed)

        # Fused normalization matching original: norm(concat(norm(h), norm(hog)))
        # Original: h_norm = h/||h||, hog_norm = hog/||hog||, then concat / ||concat||
        # Since ||h_norm||=1 and ||hog_norm||=1, ||concat(h_norm, hog_norm)|| = sqrt(2)
        hist_norm = np.linalg.norm(histogram)
        hog_norm = np.linalg.norm(hog_features)
        if hist_norm == 0 or hog_norm == 0:
            # Fallback to original behavior for edge cases
            histogram = histogram / (hist_norm + 1e-12)
            hog_features = hog_features / (hog_norm + 1e-12)
        else:
            histogram = histogram / hist_norm
            hog_features = hog_features / hog_norm
        # Original: feature_vector = concat(h_norm, hog_norm) / sqrt(2)
        feature_vector = np.concatenate((histogram, hog_features)) / np.sqrt(2.0)

        scaled_features = self._scaler.transform([feature_vector])
        normalized_features = self._normalizer.transform(scaled_features)
        distances = self._model.decision_function(normalized_features)
        confidence = float(np.max(distances))
        rejected = confidence < self._rejection_threshold
        if rejected:
            return MaterialPrediction(label=None, confidence=confidence, rejected=True)

        predicted_id = int(self._model.predict(normalized_features)[0])
        return MaterialPrediction(
            label=self._inverse_label_map[predicted_id],
            confidence=confidence,
            rejected=False,
        )
