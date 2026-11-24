"""Local integration test that protects v1 model-output parity."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import unittest

import cv2
import joblib
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preprocessing.feature_extraction import FeatureExtractor  # noqa: E402


MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks" / "fixtures" / "v1_eval_manifest.csv"
ORACLE_PATH = REPOSITORY_ROOT / "benchmarks" / "results" / "v1_baseline_predictions.csv"
MODEL_DIRECTORY = SOURCE_ROOT / "models" / "saved_models"


def v1_webcam_label(
    frame: np.ndarray,
    extractor: FeatureExtractor,
    scaler: object,
    normalizer: object,
    model: object,
    threshold: float,
    inverse_label_map: dict[int, str],
) -> str:
    """Mirror the v1 webcam application's single-frame SVM decision path."""

    processed = cv2.resize(frame, (64, 64))
    histogram = extractor.extract_color_histogram(processed)
    histogram = histogram / np.linalg.norm(histogram)
    hog_features = extractor.extract_hog(processed)
    hog_features = hog_features / np.linalg.norm(hog_features)
    feature_vector = np.concatenate((histogram, hog_features))
    feature_vector = feature_vector / np.linalg.norm(feature_vector)

    scaled_features = scaler.transform([feature_vector])
    normalized_features = normalizer.transform(scaled_features)
    distances = model.decision_function(normalized_features)
    max_confidence = float(np.max(distances))
    if max_confidence < threshold:
        return "unknown"
    return inverse_label_map[int(model.predict(normalized_features)[0])]


@unittest.skipUnless(
    MANIFEST_PATH.exists() and ORACLE_PATH.exists() and (REPOSITORY_ROOT / "dataset").is_dir(),
    "requires the local dataset plus versioned benchmark artifacts",
)
class V1PipelineParityTests(unittest.TestCase):
    """Verify that refactors retain every frozen v1 benchmark prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = FeatureExtractor()
        cls.scaler = joblib.load(MODEL_DIRECTORY / "svm_scaler.joblib")
        cls.normalizer = joblib.load(MODEL_DIRECTORY / "svm_normalizer.joblib")
        cls.model = joblib.load(MODEL_DIRECTORY / "svm_model.joblib")
        cls.threshold = float(joblib.load(MODEL_DIRECTORY / "svm_rejection_threshold.joblib"))
        label_map = joblib.load(MODEL_DIRECTORY / "svm_label_map.joblib")
        cls.inverse_label_map = {value: key for key, value in label_map.items()}
        with ORACLE_PATH.open(newline="", encoding="utf-8") as file:
            cls.oracle = list(csv.DictReader(file))

    def test_local_pipeline_matches_frozen_v1_predictions(self) -> None:
        self.assertEqual(len(self.oracle), 600)
        for expected in self.oracle:
            with self.subTest(path=expected["path"]):
                frame = cv2.imread(str(REPOSITORY_ROOT / expected["path"]))
                self.assertIsNotNone(frame)
                prediction = v1_webcam_label(
                    frame,
                    self.extractor,
                    self.scaler,
                    self.normalizer,
                    self.model,
                    self.threshold,
                    self.inverse_label_map,
                )
                self.assertEqual(prediction, expected["prediction"])


if __name__ == "__main__":
    unittest.main()
