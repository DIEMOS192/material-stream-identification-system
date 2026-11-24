"""Local integration test that protects v1 model-output parity."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import unittest

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from inference import SVMFrameClassifier  # noqa: E402


MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks" / "fixtures" / "v1_eval_manifest.csv"
ORACLE_PATH = REPOSITORY_ROOT / "benchmarks" / "results" / "v1_baseline_predictions.csv"
@unittest.skipUnless(
    MANIFEST_PATH.exists() and ORACLE_PATH.exists() and (REPOSITORY_ROOT / "dataset").is_dir(),
    "requires the local dataset plus versioned benchmark artifacts",
)
class V1PipelineParityTests(unittest.TestCase):
    """Verify that refactors retain every frozen v1 benchmark prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = SVMFrameClassifier.from_model_directory(
            SOURCE_ROOT / "models" / "saved_models"
        )
        with ORACLE_PATH.open(newline="", encoding="utf-8") as file:
            cls.oracle = list(csv.DictReader(file))

    def test_local_pipeline_matches_frozen_v1_predictions(self) -> None:
        self.assertEqual(len(self.oracle), 600)
        for expected in self.oracle:
            with self.subTest(path=expected["path"]):
                frame = cv2.imread(str(REPOSITORY_ROOT / expected["path"]))
                self.assertIsNotNone(frame)
                result = self.classifier.predict(frame)
                prediction = "unknown" if result.rejected else result.label
                self.assertEqual(prediction, expected["prediction"])


if __name__ == "__main__":
    unittest.main()
