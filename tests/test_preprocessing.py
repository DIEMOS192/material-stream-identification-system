"""Deterministic unit tests for the v1 feature representation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preprocessing.feature_extraction import FeatureExtractor  # noqa: E402


class FeatureExtractorTests(unittest.TestCase):
    """Protect the feature shape and deterministic behavior used by saved models."""

    def setUp(self) -> None:
        self.extractor = FeatureExtractor()
        rows, columns = np.indices((64, 64))
        self.frame = np.dstack(
            (
                (rows * 3 % 256).astype(np.uint8),
                (columns * 5 % 256).astype(np.uint8),
                ((rows + columns) * 7 % 256).astype(np.uint8),
            )
        )

    def test_color_histogram_has_expected_shape_and_unit_norm(self) -> None:
        histogram = self.extractor.extract_color_histogram(self.frame)

        self.assertEqual(histogram.shape, (14 * 12 * 12,))
        self.assertAlmostEqual(float(np.linalg.norm(histogram)), 1.0, places=6)

    def test_hog_has_expected_shape_and_is_deterministic(self) -> None:
        first = self.extractor.extract_hog(self.frame)
        second = self.extractor.extract_hog(self.frame)

        self.assertEqual(first.shape, (1764,))
        np.testing.assert_array_equal(first, second)

    def test_combined_feature_has_saved_model_dimension(self) -> None:
        histogram = self.extractor.extract_color_histogram(self.frame)
        hog_features = self.extractor.extract_hog(self.frame)
        combined = np.concatenate((histogram, hog_features))

        self.assertEqual(combined.shape, (3780,))


if __name__ == "__main__":
    unittest.main()
