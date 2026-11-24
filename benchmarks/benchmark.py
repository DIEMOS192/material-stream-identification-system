"""Reproducible baseline benchmark for the deployed SVM webcam pipeline.

The benchmark draws a deterministic, balanced held-out set from ``dataset/``.
It excludes every image tracked as ``data/raw`` at the v1 baseline tag, so the
reported accuracy is not evaluated on v1 training images.  Image decoding is
performed before timing; FPS and latency therefore describe in-memory frame
processing, matching webcam inference rather than disk throughput.

Run from the repository root:
    py -3.14 benchmarks/benchmark.py --profile
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import sklearn
import skimage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from preprocessing.feature_extraction import FeatureExtractor  # noqa: E402


BASELINE_TAG = "v1.0-before-enhancement"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
BASELINE_RESULTS_PATH = REPOSITORY_ROOT / "benchmarks" / "results" / "v1_baseline.json"


@dataclass(frozen=True)
class TimedPrediction:
    """A v1-model result together with per-frame stage timings."""

    label: str
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


def load_v1_training_paths() -> set[tuple[str, str]]:
    """Return ``(class, filename)`` pairs used by the tagged v1 training data."""

    command = ["git", "ls-tree", "-r", "--name-only", BASELINE_TAG, "data/raw"]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: set[tuple[str, str]] = set()
    for line in completed.stdout.splitlines():
        path = Path(line)
        if len(path.parts) == 4:
            paths.add((path.parts[2], path.name))
    return paths


def create_manifest(dataset_root: Path, samples_per_class: int, seed: int) -> list[dict[str, str]]:
    """Select a deterministic balanced evaluation subset outside v1 training data."""

    v1_training_paths = load_v1_training_paths()
    random_generator = random.Random(seed)
    manifest: list[dict[str, str]] = []

    for class_name in CLASSES:
        class_directory = dataset_root / class_name
        candidates = sorted(
            path
            for path in class_directory.glob("*.jpg")
            if (class_name, path.name) not in v1_training_paths
        )
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Only {len(candidates)} held-out images available for {class_name}; "
                f"need {samples_per_class}."
            )
        for path in random_generator.sample(candidates, samples_per_class):
            manifest.append(
                {"path": path.relative_to(REPOSITORY_ROOT).as_posix(), "label": class_name}
            )

    return sorted(manifest, key=lambda item: (item["label"], item["path"]))


def write_manifest(manifest: list[dict[str, str]], path: Path) -> None:
    """Persist the fixed evaluation subset in a reviewable CSV format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("path", "label"))
        writer.writeheader()
        writer.writerows(manifest)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate an existing evaluation manifest."""

    with path.open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    if not manifest or any(set(row) != {"path", "label"} for row in manifest):
        raise ValueError(f"Invalid benchmark manifest: {path}")
    return manifest


def load_model_artifacts() -> tuple[Any, Any, Any, float, dict[int, str]]:
    """Load the exact SVM artifacts used by the v1 webcam application."""

    models_directory = SOURCE_ROOT / "models" / "saved_models"
    scaler = joblib.load(models_directory / "svm_scaler.joblib")
    normalizer = joblib.load(models_directory / "svm_normalizer.joblib")
    model = joblib.load(models_directory / "svm_model.joblib")
    threshold = float(joblib.load(models_directory / "svm_rejection_threshold.joblib"))
    label_map = joblib.load(models_directory / "svm_label_map.joblib")
    return scaler, normalizer, model, threshold, {value: key for key, value in label_map.items()}


def predict_frame(
    frame: np.ndarray,
    extractor: FeatureExtractor,
    scaler: Any,
    normalizer: Any,
    model: Any,
    threshold: float,
    inverse_label_map: dict[int, str],
) -> TimedPrediction:
    """Run one frame through the v1 preprocessing and SVM rejection logic."""

    preprocess_start = time.perf_counter_ns()
    processed = cv2.resize(frame, (64, 64))
    histogram = extractor.extract_color_histogram(processed)
    histogram = histogram / np.linalg.norm(histogram)
    hog_features = extractor.extract_hog(processed)
    hog_features = hog_features / np.linalg.norm(hog_features)
    feature_vector = np.concatenate((histogram, hog_features))
    feature_vector = feature_vector / np.linalg.norm(feature_vector)
    preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000

    inference_start = time.perf_counter_ns()
    scaled_features = scaler.transform([feature_vector])
    normalized_features = normalizer.transform(scaled_features)
    distances = model.decision_function(normalized_features)
    max_confidence = float(np.max(distances))
    rejected = max_confidence < threshold
    predicted_id = -1 if rejected else int(model.predict(normalized_features)[0])
    inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000

    postprocess_start = time.perf_counter_ns()
    label = "unknown" if predicted_id == -1 else inverse_label_map[predicted_id]
    postprocess_ms = (time.perf_counter_ns() - postprocess_start) / 1_000_000
    return TimedPrediction(label, preprocess_ms, inference_ms, postprocess_ms)


def percentile(values: list[float], value: float) -> float:
    """Return a JSON-serializable percentile."""

    return float(np.percentile(values, value))


def run_benchmark(manifest: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Measure v1 frame processing over already-decoded held-out images."""

    frames: list[tuple[dict[str, str], np.ndarray]] = []
    for row in manifest:
        frame = cv2.imread(str(REPOSITORY_ROOT / row["path"]))
        if frame is None:
            raise ValueError(f"Could not read benchmark image: {row['path']}")
        frames.append((row, frame))

    scaler, normalizer, model, threshold, inverse_label_map = load_model_artifacts()
    extractor = FeatureExtractor()
    predict_frame(frames[0][1], extractor, scaler, normalizer, model, threshold, inverse_label_map)

    predictions: list[dict[str, str]] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    total_ms: list[float] = []

    for row, frame in frames:
        frame_start = time.perf_counter_ns()
        result = predict_frame(frame, extractor, scaler, normalizer, model, threshold, inverse_label_map)
        elapsed_ms = (time.perf_counter_ns() - frame_start) / 1_000_000
        predictions.append({**row, "prediction": result.label})
        preprocess_ms.append(result.preprocess_ms)
        inference_ms.append(result.inference_ms)
        postprocess_ms.append(result.postprocess_ms)
        total_ms.append(elapsed_ms)

    correct = sum(row["label"] == row["prediction"] for row in predictions)
    rejected = sum(row["prediction"] == "unknown" for row in predictions)
    metrics = {
        "baseline_tag": BASELINE_TAG,
        "benchmark_scope": "in-memory per-frame v1 SVM pipeline; image decoding excluded",
        "frames": len(predictions),
        "accuracy": correct / len(predictions),
        "rejection_rate": rejected / len(predictions),
        "fps": 1_000 / float(np.mean(total_ms)),
        "latency_ms": {
            "preprocess_mean": float(np.mean(preprocess_ms)),
            "inference_mean": float(np.mean(inference_ms)),
            "postprocess_mean": float(np.mean(postprocess_ms)),
            "total_mean": float(np.mean(total_ms)),
            "total_p50": percentile(total_ms, 50),
            "total_p95": percentile(total_ms, 95),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    return metrics, predictions


def write_results(metrics: dict[str, Any], predictions: list[dict[str, str]], output: Path) -> None:
    """Write metrics plus a prediction oracle for future parity tests."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions_path = output.with_name(f"{output.stem}_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("path", "label", "prediction"))
        writer.writeheader()
        writer.writerows(predictions)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPOSITORY_ROOT / "dataset")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "benchmarks" / "fixtures" / "v1_eval_manifest.csv")
    parser.add_argument("--output", type=Path, default=BASELINE_RESULTS_PATH)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--overwrite-baseline",
        action="store_true",
        help="Allow replacement of the committed v1 baseline metrics file.",
    )
    parser.add_argument("--profile", action="store_true", help="Save cProfile data next to the results JSON.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.output == BASELINE_RESULTS_PATH and arguments.output.exists() and not arguments.overwrite_baseline:
        raise SystemExit(
            "Refusing to overwrite the committed v1 baseline. "
            "Use --output benchmarks/results/<candidate>.json for a new run, "
            "or --overwrite-baseline to replace the baseline deliberately."
        )
    if arguments.rebuild_manifest or not arguments.manifest.exists():
        manifest = create_manifest(arguments.dataset, arguments.samples_per_class, arguments.seed)
        write_manifest(manifest, arguments.manifest)
    else:
        manifest = read_manifest(arguments.manifest)

    if arguments.profile:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        metrics, predictions = run_benchmark(manifest)
        profiler.disable()
        profiler.dump_stats(arguments.output.with_suffix(".prof"))
    else:
        metrics, predictions = run_benchmark(manifest)

    write_results(metrics, predictions, arguments.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
