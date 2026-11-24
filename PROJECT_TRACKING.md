# v2 Enhancement Tracking

This file is the persistent handoff log for the portfolio-grade v2 upgrade.
Update it after each completed step and commit it with the related work.

## Project identity

- Repository: `https://github.com/DIEMOS192/material-stream-identification-system`
- Purpose: classify material from a live camera feed using HSV color-histogram
  and HOG features with a rejection-capable SVM model.
- History: this was a university team project completed before graduation; it
  was not a graduation/capstone project. The repository owner created the repo
  and wrote almost all of the code. The v2 README must credit teammates and
  describe v2 as extended and optimized after graduation, without implying the
  original was a graduation project.

## Git state and milestones

- Baseline tag: `v1.0-before-enhancement` at `c1bd7b69f8a43080f0cc4aa16f1347867d3aec0d`
- Active branch: `v2-refactor`
- First v2 commit: `158dda8 add v1 baseline benchmark`
- Do not rewrite history or work directly on `main`.
- The worktree already contained substantial user changes before v2 began:
  2,401 deleted tracked files under `data/`, nine modified files, and an
  untracked prediction workbook. These must not be staged, reverted, or
  committed accidentally. In particular, do not blindly commit the `data/`
  deletions.

## Completed work

### Audit (2026-11-24)

- Audited the source layout, training scripts, webcam application, saved model
  artifacts, requirements, documentation, and Git status.
- Found no test suite, CI workflow, project/package configuration, license, or
  useful README (the current README was empty).
- Confirmed duplicated feature/augmentation logic, tracked editor temporary
  files, `sys.path` manipulation, hard-coded paths/settings, and unpinned
  dependencies.
- Confirmed that `src/test.py` is an original TA final-assessment entry point,
  not arbitrary duplicate code. In v2, preserve its behavior/compatibility and
  extract shared internals rather than deleting it outright.

### Environment discovery

- No project-local virtual environment or Conda environment exists.
- Python 3.14 is usable and currently has these relevant packages:
  `opencv-python 4.11.0.86`, `scikit-image 0.25.2`, `numpy 2.3.5`,
  `scikit-learn 1.8.0`, and `joblib 1.5.2`.
- Python 3.12 lacks OpenCV; use `py -3.14` for current benchmark work.
- Do not create a new environment solely for the baseline. Later, pin the
  verified versions in project metadata for reproducibility.

### Baseline benchmark

- Added [benchmark script](benchmarks/benchmark.py), a fixed evaluation
  [manifest](benchmarks/fixtures/v1_eval_manifest.csv), saved
  [metrics](benchmarks/results/v1_baseline.json), and a per-image prediction
  [oracle](benchmarks/results/v1_baseline_predictions.csv).
- The local `dataset/` directory has the six expected classes and 5,690 JPEGs.
  It is a superset of the v1 raw training set: all 1,960 images tracked under
  `v1.0-before-enhancement:data/raw` are present there.
- The original TA test set (`data/testset-C`) was not committed. To prevent
  training-data leakage, the benchmark deterministically selects 100 images
  per class from the 3,730 images not present in v1 `data/raw`.
- Measurements describe the in-memory webcam-style pipeline; disk decoding is
  explicitly excluded from FPS and per-frame latency.

| Baseline metric | Value |
| --- | ---: |
| Frames | 600 (100/class) |
| Accuracy | 64.83% |
| Rejection rate | 20.00% |
| FPS | 96.14 |
| Mean total latency | 10.40 ms |
| p95 total latency | 15.42 ms |

- These figures are the only accepted v1-v2 comparison baseline. Historical
  report figures are not directly comparable because their TA evaluation split
  was never committed or documented.
- Re-run command: `py -3.14 benchmarks/benchmark.py --profile`
- The `--profile` switch writes an ignored `.prof` file. The versioned JSON and
  CSV files are the durable benchmark record.

### Measured profile findings

`cProfile` on the baseline run recorded 7.051 seconds across 600 frames plus a
warm-up. The largest cumulative costs in the frame pipeline were:

1. SVM `decision_function`: about 2.335 s.
2. SVM `predict` for non-rejected frames: about 1.811 s.
3. HOG extraction: about 0.945 s.
4. Image decoding: about 0.553 s; excluded from the reported in-memory timing.

Therefore, optimize and verify the duplicate SVM scoring/prediction path before
spending time on minor Python-loop cleanup. Every optimization must preserve
the per-image labels in the baseline oracle and re-run the benchmark.

## Current constraints and decisions

- Preserve behavior. Before and after every optimization, compare outputs with
  `benchmarks/results/v1_baseline_predictions.csv` and record a fresh
  benchmark. Never claim unmeasured speedups.
- Keep commits small and scoped (for example, `vectorize preprocessing`,
  `batch inference`, or `add benchmark script`).
- Do not add ONNX Runtime, TorchScript, quantization, threaded capture, or
  batching unless the measured profile and latency trade-offs justify it.
- Licensing is important before public release but does not block technical
  baseline work. Choose a license only after confirming the owner has authority
  to license all teammate contributions and the dataset terms permit it.
- Build the polished README after reproducible results exist; its benchmark
  table must use the saved same-machine v1/v2 runs.

## Next work, in order

1. Resolve repository/data consistency: decide the canonical documented dataset
   location, document its source/license, and deliberately handle the pre-v2
   `data/` deletions without absorbing unrelated user changes.
2. Add output-parity tests around preprocessing and the deployed SVM pipeline.
3. Implement only the highest-confidence measured inference optimization, then
   compare its output oracle and rerun the benchmark before committing.
4. Continue with package layout, configuration, pinned dependencies, training
   reproducibility, pytest/CI, and README/licensing as planned.

## Resume checklist

1. Run `git branch --show-current` and confirm `v2-refactor`.
2. Review `git status --short` before staging anything; the pre-existing dirty
   changes are intentional user state until explicitly resolved.
3. Run `py -3.14 benchmarks/benchmark.py --profile` before changing pipeline
   behavior and save the new output only if it replaces a deliberate baseline.
4. Read this file and the latest commit before starting the next task.
