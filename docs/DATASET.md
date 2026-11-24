# Dataset contract

The image dataset is intentionally local-only and is not committed to Git.
Place it at `dataset/` in the repository root with this layout:

```text
dataset/
├── cardboard/*.jpg
├── glass/*.jpg
├── metal/*.jpg
├── paper/*.jpg
├── plastic/*.jpg
└── trash/*.jpg
```

The v2 baseline expects the six directory names above as the class labels.
`dataset/` is ignored by Git so that image data is not accidentally committed.

## Benchmark split

The versioned benchmark manifest is
`benchmarks/fixtures/v1_eval_manifest.csv`. It contains a deterministic,
balanced 600-image evaluation subset: 100 images for each class. The benchmark
creates it by excluding every path present in
`v1.0-before-enhancement:data/raw`, then sampling with seed `42` from the
remaining local images. This prevents the v1 saved model from being evaluated
on its recorded raw training images.

Run the baseline benchmark with:

```powershell
py -3.14 benchmarks/benchmark.py --profile
```

The benchmark validates that all manifest paths are readable before measuring
the in-memory frame pipeline. It does not measure disk decoding as webcam
latency.

## Provenance and licensing status

The original repository did not record a dataset source, download URL, or
license, and the image files contain no source metadata. Do not publish,
redistribute, or state a dataset license until the original source and its
terms have been confirmed by the repository owner.

Once confirmed, replace this section with the dataset name, source URL,
license, version or download date, and any required attribution. If the data
cannot be redistributed, document repeatable download instructions instead of
adding it to Git.

## v1 data history

The `v1.0-before-enhancement` tag preserves the original tracked `data/raw`
and `data/augmented` inputs for historical reproducibility. The active v2
worktree has pre-existing deletions in that old `data/` tree; they are not part
of the v2 refactor unless explicitly staged in a future data-cleanup commit.
