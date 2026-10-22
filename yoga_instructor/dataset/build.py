"""Build leakage-aware model samples and deterministic train/validation/test splits."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from yoga_instructor.dataset.paths import (
    ASSEMBLED_DATASET_PATH, CLASS_TO_INDEX, DATA_DIR, POSE_LABELS,
    SOURCE_CLASS_NAMES, SOURCE_TO_POSE_LABEL,
)
from yoga_instructor.features.extraction import body_normalize_3d, compute_features_3d
from yoga_instructor.features.schema import MODEL_FEATURE_NAMES, POSE_FEATURE_NAMES, SCHEMA_VERSION


def create_dataset(
    landmarks_path=DATA_DIR / "landmarks_3d.npz",
    output_path=ASSEMBLED_DATASET_PATH,
    seed=42,
    validation_fraction=0.15,
):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    with np.load(landmarks_path, allow_pickle=False) as source:
        points = source["points"]
        labels = source["labels"].astype(str)
        source_splits = source["source_splits"].astype(str)
        relative_paths = source["relative_paths"].astype(str)
        hashes = source["sha256"].astype(str)
    if points.shape != (len(labels), 33, 3) or not all(len(values) == len(points) for values in (source_splits, relative_paths, hashes)):
        raise ValueError("landmarks archive has inconsistent arrays or points not shaped (n, 33, 3)")
    unknown_labels = sorted(set(labels) - set(SOURCE_CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"landmarks contain unconfigured class labels: {unknown_labels}")
    extraction_summary_path = Path(landmarks_path).with_suffix(".summary.json")
    extraction_parameters = (
        json.loads(extraction_summary_path.read_text(encoding="utf-8"))
        if extraction_summary_path.is_file() else None
    )

    grouped = defaultdict(list)
    for index, digest in enumerate(hashes):
        # Fall back to a stable path hash only for old extraction archives without a hash.
        key = digest or hashlib.sha256(relative_paths[index].encode()).hexdigest()
        grouped[key].append(index)
    unique_groups = []
    conflicting_duplicate_groups = 0
    for digest, indices in sorted(grouped.items()):
        class_names = {labels[index] for index in indices}
        if len(class_names) != 1:
            conflicting_duplicate_groups += 1
            continue
        label = next(iter(class_names))
        split = "test" if any(source_splits[index].upper() == "TEST" for index in indices) else "train_pool"
        unique_groups.append((digest, indices[0], label, split))

    # Keep official TEST images as test. Deterministically reserve a stratified
    # fraction of remaining unique training images for validation by class.
    rng = np.random.default_rng(seed)
    split_by_hash = {}
    for label in SOURCE_CLASS_NAMES:
        class_groups = [group for group in unique_groups if group[2] == label and group[3] == "train_pool"]
        order = rng.permutation(len(class_groups))
        validation_count = int(round(len(class_groups) * validation_fraction))
        if len(class_groups) > 1:
            validation_count = max(1, min(validation_count, len(class_groups) - 1))
        else:
            validation_count = 0
        validation_set = {class_groups[int(position)][0] for position in order[:validation_count]}
        for digest, _, _, _ in class_groups:
            split_by_hash[digest] = "validation" if digest in validation_set else "train"
    for digest, _, _, split in unique_groups:
        if split == "test":
            split_by_hash[digest] = "test"

    vectors, targets, sample_splits, sample_ids = [], [], [], []
    skipped_degenerate = []
    for digest, index, label, _ in unique_groups:
        try:
            normalized = body_normalize_3d(points[index])
            _, engineered = compute_features_3d(points[index], normalized)
        except ValueError as error:
            skipped_degenerate.append({"sample": relative_paths[index], "reason": str(error)})
            continue
        vectors.append(np.concatenate((normalized, engineered)))
        targets.append(CLASS_TO_INDEX[SOURCE_TO_POSE_LABEL[label]])
        sample_splits.append(split_by_hash[digest])
        sample_ids.append(relative_paths[index])

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int64)
    splits = np.asarray(sample_splits, dtype="U16")
    sample_ids = np.asarray(sample_ids, dtype="U512")
    if X.ndim != 2 or X.shape[1] != len(MODEL_FEATURE_NAMES):
        raise ValueError(f"generated model inputs have shape {X.shape}, expected (*, {len(MODEL_FEATURE_NAMES)})")
    if not {"train", "validation", "test"}.issubset(set(splits)):
        raise ValueError(f"all three splits must be non-empty; found {sorted(set(splits))}")
    for split in ("train", "validation", "test"):
        split_labels = set(y[splits == split].tolist())
        expected_labels = set(range(len(POSE_LABELS)))
        if split_labels != expected_labels:
            missing = [POSE_LABELS[index] for index in sorted(expected_labels - split_labels)]
            raise ValueError(f"{split} split is missing pose classes: {missing}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, splits=splits, sample_ids=sample_ids)
    counts = {
        split: {POSE_LABELS[index]: int(np.sum((splits == split) & (y == index))) for index in range(len(POSE_LABELS))}
        for split in ("train", "validation", "test")
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "pose_feature_names": list(POSE_FEATURE_NAMES),
        "landmark_shape": [33, 3],
        "class_labels": list(POSE_LABELS),
        "source_class_names": list(SOURCE_CLASS_NAMES),
        "class_to_index": {label: index for index, label in enumerate(POSE_LABELS)},
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_policy": "official TEST retained; TRAIN split stratified by class; exact duplicate images grouped by SHA-256",
        "subject_or_recording_split": "unavailable: source images have no subject or recording identifiers",
        "normalization": "hip-midpoint centered; divided by 3D distance from hip midpoint to shoulder midpoint",
        "missing_feature_value": -1.0,
        "landmark_extraction": extraction_parameters,
        "samples_per_split_and_class": counts,
        "source_landmark_rows": int(len(points)),
        "unique_nonconflicting_hash_groups": int(len(unique_groups)),
        "duplicate_rows_collapsed": int(sum(len(grouped[digest]) - 1 for digest, _, _, _ in unique_groups)),
        "conflicting_label_duplicate_groups_dropped": conflicting_duplicate_groups,
        "degenerate_landmark_samples_dropped": skipped_degenerate,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(output_path), "samples": len(X), "counts": counts, "metadata": str(metadata_path)}, indent=2))
    return X, y, splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, default=DATA_DIR / "landmarks_3d.npz")
    parser.add_argument("--output", type=Path, default=ASSEMBLED_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    create_dataset(args.landmarks, args.output, args.seed, args.validation_fraction)


if __name__ == "__main__":
    main()
