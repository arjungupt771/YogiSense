"""Train the pose classifier from an explicitly split feature dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from yoga_instructor.dataset.paths import ASSEMBLED_DATASET_PATH, DATA_DIR, MODEL_PATH, POSE_LABELS
from yoga_instructor.features.schema import MODEL_FEATURE_NAMES, SCHEMA_VERSION


def train(dataset_path=ASSEMBLED_DATASET_PATH, output_dir=None, seed=42, epochs=40, batch_size=32, learning_rate=1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir or MODEL_PATH.parent)
    metadata_path = dataset_path.with_suffix(".metadata.json")
    if not dataset_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Expected dataset and metadata: {dataset_path} and {metadata_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        X, y, splits = archive["X"], archive["y"], archive["splits"].astype(str)
    dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if X.ndim != 2 or X.shape[1] != len(MODEL_FEATURE_NAMES) or not np.isfinite(X).all():
        raise ValueError(f"Expected finite X shaped (n, {len(MODEL_FEATURE_NAMES)}), got {X.shape}")
    if y.shape != (len(X),) or splits.shape != (len(X),):
        raise ValueError("X, y, and splits arrays have inconsistent lengths")
    if dataset_metadata.get("feature_names") != list(MODEL_FEATURE_NAMES):
        raise ValueError("Dataset feature schema does not match yoga_instructor.features.schema")
    if dataset_metadata.get("class_labels") != list(POSE_LABELS):
        raise ValueError("Dataset class mapping does not match dataset_paths.py")
    if not np.array_equal(np.unique(y), np.arange(len(POSE_LABELS))):
        raise ValueError("Dataset must contain every contiguous class index from zero")
    train_mask, validation_mask, test_mask = (splits == name for name in ("train", "validation", "test"))
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Dataset must contain non-empty train, validation, and test splits")
    if len(np.unique(y[train_mask])) != len(POSE_LABELS):
        raise ValueError("Every configured pose class must have training samples")
    train_counts = np.bincount(y[train_mask], minlength=len(POSE_LABELS))
    class_weights = {
        index: float(train_mask.sum() / (len(POSE_LABELS) * count))
        for index, count in enumerate(train_counts)
    }

    model = models.Sequential([
        layers.Input(shape=(len(MODEL_FEATURE_NAMES),)),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dense(len(POSE_LABELS), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X[train_mask], y[train_mask],
        validation_data=(X[validation_mask], y[validation_mask]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weights,
        verbose=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "yogisense_model_3d.keras"
    model.save(model_path)
    (output_dir / "class_mapping.json").write_text(
        json.dumps({"class_to_index": {label: index for index, label in enumerate(POSE_LABELS)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "missing_feature_value": dataset_metadata.get("missing_feature_value", -1.0),
        "landmark_shape": dataset_metadata.get("landmark_shape", [33, 3]),
        "normalization": dataset_metadata.get("normalization"),
        "landmark_extraction": dataset_metadata.get("landmark_extraction"),
    }
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model_file": model_path.name,
        "class_labels": list(POSE_LABELS),
        "input_dimension": len(MODEL_FEATURE_NAMES),
        "output_dimension": len(POSE_LABELS),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "tensorflow_version": tf.__version__,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_split_policy": dataset_metadata.get("split_policy"),
        "dataset_split_seed": dataset_metadata.get("seed"),
        "validation_fraction": dataset_metadata.get("validation_fraction"),
        "samples_per_split_and_class": dataset_metadata.get("samples_per_split_and_class"),
        "normalization": dataset_metadata.get("normalization"),
        "split_counts": {name: int(np.sum(splits == name)) for name in ("train", "validation", "test")},
        "best_validation_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "class_weights": {POSE_LABELS[index]: weight for index, weight in class_weights.items()},
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model and metadata to {output_dir}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ASSEMBLED_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_PATH.parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.seed, args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
