"""Evaluate the trained classifier only on the held-out official test split."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from yoga_instructor.dataset.paths import ASSEMBLED_DATASET_PATH, MODEL_PATH, POSE_LABELS


def evaluate(dataset_path=ASSEMBLED_DATASET_PATH, model_path=MODEL_PATH):
    import tensorflow as tf

    dataset_path, model_path = Path(dataset_path), Path(model_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        X, y, splits = archive["X"], archive["y"], archive["splits"].astype(str)
    test = splits == "test"
    if not test.any():
        raise ValueError("No held-out test samples were found")
    classifier = tf.keras.models.load_model(model_path)
    probabilities = classifier.predict(X[test], verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        y[test], predictions, labels=np.arange(len(POSE_LABELS)),
        target_names=POSE_LABELS, output_dict=True, zero_division=0,
    )
    result = {"test_accuracy": float(accuracy_score(y[test], predictions)), "classification_report": report}
    output = model_path.parent / "test_evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved test-only evaluation to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ASSEMBLED_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.dataset, args.model)


if __name__ == "__main__":
    main()
