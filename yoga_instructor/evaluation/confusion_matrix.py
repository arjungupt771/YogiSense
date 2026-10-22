"""Plot a confusion matrix using only the held-out official test split."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from yoga_instructor.dataset.paths import ASSEMBLED_DATASET_PATH, MODEL_PATH, POSE_LABELS


def plot(dataset_path=ASSEMBLED_DATASET_PATH, model_path=MODEL_PATH, output_path=None):
    import tensorflow as tf

    with np.load(dataset_path, allow_pickle=False) as archive:
        X, y, splits = archive["X"], archive["y"], archive["splits"].astype(str)
    test = splits == "test"
    if not test.any():
        raise ValueError("No held-out test samples were found")
    model = tf.keras.models.load_model(model_path)
    predictions = np.argmax(model.predict(X[test], verbose=0), axis=1)
    matrix = confusion_matrix(y[test], predictions, labels=np.arange(len(POSE_LABELS)))
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=POSE_LABELS).plot(
        cmap="Blues", ax=axis, xticks_rotation=45, colorbar=False,
    )
    figure.tight_layout()
    output_path = Path(output_path or Path(model_path).parent / "test_confusion_matrix.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved test confusion matrix to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ASSEMBLED_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()
