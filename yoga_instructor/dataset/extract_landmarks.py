"""Run MediaPipe Pose on source images and save quality-checked landmarks."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from yoga_instructor.dataset.paths import DATA_DIR, RAW_IMAGE_DIR, SOURCE_CLASS_NAMES
from yoga_instructor.features.extraction import mp_to_landmarks3
from yoga_instructor.pose import PoseDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_dataset(raw_dir=RAW_IMAGE_DIR, output_path=None, min_detection_confidence=0.5, min_visibility=0.35):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path or DATA_DIR / "landmarks_3d.npz")
    points, labels, splits, relative_paths, hashes = [], [], [], [], []
    failures = []
    files = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    with PoseDetector(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        for index, path in enumerate(files, 1):
            relative = path.relative_to(raw_dir)
            parts = relative.parts
            if len(parts) < 3 or parts[1] not in SOURCE_CLASS_NAMES or parts[0].upper() not in {"TRAIN", "TEST"}:
                failures.append({"path": str(relative), "reason": "unexpected directory layout or class"})
                continue
            image = cv2.imread(str(path))
            if image is None:
                failures.append({"path": str(relative), "reason": "image could not be decoded"})
                continue
            result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = mp_to_landmarks3(result, min_visibility=min_visibility)
            if landmarks is None:
                failures.append({"path": str(relative), "reason": "no pose or low landmark visibility"})
                continue
            points.append(landmarks)
            labels.append(parts[1])
            splits.append(parts[0].upper())
            relative_paths.append(relative.as_posix())
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            if index % 100 == 0:
                print(f"Processed {index}/{len(files)} images; retained {len(points)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        raise RuntimeError(f"No valid pose landmarks extracted from {raw_dir}")
    np.savez_compressed(
        output_path,
        points=np.stack(points).astype(np.float32),
        labels=np.asarray(labels, dtype="U32"),
        source_splits=np.asarray(splits, dtype="U8"),
        relative_paths=np.asarray(relative_paths, dtype="U512"),
        sha256=np.asarray(hashes, dtype="U64"),
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "images_found": len(files), "samples_retained": len(points),
        "failures": failures,
        "pose_model_complexity": 1,
        "static_image_mode": True,
        "min_detection_confidence": min_detection_confidence,
        "min_visibility": min_visibility,
        "image_color_conversion": "OpenCV BGR to MediaPipe RGB",
        "landmark_shape": [33, 3],
        "output": str(output_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))
    print(f"Detailed per-image failures: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "landmarks_3d.npz")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    args = parser.parse_args()
    extract_dataset(args.raw_dir, args.output, args.min_detection_confidence, args.min_visibility)


if __name__ == "__main__":
    main()
