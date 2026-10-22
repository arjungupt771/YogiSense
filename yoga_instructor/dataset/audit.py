"""Audit source images for class coverage, corruption, duplicates, and balance."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from yoga_instructor.dataset.paths import RAW_IMAGE_DIR, SOURCE_CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit_dataset(root=RAW_IMAGE_DIR):
    root = Path(root)
    counts = Counter()
    corrupt = []
    hashes = defaultdict(list)
    unexpected_classes = set()
    seen_splits = set()
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        split = split_dir.name.upper()
        seen_splits.add(split)
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if class_dir.name not in SOURCE_CLASS_NAMES:
                unexpected_classes.add(class_dir.name)
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                counts[(split, class_dir.name)] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hashes[digest].append((split, class_dir.name, str(path.relative_to(root))))
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    corrupt.append({"path": str(path), "error": str(error)})
    duplicates = [items for items in hashes.values() if len(items) > 1]
    cross_split_duplicates = [items for items in duplicates if len({item[0] for item in items}) > 1]
    expected_splits = {"TRAIN", "TEST"}
    missing_by_split = {
        split: sorted(set(SOURCE_CLASS_NAMES) - {label for found_split, label in counts if found_split == split})
        for split in expected_splits
    }
    balance = {}
    for split in expected_splits:
        class_counts = {label: int(counts[(split, label)]) for label in SOURCE_CLASS_NAMES}
        nonzero = [count for count in class_counts.values() if count]
        balance[split] = {
            "counts": class_counts,
            "max_to_min_ratio": (max(nonzero) / min(nonzero)) if nonzero else None,
        }
    report = {
        "root": str(root.resolve()),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
        "total_images": sum(counts.values()),
        "corrupt_images": corrupt,
        "duplicate_groups": duplicates,
        "cross_split_duplicate_groups": cross_split_duplicates,
        "unexpected_classes": sorted(unexpected_classes),
        "missing_classes": sorted(set(SOURCE_CLASS_NAMES) - {label for _, label in counts}),
        "missing_classes_by_split": missing_by_split,
        "class_balance": balance,
        "splits": sorted(seen_splits),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RAW_IMAGE_DIR)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = audit_dataset(args.root)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if (
        report["corrupt_images"]
        or report["missing_classes"]
        or report["unexpected_classes"]
        or any(report["missing_classes_by_split"].values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
