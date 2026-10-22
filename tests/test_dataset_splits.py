import tempfile
import unittest
from pathlib import Path

import numpy as np

from yoga_instructor.dataset.build import create_dataset
from yoga_instructor.dataset.paths import SOURCE_CLASS_NAMES


def landmarks(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 0.01, (33, 3)).astype(np.float32)
    points[11], points[12] = (-0.5, 0, 0), (0.5, 0, 0)
    points[23], points[24] = (-0.3, 1, 0), (0.3, 1, 0)
    points[25], points[26] = (-0.3, 2, 0), (0.3, 2, 0)
    points[27], points[28] = (-0.3, 3, 0), (0.3, 3, 0)
    return points


class DatasetSplitTests(unittest.TestCase):
    def test_splits_are_reproducible_and_exact_duplicates_stay_together(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            rows, labels, source_splits, paths, hashes = [], [], [], [], []
            for class_index, label in enumerate(SOURCE_CLASS_NAMES):
                for number in range(5):
                    rows.append(landmarks(class_index * 10 + number))
                    labels.append(label)
                    source_splits.append("TEST" if number < 2 else "TRAIN")
                    paths.append(f"{label}/{number}.jpg")
                    hashes.append(f"{label}-{number}")
                # A duplicate of a TEST sample in TRAIN must remain assigned to TEST and be deduplicated.
                rows.append(rows[-5].copy())
                labels.append(label)
                source_splits.append("TRAIN")
                paths.append(f"{label}/duplicate.jpg")
                hashes.append(f"{label}-0")
            source_file = directory / "landmarks.npz"
            np.savez_compressed(
                source_file,
                points=np.stack(rows), labels=np.asarray(labels),
                source_splits=np.asarray(source_splits), relative_paths=np.asarray(paths),
                sha256=np.asarray(hashes),
            )
            output = directory / "assembled.npz"
            first = create_dataset(source_file, output, seed=7, validation_fraction=0.25)
            first_archive = np.load(output, allow_pickle=False)
            first_splits, first_ids = first_archive["splits"].copy(), first_archive["sample_ids"].copy()
            first_archive.close()
            second = create_dataset(source_file, output, seed=7, validation_fraction=0.25)
            self.assertTrue(np.array_equal(first[0], second[0]))
            self.assertTrue(np.array_equal(first[1], second[1]))
            with np.load(output, allow_pickle=False) as second_archive:
                self.assertTrue(np.array_equal(first_splits, second_archive["splits"]))
            self.assertEqual(len(first_ids), len(set(first_ids.tolist())))
            self.assertEqual(set(first_splits.tolist()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
