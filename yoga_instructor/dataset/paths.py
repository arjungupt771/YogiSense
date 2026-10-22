"""Dataset paths and explicit dataset assembly helpers.

Importing this module must not scan data or start processing images.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"
ASSEMBLED_DATASET_PATH = DATA_DIR / "assembled_dataset_3d.npz"
MODEL_ARTIFACT_DIR = DATA_DIR / "model_artifacts"
MODEL_PATH = MODEL_ARTIFACT_DIR / "yogisense_model_3d.keras"
MODEL_METADATA_PATH = MODEL_ARTIFACT_DIR / "model_metadata.json"
SOURCE_CLASS_NAMES = ("downdog", "goddess", "plank", "tree", "warrior2")
POSE_LABELS = ("downward_dog", "goddess", "plank", "tree_pose", "warrior_ii")
SOURCE_TO_POSE_LABEL = dict(zip(SOURCE_CLASS_NAMES, POSE_LABELS))
CLASS_TO_INDEX = {label: index for index, label in enumerate(POSE_LABELS)}
