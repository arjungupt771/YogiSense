"""Canonical ordered contract for model inputs and engineered pose features."""

SCHEMA_VERSION = 1
LANDMARK_COUNT = 33
COORDINATE_COUNT = 3
LANDMARK_FEATURE_NAMES = tuple(
    f"landmark_{index:02d}_{axis}"
    for index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
)
POSE_FEATURE_NAMES = (
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "spine_angle",
    "backbend_amount",
    "shoulder_width_3d",
    "shoulder_hip_offset_3d",
    "wrist_to_ankle_3d",
    "ankle_above_head",
    "torso_twist_3d",
)
MODEL_FEATURE_NAMES = LANDMARK_FEATURE_NAMES + POSE_FEATURE_NAMES
MISSING_FEATURE_VALUE = -1.0

