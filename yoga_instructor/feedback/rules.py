"""Rule-based pose cues using the canonical feature dictionary."""

import math

from yoga_instructor.features.schema import POSE_FEATURE_NAMES

DEFAULT_RULES = {
    "chakrasana": {
        "left_elbow_angle": ("min", 150), "right_elbow_angle": ("min", 150),
        "left_knee_angle": ("min", 160), "right_knee_angle": ("min", 160),
        "backbend_amount": ("min", 25), "wrist_to_ankle_3d": ("max", 1.2),
    },
    "halasana": {
        "ankle_above_head": ("min", 0.5), "left_knee_angle": ("min", 160),
        "right_knee_angle": ("min", 160), "shoulder_hip_offset_3d": ("max", 0.8),
    },
    "ardha_matsyendrasana": {
        "torso_twist_3d": ("min", 20), "left_knee_angle": ("min", 140),
        "right_knee_angle": ("min", 140),
    },
}


def pose_rule_feedback_3d(pose_label, features, rules=DEFAULT_RULES):
    """Return only cues backed by present, finite canonical features.

    Unknown or missing measurements suppress that individual cue, avoiding
    false corrections when landmarks are occluded or a vector is degenerate.
    """
    if not isinstance(features, dict):
        return []
    unknown = set(features) - set(POSE_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown pose feature name(s): {', '.join(sorted(unknown))}")

    def violates(name, direction, threshold):
        value = features.get(name)
        if value is None:
            return False
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        return value < threshold if direction == "min" else value > threshold

    cues = []
    if pose_label == "chakrasana":
        if any(violates(name, *rules[pose_label][name]) for name in ("left_elbow_angle", "right_elbow_angle")):
            cues.append("Straighten your arms more.")
        if any(violates(name, *rules[pose_label][name]) for name in ("left_knee_angle", "right_knee_angle")):
            cues.append("Straighten your legs.")
        if violates("backbend_amount", *rules[pose_label]["backbend_amount"]):
            cues.append("Arch your back more, open the chest.")
        if violates("wrist_to_ankle_3d", *rules[pose_label]["wrist_to_ankle_3d"]):
            cues.append("Bring your hands closer to your feet and press firmly into the mat.")
    elif pose_label == "halasana":
        if violates("ankle_above_head", *rules[pose_label]["ankle_above_head"]):
            cues.append("Bring your feet further over your head.")
        if any(violates(name, *rules[pose_label][name]) for name in ("left_knee_angle", "right_knee_angle")):
            cues.append("Straighten your legs fully.")
        if violates("shoulder_hip_offset_3d", *rules[pose_label]["shoulder_hip_offset_3d"]):
            cues.append("Roll your shoulders under your back for support.")
    elif pose_label == "ardha_matsyendrasana":
        if violates("torso_twist_3d", *rules[pose_label]["torso_twist_3d"]):
            cues.append("Rotate your torso more to deepen the twist.")
        if any(violates(name, *rules[pose_label][name]) for name in ("left_knee_angle", "right_knee_angle")):
            cues.append("Keep your base leg active and straight.")
    else:
        if violates("left_elbow_angle", "min", 150):
            cues.append("Extend your left arm more.")
        if violates("right_elbow_angle", "min", 150):
            cues.append("Extend your right arm more.")
    return cues

