"""Deterministic MediaPipe landmark validation and 3D pose features."""
from __future__ import annotations

import math

import numpy as np

from yoga_instructor.features.schema import MISSING_FEATURE_VALUE, POSE_FEATURE_NAMES

LANDMARK_SHAPE = (33, 3)
QUALITY_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def mp_to_landmarks3(results, min_visibility: float = 0.35):
    """Return a quality-checked (33, 3) float32 array, or None if detection is weak."""
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return None
    landmarks = pose_landmarks.landmark
    if len(landmarks) != LANDMARK_SHAPE[0]:
        return None
    points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    visibility = np.asarray([getattr(point, "visibility", 1.0) for point in landmarks], dtype=np.float32)
    if not validate_landmarks(points, visibility=visibility, min_visibility=min_visibility):
        return None
    return points


def validate_landmarks(points, visibility=None, min_visibility: float = 0.35) -> bool:
    """Check shape, finite coordinates, and visibility of pose-critical landmarks."""
    try:
        points = np.asarray(points)
    except (TypeError, ValueError):
        return False
    if points.shape != LANDMARK_SHAPE or not np.isfinite(points).all():
        return False
    if visibility is None:
        return True
    try:
        visibility = np.asarray(visibility, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if visibility.shape != (LANDMARK_SHAPE[0],) or not np.isfinite(visibility).all():
        return False
    if np.any((visibility < 0.0) | (visibility > 1.0)):
        return False
    return bool(np.all(visibility[list(QUALITY_LANDMARKS)] >= min_visibility))


def body_normalize_3d(raw_pts):
    """Center at the hip midpoint and scale by shoulder-midpoint to hip-midpoint length."""
    points = np.asarray(raw_pts, dtype=np.float32)
    if not validate_landmarks(points):
        raise ValueError("landmarks must be a finite array with shape (33, 3)")
    mid_hip = (points[23] + points[24]) / 2.0
    mid_shoulder = (points[11] + points[12]) / 2.0
    torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))
    if not math.isfinite(torso_length) or torso_length <= 1e-6:
        raise ValueError("cannot normalize landmarks: torso length is zero")
    return ((points - mid_hip) / torso_length).astype(np.float32).reshape(-1)


def angle_3d(a, b, c):
    """Return the angle at b in degrees; return None for zero-length/non-finite limbs."""
    a, b, c = (np.asarray(point, dtype=np.float64) for point in (a, b, c))
    ba, bc = a - b, c - b
    norm_product = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if not math.isfinite(norm_product) or norm_product <= 1e-12:
        return None
    cosine = float(np.dot(ba, bc) / norm_product)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _axis_angle(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_features_3d(raw_pts, body_norm_flat=None):
    """Return ``(feature_dict, ordered_vector)`` using ``POSE_FEATURE_NAMES``.

    Missing landmarks return ``(None, None)``. Degenerate angles are represented
    as ``None`` in the dictionary and the documented -1 sentinel in the vector.
    """
    if raw_pts is None:
        return None, None
    points = np.asarray(raw_pts, dtype=np.float32)
    if not validate_landmarks(points):
        raise ValueError("landmarks must be a finite array with shape (33, 3)")
    normalized = body_normalize_3d(points) if body_norm_flat is None else np.asarray(body_norm_flat, dtype=np.float32).reshape(-1)
    if normalized.shape != (99,) or not np.isfinite(normalized).all():
        raise ValueError("normalized landmarks must contain 99 finite values")
    body = normalized.reshape(LANDMARK_SHAPE)

    mid_shoulder = (body[11] + body[12]) / 2.0
    mid_hip = (body[23] + body[24]) / 2.0
    features = {
        "left_knee_angle": angle_3d(body[23], body[25], body[27]),
        "right_knee_angle": angle_3d(body[24], body[26], body[28]),
        "left_elbow_angle": angle_3d(body[11], body[13], body[15]),
        "right_elbow_angle": angle_3d(body[12], body[14], body[16]),
        "spine_angle": angle_3d(mid_shoulder, mid_hip, (body[25] + body[26]) / 2.0),
        "backbend_amount": None,
        "shoulder_width_3d": float(np.linalg.norm(body[11] - body[12])),
        "shoulder_hip_offset_3d": float(np.linalg.norm((mid_shoulder - mid_hip)[[0, 2]])),
        "wrist_to_ankle_3d": float((np.linalg.norm(body[15] - body[27]) + np.linalg.norm(body[16] - body[28])) / 2.0),
        "ankle_above_head": float(points[27, 1] < points[0, 1] or points[28, 1] < points[0, 1]),
        "torso_twist_3d": _axis_angle(body[12] - body[11], body[24] - body[23]),
    }
    if features["spine_angle"] is not None:
        features["backbend_amount"] = 180.0 - features["spine_angle"]
    vector = np.asarray(
        [MISSING_FEATURE_VALUE if features[name] is None else features[name] for name in POSE_FEATURE_NAMES],
        dtype=np.float32,
    )
    return features, vector

