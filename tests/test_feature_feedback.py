import unittest

import numpy as np

from yoga_instructor.features.extraction import (
    angle_3d, body_normalize_3d, compute_features_3d, validate_landmarks,
)
from yoga_instructor.features.schema import MISSING_FEATURE_VALUE, POSE_FEATURE_NAMES
from yoga_instructor.feedback.rules import pose_rule_feedback_3d


def synthetic_pose():
    points = np.zeros((33, 3), dtype=np.float32)
    points[:, 1] = 0.0
    points[0] = (0.0, -2.0, 0.0)
    points[11], points[12] = (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
    points[13], points[14] = (-1.5, 0.0, 0.0), (1.5, 0.0, 0.0)
    points[15], points[16] = (-2.5, 0.0, 0.0), (2.5, 0.0, 0.0)
    points[23], points[24] = (-0.3, 1.0, 0.0), (0.3, 1.0, 0.0)
    points[25], points[26] = (-0.3, 2.0, 0.0), (0.3, 2.0, 0.0)
    points[27], points[28] = (-0.3, 3.0, 0.0), (0.3, 3.0, 0.0)
    return points


class FeatureExtractionTests(unittest.TestCase):
    def test_known_angles_and_zero_length_vectors(self):
        self.assertAlmostEqual(angle_3d((0, 0, 0), (1, 0, 0), (1, 1, 0)), 90.0)
        self.assertIsNone(angle_3d((1, 0, 0), (1, 0, 0), (2, 0, 0)))

    def test_normalization_centers_and_is_scale_translation_invariant(self):
        points = synthetic_pose()
        normalized = body_normalize_3d(points).reshape(33, 3)
        self.assertTrue(np.allclose((normalized[23] + normalized[24]) / 2.0, 0.0))
        moved_scaled = points * 3.5 + np.array([4.0, -7.0, 2.0], dtype=np.float32)
        self.assertTrue(np.allclose(body_normalize_3d(points), body_normalize_3d(moved_scaled), atol=1e-6))

    def test_shape_visibility_and_degenerate_torso_are_controlled(self):
        points = synthetic_pose()
        self.assertTrue(validate_landmarks(points))
        visibility = np.ones(33, dtype=np.float32)
        visibility[27] = 0.1
        self.assertFalse(validate_landmarks(points, visibility=visibility))
        visibility[27] = 1.2
        self.assertFalse(validate_landmarks(points, visibility=visibility))
        self.assertFalse(validate_landmarks(np.zeros((32, 3))))
        with self.assertRaises(ValueError):
            body_normalize_3d(np.zeros((33, 3)))

    def test_feature_order_and_repeatability(self):
        points = synthetic_pose()
        first_dict, first_vector = compute_features_3d(points)
        second_dict, second_vector = compute_features_3d(points.copy())
        self.assertEqual(tuple(first_dict), POSE_FEATURE_NAMES)
        self.assertEqual(first_vector.shape, (len(POSE_FEATURE_NAMES),))
        self.assertTrue(np.array_equal(first_vector, second_vector))
        self.assertAlmostEqual(first_dict["left_knee_angle"], 180.0)
        self.assertAlmostEqual(first_dict["left_elbow_angle"], 180.0)

    def test_degenerate_angle_uses_documented_missing_sentinel(self):
        points = synthetic_pose()
        points[23] = points[25]  # Degenerate left knee angle.
        features, vector = compute_features_3d(points)
        self.assertIsNone(features["left_knee_angle"])
        self.assertEqual(vector[POSE_FEATURE_NAMES.index("left_knee_angle")], MISSING_FEATURE_VALUE)


class FeedbackRuleTests(unittest.TestCase):
    def setUp(self):
        self.features = {name: 180.0 for name in POSE_FEATURE_NAMES}
        self.features.update({
            "backbend_amount": 40.0,
            "shoulder_width_3d": 0.5,
            "shoulder_hip_offset_3d": 0.2,
            "wrist_to_ankle_3d": 0.5,
            "ankle_above_head": 1.0,
            "torso_twist_3d": 30.0,
        })

    def test_wheel_cues_use_canonical_measurements(self):
        self.features.update({
            "left_elbow_angle": 100.0, "left_knee_angle": 120.0,
            "backbend_amount": 5.0, "wrist_to_ankle_3d": 2.0,
        })
        cues = pose_rule_feedback_3d("chakrasana", self.features)
        self.assertEqual(len(cues), 4)

    def test_plough_and_twist_rules(self):
        self.features.update({"ankle_above_head": 0.0, "right_knee_angle": 120.0, "shoulder_hip_offset_3d": 1.2})
        self.assertEqual(len(pose_rule_feedback_3d("halasana", self.features)), 3)
        self.features.update({"torso_twist_3d": 5.0, "left_knee_angle": 100.0})
        self.assertEqual(len(pose_rule_feedback_3d("ardha_matsyendrasana", self.features)), 2)

    def test_unavailable_measurements_do_not_trigger_false_feedback(self):
        cues = pose_rule_feedback_3d("chakrasana", {"left_elbow_angle": None})
        self.assertEqual(cues, [])

    def test_generic_pose_rule(self):
        self.features["right_elbow_angle"] = 100.0
        cues = pose_rule_feedback_3d("tree_pose", self.features)
        self.assertEqual(cues, ["Extend your right arm more."])

    def test_unrecognized_feature_name_is_rejected(self):
        with self.assertRaises(ValueError):
            pose_rule_feedback_3d("chakrasana", {"left_elbow": 0})


if __name__ == "__main__":
    unittest.main()
