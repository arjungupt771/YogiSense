"""MediaPipe Pose Landmarker compatibility for image and webcam inference."""
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

import mediapipe as mp

from yoga_instructor.dataset.paths import DATA_DIR

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = DATA_DIR / "pose_landmarker_lite.task"


def _ensure_model():
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".task.download")
    try:
        urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(MODEL_PATH)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe Pose Landmarker model. "
            f"Check your internet connection or place the model at {MODEL_PATH}."
        ) from error
    return MODEL_PATH


class PoseDetector(AbstractContextManager):
    """Wrap current MediaPipe Tasks and expose the result shape used by this app."""

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        del model_complexity  # Tasks' lite model is the closest equivalent.
        self.running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if static_image_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, rgb_image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        if self.running_mode == mp.tasks.vision.RunningMode.IMAGE:
            result = self._detector.detect(image)
        else:
            self._last_timestamp_ms += 1
            result = self._detector.detect_for_video(image, self._last_timestamp_ms)
        poses = result.pose_landmarks
        pose_landmarks = SimpleNamespace(landmark=poses[0]) if poses else None
        return SimpleNamespace(pose_landmarks=pose_landmarks)

    def close(self):
        self._detector.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def draw_pose_landmarks(image, landmarks):
    """Draw the pose using OpenCV, without relying on the removed Solutions API."""
    import cv2

    height, width = image.shape[:2]
    connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    for connection in connections:
        start, end = int(connection.start), int(connection.end)
        first, second = landmarks.landmark[start], landmarks.landmark[end]
        p1 = (int(first.x * width), int(first.y * height))
        p2 = (int(second.x * width), int(second.y * height))
        cv2.line(image, p1, p2, (0, 255, 0), 2)
    return image
