"""Standalone OpenCV webcam prediction demo (run as a script to open camera)."""
import time

import cv2
import numpy as np
import tensorflow as tf

from yoga_instructor.dataset.paths import MODEL_PATH, POSE_LABELS
from yoga_instructor.features.extraction import body_normalize_3d, compute_features_3d, mp_to_landmarks3
from yoga_instructor.pose import PoseDetector


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    classifier = tf.keras.models.load_model(MODEL_PATH)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")
    history = []
    previous_time = time.time()
    try:
        with PoseDetector(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = mp_to_landmarks3(results)
                label_text = "Pose not detected clearly"
                if points is not None:
                    normalized = body_normalize_3d(points)
                    _, features = compute_features_3d(points, normalized)
                    history.append(np.concatenate((normalized, features)))
                    history[:] = history[-5:]
                    model_input = np.mean(np.stack(history), axis=0).reshape(1, -1)
                    probabilities = classifier.predict(model_input, verbose=0)[0]
                    class_index = int(np.argmax(probabilities))
                    label_text = f"{POSE_LABELS[class_index]} ({probabilities[class_index]:.2f})"
                cv2.putText(frame, f"Pose: {label_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                now = time.time()
                fps = 1.0 / max(now - previous_time, 1e-8)
                previous_time = now
                cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("YogiSense 3D", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
