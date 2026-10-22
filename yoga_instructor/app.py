"""Streamlit webcam yoga pose classifier."""
from collections import deque
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from yoga_instructor.dataset.paths import MODEL_METADATA_PATH, MODEL_PATH, POSE_LABELS
from yoga_instructor.features.extraction import body_normalize_3d, compute_features_3d, mp_to_landmarks3
from yoga_instructor.features.schema import MODEL_FEATURE_NAMES
from yoga_instructor.feedback.rules import pose_rule_feedback_3d
from yoga_instructor.pose import PoseDetector, draw_pose_landmarks



def make_model_input(landmarks):
    """Build the trained vector from raw MediaPipe (33, 3) points."""
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (33, 3) or not np.isfinite(points).all():
        return None
    normalized = body_normalize_3d(points)
    features, feature_vector = compute_features_3d(points, normalized)
    if feature_vector is None:
        return None
    vector = np.concatenate((normalized, feature_vector)).astype(np.float32).reshape(1, -1)
    if vector.shape != (1, len(MODEL_FEATURE_NAMES)):
        raise ValueError(f"live feature vector has unexpected shape {vector.shape}")
    return vector, features


def load_classifier():
    if not MODEL_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model artifacts are missing under {MODEL_PATH.parent}. Run the documented dataset and training pipeline first."
        )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    expected_class_map = {label: index for index, label in enumerate(POSE_LABELS)}
    if metadata.get("input_dimension") != len(MODEL_FEATURE_NAMES):
        raise ValueError("Model metadata has an incompatible input feature dimension.")
    if metadata.get("class_labels") != list(POSE_LABELS):
        raise ValueError("Model class labels do not match the dataset class mapping.")
    class_mapping_path = MODEL_PATH.parent / "class_mapping.json"
    if not class_mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping missing at {class_mapping_path}.")
    class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    if class_mapping.get("class_to_index") != expected_class_map:
        raise ValueError("Model class index mapping is inconsistent with dataset_paths.py.")
    schema_path = MODEL_PATH.parent / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema metadata missing at {schema_path}.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("feature_names") != list(MODEL_FEATURE_NAMES):
        raise ValueError("Model feature schema does not match the live feature ordering.")
    classifier = tf.keras.models.load_model(MODEL_PATH)
    input_dim = int(classifier.input_shape[-1])
    if input_dim != len(MODEL_FEATURE_NAMES):
        raise ValueError(
            f"Model expects {input_dim} inputs; live extraction produces {len(MODEL_FEATURE_NAMES)}."
        )
    output_dim = int(classifier.output_shape[-1])
    if output_dim != len(POSE_LABELS):
        raise ValueError(f"Model has {output_dim} outputs but {len(POSE_LABELS)} pose labels are configured.")
    return classifier


class VideoProcessor(VideoTransformerBase):
    def __init__(self, classifier, confidence_history):
        self.classifier = classifier
        self.confidence_history = confidence_history
        self.pose = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detected_landmarks = getattr(results, "pose_landmarks", None)
        if detected_landmarks is None:
            cv2.putText(image, "Pose not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            return image
        points = mp_to_landmarks3(results)
        if points is None:
            cv2.putText(image, "Pose not detected clearly", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            return image
        try:
            model_input = make_model_input(points)
            if model_input is None:
                cv2.putText(image, "Invalid landmarks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                return image
            input_vector, features = model_input
            probabilities = np.asarray(self.classifier.predict(input_vector, verbose=0))[0]
            if probabilities.size != len(POSE_LABELS) or not np.isfinite(probabilities).all():
                raise ValueError("Model returned invalid class probabilities")
            class_index = int(np.argmax(probabilities))
            label = POSE_LABELS[class_index]
            confidence = float(probabilities[class_index])
            self.confidence_history.append(confidence)
            cv2.putText(image, f"Pose: {label} ({confidence * 100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for index, cue in enumerate(pose_rule_feedback_3d(label, features)):
                cv2.putText(image, cue, (20, 80 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            draw_pose_landmarks(image, detected_landmarks)
        except (ValueError, TypeError, IndexError):
            cv2.putText(image, "Prediction unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image


def run_app():
    st.title("🧘 YogiSense 3D – Live Yoga Pose Detection")
    st.markdown("Real-time yoga pose classification with 3D feedback and confidence visualization.")
    try:
        classifier = load_classifier()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Model unavailable: {error}")
        st.stop()

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False
    start_column, stop_column = st.columns(2)
    with start_column:
        if st.button("Start Camera"):
            st.session_state.run_camera = True
    with stop_column:
        if st.button("Stop Camera"):
            st.session_state.run_camera = False

    confidence_history = deque(maxlen=50)
    if st.session_state.run_camera:
        webrtc_streamer(
            key="yogisense",
            video_processor_factory=lambda: VideoProcessor(classifier, confidence_history),
        )
    st.subheader("📈 Live Pose Confidence")
    if confidence_history:
        figure, axis = plt.subplots()
        axis.plot(list(confidence_history), label="Pose Confidence")
        axis.set_ylim(0, 1)
        axis.set_xlabel("Frames")
        axis.set_ylabel("Confidence")
        axis.set_title("Live Pose Confidence")
        axis.legend()
        st.pyplot(figure)


if __name__ == "__main__":
    run_app()
