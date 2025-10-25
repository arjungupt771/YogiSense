import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from cellb import compute_features_3d
from cellnew import pose_rule_feedback_3d
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
from database import DATA_DIR
from collections import deque

# Load model
model = tf.keras.models.load_model(DATA_DIR/'yogisense_model_3d.h5')
#model = tf.keras.models.load_model(model_path)
labels = ['warrior_ii','tree_pose','downward_dog','cobra','mountain','chakrasana','halasana','ardha_matsyendrasana']

# Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit title
st.title("🧘 YogiSense 3D – Live Yoga Pose Detection")
st.markdown("Real-time yoga pose classification with 3D feedback and accuracy visualization.")

# Start/Stop buttons
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera"):
        st.session_state.run_camera = True
with col2:
    if st.button("Stop Camera"):
        st.session_state.run_camera = False

# Pose accuracy deque for live plot
pose_confidences = deque(maxlen=50)

# Video Processor class
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not st.session_state.run_camera:
            return img

        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = {}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[f"point_{i}"] = [lm.x, lm.y, lm.z]

            feats = compute_features_3d(landmarks)
            X = np.array(list(feats.values())).reshape(1,-1)
            pred = model.predict(X)
            pose_id = np.argmax(pred)
            pose_label = labels[pose_id]
            confidence = np.max(pred)
            pose_confidences.append(confidence)

            feedback = pose_rule_feedback_3d(pose_label, feats)

            # Draw pose label & feedback
            cv2.putText(img, f"Pose: {pose_label} ({confidence*100:.1f}%)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for i, fb in enumerate(feedback):
                cv2.putText(img, fb, (20,80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img

# Start webcam stream
if st.session_state.run_camera:
    webrtc_streamer(key="yogisense", video_processor_factory=VideoProcessor)

# Live Accuracy Plot
st.subheader("📈 Live Pose Confidence")
if pose_confidences:
    fig, ax = plt.subplots()
    ax.plot(list(pose_confidences), label="Pose Confidence")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Confidence")
    ax.set_title("Live Pose Accuracy")
    ax.legend()
    st.pyplot(fig)
