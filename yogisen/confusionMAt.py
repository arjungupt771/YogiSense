import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from database import DATA_DIR
# Load dataset
d = np.load(os.path.join(DATA_DIR, 'assembled_dataset_3d.npz'))
X, y = d['X'], d['y']

# Load trained model
import tensorflow as tf
model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'yogisense_model_3d'))

# Predict on validation/test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Define pose labels in the correct order
POSE_LABELS = ['warrior_ii','tree_pose','downward_dog','cobra','mountain','chakrasana','halasana','ardha_matsyendrasana']

# Compute confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=range(len(POSE_LABELS)))

# Plot
fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=POSE_LABELS)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix – YogiSense 3D Model")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load dataset
d = np.load(os.path.join(DATA_DIR, 'assembled_dataset_3d.npz'))
X, y = d['X'], d['y']

# Load trained 3D model
model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'yogisense_model_3d'))

# Split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Pose labels
POSE_LABELS = [
    'warrior_ii','tree_pose','downward_dog','cobra',
    'mountain','chakrasana','halasana','ardha_matsyendrasana'
]

# --- Confusion Matrix ---
cm = confusion_matrix(y_val, y_pred, labels=range(len(POSE_LABELS)))

fig, ax = plt.subplots(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=POSE_LABELS)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, ax=ax)
plt.title("Confusion Matrix – YogiSense 3D Model")
plt.show()

# --- Classification Report ---
report = classification_report(y_val, y_pred, target_names=POSE_LABELS, digits=3)
print("Classification Report – YogiSense 3D Model")
print(report)
