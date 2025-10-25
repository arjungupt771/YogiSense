import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
from database import DATA_DIR

# Load dataset
d = np.load(os.path.join(DATA_DIR, 'assembled_dataset_3d.npz'))
X, y = d['X'], d['y']

# Load trained model
model = tf.keras.models.load_model(os.path.join(DATA_DIR, 'yogisense_model_3d'))

# Train/val split (same as training setup)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
acc = accuracy_score(y_val, y_pred)

# Classification report (precision, recall, f1)
POSE_LABELS = [
    'warrior_ii','tree_pose','downward_dog','cobra',
    'mountain','chakrasana','halasana','ardha_matsyendrasana'
]
report = classification_report(y_val, y_pred, target_names=POSE_LABELS, output_dict=True)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, labels=range(len(POSE_LABELS)))

# Compute FPR (False Positive Rate) and TPR (Recall = True Positive Rate)
fpr_dict, tpr_dict = {}, {}
for i, label in enumerate(POSE_LABELS):
    TP = cm[i,i]
    FN = np.sum(cm[i,:]) - TP
    FP = np.sum(cm[:,i]) - TP
    TN = np.sum(cm) - (TP+FP+FN)
    fpr_dict[label] = FP / (FP+TN) if (FP+TN)>0 else 0.0
    tpr_dict[label] = TP / (TP+FN) if (TP+FN)>0 else 0.0

# Print results
print(f"\n✅ Overall Accuracy: {acc:.3f}\n")

print("Per-Class Metrics (Precision, Recall/TPR, FPR):\n")
for label in POSE_LABELS:
    prec = report[label]['precision']
    rec = report[label]['recall']   # same as TPR
    fpr = fpr_dict[label]
    print(f"{label:20s} | Precision: {prec:.3f} | Recall/TPR: {rec:.3f} | FPR: {fpr:.3f}")

print("\n--- Macro Averages ---")
print(f"Precision: {report['macro avg']['precision']:.3f}")
print(f"Recall/TPR: {report['macro avg']['recall']:.3f}")
print(f"Accuracy : {acc:.3f}")
