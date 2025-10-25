import glob, os
import numpy as np
from cellb import body_normalize_3d, compute_features_3d

# Path to dataset folder
DATA_DIR = './yogi_dataset'
if not os.path.exists(DATA_DIR):
    print("⚠️ Dataset folder not found!")
    print("Please download the Yoga-82 dataset from:")
    print("https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset")
    print("Extract it so images are under:", DATA_DIR)
    exit()

# Process all landmark .npz files
all_X_landmarks, all_X_feats, all_y = [], [], []
files = glob.glob(os.path.join(DATA_DIR, '*.npz'))

if not files:
    print("⚠️ No .npz files found. Run landmark extraction first!")
    exit()

for f in files:
    d = np.load(f)
    X, y = d['X'], d['y']  # assumes stored normalized landmarks
    for i in range(len(X)):
        raw_pts = X[i].reshape((33, 3))  # adjust if needed
        body_norm = body_normalize_3d(raw_pts)  # normalization function
        feats_dict, feats_vec = compute_features_3d(raw_pts, body_norm)  # from feature_extraction.py
        all_X_landmarks.append(body_norm)
        all_X_feats.append(feats_vec)
        all_y.append(y[i])

if all_y:
    X_land = np.vstack(all_X_landmarks)
    X_feat = np.vstack(all_X_feats)
    X_combined = np.hstack([X_land, X_feat])
    y_arr = np.array(all_y)
    print("✅ 3D dataset shape:", X_combined.shape, y_arr.shape)
    np.savez_compressed(os.path.join(DATA_DIR, 'assembled_dataset_3d.npz'), X=X_combined, y=y_arr)
    print("Dataset saved at:", os.path.join(DATA_DIR, 'assembled_dataset_3d.npz'))
else:
    print("⚠️ No dataset found after processing.")
