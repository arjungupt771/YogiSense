# Enhanced helpers + feature extraction + calibration + rules
# Helpers: enhanced landmark handling, engineered features, calibration and rule engine
import cv2, os, math, time, glob
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

mp_pose = mp.solutions.pose

# ---------- basic conversions ----------
def mp_to_landmarks3(results):
    """Return raw 33x3 np array (x,y,z) from MediaPipe results (z is relative)."""
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # normalized coords; z is relative depth
    return pts

def body_normalize_3d(raw_pts):
    """
    Convert raw (33,3) normalized coords into a body-centered, scale-invariant 3D coordinate system.
    Uses mid-hip as origin and torso length for scale. Returns flattened (33*3).
    """
    pts = raw_pts.copy()
    left_hip = pts[23]; right_hip = pts[24]
    mid_hip = (left_hip + right_hip) / 2.0
    left_sh = pts[11]; right_sh = pts[12]
    mid_sh = (left_sh + right_sh) / 2.0
    # Use 3D torso length for scale (distance between mid_sh and mid_hip)
    torso_len = np.linalg.norm(mid_sh - mid_hip)
    if torso_len < 1e-6:
        torso_len = 1.0
    pts_centered = pts - mid_hip
    pts_scaled = pts_centered / torso_len
    return pts_scaled.flatten()

# ---------- geometric helpers ----------
def dist(a,b):
    return float(np.linalg.norm(a - b))

def angle_3d(a,b,c):
    # angle at b formed by a-b-c (works in 3D)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def vec(a,b):
    return b - a

def compute_features_3d(raw_pts, body_norm_flat=None):
    """
    raw_pts: (33,3) image-space normalized coords (x,y,z)
    body_norm_flat: flattened body-centered 3D pts (33*3) produced by body_normalize_3d
    Returns: feats_dict, feat_vector (numpy)
    """
    if raw_pts is None:
        return None, None
    if body_norm_flat is None:
        body_norm_flat = body_normalize_3d(raw_pts)
    pts3 = body_norm_flat.reshape((33,3))
    # indices
    L_SH, R_SH = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WR, R_WR = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    NOSE = 0
    feats = {}
    # 3D angles
    try:
        feats['left_knee_angle'] = angle_3d(pts3[L_HIP], pts3[L_KNEE], pts3[L_ANK])
        feats['right_knee_angle'] = angle_3d(pts3[R_HIP], pts3[R_KNEE], pts3[R_ANK])
        feats['left_elbow_angle'] = angle_3d(pts3[L_SH], pts3[L_ELB], pts3[L_WR])
        feats['right_elbow_angle'] = angle_3d(pts3[R_SH], pts3[R_ELB], pts3[R_WR])
        mid_sh = (pts3[L_SH] + pts3[R_SH]) / 2.0
        mid_hip = (pts3[L_HIP] + pts3[R_HIP]) / 2.0
        mid_knee = (pts3[L_KNEE] + pts3[R_KNEE]) / 2.0
        feats['spine_angle'] = angle_3d(mid_sh, mid_hip, mid_knee)  # 180 straight
        feats['backbend_amount'] = 180.0 - feats['spine_angle']
    except Exception:
        feats['left_knee_angle'] = feats['right_knee_angle'] = None
        feats['left_elbow_angle'] = feats['right_elbow_angle'] = None
        feats['spine_angle'] = feats['backbend_amount'] = None
    # 3D distances (normalized by torso length implicitly since pts3 are scaled)
    feats['shoulder_width_3d'] = float(np.linalg.norm(pts3[L_SH] - pts3[R_SH]))
    feats['hip_width_3d'] = float(np.linalg.norm(pts3[L_HIP] - pts3[R_HIP]))
    feats['wrist_to_ankle_3d'] = float((np.linalg.norm(pts3[L_WR]-pts3[L_ANK]) + np.linalg.norm(pts3[R_WR]-pts3[R_ANK]))/2.0)
    # ankle above head in image space: check raw image coords (use raw_pts y: smaller -> higher)
    feats['ankle_above_head'] = float((raw_pts[L_ANK,1] < raw_pts[NOSE,1]) or (raw_pts[R_ANK,1] < raw_pts[NOSE,1]))
    # torso twist: angle between shoulder axis and hip axis in 3D
    sh_axis = pts3[R_SH] - pts3[L_SH]
    hip_axis = pts3[R_HIP] - pts3[L_HIP]
    try:
        dot = np.dot(sh_axis, hip_axis)
        norms = (np.linalg.norm(sh_axis) * np.linalg.norm(hip_axis) + 1e-8)
        feats['torso_twist_3d'] = float(math.degrees(math.acos(np.clip(dot / norms, -1.0, 1.0))))
    except Exception:
        feats['torso_twist_3d'] = None
    # build vector in consistent order
    vec = np.array([
        feats.get('left_knee_angle') or -1.0,
        feats.get('right_knee_angle') or -1.0,
        feats.get('left_elbow_angle') or -1.0,
        feats.get('right_elbow_angle') or -1.0,
        feats.get('spine_angle') or -1.0,
        feats.get('backbend_amount') or -1.0,
        feats['shoulder_width_3d'],
        feats['hip_width_3d'],
        feats['wrist_to_ankle_3d'],
        feats['ankle_above_head'],
        feats.get('torso_twist_3d') or -1.0
    ], dtype=np.float32)
    return feats, vec

# ---------- Calibration (Adaptive Pose Baseline) ----------
CALIB_PATH = 'yogi_calib.json'

def calibrate_pose(label, n_samples=30, show=False):
    """
    Collects n_samples frames while user holds an (attempted) best pose and saves mean features.
    Use this to get a personalized baseline for each pose.
    """
    print("Start calibration for", label)
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        count = 0
        feat_list = []
        while count < n_samples:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img)
            raw_pts, flat_raw = mp_to_landmarks3(results)
            if raw_pts is not None:
                body_norm = body_normalize_3d(flat_raw)
                feats, vec = compute_features_3d(raw_pts, body_norm)
                feat_list.append(vec)
                count += 1
            if show:
                cv2.putText(frame, f'Calibrating {label} {count}/{n_samples}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                cv2.imshow('Calibrate', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

    if len(feat_list) == 0:
        print("No samples collected.")
        return None
    mean_vec = np.mean(np.stack(feat_list), axis=0).tolist()

    # store in JSON
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH,'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[label] = mean_vec
    with open(CALIB_PATH,'w') as f:
        json.dump(data, f, indent=2)
    print("Calibration saved for", label)
    return mean_vec

def load_calibration():
    if not os.path.exists(CALIB_PATH):
        return {}
    with open(CALIB_PATH,'r') as f:
        return json.load(f)

# ---------- Per-pose rule engine ----------
# NOTE: thresholds below are starting heuristics. Prefer per-user calibration adjustments.
DEFAULT_RULES = {
    'chakrasana': {
        'left_elbow_min': 150,    # degrees (arm should be near straight)
        'right_elbow_min': 150,
        'left_knee_min': 160,
        'right_knee_min': 160,
        'backbend_min': 25.0,     # backbend_amount = 180 - spine_angle
        'wrist_to_ankle_max': 0.9 # normalized image distance (lower means wrists near ankles/ground)
    },
    'halasana': {
        'ankle_above_head': True,
        'left_knee_min': 160,
        'right_knee_min': 160,
        'shoulder_support_max': 0.5 # shoulder_hip_dist small means shoulders tucked under
    },
    'ardha_matsyendrasana': {
        'torso_twist_min': 15.0,   # degrees
        'shoulder_hip_align_max': 30.0  # optional: large difference indicates twist
    }
}

def pose_rule_feedback(pose_label, feats, calib=None, rules=DEFAULT_RULES):
    """
    feats: dict returned by compute_features
    calib: mean baseline vector from calibration (list/np.array) or None
    returns: list of feedback strings (may be empty)
    """
    fb = []
    r = rules.get(pose_label, {})
    # Chakrasana (Wheel)
    if pose_label == 'chakrasana' or pose_label.startswith('chakra') or pose_label == 'wheel':
        # elbows
        if feats.get('left_elbow_angle') is not None and feats['left_elbow_angle'] < r['left_elbow_min']:
            fb.append("Extend your left arm until it's straighter.")
        if feats.get('right_elbow_angle') is not None and feats['right_elbow_angle'] < r['right_elbow_min']:
            fb.append("Extend your right arm until it's straighter.")
        # legs
        if feats.get('left_knee_angle') is not None and feats['left_knee_angle'] < r['left_knee_min']:
            fb.append("Straighten the left leg (knee).")
        if feats.get('right_knee_angle') is not None and feats['right_knee_angle'] < r['right_knee_min']:
            fb.append("Straighten the right leg (knee).")
        # backbend (compare to calibration if present)
        backbend_amount = feats.get('backbend_amount')
        if backbend_amount is not None:
            if calib is not None:
                # require wiggle more than baseline + some delta
                # calib is mean feature vector; find backbend index (12th index in our packing) -> use mapping instead for clarity
                # easier: if user calibration recorded a backbend, require they be near that or exceed it
                pass
            if backbend_amount < r['backbend_min']:
                fb.append("Arch your back more (open the chest).")
        # wrists near ground check
        if feats.get('wrist_to_ankle_avg') is not None and feats['wrist_to_ankle_avg'] > r['wrist_to_ankle_max']:
            fb.append("Bring hands and feet closer (place hands firmly on mat).")
    # Halasana (Plough)
    elif pose_label == 'halasana' or pose_label.startswith('plough'):
        if r.get('ankle_above_head'):
            if not feats.get('ankle_over_head', False):
                fb.append("Lift the legs over the head so the feet pass behind the head.")
        if feats.get('left_knee_angle') is not None and feats['left_knee_angle'] < r['left_knee_min']:
            fb.append("Straighten left leg.")
        if feats.get('right_knee_angle') is not None and feats['right_knee_angle'] < r['right_knee_min']:
            fb.append("Straighten right leg.")
        if feats.get('shoulder_hip_dist') is not None and feats['shoulder_hip_dist'] > r['shoulder_support_max']:
            fb.append("Bring shoulders closer to the mat for support.")
    # Ardha Matsyendrasana (Seated twist)
    elif pose_label in ('ardha_matsyendrasana','ardha','twist','agru','agrunasan'):
        if feats.get('torso_twist_angle') is not None and feats['torso_twist_angle'] < r['torso_twist_min']:
            fb.append("Rotate your torso more to deepen the twist.")
        # optionally, check shoulders vs hips misalignment
        if feats.get('torso_twist_angle') is not None and feats.get('shoulder_hip_dist') is not None:
            pass
    else:
        # generic checks: arms extended, legs straight
        if feats.get('left_elbow_angle') is not None and feats['left_elbow_angle'] < 150:
            fb.append("Extend left arm.")
        if feats.get('right_elbow_angle') is not None and feats['right_elbow_angle'] < 150:
            fb.append("Extend right arm.")
    return fb
