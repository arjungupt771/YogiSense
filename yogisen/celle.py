# #Inference: classification + features + rule-based correction + calibration use
# # Live inference loop with feature extraction and per-pose rule-based feedback
# model_dir = '/mnt/data/yogisense_cnn_model_enhanced' if os.path.exists('/mnt/data') else './yogisense_cnn_model_enhanced'
# clf = None
# if os.path.exists(model_dir):
#     import tensorflow as tf
#     try:
#         clf = tf.keras.models.load_model(model_dir)
#         print("Loaded enhanced classifier")
#     except Exception as e:
#         print("Model load failed:", e)

# # load calibration
# calib = load_calibration()  # dict: pose_label -> mean_feat_vec

# # pose labels you used during data collection (must match training)
# POSE_LABELS = ['warrior_ii', 'tree_pose', 'downward_dog', 'cobra', 'mountain', 'chakrasana', 'halasana', 'ardha_matsyendrasana']  # adjust to match your dataset

# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
#     prev_time = time.time()
#     buffer_feat = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(img)
#         raw_pts, flat_raw = mp_to_landmarks(results)
#         feedback = []
#         label_text = '---'
#         if raw_pts is not None:
#             body_norm_flat = normalize_landmarks_body(flat_raw)
#             feats_dict, feats_vec = compute_features(raw_pts, body_norm_flat)
#             # smoothing
#             buffer_feat.append(np.hstack([body_norm_flat, feats_vec]))
#             if len(buffer_feat) > 5:
#                 buffer_feat.pop(0)
#             input_vec = np.mean(np.stack(buffer_feat), axis=0).reshape(1,-1)
#             # classify if model available
#             if clf is not None:
#                 probs = clf.predict(input_vec, verbose=0)[0]
#                 idx = int(np.argmax(probs))
#                 label_text = f'{POSE_LABELS[idx]} ({probs[idx]:.2f})'
#                 predicted_pose = POSE_LABELS[idx]
#             else:
#                 predicted_pose = 'unknown'
#             # compute feature dict again using mean smoothing (map location of feats)
#             # (we have feats_dict from the immediate frame; for stability, use averaged features extracted by splitting input_vec)
#             # Format: input_vec = [33*3 landmarks | feat_vec] -> separate
#             num_land = 33*3
#             avg_feats_vec = input_vec[0, num_land:]
#             # make a quick feats_dict_avg mapping by reusing order from compute_features
#             keys = ['left_knee_angle','right_knee_angle','left_elbow_angle','right_elbow_angle','spine_angle',
#                     'shoulder_width','hip_width','shoulder_hip_dist','mid_sh_to_ankle','wrist_to_ankle_avg',
#                     'ankle_over_head','torso_twist_angle','backbend_amount']
#             feats_dict_avg = {}
#             for i,k in enumerate(keys):
#                 feats_dict_avg[k] = float(avg_feats_vec[i])
#             # calibration vector for this pose if exists
#             calib_vec = None
#             if predicted_pose in calib:
#                 calib_vec = np.array(calib[predicted_pose], dtype=np.float32)
#             # get rule-based feedback
#             pose_fb = pose_rule_feedback(predicted_pose, feats_dict_avg, calib=calib_vec)
#             feedback.extend(pose_fb)
#         # draw overlay
#         h,w = frame.shape[:2]
#         cv2.putText(frame, 'Pose: ' + label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
#         y0 = 60
#         for i,f in enumerate(feedback[:6]):  # limit to 6 lines
#             cv2.putText(frame, f, (10, y0 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#         # fps
#         curr_time = time.time()
#         fps = 1.0 / (curr_time - prev_time + 1e-8)
#         prev_time = curr_time
#         cv2.putText(frame, f'FPS: {int(fps)}', (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#         cv2.imshow('YogiSense Enhanced Live', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()


# Inference loop with 3D-aware features
import cv2,time,tensorflow as tf
import os
from database import DATA_DIR
from cellb import mp_to_landmarks3
from cellb import body_normalize_3d
from cellb import compute_features_3d
import mediapipe as mp_pose
import numpy as np 


clf = tf.keras.models.load_model(os.path.join(DATA_DIR,'yogisense_model_3d'))
POSE_LABELS = ['warrior_ii','tree_pose','downward_dog','cobra','mountain','chakrasana','halasana','ardha_matsyendrasana']

cap = cv2.VideoCapture(0)
with mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5) as pose:
    prev_time=time.time()
    buffer=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose.process(img)
        raw_pts=mp_to_landmarks3(res)
        feedback=[]
        label_text='---'
        if raw_pts is not None:
            body_norm=body_normalize_3d(raw_pts)
            feats_dict,feats_vec=compute_features_3d(raw_pts,body_norm)
            buffer.append(np.hstack([body_norm,feats_vec]))
            if len(buffer)>5: buffer.pop(0)
            inp=np.mean(np.stack(buffer),axis=0).reshape(1,-1)
            probs=clf.predict(inp,verbose=0)[0]
            idx=int(np.argmax(probs))
            label_text=f'{POSE_LABELS[idx]} ({probs[idx]:.2f})'
        cv2.putText(frame,'Pose: '+label_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        fps=1.0/(time.time()-prev_time+1e-8);prev_time=time.time()
        cv2.putText(frame,f'FPS:{int(fps)}',(frame.shape[1]-120,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow('YogiSense 3D',frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release();cv2.destroyAllWindows()
