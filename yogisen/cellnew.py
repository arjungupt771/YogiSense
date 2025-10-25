# --- 3D Rule Engine for advanced poses ---

DEFAULT_RULES_3D = {
    'chakrasana': {
        'elbow_min': 150,      # arms should be near straight
        'knee_min': 160,       # legs straight
        'backbend_min': 25.0,  # spine arch at least this much
        'wrist_to_ankle_max': 1.2,  # wrists should be close to ankles (normalized units)
    },
    'halasana': {
        'ankle_over_head': True, # feet should cross head line
        'knee_min': 160,         # legs straight
        'shoulder_hip_max': 0.8, # shoulders tucked under (hips not too far)
    },
    'ardha_matsyendrasana': {
        'torso_twist_min': 20.0, # torso twist angle
        'knee_min': 140,         # ensure base leg is not collapsing
    }
}

def pose_rule_feedback_3d(pose_label, feats, rules=DEFAULT_RULES_3D):
    """
    feats: dict from compute_features_3d
    rules: thresholds
    returns: list of corrections
    """
    fb = []
    r = rules.get(pose_label, {})
    
    # Chakrasana (Wheel Pose)
    if pose_label == 'chakrasana':
        if feats.get('left_elbow',180) < r['elbow_min'] or feats.get('right_elbow',180) < r['elbow_min']:
            fb.append("Straighten your arms more.")
        if feats.get('left_knee',180) < r['knee_min'] or feats.get('right_knee',180) < r['knee_min']:
            fb.append("Straighten your legs.")
        if feats.get('backbend',0) < r['backbend_min']:
            fb.append("Arch your back more, open the chest.")
        if feats.get('wrist_to_ankle',99) > r['wrist_to_ankle_max']:
            fb.append("Bring your hands closer to your feet, press firmly into the mat.")
    
    # Halasana (Plough Pose)
    elif pose_label == 'halasana':
        if r['ankle_over_head'] and feats.get('ankle_over_head',0) < 0.5:
            fb.append("Bring your feet further over the head until they pass the head line.")
        if feats.get('left_knee',180) < r['knee_min'] or feats.get('right_knee',180) < r['knee_min']:
            fb.append("Straighten your legs fully.")
        if feats.get('shoulder_width',1) > r['shoulder_hip_max']:
            fb.append("Roll shoulders under the back for better support.")
    
    # Ardha Matsyendrasana (Seated Twist)
    elif pose_label == 'ardha_matsyendrasana':
        if feats.get('torso_twist',0) < r['torso_twist_min']:
            fb.append("Rotate your torso more to deepen the twist.")
        if feats.get('left_knee',180) < r['knee_min'] or feats.get('right_knee',180) < r['knee_min']:
            fb.append("Keep your base leg active and straight.")
    
    # Generic fallback
    else:
        if feats.get('left_elbow',180) < 150:
            fb.append("Extend left arm more.")
        if feats.get('right_elbow',180) < 150:
            fb.append("Extend right arm more.")
    
    return fb
