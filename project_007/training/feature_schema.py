"""
PROJECT 007 — P4.0 Feature Schema
Defines the feature vector structure for ML training.
"""

# Feature groups and their columns
MOTION_FEATURES = [
    "arm_velocity",
    "body_displacement",
    "fall_score",
    "fall_score_delta",
    "uncertainty",
]

TRACKING_FEATURES = [
    "track_count",
    "nearest_person_distance",
    "overlap_duration",
]

SCENE_FEATURES = [
    "optical_flow_magnitude",
    "optical_flow_instability",
    "frame_brightness",
    "occlusion_ratio",
    "scene_stability",
]

RISK_FEATURES = [
    "active_rules_count",
    "rule_confidence_sum",
    "current_risk_score",
]

PAIRWISE_FEATURES = [
    "min_pair_distance",
    "max_approach_velocity",
]

# Temporal window features (aggregated over N frames)
TEMPORAL_FEATURES = [
    "arm_velocity_mean",
    "arm_velocity_max",
    "arm_velocity_std",
    "body_displacement_mean",
    "body_displacement_max",
    "fall_score_mean",
    "fall_score_max",
    "risk_score_mean",
    "risk_score_max",
    "rule_count_sum",
]

# All features combined (ordering matters for the model)
ALL_FEATURES = (
    MOTION_FEATURES
    + TRACKING_FEATURES
    + SCENE_FEATURES
    + RISK_FEATURES
    + PAIRWISE_FEATURES
    + TEMPORAL_FEATURES
)

# Class mapping
CLASS_LABELS = {
    0: "normal",
    1: "camera_tamper",
    2: "intrusion",
    3: "interaction",
    4: "occlusion",
}

LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

# Annotation label to class mapping
ANNOTATION_TO_CLASS = {
    "normal": 0,
    "camera_rush": 2,           # intrusion
    "proximity_intrusion": 2,   # intrusion
    "camera_shake": 1,          # camera_tamper
    "lens_occlusion": 4,        # occlusion
    "high_energy_interaction": 3,  # interaction
}
