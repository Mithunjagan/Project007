"""
PROJECT 007 — Central Configuration
All thresholds, device settings, and constants live here.
No values are hardcoded elsewhere in the codebase.
"""

# ═══════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FALLBACK_WIDTH = 640
CAMERA_FALLBACK_HEIGHT = 480

# ═══════════════════════════════════════
# YOLO DETECTION
# ═══════════════════════════════════════
YOLO_MODEL = "yolov8n.pt"
YOLO_DEVICE = "cuda"
YOLO_CONF_THRESHOLD = 0.5
YOLO_PERSON_CLASS = 0
YOLO_HALF = True            # FP16 inference on CUDA
YOLO_TRACKER = "bytetrack.yaml"

# ═══════════════════════════════════════
# MEDIAPIPE POSE
# ═══════════════════════════════════════
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_EVERY_N_FRAMES = 2     # Run pose every Nth frame
MAX_POSE_CROP = 256          # Resize larger crops before pose extraction

# ═══════════════════════════════════════
# ROLLING BUFFER
# ═══════════════════════════════════════
BUFFER_SIZE = 16             # deque maxlen per track
STALE_TIMEOUT = 2.0          # seconds before track is considered stale

# ═══════════════════════════════════════
# MOTION THRESHOLDS
# ═══════════════════════════════════════
ARM_VELOCITY_HIGH = 0.15     # Arm velocity threshold for warning
FALL_SCORE_HIGH = 0.6        # Fall score threshold for danger

# ═══════════════════════════════════════
# OVERLAY COLORS (BGR format for OpenCV)
# ═══════════════════════════════════════
COLOR_NORMAL = (0, 255, 0)       # Green — normal state
COLOR_WARNING = (0, 165, 255)    # Orange — high arm velocity
COLOR_DANGER = (0, 0, 255)       # Red — high fall score
COLOR_TEXT_BG = (0, 0, 0)        # Black — text background

# ═══════════════════════════════════════
# FPS
# ═══════════════════════════════════════
TARGET_FPS = 25
FPS_AVERAGE_WINDOW = 30          # Rolling window for FPS averaging

# ═══════════════════════════════════════
# WINDOW
# ═══════════════════════════════════════
WINDOW_TITLE = "PROJECT 007 \u2014 Motion Analysis Pipeline"

# ═══════════════════════════════════════
# STALENESS (P0.5)
# ═══════════════════════════════════════
MAX_RESULT_AGE_MS = 300          # Discard inference older than this

# ═══════════════════════════════════════
# QUEUE BACKPRESSURE (P0.5)
# ═══════════════════════════════════════
POSE_QUEUE_MAX = 8               # Max items in pose crop queue

# ═══════════════════════════════════════
# TELEMETRY (P0.5)
# ═══════════════════════════════════════
TELEMETRY_LOG_DIR = "logs"
TELEMETRY_MAX_FILE_MB = 50
TELEMETRY_FLUSH_INTERVAL = 10    # Flush JSONL buffer every N frames

# ═══════════════════════════════════════
# STRESS TEST (P0.5)
# ═══════════════════════════════════════
ENABLE_STRESS_TEST = False
STRESS_DELAY_MIN = 0.02          # seconds
STRESS_DELAY_MAX = 0.15          # seconds

# ═══════════════════════════════════════
# OVERLAY HEALTH THRESHOLDS (P0.5)
# ═══════════════════════════════════════
FRAME_AGE_WARN_MS = 150
FRAME_AGE_DANGER_MS = 300
QUEUE_WARN_RATIO = 0.5
QUEUE_DANGER_RATIO = 0.8

# ═══════════════════════════════════════
# THREAD HEARTBEAT (P0.5)
# ═══════════════════════════════════════
HEARTBEAT_TIMEOUT_S = 5.0        # Warn if thread silent longer

# ═══════════════════════════════════════
# PROXY RULES THRESHOLDS (P1)
# ═══════════════════════════════════════
# Normalized thresholds (e.g. units of bounding box height / sec)
APPROACH_VELOCITY_THRESHOLD = 2.0
ARM_SWING_THRESHOLD = 3.0
FALL_SCORE_THRESHOLD = 0.6
CONTACT_DISTANCE_THRESHOLD = 1.5
CROWD_DISPERSION_THRESHOLD = 1.5

DIRECTION_THRESHOLD = 0.7        # Dot product for directed arm swing

# ═══════════════════════════════════════
# RULE PERSISTENCE & SCORING (P1)
# ═══════════════════════════════════════
RULE_ACTIVE_MIN_FRAMES = 5
PERSISTENCE_DECAY = 0.85
RISK_DECAY = 0.92
MAX_RISK_SCORE = 1.0
EVENT_COOLDOWN_SECONDS = 10.0

# ═══════════════════════════════════════
# VIDEO RECORDER (P1)
# ═══════════════════════════════════════
SAVE_EVENT_CLIPS = True
PRE_EVENT_BUFFER_SECONDS = 5
POST_EVENT_BUFFER_SECONDS = 5

# ═══════════════════════════════════════
# CAMERA THREAT & TAMPER (P1.5)
# ═══════════════════════════════════════
FLOW_DOWNSCALE_WIDTH = 160
FLOW_EVERY_N_FRAMES = 3

TAMPER_DARK_PIXEL_RATIO = 0.85
TAMPER_SHAKE_MAGNITUDE = 15.0

INTRUSION_AREA_GROWTH_RATE = 1.2
INTRUSION_MAX_OCCUPANCY = 0.60

# ═══════════════════════════════════════
# FAILURE CAPTURE (P2.5)
# ═══════════════════════════════════════
FAILURE_CLIP_ENABLED = True
FAILURE_CLIP_SECONDS_BEFORE = 5
FAILURE_CLIP_SECONDS_AFTER = 5

# ═══════════════════════════════════════
# DATASET & ANNOTATION (P3.0)
# ═══════════════════════════════════════
DATASET_DIR = "dataset"
ANNOTATIONS_DIR = "dataset/annotations"

SCENARIO_CATEGORIES = [
    "normal",
    "camera_tamper",
    "intrusion",
    "interaction",
    "crowded",
    "occlusion",
    "lighting_change",
]

EVENT_LABELS = [
    "normal",
    "camera_rush",
    "proximity_intrusion",
    "camera_shake",
    "lens_occlusion",
    "high_energy_interaction",
]

# ═══════════════════════════════════════
# HYBRID ML LAYER (P4.0)
# ═══════════════════════════════════════
ML_MODEL_PATH = "models/saved/xgboost.pkl"
ML_FUSION_DETERMINISTIC_WEIGHT = 0.70
ML_FUSION_ML_WEIGHT = 0.30
ML_FUSION_ML_CAP = 0.3              # Max ML adjustment (+/-)
ML_TEMPORAL_WINDOW = 16              # Frames for temporal features

# ═══════════════════════════════════════
# DEEP LEARNING VIOLENCE CLASSIFIER (P5.0)
# ═══════════════════════════════════════
DL_ENCODER_MODEL = "mobilenet_v3_small"
DL_ENCODER_DEVICE = "cuda"
DL_CLASSIFIER_PATH = "models/saved/violence_classifier.pt"
DL_TEMPORAL_WINDOW = 16               # Frames in LSTM sliding window
DL_ENCODE_EVERY_N_FRAMES = 2          # Encode every Nth frame (match pose)
DL_FUSION_WEIGHT = 0.50               # Weight for DL prediction in fusion
RULE_FUSION_WEIGHT = 0.30             # Weight for rule-based in fusion
ML_FUSION_WEIGHT_V2 = 0.20            # Weight for XGBoost/RF in fusion

# ═══════════════════════════════════════
# IDENTITY PERSISTENCE & PERSISTENT RE-ID (P5.1)
# ═══════════════════════════════════════
ENABLE_SOFT_REID = True
REID_GHOST_TIMEOUT_S = 3.0            # Keep inactive tracks in memory for 3s
REID_SIMILARITY_THRESHOLD = 0.65      # Similarity required to merge identity
REID_COLOR_HIST_WEIGHT = 0.40         # Weight of color histogram similarity
REID_GEOMETRY_WEIGHT = 0.60           # Weight of spatial proximity + scale similarity

# ═══════════════════════════════════════
# EGOMOTION COMPENSATION (P6.0-A)
# ═══════════════════════════════════════
ENABLE_EGOMOTION = True               # Enable ego-motion compensation layer
EGOMOTION_MAX_FEATURES = 150          # Max background feature points to track
EGOMOTION_MIN_FEATURES = 30           # Redetect when features drop below this
EGOMOTION_RANSAC_THRESHOLD = 3.0      # RANSAC reprojection threshold (pixels)
EGOMOTION_STABILITY_PENALTY_CAP = 0.3 # Max stability reduction from vibration

