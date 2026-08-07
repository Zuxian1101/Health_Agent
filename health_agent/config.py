"""Central configuration for the Health Agent.

Every tunable number lives here. No other module should hard-code a value
a user might reasonably want to change.
"""

from pathlib import Path

# --------------------------------------------------------------- camera --
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ----------------------------------------------------------- face model --
# MediaPipe Tasks FaceLandmarker bundle: 478 landmarks + 52 blendshapes
# from a single inference pass. Run `python download_model.py` to fetch it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_FACE_TRACKING_CONFIDENCE = 0.5

# ----------------------------------------------------------------- rPPG --
HR_WINDOW_SECONDS = 10.0
HR_MIN_WINDOW_SECONDS = 6.0        # refuse to estimate on a shorter window

# 0.75-3.0 Hz == 45-180 BPM. The original 4.0 Hz (240 BPM) ceiling admitted
# a lot of motion noise for no realistic benefit.
LOW_HR_HZ = 0.75
HIGH_HR_HZ = 3.0

RESAMPLE_FPS = 30.0                # uniform grid irregular frames map onto
POS_WINDOW_SECONDS = 1.6           # per Wang et al. 2017
FFT_ZERO_PAD_FACTOR = 8            # finer bins -> finer BPM resolution
HR_SMOOTHING_ALPHA = 0.3           # EMA weight on each new BPM estimate
MIN_HR_SNR_DB = 2.0                # below this the estimate is "unreliable"
BANDPASS_ORDER = 3

# -------------------------------------------------------------- fatigue --
EAR_CALIBRATION_SECONDS = 5.0      # per-person open-eye baseline
EYE_CLOSED_RATIO = 0.75            # closed if EAR < baseline * this
BLINK_MIN_FRAMES = 2               # debounce 1-frame EAR dropouts
PERCLOS_WINDOW_SECONDS = 60.0
BLINK_RATE_WINDOW_SECONDS = 60.0

# PERCLOS = fraction of time the eyes are closed. The standard drowsiness
# measure (Wierwille et al.); anchors below define the 0-100 score.
PERCLOS_ALERT = 0.15               # maps to score 50
PERCLOS_DROWSY = 0.30              # maps to score 100
BLINK_RATE_NORMAL = (8.0, 21.0)    # blinks/min; outside this is a signal too
PERCLOS_WEIGHT = 0.8
BLINK_WEIGHT = 0.2

FATIGUE_THRESHOLD = 70.0           # 0-100 score at which the agent speaks up

# ----------------------------------------------------------- expression --
EXPRESSION_MIN_SCORE = 0.25        # below this we report "neutral"
NEGATIVE_EXPRESSIONS = ("sad", "angry", "fearful", "disgusted")

# ------------------------------------------------------------- reminders --
WATER_REMIND_INTERVAL_MIN = 60
REST_REMIND_INTERVAL_MIN = 90
POSTURE_REMIND_INTERVAL_MIN = 30
STRESS_REMIND_INTERVAL_MIN = 15
HR_REMIND_INTERVAL_MIN = 10

ADVICE_HYSTERESIS_SECONDS = 10.0   # condition must hold this long before firing
ADVICE_MAX_SHOWN = 2

HR_HIGH_BPM = 100.0
HR_LOW_BPM = 50.0

# --------------------------------------------------------------- display --
WINDOW_NAME = "AI Health Agent"
DISCLAIMER = "Not a medical device - informational use only"
