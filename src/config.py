"""Central settings: paths, image size, class list, thresholds, retrain rules.

Everything other modules need to agree on lives here. Do not hard-code these
values anywhere else. (The GVLiD metadata file is NOT used by any script.)
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "split"
FIELD_TEST_DIR = DATA_DIR / "field_test"
OUT_OF_SCOPE_DIR = FIELD_TEST_DIR / "out_of_scope"   # mildew photos, never trained on
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_PENDING_DIR = FEEDBACK_DIR / "pending"
FEEDBACK_APPROVED_DIR = FEEDBACK_DIR / "approved"    # approved/<class>/ created by code
FEEDBACK_REJECTED_DIR = FEEDBACK_DIR / "rejected"
MODELS_DIR = ROOT / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
PRODUCTION_DIR = MODELS_DIR / "production"
REPORTS_DIR = ROOT / "reports"
REVIEW_DIR = REPORTS_DIR / "review"
LOGS_DIR = ROOT / "logs"
CLASS_NAMES_FILE = VERSIONS_DIR / "class_names.json"

# ------------------------------------------------------------------ basics
SEED = 42
IMG_SIZE = (224, 224)            # (height, width) fed to the model
IMG_CHANNELS = 3
MIN_IMAGE_PX = 100               # images with a shorter side below this are removed
STORE_MAX_SIDE = 512             # cleaned images are stored with longest side <= 512
JPEG_QUALITY = 95
RAW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
GRAYSCALE_MODES = ("1", "L", "LA")   # treated as an oddity for the 4 grape classes

# ----------------------------------------------------------------- classes
# Sorted alphabetically. This is the model output order. The app must load the
# order from class_names.json (written at training), never from this list.
CLASS_NAMES = [
    "Black_Rot",
    "Esca",
    "Healthy_Leaves",
    "Leaf_Blight",
    "Not_a_Grape_Leaf",
]
assert CLASS_NAMES == sorted(CLASS_NAMES), "CLASS_NAMES must stay sorted"
NUM_CLASSES = len(CLASS_NAMES)
NOT_LEAF_CLASS = "Not_a_Grape_Leaf"
HEALTHY_CLASS = "Healthy_Leaves"
DISEASE_CLASSES = ["Black_Rot", "Esca", "Leaf_Blight"]

# model label (folder name) -> name shown in the app
DISPLAY_NAMES = {
    "Black_Rot": "Black Rot",
    "Esca": "Esca",
    "Healthy_Leaves": "Healthy Leaves",
    "Leaf_Blight": "Leaf Blight",
    "Not_a_Grape_Leaf": "Not a Grape Leaf",
}

# model label -> key in the advice map (src/pesticide_map.py). None = no advice.
ADVICE_KEYS = {
    "Black_Rot": "Black_Rot",
    "Esca": "Esca",
    "Healthy_Leaves": "Healthy",
    "Leaf_Blight": "Leaf_Blight",
    "Not_a_Grape_Leaf": None,
}

UNSUPPORTED_NOTICE = (
    "Downy Mildew and Powdery Mildew are not supported. A leaf with these "
    "diseases may be misclassified, including as Healthy."
)

# ------------------------------------------------------------------- split
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP_MAX_HAMMING = 3         # dhash distance (of 64 bits) counted as near-duplicate
NEAR_DUP_KEEP = 3                # keep at most this many images per near-duplicate cluster
FLAT_IMAGE_STD = 2.0             # nearly blank images are not near-dup hashed

# ---------------------------------------------------------------- training
BATCH_SIZE = 32
DROPOUT = 0.4
LABEL_SMOOTHING = 0.05
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3
PHASE1_PATIENCE = 4
PHASE2_EPOCHS = 25
PHASE2_LR = 1e-5
PHASE2_PATIENCE = 6
FINE_TUNE_LAYERS = 40            # top layers of the base model unfrozen in phase 2
SANITY_TRAIN_IMAGES_PER_CLASS = 4   # 4 x 5 classes = 20 train images
SANITY_MIN_TRAIN_ACC = 0.95
SANITY_MAX_SINGLE_CLASS_SHARE = 0.50  # one class may not take more than half of val predictions

# ------------------------------------------------- inference thresholds (doc 04)
# Starting values. Tune on validation + field_test (never test) in step 7.
THRESHOLDS = {
    "not_a_leaf_min_conf": 0.60,
    "confident_min_conf": 0.85,
    "confident_min_margin": 0.25,
    "healthy_min_conf": 0.90,
    "low_conf_below": 0.60,
}
MAX_UPLOAD_MB = 10
LATENCY_TARGET_MS = 500

# -------------------------------------------------------- evaluation targets
TARGETS = {
    "test_accuracy": 0.90,
    "class_recall": 0.85,
    "not_leaf_recall": 0.90,
    "field_accuracy_aim": 0.75,
}

# -------------------------------------------------------------- retraining
RETRAIN = {
    "min_total_approved": 100,
    "min_per_class": 15,
    "max_days_between": 30,
    "feedback_cap_fraction": 0.30,
    "feedback_holdout_fraction": 0.15,
    "learning_rate": 1e-5,
}
PROMOTION = {
    "min_test_macro_f1_gain": 0.0,        # candidate >= production
    "min_field_gain_points": 0.005,       # +0.5 pt or more on field_test
    "max_class_recall_drop": 0.02,        # no class recall may drop > 2 points
    "not_leaf_false_accept_no_worse": True,
    "max_latency_ms": LATENCY_TARGET_MS,
}


def snapshot() -> dict:
    """JSON-safe copy of all settings, saved next to every trained model."""
    out = {}
    for key, value in globals().items():
        if key.isupper():
            out[key] = str(value) if isinstance(value, Path) else value
    return out