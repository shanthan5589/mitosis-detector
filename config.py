from pathlib import Path

# ---------------------------------------------------------------------------
# Raw data locations (your DICOM files + CSVs)
# ---------------------------------------------------------------------------
TRAIN_DATA_DIR = Path("D:/Mitosis WSI CCMCT/training_data")
TEST_DATA_DIR  = Path("D:/Mitosis WSI CCMCT/testing_data")

# ---------------------------------------------------------------------------
# Output locations (all generated files go here, alongside this config)
# ---------------------------------------------------------------------------
TWO_STAGE_DIR   = Path(__file__).parent
YOLO_DATA_DIR   = TWO_STAGE_DIR / "yolo_data"     # YOLO tiles + labels
STAGE2_DATA_DIR = TWO_STAGE_DIR / "stage2_data"   # crops for Stage 2 classifier
MODELS_DIR      = TWO_STAGE_DIR / "models"         # saved weights

# ---------------------------------------------------------------------------
# Slide splits — same as original experiment so F1 is comparable
# ---------------------------------------------------------------------------
TRAIN_SLIDES = [4, 12, 13, 15, 17, 19, 21, 22, 24, 25, 26, 28, 29, 32, 34, 35, 36]
VAL_SLIDES   = [7, 8, 14, 23]
# Held-out slides for final pipeline evaluation — must NOT overlap VAL_SLIDES.
# Populate with slide IDs from TEST_DATA_DIR before running evaluate.py --split test.
TEST_SLIDES: list[int] = [1, 2, 3, 6, 9, 11, 18, 20, 27, 30, 31]

# ---------------------------------------------------------------------------
# Stage 1 — YOLO data preparation
# ---------------------------------------------------------------------------
# Fixed bounding box size (pixels) drawn around each mitotic annotation point.
# Annotations are just (x, y) center points, not boxes, so we fix the size.
BOX_SIZE = 64

# How many unannotated background tiles to sample per slide.
# These teach YOLO what "no mitosis" looks like.
MAX_BG_TILES_PER_SLIDE = 30

# Max worker processes for Stage 1 data preparation.
# Lower values reduce memory pressure on machines with limited RAM.
PREPARE_WORKERS = 4

# Annotation class IDs (from the dataset's Classes table)
MITOTIC_CLASS = 2  # the only positive class

# ---------------------------------------------------------------------------
# Stage 1 — YOLO training
# ---------------------------------------------------------------------------
YOLO_MODEL      = "yolov8s.pt"  # YOLOv8 small — good balance of speed vs accuracy
YOLO_IMG_SIZE   = 640           # YOLO internally resizes every tile to this
YOLO_EPOCHS     = 50
YOLO_BATCH      = 16
YOLO_CONF_THRESH = 0.10         # LOW threshold at inference — we want high recall here,
                                 # Stage 2 will filter out the false positives

# ---------------------------------------------------------------------------
# Stage 2 — classifier data preparation
# ---------------------------------------------------------------------------
# When we crop a YOLO detection from the tile, how big should the crop be?
# Slightly larger than BOX_SIZE to give the classifier some surrounding context.
STAGE2_CROP_SIZE = 96

# ---------------------------------------------------------------------------
# Stage 2 — EfficientNet classifier training
# ---------------------------------------------------------------------------
STAGE2_IMG_SIZE  = 224    # resize every crop to this before feeding EfficientNet
STAGE2_EPOCHS    = 30
STAGE2_BATCH     = 128
STAGE2_LR        = 1e-4
STAGE2_CONF_THRESH = 0.50 # final decision threshold for Stage 2 output
