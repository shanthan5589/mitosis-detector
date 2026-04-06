"""
STEP 2 OF 2-STAGE PIPELINE: Train YOLOv8 (Stage 1 detector).

What this script does
---------------------
Trains a YOLOv8s object detector on the tile images prepared by prepare_yolo_data.py.
The detector's job is NOT to be perfectly precise — it is to have HIGH RECALL:
catch every real mitotic figure, even if it also flags some non-mitotic cells.
Stage 2 will filter out those false positives.

This is why YOLO_CONF_THRESH in config.py is set to 0.10 (very low).
We are intentionally biased toward recall over precision at this stage.

Prerequisites
-------------
    pip install ultralytics

    Run prepare_yolo_data.py first to generate yolo_data/.

Usage
-----
    python stage1_yolo/train_yolo.py
    (run from the repo root: D:/Github/mitosis-detector)

Output
------
    models/yolo/weights/best.pt   ← best checkpoint (use this)
    models/yolo/weights/last.pt   ← last epoch checkpoint
    models/yolo/results.csv       ← per-epoch metrics
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    YOLO_DATA_DIR,
    MODELS_DIR,
    YOLO_MODEL,
    YOLO_IMG_SIZE,
    YOLO_EPOCHS,
    YOLO_BATCH,
)

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics is not installed. Run: pip install ultralytics"
    )


def main() -> None:

    device = 0 if torch.cuda.is_available() else "cpu"

    data_yaml = YOLO_DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Run prepare_yolo_data.py first."
        )

    print(f"Loading base model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)   # downloads pretrained weights on first run

    print(f"Starting YOLO training — {YOLO_EPOCHS} epochs, batch {YOLO_BATCH}, img {YOLO_IMG_SIZE}")
    model.train(
        data=str(data_yaml),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
        project=str(MODELS_DIR),
        device=device,
        workers=0, 
        name="yolo",
        exist_ok=True,          # overwrite previous run if re-running

        # ── Augmentation tweaks for histology tiles ───────────────────────
        # Mosaic stitches 4 random images together — bad for tissue slides
        # because it creates unnatural tissue boundaries.
        mosaic=0.0,

        # Mild colour augmentation is fine — staining variation is real in WSIs
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,

        # Standard geometric augmentations
        flipud=0.5,
        fliplr=0.5,
        degrees=90,             # mitoses can be in any orientation

        # No mixup or copy-paste — they distort cell morphology
        mixup=0.0,
        copy_paste=0.0,

        # Use automatic mixed precision (FP16) for faster training on CUDA
        amp=True,

        # Save best model based on mAP50
        save=True,
        save_period=-1,         # only save best + last (not every epoch)
    )

    best_weights = MODELS_DIR / "yolo" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights saved to:\n  {best_weights}")
    print("\nNext step: run prepare_stage2_data.py")


if __name__ == "__main__":
    main()
