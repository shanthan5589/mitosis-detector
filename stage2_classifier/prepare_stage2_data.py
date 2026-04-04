"""
STEP 3 OF 2-STAGE PIPELINE: Build Stage 2 classifier training data.

What this script does
---------------------
Takes the trained YOLO model and runs it on every tile in the train and val splits.
For each tile, it:

  1. Loads YOLO's detections (at a low confidence threshold — see config.YOLO_CONF_THRESH)
  2. Loads the ground-truth mitotic boxes from the corresponding label file
  3. Matches detections to ground-truth using center-distance (same as evaluate.py):
       - Detection center within BOX_SIZE/2 px of a GT point → True Positive  → crop label = 1
       - Detection center outside that radius                 → False Positive → crop label = 0
  4. Saves crops for any GT boxes that YOLO MISSED entirely (False Negatives) → label = 1
     (We don't want Stage 2 to also miss these)

Each crop is STAGE2_CROP_SIZE x STAGE2_CROP_SIZE pixels centered on the detection/GT center,
saved as a PNG file. A CSV manifest is written for each split.

Why this is better than the old approach
-----------------------------------------
The old model trained on all extracted patches. It saw easy negatives and never learned
to distinguish hard cases (things that look like mitoses but aren't).

Stage 2 trains ONLY on:
  - What YOLO correctly found (true positives)
  - What YOLO WRONGLY flagged (hard negatives — exactly the confusing cases)
  - What YOLO missed (to fill in coverage gaps)

This forces the classifier to solve the hard problem, not the easy one.

Prerequisites
-------------
  Run prepare_yolo_data.py and train_yolo.py first.

Usage
-----
    python stage2_classifier/prepare_stage2_data.py
    (run from the repo root: D:/Github/mitosis-detector)

Output
------
    stage2_data/train/crops/  *.png files
    stage2_data/train/train.csv  (columns: path, label)
    stage2_data/val/crops/    *.png files
    stage2_data/val/val.csv
"""

import sys
import csv
import shutil
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    YOLO_DATA_DIR,
    STAGE2_DATA_DIR,
    MODELS_DIR,
    YOLO_CONF_THRESH,
    STAGE2_CROP_SIZE,
    BOX_SIZE,
)

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is not installed. Run: pip install ultralytics")


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def extract_crop(image: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    """
    Extract a square crop of shape (crop_size, crop_size, 3) centered at (cx, cy).
    Coordinates are in pixels relative to the image.
    Regions outside image boundaries are zero-padded.
    """
    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(w, x2);  y2c = min(h, y2)

    crop = image[y1c:y2c, x1c:x2c]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return crop


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split_name: str, yolo_model) -> None:
    img_dir  = YOLO_DATA_DIR  / "images" / split_name
    lbl_dir  = YOLO_DATA_DIR  / "labels" / split_name
    out_dir  = STAGE2_DATA_DIR / split_name
    crop_dir = out_dir / "crops"

    # Clear stale outputs so reruns never train on leftover data.
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{split_name}.csv"
    if csv_path.exists():
        csv_path.unlink()

    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        print(f"  [{split_name}] No images found in {img_dir}")
        return

    manifest_rows = []   # will become the CSV
    crop_idx      = 0
    tp_count = fp_count = fn_count = 0

    for img_path in image_paths:
        # ── Load tile image (BGR from OpenCV) ─────────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        # ── Load ground-truth YOLO labels (normalized → pixel boxes) ──────
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_boxes = []   # list of (x1, y1, x2, y2) in pixels

        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # YOLO label: class cx cy w h (normalized)
                _, cx_n, cy_n, bw_n, bh_n = map(float, parts)
                cx_px = cx_n * img_w;  cy_px = cy_n * img_h
                bw_px = bw_n * img_w;  bh_px = bh_n * img_h
                gt_boxes.append(np.array([
                    cx_px - bw_px / 2,
                    cy_px - bh_px / 2,
                    cx_px + bw_px / 2,
                    cy_px + bh_px / 2,
                ]))

        # ── Run YOLO inference on this tile ───────────────────────────────
        # verbose=False suppresses per-image console spam
        results   = yolo_model(str(img_path), conf=YOLO_CONF_THRESH, verbose=False)
        det_boxes = []   # list of (x1, y1, x2, y2) in pixels

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.xyxy.cpu().numpy():
                det_boxes.append(box[:4])

        # ── Match detections to ground truth using center-distance ───────
        # Uses the same criterion as final evaluation (center distance <= BOX_SIZE/2)
        # so a detection is a TP in training iff it would be a TP at eval time.
        # Global one-to-one assignment: sort all within-threshold pairs by distance
        # ascending (closest first), then greedily assign free pairs.
        match_dist = BOX_SIZE / 2   # 32 px — identical to evaluate.py

        det_centers = np.array([
            [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in det_boxes
        ]) if det_boxes else np.zeros((0, 2))
        gt_centers = np.array([
            [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in gt_boxes
        ]) if gt_boxes else np.zeros((0, 2))

        if det_boxes and gt_boxes:
            dist_matrix = np.linalg.norm(
                det_centers[:, None] - gt_centers[None, :], axis=2
            )   # shape: (num_dets, num_gts)
        else:
            dist_matrix = np.full((len(det_boxes), len(gt_boxes)), np.inf)

        candidate_pairs = sorted(
            (dist_matrix[di, gi], di, gi)
            for di in range(len(det_boxes))
            for gi in range(len(gt_boxes))
            if dist_matrix[di, gi] <= match_dist
        )   # ascending distance — closest pair assigned first

        matched_dets: dict[int, int] = {}   # det_i → gt_i for confirmed TPs
        matched_gts:  set[int]       = set()

        for _dist, det_i, gt_i in candidate_pairs:
            if det_i not in matched_dets and gt_i not in matched_gts:
                matched_dets[det_i] = gt_i
                matched_gts.add(gt_i)

        # ── Write crops for each detection ────────────────────────────────
        for det_i, det_box in enumerate(det_boxes):
            det_cx = det_centers[det_i, 0]
            det_cy = det_centers[det_i, 1]

            if det_i in matched_dets:
                label = 1
                tp_count += 1
            else:
                # Discard if center is near any matched GT — its crop still
                # contains the mitotic figure so label 0 would corrupt training.
                min_dist_to_matched = (
                    min(dist_matrix[det_i, gi] for gi in matched_gts)
                    if matched_gts else np.inf
                )
                if min_dist_to_matched <= match_dist:
                    continue   # near a matched GT — discard
                label = 0
                fp_count += 1

            crop      = extract_crop(image, det_cx, det_cy, STAGE2_CROP_SIZE)
            crop_name = f"{img_path.stem}_det{crop_idx:06d}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
            manifest_rows.append((str(crop_dir / crop_name), label))
            crop_idx += 1

        # ── Save crops for GT boxes YOLO completely missed (False Negatives) ──
        for gt_i, gt_box in enumerate(gt_boxes):
            if gt_i in matched_gts:
                continue   # already captured as a TP above
            # YOLO missed this mitotic figure — save it as a positive example
            gt_cx = (gt_box[0] + gt_box[2]) / 2
            gt_cy = (gt_box[1] + gt_box[3]) / 2
            crop      = extract_crop(image, gt_cx, gt_cy, STAGE2_CROP_SIZE)
            crop_name = f"{img_path.stem}_fn{crop_idx:06d}.png"
            cv2.imwrite(str(crop_dir / crop_name), crop)
            manifest_rows.append((str(crop_dir / crop_name), 1))
            crop_idx   += 1
            fn_count   += 1

    # ── Write CSV manifest ─────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(manifest_rows)

    print(f"  [{split_name}] TP={tp_count}  FP(hard neg)={fp_count}  FN(missed)={fn_count}")
    print(f"  [{split_name}] {len(manifest_rows)} crops saved → {csv_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    weights_path = MODELS_DIR / "yolo" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path}. "
            "Run train_yolo.py first."
        )

    print(f"Loading YOLO weights from {weights_path}\n")
    yolo_model = YOLO(str(weights_path))

    process_split("train", yolo_model)
    process_split("val",   yolo_model)

    print("Done. Run stage2_classifier/train.py next.")


if __name__ == "__main__":
    main()
