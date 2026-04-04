"""
STEP 5 OF 2-STAGE PIPELINE: Evaluate the full two-stage pipeline on val slides.

What this script does
---------------------
Runs the complete detection pipeline (Stage 1 YOLO → Stage 2 EfficientNet) on
every val slide directly from raw DICOM files, then computes Precision, Recall
and F1 against the ground-truth mitotic annotations.

Matching strategy
-----------------
Ground-truth annotations are POINTS (x, y) on the slide. A detection is
considered a True Positive if its center is within BOX_SIZE/2 pixels of an
unmatched ground-truth point. This is standard for point-annotation datasets.

  TP = detection within distance threshold of a GT point
  FP = detection with no nearby GT point
  FN = GT point with no nearby detection

Usage
-----
    python evaluate.py --split val
    python evaluate.py --split test
    (run from the repo root: D:/Github/mitosis-detector)
"""

import sys
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import pydicom
from pydicom.pixels import iter_pixels
from torchvision import models, transforms
from torchvision.ops import nms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    MODELS_DIR,
    VAL_SLIDES,
    TEST_SLIDES,
    YOLO_CONF_THRESH,
    STAGE2_CROP_SIZE,
    STAGE2_IMG_SIZE,
    STAGE2_CONF_THRESH,
    MITOTIC_CLASS,
    BOX_SIZE,
)
from stage2_classifier.prepare_stage2_data import extract_crop

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("pip install ultralytics")


# ---------------------------------------------------------------------------
# Normalisation (same as prepare_yolo_data.py)
# ---------------------------------------------------------------------------

def normalize_tile(frame: np.ndarray) -> np.ndarray:
    if frame.max() > 255:
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
    return frame.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2 model loader
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_stage2_model(weights_path: Path, device: torch.device) -> nn.Module:
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=False))
    model.to(device).eval()
    return model


stage2_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def run_stage2(model, crops_bgr: list[np.ndarray], device: torch.device) -> list[float]:
    """
    Run Stage 2 classifier on a batch of BGR crops.
    Returns a list of sigmoid scores (one per crop).
    """
    if not crops_bgr:
        return []

    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(stage2_transform(pil))

    batch  = torch.stack(tensors).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(batch)).cpu().squeeze(1).tolist()
    return scores


# ---------------------------------------------------------------------------
# Distance-based matching
# ---------------------------------------------------------------------------

def match_detections_to_gt(
    detections: list[tuple[float, float]],   # (slide_x, slide_y)
    gt_points:  list[tuple[float, float]],   # (slide_x, slide_y)
    dist_thresh: float,
) -> tuple[int, int, int]:
    """
    Global distance-based one-to-one matching (order-independent).
    All within-threshold (det, gt) pairs are sorted by distance ascending;
    the closest pair is assigned first, then the next-closest among still-free
    pairs, and so on.  This is equivalent to a greedy approximation of the
    minimum-cost bipartite matching and avoids the order-dependence of a naive
    per-detection loop.
    Returns (TP, FP, FN).
    """
    if not detections:
        return 0, 0, len(gt_points)
    if not gt_points:
        return 0, len(detections), 0

    det_arr     = np.array(detections)                                        # (M, 2)
    gt_arr      = np.array(gt_points)                                         # (N, 2)
    dist_matrix = np.linalg.norm(det_arr[:, None] - gt_arr[None, :], axis=2) # (M, N)

    # Collect all pairs within threshold, sorted by distance ascending.
    candidate_pairs = sorted(
        (dist_matrix[i, j], i, j)
        for i in range(len(detections))
        for j in range(len(gt_points))
        if dist_matrix[i, j] <= dist_thresh
    )

    matched_dets = set()
    matched_gts  = set()
    tp = 0

    for _dist, det_i, gt_j in candidate_pairs:
        if det_i not in matched_dets and gt_j not in matched_gts:
            matched_dets.add(det_i)
            matched_gts.add(gt_j)
            tp += 1

    fp = len(detections) - len(matched_dets)
    fn = len(gt_points)  - len(matched_gts)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Per-slide evaluation
# ---------------------------------------------------------------------------

def evaluate_slide(
    slide_id: int,
    dcm_path: Path,
    gt_points: list[tuple[float, float]],
    yolo_model,
    stage2_model: nn.Module,
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Run the full pipeline on one slide and return (TP, FP, FN).
    """
    dcm      = pydicom.dcmread(str(dcm_path))
    tile_h   = int(dcm.Rows)
    tile_w   = int(dcm.Columns)
    slide_w  = int(dcm.TotalPixelMatrixColumns)
    tiles_per_row = math.ceil(slide_w / tile_w)

    all_detections = []  # (slide_x, slide_y, score) for every surviving detection

    for tile_i, frame in enumerate(iter_pixels(dcm)):
        pixel_data = normalize_tile(frame)
        img_bgr    = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)

        # ── Stage 1: YOLO ─────────────────────────────────────────────────
        results = yolo_model(img_bgr, conf=YOLO_CONF_THRESH, verbose=False)
        candidate_boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.xyxy.cpu().numpy():
                candidate_boxes.append(box[:4])

        if not candidate_boxes:
            continue

        # ── Stage 2: EfficientNet on each candidate crop ──────────────────
        # extract_crop uses zero-padding at tile edges, matching training-data
        # preparation in prepare_stage2_data.py exactly.
        crops = []
        for box in candidate_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            crops.append(extract_crop(img_bgr, cx, cy, STAGE2_CROP_SIZE))

        scores = run_stage2(stage2_model, crops, device)

        # ── NMS: suppress duplicate boxes around the same mitosis ─────────
        passing = [(box, s) for box, s in zip(candidate_boxes, scores)
                   if s >= STAGE2_CONF_THRESH]
        if passing:
            pass_boxes  = torch.tensor([b for b, _ in passing], dtype=torch.float32)
            pass_scores = torch.tensor([s for _, s in passing], dtype=torch.float32)
            keep        = nms(pass_boxes, pass_scores, iou_threshold=0.3)
            passing     = [passing[i] for i in keep.tolist()]
        candidate_boxes = [b for b, _ in passing]
        scores          = [s for _, s in passing]

        # ── Convert surviving detections to slide coordinates ─────────────
        tile_col    = tile_i % tiles_per_row
        tile_row    = tile_i // tiles_per_row
        tile_origin_x = tile_col * tile_w
        tile_origin_y = tile_row * tile_h

        for box, score in zip(candidate_boxes, scores):
            cx_tile = (box[0] + box[2]) / 2
            cy_tile = (box[1] + box[3]) / 2
            slide_x = tile_origin_x + cx_tile
            slide_y = tile_origin_y + cy_tile
            all_detections.append((slide_x, slide_y, score))

    # ── Slide-level NMS: suppress cross-tile duplicates at tile boundaries ──
    # A mitosis near a tile edge can be detected in two adjacent tiles.
    # Tile-local NMS cannot catch these; one final NMS pass in slide coords can.
    if len(all_detections) > 1:
        half = BOX_SIZE / 2
        sl_boxes  = torch.tensor(
            [[x - half, y - half, x + half, y + half] for x, y, _ in all_detections],
            dtype=torch.float32,
        )
        sl_scores = torch.tensor([s for _, _, s in all_detections], dtype=torch.float32)
        keep = nms(sl_boxes, sl_scores, iou_threshold=0.3)
        all_detections = [all_detections[i] for i in keep.tolist()]

    # ── Match detections to ground truth ──────────────────────────────────
    dist_thresh = BOX_SIZE / 2   # 32 pixels by default
    det_points  = [(x, y) for x, y, _ in all_detections]
    tp, fp, fn  = match_detections_to_gt(det_points, gt_points, dist_thresh)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["val", "test"], required=True,
        help="Evaluation split: 'val' (same slides used for Stage 2 checkpoint selection, "
             "F1 is optimistically biased) or 'test' (independent holdout, requires "
             "TEST_SLIDES populated in config.py).",
    )
    args = parser.parse_args()

    if args.split == "val":
        print("WARNING: --split val uses the same slides that were used to select "
              "the Stage 2 checkpoint. The resulting F1 is optimistically biased "
              "and should not be reported as a final result.\n")
        eval_slides = VAL_SLIDES
        data_dir    = TRAIN_DATA_DIR
    else:
        if not TEST_SLIDES:
            raise ValueError(
                "TEST_SLIDES is empty in config.py. "
                "Populate it with held-out slide IDs from TEST_DATA_DIR, "
                "or run with --split val."
            )
        eval_slides = TEST_SLIDES
        data_dir    = TEST_DATA_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split:  {args.split} ({len(eval_slides)} slides)\n")

    # ── Load models ───────────────────────────────────────────────────────
    yolo_weights   = MODELS_DIR / "yolo" / "weights" / "best.pt"
    stage2_weights = MODELS_DIR / "stage2_best.pth"
    for p in [yolo_weights, stage2_weights]:
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")

    yolo_model   = YOLO(str(yolo_weights))
    stage2_model = load_stage2_model(stage2_weights, device)
    print("Both models loaded.\n")

    # ── Load ground-truth annotations ─────────────────────────────────────
    meta_dir  = data_dir / "meta_data"
    df_annots = pd.read_csv(meta_dir / "Annotations.csv")
    df_coords = pd.read_csv(meta_dir / "Annotations_coordinates.csv")
    df_slides = pd.read_csv(meta_dir / "Slides.csv")

    df = pd.merge(df_annots, df_coords, on="uid")
    df = df[(df["deleted"] == 0) & (df["agreedClass"] == MITOTIC_CLASS)]
    df = df[df["slide_x"].isin(eval_slides)]

    slide_to_path = {
        int(row["uid"]): data_dir / row["filename"]
        for _, row in df_slides.iterrows()
    }

    # ── Verify all eval slides are resolvable before starting ────────────
    missing = [
        slide_id for slide_id in eval_slides
        if slide_to_path.get(slide_id) is None or not slide_to_path[slide_id].exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot evaluate: {len(missing)} slide(s) not found: {missing}. "
            "Fix the DICOM paths or update eval_slides in config.py."
        )

    # ── Evaluate each slide ───────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0

    for slide_id in eval_slides:
        dcm_path = slide_to_path[slide_id]

        slide_gt = df[df["slide_x"] == slide_id]
        gt_points = list(zip(
            slide_gt["coordinateX"].astype(float),
            slide_gt["coordinateY"].astype(float),
        ))

        print(f"Evaluating slide {slide_id} ({len(gt_points)} GT mitotic cells)...")
        tp, fp, fn = evaluate_slide(
            slide_id, dcm_path, gt_points, yolo_model, stage2_model, device
        )
        total_tp += tp;  total_fp += fp;  total_fn += fn
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"  Slide {slide_id}: TP={tp} FP={fp} FN={fn} → P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # ── Overall metrics ───────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n" + "=" * 50)
    print(f"Overall  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
