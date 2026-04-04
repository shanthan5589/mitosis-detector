"""
STEP 1 OF 2-STAGE PIPELINE: Prepare YOLO training data from raw DICOM slides.

What this script does
---------------------
Each DICOM file is a whole-slide image stored as a grid of tiles (e.g. 512x512 each).
This script streams through those tiles and saves two kinds of images:

  1. Annotated tiles  — tiles that contain at least one cell annotation.
                        For mitotic cells (class 2), it writes a YOLO bounding box label.
                        For all other cell types, no label is written (they are background).

  2. Background tiles — a random sample of tiles with zero annotations at all.
                        These are saved with empty label files so YOLO learns what
                        plain tissue looks like.

Slides are processed in parallel (one worker per slide) using all available CPU cores.

Output layout
-------------
  yolo_data/
    images/train/   slide4_tile000123.png  ...
    images/val/     slide7_tile000045.png  ...
    labels/train/   slide4_tile000123.txt  ...   (YOLO format: class cx cy w h)
    labels/val/     slide7_tile000045.txt  ...
    data.yaml       YOLO dataset config

Usage
-----
    python stage1_yolo/prepare_yolo_data.py
    (run from the repo root: D:/Github/mitosis-detector)
"""

import sys
import math
import random
import shutil
import yaml
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import pydicom
from pydicom.pixels import iter_pixels

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_DATA_DIR,
    YOLO_DATA_DIR,
    TRAIN_SLIDES,
    VAL_SLIDES,
    BOX_SIZE,
    MITOTIC_CLASS,
    MAX_BG_TILES_PER_SLIDE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_tile(frame: np.ndarray) -> np.ndarray:
    if frame.max() > 255:
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
    return frame.astype(np.uint8)


def compute_yolo_box(coord_x, coord_y, tile_w, tile_h, box_size):
    local_x = coord_x % tile_w
    local_y = coord_y % tile_h
    cx = local_x / tile_w
    cy = local_y / tile_h
    bw = box_size / tile_w
    bh = box_size / tile_h
    return cx, cy, bw, bh


# ---------------------------------------------------------------------------
# Per-slide worker (runs in its own process)
# ---------------------------------------------------------------------------

def process_one_slide(args):
    slide_id, img_dir, lbl_dir, slide_df, dcm_path = args

    # Per-slide seed so background sampling is deterministic regardless of
    # the order workers finish (imap_unordered gives no ordering guarantee).
    random.seed(42 + slide_id)

    dcm           = pydicom.dcmread(str(dcm_path))
    tile_h        = int(dcm.Rows)
    tile_w        = int(dcm.Columns)
    slide_w       = int(dcm.TotalPixelMatrixColumns)
    slide_h       = int(dcm.TotalPixelMatrixRows)
    tiles_per_row = math.ceil(slide_w / tile_w)

    slide_df = slide_df.copy()
    slide_df["coordinateX"] = slide_df["coordinateX"].astype(int)
    slide_df["coordinateY"] = slide_df["coordinateY"].astype(int)
    slide_df = slide_df[
        (slide_df["coordinateX"] >= 0) & (slide_df["coordinateX"] < slide_w) &
        (slide_df["coordinateY"] >= 0) & (slide_df["coordinateY"] < slide_h)
    ].copy()

    slide_df["tile_col"]   = slide_df["coordinateX"] // tile_w
    slide_df["tile_row"]   = slide_df["coordinateY"] // tile_h
    slide_df["tile_index"] = slide_df["tile_row"] * tiles_per_row + slide_df["tile_col"]

    annotated_set     = set(slide_df["tile_index"].unique())
    max_annotated_idx = max(annotated_set) if annotated_set else 0
    bg_pool           = [i for i in range(max_annotated_idx) if i not in annotated_set]
    bg_sample         = set(random.sample(bg_pool, min(MAX_BG_TILES_PER_SLIDE, len(bg_pool))))

    target_tiles   = sorted(annotated_set | bg_sample)
    tile_to_annots = {ti: tg for ti, tg in slide_df.groupby("tile_index")}

    ptr       = 0
    n_images  = 0
    n_mitotic = 0

    for frame_i, frame in enumerate(iter_pixels(dcm)):
        if ptr >= len(target_tiles):
            break
        if frame_i != target_tiles[ptr]:
            continue

        pixel_data = normalize_tile(frame)
        tile_name  = f"slide{slide_id}_tile{frame_i:06d}"
        img_bgr    = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_dir / f"{tile_name}.png"), img_bgr)

        lines = []
        if frame_i in tile_to_annots:
            for _, row in tile_to_annots[frame_i].iterrows():
                if int(row["agreedClass"]) != MITOTIC_CLASS:
                    continue
                cx, cy, bw, bh = compute_yolo_box(
                    int(row["coordinateX"]), int(row["coordinateY"]),
                    tile_w, tile_h, BOX_SIZE,
                )
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_mitotic += 1

        (lbl_dir / f"{tile_name}.txt").write_text("\n".join(lines))
        n_images += 1
        ptr += 1

    return slide_id, len(annotated_set), len(bg_sample), n_images, n_mitotic


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def prepare_split(split_name: str, slide_ids: list,
                  img_dir: Path, lbl_dir: Path) -> None:
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    meta_dir  = TRAIN_DATA_DIR / "meta_data"
    df_annots = pd.read_csv(meta_dir / "Annotations.csv")
    df_coords = pd.read_csv(meta_dir / "Annotations_coordinates.csv")
    df_slides = pd.read_csv(meta_dir / "Slides.csv")

    df = pd.merge(df_annots, df_coords, on="uid")
    df = df[df["deleted"] == 0]
    df = df[df["slide_x"].isin(slide_ids)]

    slide_to_path = {
        int(row["uid"]): TRAIN_DATA_DIR / row["filename"]
        for _, row in df_slides.iterrows()
    }

    args_list = []
    for slide_id in slide_ids:
        dcm_path = slide_to_path.get(slide_id)
        if dcm_path is None or not dcm_path.exists():
            print(f"  [{split_name}] Slide {slide_id}: DICOM not found — skipping")
            continue
        slide_df = df[df["slide_x"] == slide_id]
        args_list.append((slide_id, img_dir, lbl_dir, slide_df, dcm_path))

    n_workers = min(len(args_list), cpu_count())
    print(f"  [{split_name}] {len(args_list)} slides, {n_workers} workers\n")

    total_images = total_mitotic = 0
    with Pool(processes=n_workers) as pool:
        for slide_id, n_annot, n_bg, n_img, n_mit in pool.imap_unordered(process_one_slide, args_list):
            print(f"  [{split_name}] Slide {slide_id}: {n_annot} annotated + {n_bg} bg tiles saved")
            total_images  += n_img
            total_mitotic += n_mit

    print(f"\n  {split_name} total: {total_images} images, {total_mitotic} mitotic boxes\n")


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml() -> None:
    data = {
        "path":  str(YOLO_DATA_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["mitotic"],
    }
    yaml_path = YOLO_DATA_DIR / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"data.yaml written → {yaml_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"CPU cores available: {cpu_count()}")
    print("Preparing YOLO dataset from raw DICOM slides...\n")

    prepare_split(
        "train", TRAIN_SLIDES,
        YOLO_DATA_DIR / "images" / "train",
        YOLO_DATA_DIR / "labels" / "train",
    )
    prepare_split(
        "val", VAL_SLIDES,
        YOLO_DATA_DIR / "images" / "val",
        YOLO_DATA_DIR / "labels" / "val",
    )
    write_data_yaml()
    print("\nDone. Run train_yolo.py next.")


if __name__ == "__main__":
    main()
