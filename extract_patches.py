"""
Patch extraction: Reads DICOM whole-slide images, maps annotations to tiles,
and extracts 64x64 patches into mitotic/ and non_mitotic/ directories.
Uses multiprocessing. Supports resumable extraction.

Usage:
    python extract_patches.py --split train
    python extract_patches.py --split test
"""

import argparse
import pandas as pd
import multiprocessing as mp

from config import TRAIN_META_DATA_DIR, TEST_META_DATA_DIR, TRAINING_DATA_DIR, TEST_DATA_DIR
from wsi_utils import process_slide
from model_utils import NUM_WORKERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Which split to extract patches for: train or test")
    args = parser.parse_args()

    if args.split == "train":
        meta_dir = TRAIN_META_DATA_DIR
        data_dir = TRAINING_DATA_DIR
    else:
        meta_dir = TEST_META_DATA_DIR
        data_dir = TEST_DATA_DIR

    for csv_file in ["Annotations.csv", "Annotations_coordinates.csv", "Slides.csv"]:
        if not (meta_dir / csv_file).exists():
            raise FileNotFoundError(
                f"{csv_file} not found in {meta_dir}. "
                f"Run setup_data.py --split {args.split} first."
            )

    # Load and merge annotation tables
    df_annots = pd.read_csv(meta_dir / "Annotations.csv")
    df_coords = pd.read_csv(meta_dir / "Annotations_coordinates.csv")
    df = pd.merge(df_annots, df_coords, on="uid")

    # Build slide_id -> DICOM path mapping
    df_slides = pd.read_csv(meta_dir / "Slides.csv")
    slide_to_path = {
        row["uid"]: data_dir / row["filename"]
        for _, row in df_slides.iterrows()
    }

    # Group annotations by slide
    grouped_list = [(slide_id, group.copy()) for slide_id, group in df.groupby("slide_x")]

    with mp.Pool(processes=NUM_WORKERS) as pool:
        pool.starmap(
            process_slide,
            [(sid, grp, slide_to_path, data_dir) for sid, grp in grouped_list]
        )
    print("Extraction complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()