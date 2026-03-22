"""
Patch extraction: Reads DICOM whole-slide images, maps annotations to tiles,
and extracts 64x64 patches into mitotic/ and non_mitotic/ directories.
Uses multiprocessing. Supports resumable extraction.

Usage:
    python extract_patches.py
"""

import pandas as pd
import multiprocessing as mp
from config import BASE_DIR, META_DATA_DIR, TRAINING_DATA_DIR
from wsi_utils import process_slide


def main():
    # Load and merge annotation tables
    df_annots = pd.read_csv(META_DATA_DIR / "Annotations.csv")
    df_coords = pd.read_csv(META_DATA_DIR / "Annotations_coordinates.csv")
    df = pd.merge(df_annots, df_coords, on="uid")

    # Build slide_id -> DICOM path mapping
    df_slides = pd.read_csv(META_DATA_DIR / "Slides.csv")
    slide_to_path = {
        row["uid"]: TRAINING_DATA_DIR / row["filename"]
        for _, row in df_slides.iterrows()
    }

    # Group annotations by slide
    grouped_list = [(slide_id, group.copy()) for slide_id, group in df.groupby("slide_x")]

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=6) as pool:
        pool.starmap(
            process_slide,
            [(sid, grp, slide_to_path, BASE_DIR) for sid, grp in grouped_list]
        )
    print("Extraction complete!")


if __name__ == "__main__":
    main()
