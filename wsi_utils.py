import pydicom
from pydicom.pixels import iter_pixels

import numpy as np
import pandas as pd
import cv2

from pathlib import Path
import os


def process_dicom_image(dcm, pixel_data):
    """
    Normalizes raw pixel data to 8-bit (0-255) range using Min-Max normalization.
    Only applies if pixel values exceed 255 (e.g., 12-bit DICOM data).

    Parameters:
        dcm:        pydicom Dataset object
        pixel_data: numpy array of raw pixel values from a single tile

    Returns:
        Normalized pixel data as numpy array, or None if inputs are invalid.
    """

    if dcm is None:
        print("Invalid: No DICOM data given")
        return None

    if pixel_data is None:
        print("Invalid: No pixel data found")
        return None

    if pixel_data.max() > 255:
        pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min() + 1e-8)) * 255

    return pixel_data


def extract_patch(dcm, processed_data, coord_x, coord_y, patch_size=64):
    """
    Extracts a square patch centered on (coord_x, coord_y) from a processed tile.
    Coordinates are converted from slide-level to tile-local before extraction.
    Patches near tile edges are zero-padded to maintain consistent size.

    Parameters:
        dcm:            pydicom Dataset object (used for tile dimensions)
        processed_data: numpy array of normalized pixel data for one tile
        coord_x:        slide-level x coordinate of the cell center
        coord_y:        slide-level y coordinate of the cell center
        patch_size:     side length of the square patch (default: 64)

    Returns:
        numpy array of shape (patch_size, patch_size, channels), or None if invalid.
    """

    if dcm is None:
        print("No valid DICOM data")
        return None

    if processed_data is None:
        return None

    tile_height = int(dcm.Rows)
    tile_width = int(dcm.Columns)

    # Convert slide-level coordinates to tile-local coordinates
    local_x = coord_x % tile_width
    local_y = coord_y % tile_height

    # Build patch boundaries centered on the cell
    half_patch = patch_size // 2
    start_x = local_x - half_patch
    start_y = local_y - half_patch
    end_x = local_x + half_patch
    end_y = local_y + half_patch

    # Calculate padding needed if patch extends beyond tile edges
    pad_left = max(0, -start_x)
    pad_top = max(0, -start_y)
    pad_right = max(0, -(tile_width - end_x))
    pad_bottom = max(0, -(tile_height - end_y))

    # Clamp coordinates to tile boundaries
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(tile_width, end_x)
    end_y = min(tile_height, end_y)

    patch = processed_data[start_y:end_y, start_x:end_x]

    if pad_left or pad_top or pad_right or pad_bottom:
        patch = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )

    return patch


def process_slide(slide_id, group, slide_to_path, output_root):
    """
    Processes one WSI slide: reads the DICOM file, iterates through tiles
    that contain annotated cells, extracts 64x64 patches around each annotation,
    and saves them as PNGs into mitotic/ or non_mitotic/ directories.

    Only decodes tiles that contain annotations (skips the rest for speed).
    Skips patches that already exist on disk (supports resumable extraction).

    Parameters:
        slide_id:       integer slide identifier
        group:          DataFrame of annotations belonging to this slide
        slide_to_path:  dict mapping slide_id -> Path to DICOM file
        output_root:    Path to the directory containing mitotic/ and non_mitotic/
    """

    dcm_path = slide_to_path[slide_id]

    if dcm_path is None:
        print(f"Skipping slide {slide_id}: No DICOM path found.")
        return

    dcm = pydicom.dcmread(dcm_path)

    slide_width = dcm.TotalPixelMatrixColumns
    slide_height = dcm.TotalPixelMatrixRows

    tile_height = int(dcm.Rows)
    tile_width = int(dcm.Columns)

    # Map each annotation to its tile index
    group['coordinateX'] = group['coordinateX'].astype(int)
    group['coordinateY'] = group['coordinateY'].astype(int)

    group['tile_col'] = group['coordinateX'] // tile_width
    group['tile_row'] = group['coordinateY'] // tile_height

    patches_per_row = slide_width // tile_width
    group['tile_index'] = group['tile_row'] * patches_per_row + group['tile_col']

    # Filter out annotations with coordinates outside the slide bounds
    valid_mask = (
        (group['coordinateX'] >= 0) & (group['coordinateX'] < slide_width) &
        (group['coordinateY'] >= 0) & (group['coordinateY'] < slide_height)
    )

    original_len = len(group)
    group = group[valid_mask].copy()

    skipped_invalid = original_len - len(group)
    if skipped_invalid > 0:
        print(f"Slide {slide_id}: Skipped {skipped_invalid} invalid annotations.")

    if group.empty:
        print(f"Slide {slide_id}: No valid annotations.")
        return

    # Group annotations by tile so we only decode each tile once
    tile_lookup = {}
    for tile_index, tile_group in group.groupby('tile_index'):
        tile_lookup[tile_index] = tile_group

    next_tile = sorted(tile_lookup.keys())

    if not next_tile:
        return

    # Stream through tiles, only stopping to decode tiles that have annotations
    pixel_generator = iter_pixels(dcm)
    current_target_idx = 0
    tile_index = next_tile[current_target_idx]

    for i, frame in enumerate(pixel_generator):

        if i == tile_index:

            pixel_data = process_dicom_image(dcm, frame)
            tile_group = tile_lookup[tile_index]

            if pixel_data is None:
                print(f"Slide {slide_id}: Skipping tile {tile_index} (no pixel data)")
                continue

            for _, row in tile_group.iterrows():

                label_dir = 'mitotic' if row['agreedClass'] == 2 else 'non_mitotic'
                output_path = output_root / label_dir / f"{row['uid']}_{row['agreedClass']}.png"

                # Skip if already extracted (resumable)
                if os.path.exists(output_path):
                    continue

                patch = extract_patch(dcm, pixel_data, row['coordinateX'], row['coordinateY'])

                if patch is None:
                    continue

                cv2.imwrite(str(output_path), cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR))

            current_target_idx += 1

            if current_target_idx >= len(next_tile):
                break

            tile_index = next_tile[current_target_idx]
