import pydicom
from pydicom.pixels import iter_pixels

import numpy as np

import pandas as pd

import cv2

from pathlib import Path

import os

import time

base_dir = Path("D:/Mitosis WSI CCMCT")

def process_dicom_image(dcm,pixel_data):

    '''
    Parameters: 
        path    (Required Param)
        index   (Optional Param) 

    Returns:
        Pixel Data if index in range, else returns None
    '''

    '''
    This function takes a raw 12-bit image pixel data and returns normalized 8-bit pixel data.
    Min-Max Normalization is used to normalize the pixel data.
    '''
    
    if dcm is None:
        print("Invalid No Data Given")
        return

    if pixel_data is not None:
        if pixel_data.max() > 255:
            pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min() * 1e-8)) * 255
    else:
        print("No data found.")

    return pixel_data


def extract_patch(dcm, processed_data,coord_x, coord_y,patch_size=64):

    if dcm is None:
        print("No valid data")
        return 

    tile_height = int(dcm.Rows)
    tile_width  = int(dcm.Columns)

    pixel_data = processed_data
    if pixel_data is None:
        return None

    # local_x and local_y are local coordinates of the center of the cell.
    local_x  = coord_x % tile_width
    local_y = coord_y % tile_height
    
    # We are trying to make a patch from local coordinate of the center of the cell
    # by stretching center of the cell coordinate on all four directions.
    half_patch = patch_size // 2
    start_x = local_x - half_patch
    start_y = local_y - half_patch
    end_x = local_x + half_patch
    end_y = local_y + half_patch

    # Pad if near edges
    pad_left = max(0, -start_x)
    pad_top = max(0, -start_y)
    pad_right = max(0, -(tile_width -  end_x))
    pad_bottom = max(0, -(tile_height - end_y))

    # Making sure that starts and ends to be within tile
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(tile_width, end_x)
    end_y = min(tile_height, end_y)

    patch = pixel_data[start_y:end_y, start_x:end_x]

    # Apply padding if needed (with zeros/black)
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return patch


def process_slide(slide_id, group, slide_to_path, base_dir):

    # start_time = time.time()  # Start per-slide clock
    # patches_extracted = 0  # Your counter
        
    dcm_path = slide_to_path[slide_id]
    
    if dcm_path is None:
        print(f"Skipping slide {slide_id}: No DICOM path found.")
        return
    
    dcm = pydicom.dcmread(dcm_path)

    slide_width = dcm.TotalPixelMatrixColumns
    slide_height = dcm.TotalPixelMatrixRows

    tile_height = int(dcm.Rows)
    tile_width  = int(dcm.Columns)

    group['coordinateX'] = group['coordinateX'].astype(int)
    group['coordinateY'] = group['coordinateY'].astype(int)

    group['tile_col'] = group['coordinateX'] // tile_width
    group['tile_row'] = group['coordinateY'] // tile_height

    patches_per_row = slide_width // tile_width

    group['tile_index'] = group['tile_row'] * patches_per_row + group['tile_col']

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
    
    tile_lookup = {}

    for tile_index,tile_group in group.groupby('tile_index'):
        tile_lookup[tile_index] = tile_group

    next_tile = sorted(tile_lookup.keys())

    if not next_tile:
        return
    
    pixel_generator = iter_pixels(dcm)

    current_target_idx = 0
    tile_index = next_tile[current_target_idx]

    for i,frame in enumerate(pixel_generator):

        if i == tile_index:

            pixel_data = process_dicom_image(dcm,frame)

            tile_group = tile_lookup[tile_index]

            if pixel_data is None:
                print(f"Slide {slide_id}: Skipping tile {tile_index} (no pixel data)")
                continue

            # batch_size = 100
            # patches_batch = []
            # output_paths_batch = []
                
            for _, row in tile_group.iterrows():
            
                output_path = base_dir / ('mitotic' if row['agreedClass'] == 2 else 'non_mitotic') / f"{row['uid']}_{row['agreedClass']}.png"
            
                # Very important line
                if os.path.exists(output_path):
                    continue

                patch = extract_patch(dcm, pixel_data, row['coordinateX'], row['coordinateY'])

                if patch is None:
                    continue
                
                # output_paths_batch.append(output_path)
                # patches_batch.append(patch)

                # if len(patches_batch) == batch_size:

                #     for patch, path in zip(patches_batch, output_paths_batch):
                #         cv2.imwrite(str(path), cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR))
                        
                #     patches_batch = []
                #     output_paths_batch = []

                cv2.imwrite(str(output_path), cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR)) 

            current_target_idx += 1

            if current_target_idx >= len(next_tile):
                break 
            
            tile_index = next_tile[current_target_idx]


            # If any batch with less than batch size is leftover.
            # for patch, path in zip(patches_batch, output_paths_batch):
            #         cv2.imwrite(str(path), cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR)) 

        # # <--- PASTE HERE: End timing and print (at the very end of the function)
        # end_time = time.time()
        # slide_time = end_time - start_time
        # slide_speed = patches_extracted / (slide_time / 60) if slide_time > 0 else 0
        # print(f"Slide {slide_id}: Extracted {patches_extracted} patches in {slide_time:.2f} seconds. Speed: {slide_speed:.2f} patches/min.")