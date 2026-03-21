import pydicom
from pydicom.pixels import iter_pixels

import pandas as pd

import cv2

import numpy as np

import os

def process_dicom_image(index=0,dcm=None):

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

    pixel_generator = iter_pixels(dcm)

    pixel_data = None
    for i,frame in enumerate(pixel_generator):
        if i == index:
            pixel_data = frame
            break

    if pixel_data is not None:
        if pixel_data.max() > 255:
            pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min() * 1e-8)) * 255
    else:
        print("No data found.")

    return pixel_data


def extract_patch(processed_data,coord_x, coord_y,dcm=None,patch_size=64):

    if dcm is None:
        print("No valid data")
        return 
    
    # slide_width = dcm.TotalPixelMatrixColumns
    # slide_height = dcm.TotalPixelMatrixRows

    tile_height = int(dcm.Rows)
    tile_width  = int(dcm.Columns)
    
    # # coord_x and coord_y are global coordinates of the centre of the cell.
    # if coord_x >= slide_width or coord_y >= slide_height or coord_x < 0 or coord_y < 0:
    #     return None

    # patches_per_row = slide_width // tile_width
    
    # tile_col = coord_x // tile_width
    # tile_row = coord_y // tile_height
    # tile_index = tile_row * patches_per_row + tile_col

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
        
    dcm_path = slide_to_path.get(slide_id)
    
    if dcm_path is None:
        print(f"Skipping slide {slide_id}: No DICOM path found.")
        return
    
    dcm = pydicom.dcmread(dcm_path)

    slide_width = dcm.TotalPixelMatrixColumns
    slide_height = dcm.TotalPixelMatrixRows

    tile_height = int(dcm.Rows)
    tile_width  = int(dcm.Columns)

    group['coordinateX'] = group['coordinateX'].astype(int)  # Ensure int for safe ops
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

    for tile_index,tile_group in group.groupby('tile_index'):

        pixel_data = process_dicom_image(index=tile_index,dcm=dcm)

        if pixel_data is None:
            print(f"Slide {slide_id}: Skipping tile {tile_index} (no pixel data)")
            continue
            
        for _, row in tile_group.iterrows():
        
            output_path = base_dir / ('mitotic' if row['agreedClass'] == 2 else 'non_mitotic') / f"{row['uid']}_{row['agreedClass']}.png"
        
            if os.path.exists(output_path):
                continue

            patch = extract_patch(pixel_data,row['coordinateX'], row['coordinateY'], dcm=dcm)

            if patch is None:
                continue
            
            cv2.imwrite(str(output_path), cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR))