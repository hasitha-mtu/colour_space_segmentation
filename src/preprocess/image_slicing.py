
import os
import cv2
import numpy as np
from tqdm import tqdm

def slice_images(input_dir, output_dir, tile_size=512, stride=400, bg_threshold=0.01, bg_keep_prob=0.1):
    img_in = os.path.join(input_dir, 'images')
    mask_in = os.path.join(input_dir, 'masks')
    
    img_out = os.path.join(output_dir, 'images')
    mask_out = os.path.join(output_dir, 'masks')
    
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # Get all .jpg images
    filenames = [f for f in os.listdir(img_in) if f.lower().endswith('.jpg')]

    for fname in tqdm(filenames, desc=f"Slicing {os.path.basename(input_dir)}"):
        # Load Image (.jpg)
        img = cv2.imread(os.path.join(img_in, fname))
        
        # Determine Mask Filename (Change .jpg to .png)
        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(mask_in, mask_fname)
        
        # Load Mask (.png)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {fname}, skipping.")
            continue
            
        mask = cv2.imread(mask_path, 0) # Read as 1-channel grayscale
        
        h, w, _ = img.shape
        
        # Sliding window
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                img_patch = img[y:y+tile_size, x:x+tile_size]
                mask_patch = mask[y:y+tile_size, x:x+tile_size]
                
                # Check for water content
                water_pixels = np.count_nonzero(mask_patch)
                water_ratio = water_pixels / (tile_size * tile_size)
                
                keep = False
                if water_ratio >= bg_threshold:
                    keep = True
                elif np.random.random() < bg_keep_prob:
                    keep = True
                
                if keep:
                    # Save both as .png for the final dataset to maintain quality
                    patch_name = f"{os.path.splitext(fname)[0]}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(img_out, patch_name), img_patch)
                    cv2.imwrite(os.path.join(mask_out, patch_name), mask_patch)

def main():
    # Run for all sets
    for s in ['train', 'val', 'test']:
        slice_images(input_dir=f'./dataset/split_stratified/{s}', output_dir=f'./dataset/data_sliced/{s}')


if __name__ == "__main__":
    main()
