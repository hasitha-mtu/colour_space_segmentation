import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def verify_binary_mask(mask_path):
    """
    Verify that a mask is binary (contains only 0 and 255, or 0 and 1).
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        tuple: (is_binary, unique_values, fixed_mask_array or None)
    """
    mask = np.array(Image.open(mask_path))
    unique_values = np.unique(mask)
    
    # Check if binary (0 and 255) or (0 and 1) or just (0) or just (255)
    if len(unique_values) <= 2 and all(v in [0, 1, 255] for v in unique_values):
        # Normalize to 0 and 255 if needed
        if np.max(unique_values) == 1:
            mask = (mask * 255).astype(np.uint8)
            return True, unique_values, mask
        return True, unique_values, None
    else:
        return False, unique_values, None

def organize_dataset(
    raw_images_dir='dataset/raw/images',
    raw_masks_dir='dataset/raw/masks',
    output_dir='dataset/processed',
    verify_masks=True,
    fix_non_binary=True
):
    """
    Organize raw images and masks into a proper paired dataset.
    Only includes images that have corresponding masks.
    Verifies that masks are binary.
    
    Args:
        raw_images_dir: Directory containing raw images (.jpg)
        raw_masks_dir: Directory containing masks (.png)
        output_dir: Output directory for organized dataset
        verify_masks: Whether to verify masks are binary
        fix_non_binary: Whether to attempt to fix non-binary masks
    """
    
    # Create output directories
    output_path = Path(output_dir)
    images_out = output_path / 'images'
    masks_out = output_path / 'masks'
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    
    # Get all images and masks
    raw_images_path = Path(raw_images_dir)
    raw_masks_path = Path(raw_masks_dir)
    
    images = {f.stem: f for f in raw_images_path.glob('*.jpg')}
    masks = {f.stem: f for f in raw_masks_path.glob('*.png')}
    
    print("="*70)
    print("Dataset Organization")
    print("="*70)
    print(f"Found {len(images)} images (.jpg)")
    print(f"Found {len(masks)} masks (.png)")
    
    # Statistics
    stats = {
        'with_mask': 0,
        'without_mask': 0,
        'non_binary_masks': 0,
        'fixed_masks': 0,
        'failed_masks': []
    }
    
    # Find images without masks
    images_without_masks = set(images.keys()) - set(masks.keys())
    stats['without_mask'] = len(images_without_masks)
    
    if images_without_masks:
        print(f"\nImages without masks (will be skipped): {len(images_without_masks)}")
        if len(images_without_masks) <= 10:
            for img_name in sorted(images_without_masks):
                print(f"  - {img_name}")
        else:
            print(f"  (showing first 10)")
            for img_name in sorted(list(images_without_masks)[:10]):
                print(f"  - {img_name}")
    
    # Find masks without images (orphan masks)
    orphan_masks = set(masks.keys()) - set(images.keys())
    if orphan_masks:
        print(f"\nWarning: Found {len(orphan_masks)} masks without corresponding images:")
        for mask_name in sorted(orphan_masks):
            print(f"  - {mask_name}")
    
    # Process matched pairs
    matched_pairs = set(images.keys()) & set(masks.keys())
    print(f"\nProcessing {len(matched_pairs)} matched image-mask pairs...")
    
    for img_name in tqdm(sorted(matched_pairs), desc="Organizing dataset"):
        img_path = images[img_name]
        mask_path = masks[img_name]
        
        # Verify mask is binary
        if verify_masks:
            is_binary, unique_vals, fixed_mask = verify_binary_mask(mask_path)
            
            if not is_binary:
                stats['non_binary_masks'] += 1
                
                if fix_non_binary:
                    # Try to fix: threshold at 127
                    mask_array = np.array(Image.open(mask_path))
                    fixed_mask = ((mask_array > 127) * 255).astype(np.uint8)
                    stats['fixed_masks'] += 1
                    print(f"\n  Fixed non-binary mask: {img_name}")
                    print(f"    Original values: {unique_vals}")
                    print(f"    Thresholded to binary (0, 255)")
                else:
                    stats['failed_masks'].append(img_name)
                    print(f"\n  Warning: Non-binary mask: {img_name}")
                    print(f"    Unique values: {unique_vals}")
                    continue
        
        # Copy image
        shutil.copy2(img_path, images_out / img_path.name)
        
        # Copy or save mask
        if verify_masks and fixed_mask is not None:
            # Save fixed mask
            Image.fromarray(fixed_mask).save(masks_out / mask_path.name)
        else:
            # Copy original mask
            shutil.copy2(mask_path, masks_out / mask_path.name)
        
        stats['with_mask'] += 1
    
    # Print final summary
    print("\n" + "="*70)
    print("Dataset Organization Complete")
    print("="*70)
    print(f"✓ Successfully organized pairs: {stats['with_mask']}")
    print(f"✗ Images without masks (skipped): {stats['without_mask']}")
    
    if verify_masks:
        print(f"\nMask Verification:")
        if stats['non_binary_masks'] > 0:
            print(f"  ⚠ Non-binary masks found: {stats['non_binary_masks']}")
            if fix_non_binary:
                print(f"  ✓ Fixed masks: {stats['fixed_masks']}")
            if stats['failed_masks']:
                print(f"  ✗ Failed masks (not included): {len(stats['failed_masks'])}")
        else:
            print(f"  ✓ All masks are binary")
    
    print(f"\nOutput directory: {output_path}")
    print(f"  - Images: {images_out} ({stats['with_mask']} files)")
    print(f"  - Masks: {masks_out} ({stats['with_mask']} files)")
    
    # Verify final dataset
    final_images = list(images_out.glob('*.jpg'))
    final_masks = list(masks_out.glob('*.png'))
    
    print(f"\nFinal verification:")
    print(f"  Images in output: {len(final_images)}")
    print(f"  Masks in output: {len(final_masks)}")
    
    if len(final_images) == len(final_masks):
        print(f"  ✓ Dataset is balanced!")
    else:
        print(f"  ✗ Warning: Image and mask counts don't match!")
    
    return stats

if __name__ == "__main__":
    stats = organize_dataset(
        raw_images_dir='dataset/raw/images',
        raw_masks_dir='dataset/raw/masks',
        output_dir='dataset/processed',
        verify_masks=True,
        fix_non_binary=True
    )
