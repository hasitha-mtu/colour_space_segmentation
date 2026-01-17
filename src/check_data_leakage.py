"""
Check for Data Leakage in Train/Val Split
==========================================

For directory structure:
dataset/
├── images/       # Train + validation images (mixed)
├── masks/        # Train + validation masks (mixed)
└── test/         # Separate test set
    ├── images/
    └── masks/

This script checks if your train/val split (from images/masks folders) is done at:
- IMAGE-LEVEL (CORRECT): All tiles from same image in same split
- TILE-LEVEL (WRONG): Tiles from same image in different splits → DATA LEAKAGE!

Run this BEFORE submitting your paper!
"""

import sys
from pathlib import Path
from collections import defaultdict
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_dataloaders

def extract_image_id(filename):
    """
    Extract original image ID from tile filename
    
    IMPORTANT: Adjust this function to match YOUR naming convention!
    
    Common patterns:
    1. "image_001_tile_05.png" → "image_001"
    2. "DJI_0123_patch_2_3.jpg" → "DJI_0123"
    3. "survey_20250324_001_x512_y1024.png" → "survey_20250324_001"
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern 1: name_tile_XX or name_patch_XX
    if '_tile_' in name or '_patch_' in name:
        parts = name.split('_')
        for i, part in enumerate(parts):
            if part in ['tile', 'patch']:
                return '_'.join(parts[:i])
    
    # Pattern 2: name_xXXX_yYYY (coordinate-based)
    if '_x' in name and '_y' in name:
        x_idx = name.rfind('_x')
        return name[:x_idx]
    
    # Pattern 3: Ends with digits (tile number)
    parts = name.split('_')
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) <= 3:
        return '_'.join(parts[:-1])
    
    # Fallback: return whole name
    return name

def check_data_leakage(data_dir='dataset'):
    print("="*70)
    print("CHECKING FOR DATA LEAKAGE IN TRAIN/VAL SPLIT")
    print("="*70)
    
    try:
        train_loader, val_loader = get_dataloaders(
            data_root=data_dir,
            feature_config='rgb',
            batch_size=4,
            num_workers=4,
            train_split=0.8,
            seed=42
        )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_loader.dataset)}")
        print(f"  Val:   {len(val_loader.dataset)}")
        
        # Get filenames
        train_files = []
        val_files = []
        
        if hasattr(train_loader.dataset, 'image_paths'):
            train_files = [Path(p).name for p in train_loader.dataset.image_paths]
        if hasattr(val_loader.dataset, 'image_paths'):
            val_files = [Path(p).name for p in val_loader.dataset.image_paths]
        
        if not train_files or not val_files:
            print("\n⚠ Could not extract filenames automatically")
            print("Please check your dataset class")
            return None
        
        # Extract image IDs
        train_ids = set(extract_image_id(f) for f in train_files)
        val_ids = set(extract_image_id(f) for f in val_files)
        
        overlap = train_ids & val_ids
        
        print(f"\nUnique images:")
        print(f"  Train: {len(train_ids)}")
        print(f"  Val:   {len(val_ids)}")
        
        if len(overlap) == 0:
            print("\n✅ NO DATA LEAKAGE - Image-level split detected!")
            print(f"\nReport in paper:")
            print(f"  '{len(train_ids) + len(val_ids)} UAV images, ")
            print(f"   tiled to {len(train_files) + len(val_files)} patches'")
            return False
        else:
            print(f"\n🚨 DATA LEAKAGE DETECTED!")
            print(f"  {len(overlap)} images in both train and val!")
            print("\nYou MUST fix the split before submitting!")
            return True
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    args = parser.parse_args()
    
    result = check_data_leakage(args.data_dir)
    sys.exit(0 if result == False else 1 if result == True else 2)
