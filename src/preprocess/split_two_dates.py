"""
Custom splitting for UAV dataset with two temporal groups.

Dataset structure:
- date_2025_03_24: 203 images
- date_2025_07_28: 180 images

Provides two splitting strategies optimized for this scenario.
"""

import numpy as np
import shutil
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_date_groups(source_dir='dataset/processed'):
    """
    Load and organize images by date groups.
    """
    source_path = Path(source_dir)
    images_dir = source_path / 'images'
    masks_dir = source_path / 'masks'
    
    images = {f.stem: f for f in images_dir.glob('*.jpg')}
    masks = {f.stem: f for f in masks_dir.glob('*.png')}
    matched = sorted(set(images.keys()) & set(masks.keys()))
    
    # Group by date
    groups = {
        'date_2025_03_24': [],
        'date_2025_07_28': []
    }
    
    for name in matched:
        if '2025_03_24' in name or '2025-03-24' in name or '20250324' in name:
            groups['date_2025_03_24'].append(name)
        elif '2025_07_28' in name or '2025-07-28' in name or '20250728' in name:
            groups['date_2025_07_28'].append(name)
        else:
            print(f"Warning: {name} doesn't match known date patterns")
    
    return groups, images, masks

def split_temporal_separate(
    source_dir='dataset/processed',
    output_dir='dataset/split_temporal',
    train_date='date_2025_03_24',
    test_date='date_2025_07_28',
    val_ratio_from_train=0.2,
    random_seed=42,
    copy_files=True
):
    """
    Strategy A: Date-based split for temporal generalization.
    
    One date for training (with validation split), another for testing.
    Best for assessing temporal robustness.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory
        train_date: Which date to use for training
        test_date: Which date to use for testing
        val_ratio_from_train: Proportion of train date to use for validation
        random_seed: Random seed
        copy_files: Copy vs symlink
    """
    
    groups, images, masks = get_date_groups(source_dir)
    
    print("="*70)
    print("STRATEGY A: Temporal Split (Train on one date, test on another)")
    print("="*70)
    print(f"\nDataset composition:")
    print(f"  March 24: {len(groups['date_2025_03_24'])} images")
    print(f"  July 28:  {len(groups['date_2025_07_28'])} images")
    
    # Get data for each date
    train_val_files = groups[train_date]
    test_files = groups[test_date]
    
    # Split train into train + val
    np.random.seed(random_seed)
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio_from_train,
        random_state=random_seed
    )
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    print(f"\nSplit configuration:")
    print(f"  Training date: {train_date}")
    print(f"  Testing date:  {test_date}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_files)} images (from {train_date})")
    print(f"  Val:   {len(val_files)} images (from {train_date})")
    print(f"  Test:  {len(test_files)} images (from {test_date})")
    print(f"\nTemporal gap: 4 months (March → July)")
    print(f"This split tests temporal generalization and seasonal robustness.")
    
    # Save files
    output_path = Path(output_dir)
    _save_splits(splits, images, masks, output_path, copy_files)
    
    # Save metadata
    split_info = {
        'strategy': 'temporal_separate',
        'train_date': train_date,
        'test_date': test_date,
        'temporal_gap_months': 4,
        'random_seed': random_seed,
        'splits': {
            'train': {'count': len(train_files), 'date': train_date, 'filenames': train_files},
            'val': {'count': len(val_files), 'date': train_date, 'filenames': val_files},
            'test': {'count': len(test_files), 'date': test_date, 'filenames': test_files}
        }
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Temporal split complete!")
    print(f"  Output: {output_path}")
    
    return split_info

def split_stratified_by_date(
    source_dir='dataset/processed',
    output_dir='dataset/split_stratified',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    copy_files=True
):
    """
    Strategy B: Stratified split maintaining date proportions.
    
    Each split contains data from both dates in the same proportion.
    Maximizes training data while maintaining temporal diversity.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory
        train_ratio, val_ratio, test_ratio: Split proportions
        random_seed: Random seed
        copy_files: Copy vs symlink
    """
    
    groups, images, masks = get_date_groups(source_dir)
    
    print("="*70)
    print("STRATEGY B: Stratified Split (Both dates in each split)")
    print("="*70)
    print(f"\nDataset composition:")
    print(f"  March 24: {len(groups['date_2025_03_24'])} images")
    print(f"  July 28:  {len(groups['date_2025_07_28'])} images")
    
    np.random.seed(random_seed)
    
    # Split each date separately, then combine
    all_train, all_val, all_test = [], [], []
    
    for date_name, files in groups.items():
        # Split this date's files
        train_val, test = train_test_split(
            files,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_seed
        )
        
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
        
        print(f"\n{date_name}:")
        print(f"  Train: {len(train)} ({len(train)/len(files)*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/len(files)*100:.1f}%)")
        print(f"  Test:  {len(test)} ({len(test)/len(files)*100:.1f}%)")
    
    splits = {
        'train': all_train,
        'val': all_val,
        'test': all_test
    }
    
    total = len(all_train) + len(all_val) + len(all_test)
    print(f"\nCombined splits:")
    print(f"  Train: {len(all_train)} ({len(all_train)/total*100:.1f}%)")
    print(f"  Val:   {len(all_val)} ({len(all_val)/total*100:.1f}%)")
    print(f"  Test:  {len(all_test)} ({len(all_test)/total*100:.1f}%)")
    print(f"\nEach split contains data from both March and July.")
    
    # Save files
    output_path = Path(output_dir)
    _save_splits(splits, images, masks, output_path, copy_files)
    
    # Save metadata
    split_info = {
        'strategy': 'stratified_by_date',
        'random_seed': random_seed,
        'splits': {
            'train': {'count': len(all_train), 'filenames': all_train},
            'val': {'count': len(all_val), 'filenames': all_val},
            'test': {'count': len(all_test), 'filenames': all_test}
        }
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Stratified split complete!")
    print(f"  Output: {output_path}")
    
    return split_info

def _save_splits(splits, images, masks, output_path, copy_files):
    """Helper function to save split files."""
    
    for split_name, filenames in splits.items():
        split_img_dir = output_path / split_name / 'images'
        split_mask_dir = output_path / split_name / 'masks'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for name in tqdm(filenames, desc=f"Saving {split_name}"):
            img_src = images[name]
            mask_src = masks[name]
            img_dst = split_img_dir / img_src.name
            mask_dst = split_mask_dir / mask_src.name
            
            if copy_files:
                shutil.copy2(img_src, img_dst)
                shutil.copy2(mask_src, mask_dst)
            else:
                img_dst.symlink_to(img_src.resolve())
                mask_dst.symlink_to(mask_src.resolve())

def main():
    """
    Main function providing both splitting strategies.
    """
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "UAV DATASET SPLITTING - TWO TEMPORAL GROUPS" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print()
    print("You have 2 acquisition dates:")
    print("  • March 24, 2025: 203 images")
    print("  • July 28, 2025:  180 images")
    print()
    print("Choose splitting strategy:")
    print()
    print("  [A] Temporal Split - Train on March, Test on July (RECOMMENDED)")
    print("      → Tests temporal generalization")
    print("      → Best for conference paper")
    print("      → Train: 203 (March), Val: 41 (March), Test: 180 (July)")
    print()
    print("  [B] Stratified Split - Both dates in each split")
    print("      → Maximizes training data")
    print("      → Each split has March + July data")
    print("      → Train: 268, Val: 57, Test: 58")
    print()
    
    choice = input("Enter choice [A/B] (or press Enter for A): ").strip().upper()
    
    if not choice:
        choice = 'A'
    
    print()
    
    if choice == 'A':
        split_info = split_temporal_separate(
            source_dir='dataset/processed',
            output_dir='dataset/split_temporal',
            train_date='date_2025_03_24',
            test_date='date_2025_07_28',
            val_ratio_from_train=0.2,
            random_seed=42,
            copy_files=True
        )
    elif choice == 'B':
        split_info = split_stratified_by_date(
            source_dir='dataset/processed',
            output_dir='dataset/split_stratified',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            copy_files=True
        )
    else:
        print("Invalid choice. Defaulting to Strategy A.")
        split_info = split_temporal_separate()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Load your split dataset in PyTorch:")
    print("   train_dataset = WaterSegmentationDataset('dataset/split_*/train')")
    print("   val_dataset = WaterSegmentationDataset('dataset/split_*/val')")
    print("   test_dataset = WaterSegmentationDataset('dataset/split_*/test')")
    print()
    print("2. Train your models (UNet, DeepLabv3+, DINOv2, SAM)")
    print()
    print("3. Report in your conference paper:")
    if choice == 'A':
        print("   'Temporal split with 4-month gap between train (March) and test (July)'")
    else:
        print("   'Stratified split maintaining temporal diversity across all splits'")
    print()
    print("✓ Ready for training! Good luck with your FlashFloodBreaker research!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
