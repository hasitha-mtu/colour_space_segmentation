import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

def compute_mask_statistics(mask_path):
    """
    Compute statistics for a mask to enable stratified splitting.
    
    Returns:
        dict: Statistics including water coverage ratio
    """
    mask = np.array(Image.open(mask_path))
    total_pixels = mask.size
    water_pixels = np.sum(mask > 0)
    water_ratio = water_pixels / total_pixels if total_pixels > 0 else 0
    
    return {
        'water_ratio': water_ratio,
        'has_water': water_ratio > 0
    }

def stratify_by_bins(water_ratios, n_bins=5):
    """
    Create stratification bins based on water coverage.
    
    Args:
        water_ratios: Array of water coverage ratios
        n_bins: Number of bins for stratification
        
    Returns:
        Array of bin labels for stratification
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = np.digitize(water_ratios, bins) - 1
    
    # Ensure bin labels are within valid range
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)
    
    return bin_labels

def split_dataset(
    source_dir='dataset/processed',
    output_dir='dataset/split',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    n_bins=5,
    random_seed=42,
    copy_files=True
):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        source_dir: Directory containing organized images and masks
        output_dir: Output directory for split dataset
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        stratify: Whether to stratify split based on water coverage (default: True)
        n_bins: Number of bins for stratification (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        copy_files: If True, copy files; if False, create symlinks (default: True)
    """
    
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    images_dir = source_path / 'images'
    masks_dir = source_path / 'masks'
    
    print("="*70)
    print("Dataset Splitting")
    print("="*70)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    print(f"Stratification: {'Enabled' if stratify else 'Disabled'}")
    print(f"Random seed: {random_seed}")
    
    # Get all image-mask pairs
    images = {f.stem: f for f in images_dir.glob('*.jpg')}
    masks = {f.stem: f for f in masks_dir.glob('*.png')}
    
    # Get matched pairs
    matched = sorted(set(images.keys()) & set(masks.keys()))
    
    print(f"\nTotal samples: {len(matched)}")
    
    # Compute statistics for stratification
    if stratify:
        print("\nComputing mask statistics for stratification...")
        statistics = {}
        water_ratios = []
        
        for name in tqdm(matched, desc="Analyzing masks"):
            stats = compute_mask_statistics(masks[name])
            statistics[name] = stats
            water_ratios.append(stats['water_ratio'])
        
        water_ratios = np.array(water_ratios)
        
        # Create stratification labels
        strat_labels = stratify_by_bins(water_ratios, n_bins=n_bins)
        
        print(f"\nWater coverage statistics:")
        print(f"  Mean: {np.mean(water_ratios)*100:.2f}%")
        print(f"  Median: {np.median(water_ratios)*100:.2f}%")
        print(f"  Min: {np.min(water_ratios)*100:.2f}%")
        print(f"  Max: {np.max(water_ratios)*100:.2f}%")
        print(f"  Std: {np.std(water_ratios)*100:.2f}%")
        
        print(f"\nStratification bins (n={n_bins}):")
        for i in range(n_bins):
            count = np.sum(strat_labels == i)
            bin_waters = water_ratios[strat_labels == i]
            if len(bin_waters) > 0:
                print(f"  Bin {i}: {count} samples, water coverage {np.mean(bin_waters)*100:.2f}% ± {np.std(bin_waters)*100:.2f}%")
    else:
        strat_labels = None
        water_ratios = None
    
    # Perform train/test split first
    print(f"\nSplitting dataset...")
    
    if stratify:
        train_val_names, test_names = train_test_split(
            matched,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=strat_labels
        )
        
        # Get stratification labels for train_val
        train_val_indices = [matched.index(name) for name in train_val_names]
        train_val_strat = strat_labels[train_val_indices]
        
        # Split train_val into train and val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_names, val_names = train_test_split(
            train_val_names,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=train_val_strat
        )
    else:
        train_val_names, test_names = train_test_split(
            matched,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_names, val_names = train_test_split(
            train_val_names,
            test_size=val_size_adjusted,
            random_state=random_seed
        )
    
    splits = {
        'train': train_names,
        'val': val_names,
        'test': test_names
    }
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_names)} ({len(train_names)/len(matched)*100:.1f}%)")
    print(f"  Val: {len(val_names)} ({len(val_names)/len(matched)*100:.1f}%)")
    print(f"  Test: {len(test_names)} ({len(test_names)/len(matched)*100:.1f}%)")
    
    # Verify stratification
    if stratify:
        print(f"\nVerifying stratification:")
        for split_name, names in splits.items():
            split_indices = [matched.index(name) for name in names]
            split_ratios = water_ratios[split_indices]
            print(f"  {split_name.capitalize()}: water coverage {np.mean(split_ratios)*100:.2f}% ± {np.std(split_ratios)*100:.2f}%")
    
    # Create output directories and copy/link files
    print(f"\n{'Copying' if copy_files else 'Linking'} files to output directories...")
    
    split_info = {}
    
    for split_name, names in splits.items():
        # Create directories
        split_img_dir = output_path / split_name / 'images'
        split_mask_dir = output_path / split_name / 'masks'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy or link files
        for name in tqdm(names, desc=f"Processing {split_name}"):
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
        
        # Store split info
        split_info[split_name] = {
            'filenames': names,
            'count': len(names)
        }
        
        if stratify:
            split_indices = [matched.index(name) for name in names]
            split_ratios = water_ratios[split_indices]
            split_info[split_name]['water_coverage_mean'] = float(np.mean(split_ratios))
            split_info[split_name]['water_coverage_std'] = float(np.std(split_ratios))
    
    # Save split information
    split_info_file = output_path / 'split_info.json'
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit information saved to: {split_info_file}")
    
    # Create a simple text file with filenames for each split
    for split_name, names in splits.items():
        filename_file = output_path / f'{split_name}_files.txt'
        with open(filename_file, 'w') as f:
            f.write('\n'.join(sorted(names)))
        print(f"  {split_name.capitalize()} filenames: {filename_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("Dataset Split Complete!")
    print("="*70)
    print(f"\nOutput structure:")
    print(f"  {output_path}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/ ({len(train_names)} files)")
    print(f"    │   └── masks/ ({len(train_names)} files)")
    print(f"    ├── val/")
    print(f"    │   ├── images/ ({len(val_names)} files)")
    print(f"    │   └── masks/ ({len(val_names)} files)")
    print(f"    ├── test/")
    print(f"    │   ├── images/ ({len(test_names)} files)")
    print(f"    │   └── masks/ ({len(test_names)} files)")
    print(f"    ├── split_info.json")
    print(f"    ├── train_files.txt")
    print(f"    ├── val_files.txt")
    print(f"    └── test_files.txt")
    
    return split_info

if __name__ == "__main__":
    # Standard split: 70% train, 15% val, 15% test
    split_info = split_dataset(
        source_dir='dataset/processed',
        output_dir='dataset/split',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,      # Stratify by water coverage
        n_bins=5,           # Use 5 bins for stratification
        random_seed=42,     # For reproducibility
        copy_files=True     # Copy files (set False for symlinks to save space)
    )
    
    print("\n✓ Ready for training!")
