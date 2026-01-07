"""
Spatial-aware dataset splitting to avoid spatial leakage in UAV imagery.

Spatial leakage occurs when spatially correlated samples appear in both 
training and test sets, leading to overly optimistic performance estimates
that don't generalize to new locations.
"""

import numpy as np
from pathlib import Path
import json
import shutil
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

# ============================================================================
# STRATEGY 1: Spatial Clustering-Based Split
# ============================================================================

def extract_spatial_info_from_filename(filename):
    """
    Extract spatial information from filename if available.
    
    Common patterns:
    - flight_001_frame_123.jpg → flight_id: 001, frame: 123
    - tile_x0_y0.jpg → x: 0, y: 0
    - IMG_2024_05_15_123.jpg → date, sequence number
    
    Customize this function based on your naming convention!
    """
    stem = Path(filename).stem
    
    # Try to extract flight/sequence ID
    flight_match = re.search(r'flight[_-]?(\d+)', stem, re.IGNORECASE)
    frame_match = re.search(r'frame[_-]?(\d+)', stem, re.IGNORECASE)
    tile_x = re.search(r'[tx][_-]?(\d+)', stem, re.IGNORECASE)
    tile_y = re.search(r'[ty][_-]?(\d+)', stem, re.IGNORECASE)
    sequence_match = re.search(r'(\d{3,})$', stem)  # Sequence number at end
    
    info = {}
    if flight_match:
        info['flight_id'] = int(flight_match.group(1))
    if frame_match:
        info['frame_id'] = int(frame_match.group(1))
    if tile_x:
        info['tile_x'] = int(tile_x.group(1))
    if tile_y:
        info['tile_y'] = int(tile_y.group(1))
    if sequence_match:
        info['sequence'] = int(sequence_match.group(1))
    
    return info

def group_by_flight_or_sequence(filenames):
    """
    Group images by flight ID or sequence clusters.
    
    Returns:
        dict: {group_id: [list of filenames]}
    """
    groups = {}
    ungrouped = []
    
    for filename in filenames:
        info = extract_spatial_info_from_filename(filename)
        
        # Group by flight ID if available
        if 'flight_id' in info:
            group_id = f"flight_{info['flight_id']}"
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(filename)
        
        # Group by tile position if available
        elif 'tile_x' in info and 'tile_y' in info:
            # Group nearby tiles together (e.g., 3x3 blocks)
            block_x = info['tile_x'] // 3
            block_y = info['tile_y'] // 3
            group_id = f"block_{block_x}_{block_y}"
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(filename)
        
        else:
            ungrouped.append(filename)
    
    # If we have ungrouped images, try sequence-based clustering
    if ungrouped:
        sequences = []
        for filename in ungrouped:
            info = extract_spatial_info_from_filename(filename)
            if 'sequence' in info:
                sequences.append(info['sequence'])
            else:
                sequences.append(0)
        
        if len(set(sequences)) > 1:
            # Cluster by sequence proximity
            sequences = np.array(sequences).reshape(-1, 1)
            n_clusters = min(10, len(ungrouped) // 5)  # ~5 images per cluster
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(sequences)
                
                for filename, label in zip(ungrouped, cluster_labels):
                    group_id = f"sequence_cluster_{label}"
                    if group_id not in groups:
                        groups[group_id] = []
                    groups[group_id].append(filename)
            else:
                groups['ungrouped'] = ungrouped
        else:
            groups['ungrouped'] = ungrouped
    
    return groups

def split_by_groups(groups, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset by groups to avoid spatial leakage.
    
    Args:
        groups: dict of {group_id: [filenames]}
        train_ratio, val_ratio, test_ratio: Split proportions
        random_seed: Random seed
        
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    np.random.seed(random_seed)
    
    # Get group IDs and sizes
    group_ids = list(groups.keys())
    group_sizes = [len(groups[gid]) for gid in group_ids]
    total_samples = sum(group_sizes)
    
    print(f"Number of spatial groups: {len(groups)}")
    print(f"Total samples: {total_samples}")
    print(f"Average group size: {np.mean(group_sizes):.1f} ± {np.std(group_sizes):.1f}")
    print(f"Min/Max group size: {min(group_sizes)} / {max(group_sizes)}")
    
    # Shuffle groups
    indices = np.random.permutation(len(group_ids))
    
    # Allocate groups to splits based on cumulative size
    train_samples, val_samples, test_samples = [], [], []
    train_target = int(total_samples * train_ratio)
    val_target = int(total_samples * val_ratio)
    
    current_train = 0
    current_val = 0
    
    for idx in indices:
        gid = group_ids[idx]
        size = group_sizes[idx]
        
        if current_train < train_target:
            train_samples.extend(groups[gid])
            current_train += size
        elif current_val < val_target:
            val_samples.extend(groups[gid])
            current_val += size
        else:
            test_samples.extend(groups[gid])
    
    return {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }

# ============================================================================
# STRATEGY 2: Spatial Block-Based Split (For Orthomosaic Tiles)
# ============================================================================

def spatial_block_split(filenames, images_dir, n_blocks=5, test_block_indices=None, 
                        val_block_indices=None, random_seed=42):
    """
    Split dataset using spatial blocks (like a checkerboard).
    
    Useful when images are tiles from an orthomosaic.
    
    Args:
        filenames: List of image filenames
        images_dir: Directory containing images (to get dimensions)
        n_blocks: Number of blocks in each dimension
        test_block_indices: Which blocks for test (e.g., [0, 4, 20, 24] for corners)
        val_block_indices: Which blocks for validation
        random_seed: Random seed
        
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    # This would require actual GPS coordinates or tile positions
    # Placeholder for demonstration
    print("Note: Spatial block split requires GPS coordinates or tile positions")
    print("Implement based on your specific data structure")
    
    # Example: if you have tile coordinates in filenames
    return None

# ============================================================================
# STRATEGY 3: Temporal Split (For Time-Series UAV Data)
# ============================================================================

def temporal_split(filenames, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split by time to avoid temporal leakage.
    
    Useful for monitoring applications where you want to predict future states.
    
    Args:
        filenames: List of filenames with temporal information
        
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    # Extract temporal information
    timestamps = []
    for filename in filenames:
        # This is a placeholder - extract actual timestamp from your filenames
        # e.g., from EXIF data, filename patterns, or metadata files
        timestamps.append(filename)  # Placeholder
    
    # Sort by time
    sorted_indices = np.argsort(timestamps)
    sorted_filenames = [filenames[i] for i in sorted_indices]
    
    # Split chronologically
    n_total = len(sorted_filenames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    return {
        'train': sorted_filenames[:n_train],
        'val': sorted_filenames[n_train:n_train+n_val],
        'test': sorted_filenames[n_train+n_val:]
    }

# ============================================================================
# MAIN FUNCTION: Spatial-Aware Split
# ============================================================================

def split_dataset_spatial_aware(
    source_dir='dataset/processed',
    output_dir='dataset/split_spatial',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    split_strategy='auto',  # 'auto', 'groups', 'blocks', 'temporal'
    random_seed=42,
    copy_files=True
):
    """
    Split dataset with spatial awareness to avoid spatial leakage.
    
    Args:
        source_dir: Directory with organized images/masks
        output_dir: Output directory
        train_ratio, val_ratio, test_ratio: Split proportions
        split_strategy: How to handle spatial structure
            - 'auto': Detect from filenames
            - 'groups': Use flight/sequence grouping
            - 'blocks': Use spatial blocks (requires coordinates)
            - 'temporal': Chronological split
        random_seed: Random seed
        copy_files: Copy files vs create symlinks
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    images_dir = source_path / 'images'
    masks_dir = source_path / 'masks'
    
    # Get filenames
    images = {f.stem: f for f in images_dir.glob('*.jpg')}
    masks = {f.stem: f for f in masks_dir.glob('*.png')}
    matched = sorted(set(images.keys()) & set(masks.keys()))
    
    print("="*70)
    print("Spatial-Aware Dataset Splitting")
    print("="*70)
    print(f"Total samples: {len(matched)}")
    print(f"Split strategy: {split_strategy}")
    
    # Determine split based on strategy
    if split_strategy == 'auto' or split_strategy == 'groups':
        print("\nDetecting spatial groups from filenames...")
        groups = group_by_flight_or_sequence(matched)
        
        if len(groups) == 1 and 'ungrouped' in groups:
            print("⚠️  WARNING: Could not detect spatial structure from filenames!")
            print("   All images treated as one group.")
            print("   This may NOT prevent spatial leakage.")
            print("\n   Recommendations:")
            print("   1. Provide metadata with flight/tile/timestamp info")
            print("   2. Use manual grouping based on your knowledge")
            print("   3. If images are truly independent, random split is OK")
            
            response = input("\nContinue with random split anyway? (y/n): ")
            if response.lower() != 'y':
                print("Splitting cancelled. Please provide spatial metadata.")
                return None
        
        print(f"\nFound {len(groups)} spatial groups:")
        for gid, members in sorted(groups.items()):
            print(f"  {gid}: {len(members)} images")
        
        splits = split_by_groups(groups, train_ratio, val_ratio, test_ratio, random_seed)
        
    elif split_strategy == 'temporal':
        print("\nPerforming temporal split...")
        splits = temporal_split(matched, train_ratio, val_ratio, test_ratio)
    
    elif split_strategy == 'blocks':
        print("\nPerforming spatial block split...")
        splits = spatial_block_split(matched, images_dir, random_seed=random_seed)
        if splits is None:
            print("ERROR: Block split requires GPS coordinates or tile positions")
            return None
    
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")
    
    # Print split statistics
    print("\n" + "="*70)
    print("Split Statistics")
    print("="*70)
    print(f"Train: {len(splits['train'])} ({len(splits['train'])/len(matched)*100:.1f}%)")
    print(f"Val:   {len(splits['val'])} ({len(splits['val'])/len(matched)*100:.1f}%)")
    print(f"Test:  {len(splits['test'])} ({len(splits['test'])/len(matched)*100:.1f}%)")
    
    # Create output directories and copy files
    print(f"\n{'Copying' if copy_files else 'Linking'} files...")
    
    for split_name, filenames in splits.items():
        split_img_dir = output_path / split_name / 'images'
        split_mask_dir = output_path / split_name / 'masks'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        for name in tqdm(filenames, desc=f"Processing {split_name}"):
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
    
    # Save split info
    split_info = {
        'strategy': split_strategy,
        'random_seed': random_seed,
        'train': {'filenames': splits['train'], 'count': len(splits['train'])},
        'val': {'filenames': splits['val'], 'count': len(splits['val'])},
        'test': {'filenames': splits['test'], 'count': len(splits['test'])}
    }
    
    if split_strategy == 'groups':
        split_info['groups'] = {gid: members for gid, members in groups.items()}
    
    with open(output_path / 'split_info_spatial.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print("\n✓ Spatial-aware split complete!")
    print(f"Output: {output_path}")
    
    return split_info

if __name__ == "__main__":
    # Example usage
    split_info = split_dataset_spatial_aware(
        source_dir='dataset/processed',
        output_dir='dataset/split_spatial',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_strategy='auto',  # Will try to detect groups from filenames
        random_seed=42,
        copy_files=True
    )
