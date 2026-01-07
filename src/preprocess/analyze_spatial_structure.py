"""
Analyze spatial structure of UAV dataset to detect potential spatial leakage.
"""

import numpy as np
from pathlib import Path
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

def analyze_filename_patterns(filenames):
    """
    Analyze filename patterns to detect spatial structure.
    """
    print("="*70)
    print("Filename Pattern Analysis")
    print("="*70)
    
    # Extract numbers from filenames
    numbers = []
    for filename in filenames:
        stem = Path(filename).stem
        # Find all numbers in filename
        nums = re.findall(r'\d+', stem)
        if nums:
            numbers.append([int(n) for n in nums])
    
    print(f"\nTotal files: {len(filenames)}")
    print(f"Files with numeric patterns: {len(numbers)}")
    
    # Check if sequential
    if numbers and len(numbers[0]) > 0:
        # Get the last number (usually sequence number)
        sequences = [n[-1] for n in numbers]
        sequences.sort()
        
        # Check if consecutive
        diffs = np.diff(sequences)
        is_sequential = np.all(diffs == 1) or np.median(diffs) <= 2
        
        print(f"\nSequence analysis (last number in filename):")
        print(f"  Min: {min(sequences)}")
        print(f"  Max: {max(sequences)}")
        print(f"  Range: {max(sequences) - min(sequences)}")
        print(f"  Step median: {np.median(diffs):.1f}")
        print(f"  Sequential: {'YES ⚠️' if is_sequential else 'NO ✓'}")
        
        if is_sequential:
            print("\n  ⚠️  WARNING: Filenames appear to be sequential!")
            print("     This suggests video frames or systematic scanning.")
            print("     High risk of spatial leakage with random splitting.")
        
        return is_sequential, sequences
    
    return False, []

def detect_groups(filenames):
    """
    Try to detect natural groupings in filenames.
    """
    print("\n" + "="*70)
    print("Group Detection")
    print("="*70)
    
    groups = defaultdict(list)
    
    for filename in filenames:
        stem = Path(filename).stem
        
        # Try different patterns
        # Pattern 1: flight_XXX_frame_YYY
        match = re.search(r'(flight|run|pass|transect)[_-]?(\d+)', stem, re.IGNORECASE)
        if match:
            group_id = f"{match.group(1)}_{match.group(2)}"
            groups[group_id].append(filename)
            continue
        
        # Pattern 2: tile_xXX_yYY
        match = re.search(r'tile[_-]?x?(\d+)[_-]?y?(\d+)', stem, re.IGNORECASE)
        if match:
            group_id = f"tile_block_{int(match.group(1))//5}_{int(match.group(2))//5}"
            groups[group_id].append(filename)
            continue
        
        # Pattern 3: Date-based (YYYYMMDD or YYYY_MM_DD)
        match = re.search(r'(20\d{2})[_-]?(\d{2})[_-]?(\d{2})', stem)
        if match:
            group_id = f"date_{match.group(1)}_{match.group(2)}_{match.group(3)}"
            groups[group_id].append(filename)
            continue
        
        # Pattern 4: Prefix-based grouping (first non-numeric part)
        match = re.match(r'([a-zA-Z_]+)', stem)
        if match:
            prefix = match.group(1).rstrip('_')
            groups[prefix].append(filename)
            continue
        
        # No group detected
        groups['ungrouped'].append(filename)
    
    # Report findings
    print(f"\nDetected {len(groups)} groups:")
    group_sizes = []
    for group_id, members in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {group_id:30s}: {len(members):4d} images")
        group_sizes.append(len(members))
    
    if len(groups) > 1:
        print(f"\nGroup statistics:")
        print(f"  Mean size: {np.mean(group_sizes):.1f}")
        print(f"  Median size: {np.median(group_sizes):.1f}")
        print(f"  Min size: {min(group_sizes)}")
        print(f"  Max size: {max(group_sizes)}")
        
        # Check if groups are balanced
        size_ratio = max(group_sizes) / min(group_sizes)
        if size_ratio > 10:
            print(f"\n  ⚠️  Groups are imbalanced (ratio: {size_ratio:.1f})")
        
    return groups

def estimate_spatial_correlation(filenames):
    """
    Estimate potential spatial correlation based on filename sequences.
    """
    print("\n" + "="*70)
    print("Spatial Correlation Risk Assessment")
    print("="*70)
    
    # Extract numeric sequences
    sequences = []
    for filename in filenames:
        stem = Path(filename).stem
        nums = re.findall(r'\d+', stem)
        if nums:
            sequences.append(int(nums[-1]))
    
    if not sequences:
        print("Cannot assess - no numeric patterns found")
        return
    
    sequences = np.array(sorted(sequences))
    
    # Check gaps in sequence
    if len(sequences) > 1:
        diffs = np.diff(sequences)
        
        gaps = np.where(diffs > 10)[0]
        
        print(f"\nSequence continuity:")
        print(f"  Continuous sequences: {len(gaps) + 1}")
        print(f"  Largest gap: {np.max(diffs) if len(diffs) > 0 else 0}")
        print(f"  Average step: {np.mean(diffs):.1f}")
        
        if np.mean(diffs) < 2:
            print("\n  🔴 HIGH RISK: Near-continuous sequences")
            print("     → Adjacent images likely have >90% overlap")
            print("     → MUST use group-based splitting")
        elif np.mean(diffs) < 10:
            print("\n  🟡 MEDIUM RISK: Some sequence gaps")
            print("     → Adjacent images may have 50-90% overlap")  
            print("     → Recommend group-based splitting")
        else:
            print("\n  🟢 LOW RISK: Large sequence gaps")
            print("     → Images may be independent")
            print("     → Random splitting may be acceptable")

def visualize_sequence_distribution(sequences, output_path='sequence_distribution.png'):
    """
    Visualize the distribution of sequence numbers.
    """
    if not sequences:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram
    axes[0].hist(sequences, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sequence Number')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sequence Numbers')
    axes[0].grid(True, alpha=0.3)
    
    # Sequence gaps
    if len(sequences) > 1:
        diffs = np.diff(sorted(sequences))
        axes[1].plot(diffs, marker='o', linestyle='-', markersize=3, alpha=0.6)
        axes[1].axhline(y=1, color='r', linestyle='--', label='Continuous (gap=1)')
        axes[1].axhline(y=np.median(diffs), color='g', linestyle='--', label=f'Median gap={np.median(diffs):.1f}')
        axes[1].set_xlabel('Position in Sorted Sequence')
        axes[1].set_ylabel('Gap to Next Image')
        axes[1].set_title('Gaps Between Sequential Images')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_path}")
    plt.close()

def main(dataset_dir='dataset/processed'):
    """
    Analyze dataset for spatial structure.
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    
    if not images_dir.exists():
        print(f"Error: Directory not found: {images_dir}")
        return
    
    # Get all images
    image_files = sorted([f.stem for f in images_dir.glob('*.jpg')])
    
    if not image_files:
        print(f"Error: No .jpg files found in {images_dir}")
        return
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SPATIAL LEAKAGE RISK ANALYSIS" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Analyze patterns
    is_sequential, sequences = analyze_filename_patterns(image_files)
    
    # Detect groups
    groups = detect_groups(image_files)
    
    # Estimate correlation risk
    estimate_spatial_correlation(image_files)
    
    # Visualize
    if sequences:
        visualize_sequence_distribution(sequences, 
                                       output_path=dataset_path / 'sequence_analysis.png')
    
    # Final recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if is_sequential or (len(groups) == 1 and 'ungrouped' in groups):
        print("\n🔴 HIGH RISK OF SPATIAL LEAKAGE")
        print("\nStrong recommendations:")
        print("  1. Use split_dataset_spatial.py instead of split_dataset.py")
        print("  2. Provide metadata (GPS coordinates or flight IDs)")
        print("  3. Manually group images by flight/area/date")
        print("  4. Consider temporal splitting if monitoring data")
        
    elif len(groups) > 1:
        print("\n🟡 MODERATE RISK - GROUPS DETECTED")
        print(f"\nFound {len(groups)} potential groups")
        print("Recommendations:")
        print("  1. Verify groups make sense (same flight/area/date)")
        print("  2. Use split_dataset_spatial.py with group-based splitting")
        print("  3. Document grouping strategy in your paper")
        
    else:
        print("\n🟢 LOW RISK - APPEARS INDEPENDENT")
        print("\nRecommendations:")
        print("  1. Random splitting (split_dataset.py) is acceptable")
        print("  2. Still verify with visual inspection")
        print("  3. Document independence assumption in paper")
    
    print("\n" + "="*70)
    print("\nNext steps:")
    print("  1. Review the analysis above")
    print("  2. Check sequence_analysis.png for visual patterns")
    print("  3. Choose appropriate splitting strategy")
    print("  4. Read SPATIAL_LEAKAGE_GUIDE.md for details")
    print("="*70 + "\n")
    
    # Save analysis report
    report_path = dataset_path / 'spatial_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Spatial Analysis Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Sequential pattern detected: {is_sequential}\n")
        f.write(f"Number of groups: {len(groups)}\n")
        if sequences:
            f.write(f"Sequence range: {min(sequences)} - {max(sequences)}\n")
        f.write(f"\nGroups:\n")
        for gid, members in groups.items():
            f.write(f"  {gid}: {len(members)} images\n")
    
    print(f"📝 Analysis report saved: {report_path}")

if __name__ == "__main__":
    main('dataset/processed')
