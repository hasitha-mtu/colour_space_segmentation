import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def verify_dataset(dataset_dir='dataset/processed'):
    """
    Verify the organized dataset:
    - Check all images have corresponding masks
    - Verify all masks are binary
    - Check image and mask dimensions match
    - Compute basic statistics
    """
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    print("="*70)
    print("Dataset Verification")
    print("="*70)
    
    # Get all files
    images = {f.stem: f for f in images_dir.glob('*.jpg')}
    masks = {f.stem: f for f in masks_dir.glob('*.png')}
    
    print(f"Images found: {len(images)}")
    print(f"Masks found: {len(masks)}")
    
    # Check correspondence
    print("\n1. Checking filename correspondence...")
    images_only = set(images.keys()) - set(masks.keys())
    masks_only = set(masks.keys()) - set(images.keys())
    matched = set(images.keys()) & set(masks.keys())
    
    if images_only:
        print(f"  ✗ Images without masks: {len(images_only)}")
        for name in sorted(images_only):
            print(f"    - {name}")
    else:
        print(f"  ✓ All images have corresponding masks")
    
    if masks_only:
        print(f"  ✗ Masks without images: {len(masks_only)}")
        for name in sorted(masks_only):
            print(f"    - {name}")
    else:
        print(f"  ✓ All masks have corresponding images")
    
    print(f"  ✓ Matched pairs: {len(matched)}")
    
    # Check dimensions and binary property
    print("\n2. Checking dimensions and mask properties...")
    
    dimension_issues = []
    non_binary_masks = []
    mask_stats = {
        'positive_pixels': [],
        'total_pixels': [],
        'positive_ratio': []
    }
    
    for name in sorted(matched):
        img = Image.open(images[name])
        mask = np.array(Image.open(masks[name]))
        
        # Check dimensions
        if img.size[::-1] != mask.shape[:2]:  # PIL uses (W, H), numpy uses (H, W)
            dimension_issues.append({
                'name': name,
                'image_size': img.size,
                'mask_shape': mask.shape
            })
        
        # Check if binary
        unique_values = np.unique(mask)
        if not (len(unique_values) <= 2 and all(v in [0, 1, 255] for v in unique_values)):
            non_binary_masks.append({
                'name': name,
                'unique_values': unique_values
            })
        
        # Collect statistics
        total_pixels = mask.size
        positive_pixels = np.sum(mask > 0)
        mask_stats['positive_pixels'].append(positive_pixels)
        mask_stats['total_pixels'].append(total_pixels)
        mask_stats['positive_ratio'].append(positive_pixels / total_pixels if total_pixels > 0 else 0)
    
    if dimension_issues:
        print(f"  ✗ Dimension mismatches: {len(dimension_issues)}")
        for issue in dimension_issues[:5]:  # Show first 5
            print(f"    - {issue['name']}: Image {issue['image_size']}, Mask {issue['mask_shape']}")
        if len(dimension_issues) > 5:
            print(f"    ... and {len(dimension_issues) - 5} more")
    else:
        print(f"  ✓ All image-mask dimensions match")
    
    if non_binary_masks:
        print(f"  ✗ Non-binary masks: {len(non_binary_masks)}")
        for issue in non_binary_masks[:5]:  # Show first 5
            print(f"    - {issue['name']}: {issue['unique_values']}")
        if len(non_binary_masks) > 5:
            print(f"    ... and {len(non_binary_masks) - 5} more")
    else:
        print(f"  ✓ All masks are binary")
    
    # Statistics
    print("\n3. Dataset Statistics:")
    print(f"  Total image-mask pairs: {len(matched)}")
    
    if mask_stats['positive_ratio']:
        pos_ratios = np.array(mask_stats['positive_ratio'])
        print(f"\n  Water pixel coverage (positive class):")
        print(f"    Mean: {np.mean(pos_ratios)*100:.2f}%")
        print(f"    Median: {np.median(pos_ratios)*100:.2f}%")
        print(f"    Min: {np.min(pos_ratios)*100:.2f}%")
        print(f"    Max: {np.max(pos_ratios)*100:.2f}%")
        print(f"    Std: {np.std(pos_ratios)*100:.2f}%")
        
        # Class imbalance
        mean_ratio = np.mean(pos_ratios)
        class_imbalance = (1 - mean_ratio) / mean_ratio if mean_ratio > 0 else float('inf')
        print(f"\n  Class imbalance (negative:positive): {class_imbalance:.2f}:1")
        
        # Distribution
        zero_masks = np.sum(pos_ratios == 0)
        if zero_masks > 0:
            print(f"  ⚠ Masks with no positive pixels: {zero_masks}")
    
    # Final verdict
    print("\n" + "="*70)
    if not images_only and not masks_only and not dimension_issues and not non_binary_masks:
        print("✓ Dataset is READY for training!")
        print("="*70)
        return True
    else:
        print("✗ Dataset has issues that need to be fixed")
        print("="*70)
        return False

def visualize_samples(dataset_dir='dataset/processed', n_samples=5, save_path=None):
    """
    Visualize random samples from the dataset
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    images = {f.stem: f for f in images_dir.glob('*.jpg')}
    masks = {f.stem: f for f in masks_dir.glob('*.png')}
    
    matched = sorted(set(images.keys()) & set(masks.keys()))
    
    # Select random samples
    import random
    random.seed(42)
    samples = random.sample(matched, min(n_samples, len(matched)))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, name in enumerate(samples):
        img = np.array(Image.open(images[name]))
        mask = np.array(Image.open(masks[name]))
        
        # Original image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Image: {name}')
        axes[idx, 0].axis('off')
        
        # Mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f'Mask (Water)')
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 255]  # Cyan for water
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title('Overlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.savefig(dataset_path / 'sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {dataset_path / 'sample_visualization.png'}")
    
    plt.close()

if __name__ == "__main__":
    # Verify dataset
    is_valid = verify_dataset('dataset/processed')
    
    # Create visualization
    if is_valid:
        print("\nCreating sample visualizations...")
        visualize_samples('dataset/processed', n_samples=5)
