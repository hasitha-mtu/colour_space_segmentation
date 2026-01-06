"""
Dataset Resizing Script
========================
Resizes all images and masks in a dataset to 224×224 resolution.

Usage:
    python resize_dataset.py --input_dir dataset --output_dir dataset_224 --size 224
    
Features:
- Preserves directory structure
- Handles images and masks separately
- Shows progress bar
- Validates output
- Supports various image formats
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil


def resize_image(image_path: Path, output_path: Path, size: int, is_mask: bool = False):
    """
    Resize a single image or mask
    
    Args:
        image_path: Path to input image
        output_path: Path to save resized image
        size: Target size (will be size×size)
        is_mask: Whether this is a binary mask (uses nearest neighbor)
    """
    # Read image
    if is_mask:
        # Masks: read as grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        # Images: read as RGB
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return False
    
    # Resize
    if is_mask:
        # Use nearest neighbor for masks to preserve binary values
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        # Use bilinear for images (better quality)
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_mask:
        # Save mask as grayscale
        cv2.imwrite(str(output_path), resized)
    else:
        # Convert back to BGR for saving
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), resized_bgr)
    
    return True


def resize_dataset(
    input_dir: str,
    output_dir: str,
    size: int = 224,
    images_subdir: str = 'images',
    masks_subdir: str = 'masks'
):
    """
    Resize entire dataset
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        size: Target size
        images_subdir: Subdirectory name for images
        masks_subdir: Subdirectory name for masks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check input exists
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")
    
    images_dir = input_path / images_subdir
    masks_dir = input_path / masks_subdir
    
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not masks_dir.exists():
        raise ValueError(f"Masks directory does not exist: {masks_dir}")
    
    # Create output directories
    output_images_dir = output_path / images_subdir
    output_masks_dir = output_path / masks_subdir
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(images_dir.glob('*.png')) + 
                        list(images_dir.glob('*.jpg')) + 
                        list(images_dir.glob('*.jpeg')))
    
    print(f"\n{'='*70}")
    print(f"Resizing Dataset to {size}×{size}")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {len(image_files)} images")
    print(f"{'='*70}\n")
    
    # Resize images
    print("Resizing images...")
    success_count = 0
    for img_path in tqdm(image_files, desc="Images"):
        output_img_path = output_images_dir / img_path.name
        if resize_image(img_path, output_img_path, size, is_mask=False):
            success_count += 1
    
    print(f"✓ Resized {success_count}/{len(image_files)} images")
    
    # Resize masks
    print("\nResizing masks...")
    success_count = 0
    for img_path in tqdm(image_files, desc="Masks"):
        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            # Try with different extension
            mask_path = masks_dir / f"{img_path.stem}.png"
        
        if mask_path.exists():
            output_mask_path = output_masks_dir / mask_path.name
            if resize_image(mask_path, output_mask_path, size, is_mask=True):
                success_count += 1
        else:
            print(f"Warning: No mask found for {img_path.name}")
    
    print(f"✓ Resized {success_count}/{len(image_files)} masks")
    
    # Validate output
    print("\nValidating output...")
    validate_dataset(output_path, size, images_subdir, masks_subdir)
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset resizing complete!")
    print(f"{'='*70}")
    print(f"Output: {output_path}")
    print(f"Images: {output_images_dir}")
    print(f"Masks:  {output_masks_dir}")
    print(f"\nYou can now use: --data_dir {output_dir} --image_size {size}")


def validate_dataset(dataset_dir: Path, expected_size: int, images_subdir: str, masks_subdir: str):
    """
    Validate resized dataset
    
    Args:
        dataset_dir: Dataset directory
        expected_size: Expected image size
        images_subdir: Images subdirectory name
        masks_subdir: Masks subdirectory name
    """
    images_dir = dataset_dir / images_subdir
    masks_dir = dataset_dir / masks_subdir
    
    # Get files
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    mask_files = sorted(list(masks_dir.glob('*.png')))
    
    print(f"  Images: {len(image_files)}")
    print(f"  Masks:  {len(mask_files)}")
    
    # Check a few random samples
    if len(image_files) > 0:
        sample_img = cv2.imread(str(image_files[0]))
        h, w = sample_img.shape[:2]
        print(f"  Sample image size: {w}×{h}")
        
        if w != expected_size or h != expected_size:
            print(f"  ⚠️  Warning: Size mismatch! Expected {expected_size}×{expected_size}")
        else:
            print(f"  ✓ Image size correct")
    
    if len(mask_files) > 0:
        sample_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        h, w = sample_mask.shape
        print(f"  Sample mask size: {w}×{h}")
        
        if w != expected_size or h != expected_size:
            print(f"  ⚠️  Warning: Size mismatch! Expected {expected_size}×{expected_size}")
        else:
            print(f"  ✓ Mask size correct")
        
        # Check mask values
        unique_values = np.unique(sample_mask)
        print(f"  Mask unique values: {unique_values}")
        if len(unique_values) <= 3:  # Should be 0 and 255 (or 0 and 1)
            print(f"  ✓ Mask values look correct")
        else:
            print(f"  ⚠️  Warning: Mask has unexpected values")


def compare_datasets(original_dir: str, resized_dir: str, num_samples: int = 3):
    """
    Display comparison of original vs resized images
    
    Args:
        original_dir: Original dataset directory
        resized_dir: Resized dataset directory
        num_samples: Number of samples to compare
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return
    
    orig_path = Path(original_dir) / 'images'
    resized_path = Path(resized_dir) / 'images'
    
    orig_files = sorted(list(orig_path.glob('*.png')) + list(orig_path.glob('*.jpg')))
    
    if len(orig_files) == 0:
        print("No images found for comparison")
        return
    
    # Select random samples
    import random
    samples = random.sample(orig_files, min(num_samples, len(orig_files)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_file in enumerate(samples):
        # Original
        orig_img = cv2.imread(str(img_file))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Resized
        resized_file = resized_path / img_file.name
        resized_img = cv2.imread(str(resized_file))
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f'Original {orig_img.shape[1]}×{orig_img.shape[0]}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(resized_img)
        axes[idx, 1].set_title(f'Resized {resized_img.shape[1]}×{resized_img.shape[0]}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path(resized_dir) / 'comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Resize dataset to specified resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize to 224×224
  python resize_dataset.py --input_dir dataset --output_dir dataset_224 --size 224
  
  # Resize to 336×336
  python resize_dataset.py --input_dir dataset --output_dir dataset_336 --size 336
  
  # With custom subdirectory names
  python resize_dataset.py --input_dir data/crookstown --output_dir data/crookstown_224 --size 224 --images_dir imgs --masks_dir labels
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input dataset directory (e.g., dataset)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output dataset directory (e.g., dataset_224)')
    parser.add_argument('--size', type=int, default=224,
                       help='Target size (default: 224)')
    parser.add_argument('--images_dir', type=str, default='images',
                       help='Images subdirectory name (default: images)')
    parser.add_argument('--masks_dir', type=str, default='masks',
                       help='Masks subdirectory name (default: masks)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate visual comparison of original vs resized')
    
    args = parser.parse_args()
    
    # Resize dataset
    resize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        images_subdir=args.images_dir,
        masks_subdir=args.masks_dir
    )
    
    # Generate comparison
    if args.compare:
        print("\nGenerating comparison...")
        compare_datasets(args.input_dir, args.output_dir, num_samples=3)


if __name__ == '__main__':
    main()
