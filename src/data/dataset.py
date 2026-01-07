"""
Multi-Channel Dataset Loader with Automatic Train/Val Splitting
================================================================
Combines best features from both implementations:
- Multi-channel support (RGB, Luminance, Chrominance, All)
- Automatic train/val splitting from single data_root
- ImageNet normalization option
- Flexible path specification

Supports:
- RGB-only (3 channels) - for foundation models like SAM, DINOv2
- Luminance features (3 channels) - L_LAB, L_range, L_texture
- Chrominance features (7 channels) - H, S, a, b, Cb, Cr, Intensity
- All features (10 channels) - Luminance + Chrominance
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import sys

# Handle imports for both package and direct execution
try:
    from .feature_extraction import FeatureExtractor
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.feature_extraction import FeatureExtractor


class RiverSegmentationDataset(Dataset):
    """
    River water segmentation dataset with multi-channel support
    
    Two initialization modes:
    1. data_root mode: Provide data_root with images/ and masks/ subdirs
    2. path mode: Provide explicit image_paths and mask_paths lists
    
    Expected structure (mode 1):
        data_root/
            images/
                img1.png, img2.png, ...
            masks/
                img1.png, img2.png, ...
    
    Args:
        data_root: Root directory with images/ and masks/ subdirectories
        image_paths: List of image paths (alternative to data_root)
        mask_paths: List of mask paths (alternative to data_root)
        image_dir: Images subdirectory name (default 'images')
        mask_dir: Masks subdirectory name (default 'masks')
        feature_config: 'rgb', 'luminance', 'chrominance', or 'all'
        transform: Albumentations transform (if None, uses default)
        image_size: Target size (H, W)
        normalize: Apply ImageNet normalization (for RGB/foundation models)
        augment: Apply data augmentation
    """
    
    def __init__(
        self, 
        data_root: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        mask_paths: Optional[List[str]] = None,
        image_dir: str = 'images',
        mask_dir: str = 'masks',
        feature_config: str = 'rgb',
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        normalize: bool = False,
        augment: bool = False
    ):
        self.feature_config = feature_config.lower()
        self.image_size = image_size
        self.normalize = normalize
        
        # Validate feature config
        valid_configs = ['rgb', 'luminance', 'chrominance', 'all']
        if self.feature_config not in valid_configs:
            raise ValueError(f"feature_config must be one of {valid_configs}")
        
        # Load paths - two modes
        if image_paths is not None and mask_paths is not None:
            # Mode 1: Direct path lists
            self.image_paths = sorted(image_paths)
            self.mask_paths = sorted(mask_paths)
        elif data_root is not None:
            # Mode 2: Directory-based with data_root
            img_dir_path = os.path.join(data_root, image_dir)
            mask_dir_path = os.path.join(data_root, mask_dir)
            
            # Get all images with multiple extensions
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
            self.image_paths = []
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(img_dir_path, ext)))
            self.image_paths = sorted(self.image_paths)
            
            # Get corresponding masks
            self.mask_paths = []
            for img_path in self.image_paths:
                img_name = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir_path, img_name)
                if not os.path.exists(mask_path):
                    # Try without extension change
                    base_name = os.path.splitext(img_name)[0]
                    mask_path = os.path.join(mask_dir_path, base_name + '.png')
                self.mask_paths.append(mask_path)
        else:
            raise ValueError("Provide either data_root or (image_paths + mask_paths)")
        
        # Validate
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found")
        
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Images ({len(self.image_paths)}) != Masks ({len(self.mask_paths)})"
            )
        
        # Check all files exist
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Initialize feature extractor (only if needed)
        if self.feature_config != 'rgb':
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = None
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms(augment)
        
        # Determine number of channels
        channel_map = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
        self.num_channels = channel_map[self.feature_config]
        
        print(f"Loaded {len(self.image_paths)} image-mask pairs")
        print(f"Feature config: {self.feature_config} ({self.num_channels} channels)")
        if normalize and self.feature_config == 'rgb':
            print("Using ImageNet normalization")

    def _get_default_transforms(self, augment: bool = False) -> A.Compose:
        """Get default transform pipeline"""
        
        transform_list = []
        
        # Always resize
        transform_list.append(
            A.Resize(self.image_size[0], self.image_size[1])
        )
        
        if augment:
            # Geometric augmentations
            transform_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
            ])
            
            # Color augmentations (only for RGB)
            if self.feature_config == 'rgb':
                transform_list.append(
                    A.OneOf([
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0
                        ),
                    ], p=0.5)
                )
            
            # Noise augmentations
            transform_list.append(
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                ], p=0.3)
            )
        
        # ImageNet normalization (only for RGB)
        if self.normalize and self.feature_config == 'rgb':
            transform_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        # Convert to tensor
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            dict with keys:
                - 'image': Feature tensor (C, H, W)
                - 'mask': Binary mask (1, H, W)
                - 'image_path': Path to original image
                - 'mask_path': Path to mask
        """
        # Load RGB image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load binary mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        # Extract features based on config
        if self.feature_config == 'rgb':
            # RGB - will be normalized by transform if needed
            features = image.astype(np.float32) / 255.0
        elif self.feature_config == 'luminance':
            features = self.feature_extractor.extract_luminance(image)
            features = self.feature_extractor.normalize_features(features)
        elif self.feature_config == 'chrominance':
            features = self.feature_extractor.extract_chrominance(image)
            features = self.feature_extractor.normalize_features(features)
        elif self.feature_config == 'all':
            features = self.feature_extractor.extract_all_features(image)
            features = self.feature_extractor.normalize_features(features)
        
        # Apply transforms
        if self.transform:
            # Note: For RGB with ImageNet norm, input should be [0,255]
            # But we've already normalized to [0,1] so scale back
            if self.normalize and self.feature_config == 'rgb':
                features = (features * 255.0).astype(np.uint8)
            
            transformed = self.transform(image=features, mask=mask)
            features = transformed['image']
            mask = transformed['mask']
        else:
            # Manual conversion to tensor
            features = torch.from_numpy(features).permute(2, 0, 1)  # HWC -> CHW
            mask = torch.from_numpy(mask)
        
        # Ensure mask has channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        
        return {
            'image': features.float(),
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }


def get_dataloaders(
    data_root: str,
    feature_config: str = 'rgb',
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    train_split: float = 0.8,
    normalize: bool = False,
    seed: int = 42,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with automatic splitting
    
    This is the recommended way to create dataloaders - just provide a
    data_root with images/ and masks/ subdirectories.
    
    Args:
        data_root: Root directory with images/ and masks/
        feature_config: 'rgb', 'luminance', 'chrominance', or 'all'
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target size (H, W)
        train_split: Train/val split ratio (default 0.8 = 80% train)
        normalize: Apply ImageNet normalization (for RGB only)
        seed: Random seed for splitting
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader
    
    Example:
        >>> train_loader, val_loader = get_dataloaders(
        ...     data_root='data/crookstown',
        ...     feature_config='all',
        ...     batch_size=4
        ... )
    """
    
    # Get all paths from data_root
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    image_paths = sorted(image_paths)
    
    # Get corresponding masks
    mask_paths = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, base_name + '.png')
        mask_paths.append(mask_path)
    
    # Split train/val
    np.random.seed(seed)
    indices = np.random.permutation(len(image_paths))
    split_idx = int(len(indices) * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    val_image_paths = [image_paths[i] for i in val_indices]
    val_mask_paths = [mask_paths[i] for i in val_indices]
    
    # Create datasets
    train_dataset = RiverSegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        feature_config=feature_config,
        image_size=image_size,
        normalize=normalize,
        augment=True  # Training with augmentation
    )
    
    val_dataset = RiverSegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        feature_config=feature_config,
        image_size=image_size,
        normalize=normalize,
        augment=False  # Validation without augmentation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataloader Summary:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature config: {feature_config}")
    
    return train_loader, val_loader


if __name__ == "__main__":

    train_loader, val_loader = get_dataloaders(
          data_root='dataset/data_sliced/train',
          feature_config='all',
          batch_size=4)

    print("Multi-Channel Dataset with Automatic Splitting")
    print("\nRecommended Usage (Automatic splitting):")
    print("  from src.data.dataset import get_dataloaders")
    print("  ")
    print("  train_loader, val_loader = get_dataloaders(")
    print("      data_root='dataset/data_sliced/train',")
    print("      feature_config='all',")
    print("      batch_size=4")
    print("  )")
    print("\nFeature configs: 'rgb', 'luminance', 'chrominance', 'all'")
    print("\nExpected structure:")
    print("  data_root/")
    print("    images/ - RGB images (.png, .jpg, .tif)")
    print("    masks/  - Binary masks (0/255)")
