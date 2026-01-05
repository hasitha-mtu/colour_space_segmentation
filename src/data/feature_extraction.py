"""
Feature Extraction for River Segmentation - PyTorch Version
============================================================
Extracts 10 features (3 luminance + 7 chrominance) from RGB images
Based on the TensorFlow implementation

Features:
    Luminance (3):   L_LAB, L_range, L_texture
    Chrominance (7): H_HSV, S_HSV, a_LAB, b_LAB, Cb_YCbCr, Cr_YCbCr, Intensity
"""

import cv2
import numpy as np
from typing import List, Optional


class FeatureExtractor:
    """Extract luminance and chrominance features from RGB images"""
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Returns ordered list of feature names"""
        luminance = ['L_LAB', 'L_range', 'L_texture']
        chrominance = ['H_HSV', 'S_HSV', 'a_LAB', 'b_LAB', 
                      'Cb_YCbCr', 'Cr_YCbCr', 'Intensity']
        return luminance + chrominance
    
    def extract_luminance(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 3 luminance channels
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
        
        Returns:
            luminance: (H, W, 3) array [L_LAB, L_range, L_texture]
        """
        # Ensure uint8 for color space conversions
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract luminance channels
        L = lab[:, :, 0].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)
        Y = ycbcr[:, :, 0].astype(np.float32)
        
        # Derived luminance features
        L_max = np.maximum.reduce([L, V, Y])
        L_min = np.minimum.reduce([L, V, Y])
        L_range = L_max - L_min
        
        # Texture feature (edge magnitude using Sobel)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        L_texture = cv2.magnitude(sobelx, sobely).astype(np.float32)
        
        luminance = np.stack([L, L_range, L_texture], axis=2)
        return luminance.astype(np.float32)
    
    def extract_chrominance(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 7 chrominance channels
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
        
        Returns:
            chrominance: (H, W, 7) array [H, S, a, b, Cb, Cr, Intensity]
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract chrominance channels
        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        Cb = ycbcr[:, :, 1].astype(np.float32)
        Cr = ycbcr[:, :, 2].astype(np.float32)
        
        # Intensity (average of RGB channels)
        Intensity = image.mean(axis=2).astype(np.float32)
        
        chrominance = np.stack([H, S, a, b, Cb, Cr, Intensity], axis=2)
        return chrominance.astype(np.float32)
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all 10 features (3 luminance + 7 chrominance)
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
        
        Returns:
            features: (H, W, 10) array of all features
        """
        luminance = self.extract_luminance(image)
        chrominance = self.extract_chrominance(image)
        return np.concatenate([luminance, chrominance], axis=2)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range per channel
        
        Args:
            features: (H, W, C) feature array
        
        Returns:
            normalized: (H, W, C) normalized features
        """
        normalized = np.zeros_like(features, dtype=np.float32)
        
        for i in range(features.shape[2]):
            channel = features[:, :, i]
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val > min_val:
                normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[:, :, i] = 0.0
        
        return normalized
    
    def get_rgb_from_features(self, features: np.ndarray) -> np.ndarray:
        """
        Extract RGB channels from feature array (if present)
        Assumes features contain RGB in some form
        
        Args:
            features: Feature array (H, W, C)
        
        Returns:
            rgb: RGB image (H, W, 3)
        """
        # This is a placeholder - you'd need to store original RGB separately
        # or reconstruct from color spaces
        raise NotImplementedError("RGB extraction from features not implemented")


if __name__ == '__main__':
    # Test feature extraction
    import matplotlib.pyplot as plt
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    extractor = FeatureExtractor()
    
    # Extract features
    luminance = extractor.extract_luminance(test_image)
    chrominance = extractor.extract_chrominance(test_image)
    all_features = extractor.extract_all_features(test_image)
    
    print(f"Feature names: {extractor.feature_names}")
    print(f"Luminance shape: {luminance.shape}")
    print(f"Chrominance shape: {chrominance.shape}")
    print(f"All features shape: {all_features.shape}")
    
    # Normalize
    normalized = extractor.normalize_features(all_features)
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, extractor.feature_names)):
        ax.imshow(normalized[:, :, i], cmap='viridis')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/pytorch_river_seg/test_features.png', dpi=150)
    print("\nTest visualization saved to test_features.png")
