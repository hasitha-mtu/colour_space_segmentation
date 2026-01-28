"""
Visualize RGB, Luminance, and Chrominance feature channels for paper figure.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple


class FeatureExtractor:
    """Extract luminance and chrominance features from RGB images"""
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Returns ordered list of feature names"""
        luminance = ['L (LAB)', 'L range', 'L texture']
        chrominance = ['H (HSV)', 'S (HSV)', 'a (LAB)', 'b (LAB)', 
                      'Cb (YCbCr)', 'Cr (YCbCr)', 'Intensity']
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
        """
        luminance = self.extract_luminance(image)
        chrominance = self.extract_chrominance(image)
        return np.concatenate([luminance, chrominance], axis=2)
    
    def normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize a single channel to [0, 1] range"""
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            return (channel - min_val) / (max_val - min_val)
        return np.zeros_like(channel)


def create_feature_visualization(image_path: str, output_path: str = 'feature_channels.png',
                                  figsize: Tuple[int, int] = (16, 10), dpi: int = 300):
    """
    Create a comprehensive visualization of RGB, luminance, and chrominance channels.
    
    Args:
        image_path: Path to input RGB image
        output_path: Path to save output figure
        figsize: Figure size in inches
        dpi: Output resolution
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    extractor = FeatureExtractor()
    luminance = extractor.extract_luminance(image)
    chrominance = extractor.extract_chrominance(image)
    
    # Feature names
    lum_names = ['L (LAB)', 'L range', 'L texture']
    chrom_names = ['H (HSV)', 'S (HSV)', 'a (LAB)', 'b (LAB)', 
                   'Cb (YCbCr)', 'Cr (YCbCr)', 'Intensity']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    
    # Layout: 3 rows
    # Row 1: RGB (larger) + 3 luminance channels
    # Row 2: 4 chrominance channels
    # Row 3: 3 chrominance channels + empty/legend
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.15)
    
    # Row 1: RGB image (spans 1 column) + 3 luminance
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(image)
    ax_rgb.set_title('(a) RGB', fontsize=11, fontweight='bold')
    ax_rgb.axis('off')
    
    # Luminance channels
    for i in range(3):
        ax = fig.add_subplot(gs[0, i+1])
        normalized = extractor.normalize_channel(luminance[:, :, i])
        im = ax.imshow(normalized, cmap='gray')
        ax.set_title(f'({chr(98+i)}) {lum_names[i]}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Row 2: First 4 chrominance channels
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        normalized = extractor.normalize_channel(chrominance[:, :, i])
        # Use different colormaps for different feature types
        if i == 0:  # Hue - use hsv colormap
            im = ax.imshow(normalized, cmap='hsv')
        elif i == 1:  # Saturation
            im = ax.imshow(normalized, cmap='gray')
        else:  # a, b channels - use diverging colormap
            im = ax.imshow(normalized, cmap='RdYlGn')
        ax.set_title(f'({chr(101+i)}) {chrom_names[i]}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Last 3 chrominance channels
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        normalized = extractor.normalize_channel(chrominance[:, :, i+4])
        if i < 2:  # Cb, Cr - use diverging colormap
            im = ax.imshow(normalized, cmap='RdYlBu')
        else:  # Intensity
            im = ax.imshow(normalized, cmap='gray')
        ax.set_title(f'({chr(105+i)}) {chrom_names[i+4]}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add category labels
    fig.text(0.5, 0.98, 'Luminance Features (3 channels)', ha='center', va='top', 
             fontsize=12, fontweight='bold', color='darkblue')
    fig.text(0.5, 0.64, 'Chrominance Features (7 channels)', ha='center', va='top', 
             fontsize=12, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved to: {output_path}")
    return output_path


def create_compact_visualization(image_path: str, output_path: str = 'feature_channels_compact.png',
                                  figsize: Tuple[int, int] = (14, 8), dpi: int = 300):
    """
    Create a compact 2-row visualization suitable for paper column width.
    
    Row 1: RGB + 3 Luminance channels (with "Luminance" label)
    Row 2: 7 Chrominance channels + mask (with "Chrominance" label)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    extractor = FeatureExtractor()
    luminance = extractor.extract_luminance(image)
    chrominance = extractor.extract_chrominance(image)
    
    # Feature names (shorter for compact view)
    lum_names = ['L', 'L_range', 'L_texture']
    chrom_names = ['H', 'S', 'a', 'b', 'Cb', 'Cr', 'I']
    
    # Create figure
    fig, axes = plt.subplots(2, 8, figsize=figsize)
    
    # Row 1: RGB + Luminance
    # RGB
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('RGB', fontsize=10)
    axes[0, 0].axis('off')
    
    # Luminance channels
    for i in range(3):
        normalized = extractor.normalize_channel(luminance[:, :, i])
        axes[0, i+1].imshow(normalized, cmap='gray')
        axes[0, i+1].set_title(lum_names[i], fontsize=10)
        axes[0, i+1].axis('off')
    
    # Hide unused axes in row 1
    for i in range(4, 8):
        axes[0, i].axis('off')
    
    # Row 2: Chrominance (7 channels)
    cmaps = ['hsv', 'gray', 'RdYlGn', 'RdYlGn', 'RdYlBu', 'RdYlBu', 'gray']
    for i in range(7):
        normalized = extractor.normalize_channel(chrominance[:, :, i])
        axes[1, i].imshow(normalized, cmap=cmaps[i])
        axes[1, i].set_title(chrom_names[i], fontsize=10)
        axes[1, i].axis('off')
    
    # Hide last axis
    axes[1, 7].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Luminance\n(3 ch)', ha='left', va='center', 
             fontsize=11, fontweight='bold', rotation=90)
    fig.text(0.02, 0.25, 'Chrominance\n(7 ch)', ha='left', va='center', 
             fontsize=11, fontweight='bold', rotation=90)
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Compact figure saved to: {output_path}")
    return output_path


def create_paper_figure(image_path: str, mask_path: str = None, 
                        output_path: str = 'fig_feature_channels.pdf',
                        figsize: Tuple[int, int] = (12, 9), dpi: int = 600):
    """
    Create a publication-ready figure for IEEE conference paper.
    Single column width (~3.5in) or double column width (~7.5in).
    
    Layout (3 rows x 4 columns):
    Row 1: (a) RGB, (b) L, (c) L_range, (d) L_texture  [+ Luminance label]
    Row 2: (e) H, (f) S, (g) a, (h) b                  [+ Chrominance label]
    Row 3: (i) Cb, (j) Cr, (k) Intensity, (l) Mask
    """
    # Set high-quality rendering parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'figure.dpi': 150,
        'savefig.dpi': dpi,
        'image.interpolation': 'lanczos',
        'image.resample': True,
    })
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load mask if provided
    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract features
    extractor = FeatureExtractor()
    luminance = extractor.extract_luminance(image)
    chrominance = extractor.extract_chrominance(image)
    
    # Create figure with larger size
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    
    # Titles with subfigure labels - larger font
    titles_row1 = ['(a) RGB', '(b) L', r'(c) L$_{\mathrm{range}}$', r'(d) L$_{\mathrm{texture}}$']
    titles_row2 = ['(e) H', '(f) S', '(g) a', '(h) b']
    titles_row3 = ['(i) Cb', '(j) Cr', '(k) Intensity', '(l) Ground Truth']
    
    title_fontsize = 14
    
    # Row 1: RGB + Luminance
    axes[0, 0].imshow(image, interpolation='lanczos')
    axes[0, 0].set_title(titles_row1[0], fontsize=title_fontsize, fontweight='bold', pad=8)
    axes[0, 0].axis('off')
    
    for i in range(3):
        normalized = extractor.normalize_channel(luminance[:, :, i])
        axes[0, i+1].imshow(normalized, cmap='gray', interpolation='lanczos')
        axes[0, i+1].set_title(titles_row1[i+1], fontsize=title_fontsize, fontweight='bold', pad=8)
        axes[0, i+1].axis('off')
    
    # Row 2: Chrominance (H, S, a, b)
    cmaps_row2 = ['hsv', 'gray', 'RdYlGn', 'RdYlGn']
    for i in range(4):
        normalized = extractor.normalize_channel(chrominance[:, :, i])
        axes[1, i].imshow(normalized, cmap=cmaps_row2[i], interpolation='lanczos')
        axes[1, i].set_title(titles_row2[i], fontsize=title_fontsize, fontweight='bold', pad=8)
        axes[1, i].axis('off')
    
    # Row 3: Chrominance (Cb, Cr, Intensity) + Mask
    cmaps_row3 = ['RdYlBu', 'RdYlBu', 'gray']
    for i in range(3):
        normalized = extractor.normalize_channel(chrominance[:, :, i+4])
        axes[2, i].imshow(normalized, cmap=cmaps_row3[i], interpolation='lanczos')
        axes[2, i].set_title(titles_row3[i], fontsize=title_fontsize, fontweight='bold', pad=8)
        axes[2, i].axis('off')
    
    # Mask or placeholder
    if mask is not None:
        axes[2, 3].imshow(mask, cmap='gray', interpolation='lanczos')
    else:
        # Create placeholder
        axes[2, 3].text(0.5, 0.5, 'Mask\nN/A', ha='center', va='center', 
                        transform=axes[2, 3].transAxes, fontsize=14, fontweight='bold')
    axes[2, 3].set_title(titles_row3[3], fontsize=title_fontsize, fontweight='bold', pad=8)
    axes[2, 3].axis('off')
    
    # Add category brackets/labels on the right side - larger font
    fig.text(0.99, 0.78, 'Luminance\n(3 ch)', ha='left', va='center', 
             fontsize=13, fontweight='bold', rotation=270)
    fig.text(0.99, 0.35, 'Chrominance\n(7 ch)', ha='left', va='center', 
             fontsize=13, fontweight='bold', rotation=270)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', 
                format=output_path.split('.')[-1], pad_inches=0.1)
    plt.close()
    
    # Reset rcParams to defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"Paper figure saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # import sys
    
    # if len(sys.argv) < 2:
    #     print("Usage: python visualize_features.py <image_path> [mask_path] [output_path]")
    #     print("\nExample:")
    #     print("  python visualize_features.py sample_image.png")
    #     print("  python visualize_features.py sample_image.png mask.png fig_features.pdf")
    #     sys.exit(1)
    
    # image_path = sys.argv[1]
    # mask_path = sys.argv[2] if len(sys.argv) > 2 else None
    # output_path = sys.argv[3] if len(sys.argv) > 3 else 'fig_feature_channels.pdf'

    image_path = 'dataset_512/images/DJI_20250728102007_0644_V_y2800_x4400.png'
    mask_path = 'dataset_512/masks/DJI_20250728102007_0644_V_y2800_x4400.png'
    output_path = 'experiments/test_features3.png'

    
    # Create publication figure
    create_paper_figure(image_path, mask_path, output_path)
