"""
Visualization script for comparing DeepLabv3+ models trained with RGB vs All Channels
For water segmentation in UAV imagery under tree canopy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List, Dict
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


from models.deeplabv3plus import DeepLabV3Plus

class SegmentationVisualizer:
    """Visualize and compare segmentation results from multiple models"""
    
    def __init__(self, 
                 rgb_model_path: str,
                 all_channels_model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize visualizer with trained models
        
        Args:
            rgb_model_path: Path to RGB-trained DeepLabv3+ checkpoint
            all_channels_model_path: Path to all-channels-trained DeepLabv3+ checkpoint
            device: Device to run inference on
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading RGB model...")
        self.rgb_model = self.load_model(rgb_model_path, in_channels=3)
        
        # Detect actual input channels
        if hasattr(self.rgb_model, 'in_channels'):
            self.rgb_channels = self.rgb_model.in_channels
        elif hasattr(self.rgb_model, 'channel_adapter') and self.rgb_model.channel_adapter is not None:
            self.rgb_channels = self.rgb_model.channel_adapter[0].in_channels
        else:
            self.rgb_channels = self.rgb_model.backbone.conv1.weight.shape[1]
        
        print("Loading All-Channels model...")
        self.all_channels_model = self.load_model(all_channels_model_path, in_channels=10)
        
        # Detect actual input channels
        if hasattr(self.all_channels_model, 'in_channels'):
            self.all_channels = self.all_channels_model.in_channels
        elif hasattr(self.all_channels_model, 'channel_adapter') and self.all_channels_model.channel_adapter is not None:
            self.all_channels = self.all_channels_model.channel_adapter[0].in_channels
        else:
            self.all_channels = self.all_channels_model.backbone.conv1.weight.shape[1]
        
        print("Models loaded successfully!")
        print(f"  RGB model uses {self.rgb_channels} input channels")
        print(f"  All-Channels model uses {self.all_channels} input channels")
        
        if self.rgb_channels == self.all_channels == 3:
            print("\n⚠️  WARNING: Both models use 3 RGB channels!")
            print("   Are you sure you're comparing different models?")
            print("   Check your model paths!")
        print()
        
    def load_model(self, checkpoint_path: str, in_channels: int):
        """Load a trained DeepLabv3+ model (supports both standard and channel adapter architectures)"""
        
        # Load checkpoint first to inspect it
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        # Check if this uses channel adapter architecture
        has_channel_adapter = 'channel_adapter.0.weight' in new_state_dict
        
        if has_channel_adapter:
            # This model uses channel adapter: in_channels → 3 → ResNet50
            actual_in_channels = new_state_dict['channel_adapter.0.weight'].shape[1]
            print(f"  Detected channel adapter: {actual_in_channels} → 3 channels")
            
            # Import custom DeepLabV3Plus
            try:
                print(f"  Using custom DeepLabV3Plus (channel adapter architecture)")
                
                model = DeepLabV3Plus(
                    in_channels=actual_in_channels,
                    num_classes=1,
                    pretrained=False  # We have trained weights
                )
            except ImportError as e:
                print(f"  ERROR: Cannot import custom DeepLabV3Plus: {e}")
                print(f"  Please ensure deeplabv3plus.py is in your Python path")
                raise
        else:
            # Standard architecture - check backbone.conv1 shape
            from torchvision.models.segmentation import deeplabv3_resnet50
            
            conv1_key = 'backbone.conv1.weight'
            if conv1_key in new_state_dict:
                actual_in_channels = new_state_dict[conv1_key].shape[1]
                print(f"  Detected standard architecture: {actual_in_channels} input channels")
            else:
                actual_in_channels = in_channels
                print(f"  Using expected {actual_in_channels} input channels")
            
            model = deeplabv3_resnet50(num_classes=1, weights=None)
            
            # Modify first conv layer if needed
            if actual_in_channels != 3:
                model.backbone.conv1 = torch.nn.Conv2d(
                    actual_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing and len(missing) > 5:  # Only warn if many keys missing
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected and len(unexpected) > 5:  # Ignore aux_classifier
            print(f"  Warning: {len(unexpected)} unexpected keys")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 10-channel features (3 luminance + 7 chrominance)
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            features: 10-channel feature array (H, W, 10)
        """
        # Convert to float
        img = image.astype(np.float32) / 255.0
        
        # RGB channels (luminance)
        rgb = img.copy()
        
        # HSV color space
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        
        # LAB color space
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        
        # YCbCr color space
        ycbcr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        
        # Stack all channels: RGB (3) + HSV (3) + LAB (3) + YCbCr (1, just Cb)
        # Total: 3 + 3 + 3 + 1 = 10 channels
        features = np.stack([
            rgb[:, :, 0],      # R
            rgb[:, :, 1],      # G
            rgb[:, :, 2],      # B
            hsv[:, :, 0],      # H
            hsv[:, :, 1],      # S
            hsv[:, :, 2],      # V
            lab[:, :, 1],      # A
            lab[:, :, 2],      # B
            ycbcr[:, :, 1],    # Cb
            ycbcr[:, :, 2],    # Cr
        ], axis=-1)
        
        return features
    
    @torch.no_grad()
    def predict(self, image: np.ndarray, model, use_all_channels: bool = False) -> np.ndarray:
        """
        Run inference on an image
        
        Args:
            image: RGB image (H, W, 3)
            model: Segmentation model
            use_all_channels: If True, extract all 10 channels (if model supports it)
            
        Returns:
            prediction: Binary mask (H, W) and probability map
        """
        # Check if model has channel adapter (custom DeepLabV3Plus)
        has_channel_adapter = hasattr(model, 'channel_adapter') and model.channel_adapter is not None
        
        if has_channel_adapter:
            # Custom model with channel adapter - determine required input
            actual_channels = model.in_channels
            
            if actual_channels == 10 and use_all_channels:
                # Extract all 10 channels
                input_data = self.extract_features(image)
            elif actual_channels == 3:
                # Just use RGB
                input_data = image.astype(np.float32) / 255.0
            else:
                # Use RGB as fallback
                input_data = image.astype(np.float32) / 255.0
                
        else:
            # Standard model - check backbone.conv1
            actual_channels = model.backbone.conv1.weight.shape[1]
            
            if actual_channels == 10 and use_all_channels:
                input_data = self.extract_features(image)
            elif actual_channels == 3:
                input_data = image.astype(np.float32) / 255.0
            else:
                input_data = image.astype(np.float32) / 255.0
        
        # Ensure input matches model's expected channels
        if input_data.shape[-1] != actual_channels:
            print(f"Warning: Input has {input_data.shape[-1]} channels but model expects {actual_channels}")
            if actual_channels == 3:
                # Use only RGB channels
                input_data = input_data[:, :, :3]
        
        # Convert to tensor (C, H, W)
        input_tensor = torch.from_numpy(input_data.transpose(2, 0, 1)).float()
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        output = model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            output = output['out']
        
        # Handle sigmoid already applied or not
        # Custom DeepLabV3Plus applies sigmoid internally, torchvision doesn't
        if has_channel_adapter:
            pred = output.squeeze().cpu().numpy()  # Already has sigmoid
        else:
            pred = torch.sigmoid(output).squeeze().cpu().numpy()  # Need to apply sigmoid
        
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        return pred_binary, pred
    
    def calculate_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Calculate segmentation metrics"""
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # Intersection and Union
        intersection = np.logical_and(pred_flat, gt_flat).sum()
        union = np.logical_or(pred_flat, gt_flat).sum()
        
        # True/False Positives/Negatives
        tp = intersection
        fp = (pred_flat & ~gt_flat).sum()
        fn = (~pred_flat & gt_flat).sum()
        tn = (~pred_flat & ~gt_flat).sum()
        
        # Metrics
        iou = intersection / (union + 1e-7)
        dice = 2 * intersection / (pred_flat.sum() + gt_flat.sum() + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'IoU': iou,
            'Dice': dice,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }
    
    def visualize_single_comparison(self, 
                                    image: np.ndarray,
                                    ground_truth: np.ndarray,
                                    save_path: str = None,
                                    show_probability: bool = False):
        """
        Create comprehensive visualization comparing both models
        
        Args:
            image: RGB image (H, W, 3)
            ground_truth: Ground truth mask (H, W)
            save_path: Path to save figure
            show_probability: If True, show probability maps instead of binary
        """
        # Get predictions
        rgb_pred_binary, rgb_pred_prob = self.predict(image, self.rgb_model, use_all_channels=False)
        all_pred_binary, all_pred_prob = self.predict(image, self.all_channels_model, use_all_channels=True)
        
        # Calculate metrics
        rgb_metrics = self.calculate_metrics(rgb_pred_binary, ground_truth)
        all_metrics = self.calculate_metrics(all_pred_binary, ground_truth)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Layout: 3 rows x 4 columns
        # Row 1: Image, GT, RGB pred, All channels pred
        # Row 2: RGB overlay, All overlay, Error maps
        # Row 3: Metrics comparison
        
        # Row 1: Main predictions
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(ground_truth, cmap='Blues')
        ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 4, 3)
        if show_probability:
            im3 = ax3.imshow(rgb_pred_prob, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            title_suffix = ' (Probability)'
        else:
            ax3.imshow(rgb_pred_binary, cmap='Blues')
            title_suffix = ''
        ax3.set_title(f'RGB Model ({self.rgb_channels} channels){title_suffix}\nIoU: {rgb_metrics["IoU"]:.3f}', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = plt.subplot(3, 4, 4)
        if show_probability:
            im4 = ax4.imshow(all_pred_prob, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im4, ax=ax4, fraction=0.046)
        else:
            ax4.imshow(all_pred_binary, cmap='Blues')
        ax4.set_title(f'All Channels Model ({self.all_channels} channels){title_suffix}\nIoU: {all_metrics["IoU"]:.3f}', 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Row 2: Overlays and comparisons
        # RGB overlay
        ax5 = plt.subplot(3, 4, 5)
        overlay_rgb = image.copy()
        overlay_rgb[rgb_pred_binary == 1] = [0, 255, 0]  # Green for prediction
        overlay_rgb[ground_truth == 1] = [255, 0, 0]     # Red for GT
        overlap = np.logical_and(rgb_pred_binary, ground_truth)
        overlay_rgb[overlap] = [255, 255, 0]              # Yellow for overlap
        ax5.imshow(overlay_rgb)
        ax5.set_title('RGB Model Overlay\n🟡 Correct 🟢 False Pos 🔴 False Neg', 
                     fontsize=10, fontweight='bold')
        ax5.axis('off')
        
        # All channels overlay
        ax6 = plt.subplot(3, 4, 6)
        overlay_all = image.copy()
        overlay_all[all_pred_binary == 1] = [0, 255, 0]
        overlay_all[ground_truth == 1] = [255, 0, 0]
        overlap = np.logical_and(all_pred_binary, ground_truth)
        overlay_all[overlap] = [255, 255, 0]
        ax6.imshow(overlay_all)
        ax6.set_title('All Channels Overlay\n🟡 Correct 🟢 False Pos 🔴 False Neg', 
                     fontsize=10, fontweight='bold')
        ax6.axis('off')
        
        # Error map RGB
        ax7 = plt.subplot(3, 4, 7)
        error_rgb = np.zeros((*rgb_pred_binary.shape, 3), dtype=np.uint8)
        error_rgb[np.logical_and(rgb_pred_binary, ground_truth)] = [0, 255, 0]  # TP: Green
        error_rgb[np.logical_and(rgb_pred_binary, ~ground_truth.astype(bool))] = [255, 165, 0]  # FP: Orange
        error_rgb[np.logical_and(~rgb_pred_binary.astype(bool), ground_truth)] = [255, 0, 0]  # FN: Red
        ax7.imshow(error_rgb)
        ax7.set_title('RGB Error Map\n🟢 TP 🟠 FP 🔴 FN', fontsize=10, fontweight='bold')
        ax7.axis('off')
        
        # Error map All channels
        ax8 = plt.subplot(3, 4, 8)
        error_all = np.zeros((*all_pred_binary.shape, 3), dtype=np.uint8)
        error_all[np.logical_and(all_pred_binary, ground_truth)] = [0, 255, 0]  # TP
        error_all[np.logical_and(all_pred_binary, ~ground_truth.astype(bool))] = [255, 165, 0]  # FP
        error_all[np.logical_and(~all_pred_binary.astype(bool), ground_truth)] = [255, 0, 0]  # FN
        ax8.imshow(error_all)
        ax8.set_title('All Channels Error Map\n🟢 TP 🟠 FP 🔴 FN', fontsize=10, fontweight='bold')
        ax8.axis('off')
        
        # Row 3: Metrics comparison
        ax9 = plt.subplot(3, 4, 9)
        metrics_names = list(rgb_metrics.keys())
        rgb_values = list(rgb_metrics.values())
        all_values = list(all_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax9.bar(x - width/2, rgb_values, width, 
                       label=f'RGB ({self.rgb_channels}ch)', alpha=0.8, color='steelblue')
        bars2 = ax9.bar(x + width/2, all_values, width, 
                       label=f'All Channels ({self.all_channels}ch)', alpha=0.8, color='coral')
        
        ax9.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax9.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Difference heatmap
        ax10 = plt.subplot(3, 4, 10)
        difference = rgb_pred_binary.astype(int) - all_pred_binary.astype(int)
        im10 = ax10.imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
        ax10.set_title('Prediction Difference\n(RGB - All Channels)', fontsize=11, fontweight='bold')
        ax10.axis('off')
        plt.colorbar(im10, ax=ax10, fraction=0.046)
        
        # Confusion matrices
        ax11 = plt.subplot(3, 4, 11)
        cm_rgb = confusion_matrix(ground_truth.flatten(), rgb_pred_binary.flatten())
        sns.heatmap(cm_rgb, annot=True, fmt='d', cmap='Blues', ax=ax11, 
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax11.set_title('RGB Confusion Matrix', fontsize=11, fontweight='bold')
        ax11.set_ylabel('True')
        ax11.set_xlabel('Predicted')
        
        ax12 = plt.subplot(3, 4, 12)
        cm_all = confusion_matrix(ground_truth.flatten(), all_pred_binary.flatten())
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Oranges', ax=ax12,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax12.set_title('All Channels Confusion Matrix', fontsize=11, fontweight='bold')
        ax12.set_ylabel('True')
        ax12.set_xlabel('Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig, (rgb_metrics, all_metrics)
    
    def visualize_batch(self,
                       images: List[np.ndarray],
                       ground_truths: List[np.ndarray],
                       output_dir: str,
                       image_names: List[str] = None):
        """
        Visualize multiple images and save individual comparisons
        
        Args:
            images: List of RGB images
            ground_truths: List of ground truth masks
            output_dir: Directory to save visualizations
            image_names: Optional names for saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_rgb_metrics = []
        all_channel_metrics = []
        
        for i, (img, gt) in enumerate(zip(images, ground_truths)):
            name = image_names[i] if image_names else f"sample_{i:04d}"
            save_path = output_path / f"{name}_comparison.png"
            
            print(f"\nProcessing {name}...")
            fig, (rgb_met, all_met) = self.visualize_single_comparison(
                img, gt, save_path=str(save_path)
            )
            
            all_rgb_metrics.append(rgb_met)
            all_channel_metrics.append(all_met)
            
            plt.close(fig)
        
        # Create summary statistics
        self.create_summary_report(all_rgb_metrics, all_channel_metrics, output_dir)
    
    def create_summary_report(self,
                             rgb_metrics: List[Dict],
                             all_metrics: List[Dict],
                             output_dir: str):
        """Create summary statistics and comparison plots"""
        output_path = Path(output_dir)
        
        # Convert to arrays
        metrics_names = list(rgb_metrics[0].keys())
        rgb_array = np.array([[m[k] for k in metrics_names] for m in rgb_metrics])
        all_array = np.array([[m[k] for k in metrics_names] for m in all_metrics])
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Comparison Summary Statistics\nRGB ({self.rgb_channels}ch) vs All Channels ({self.all_channels}ch)', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric_name in enumerate(metrics_names):
            ax = axes[idx // 3, idx % 3]
            
            rgb_vals = rgb_array[:, idx]
            all_vals = all_array[:, idx]
            
            # Box plot
            bp = ax.boxplot([rgb_vals, all_vals],
                           labels=[f'RGB\n({self.rgb_channels}ch)', f'All Channels\n({self.all_channels}ch)'],
                           patch_artist=True)
            
            bp['boxes'][0].set_facecolor('steelblue')
            bp['boxes'][1].set_facecolor('coral')
            
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            rgb_mean = rgb_vals.mean()
            all_mean = all_vals.mean()
            ax.text(1, rgb_mean, f'μ={rgb_mean:.3f}', ha='right', va='center', fontsize=9)
            ax.text(2, all_mean, f'μ={all_mean:.3f}', ha='left', va='center', fontsize=9)
        
        # Hide extra subplot
        if len(metrics_names) < 6:
            axes.flatten()[len(metrics_names)].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved summary statistics to {output_path / 'summary_statistics.png'}")
        
        # Print numerical summary
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print(f"RGB Model: {self.rgb_channels} channels")
        print(f"All Channels Model: {self.all_channels} channels")
        print("="*60)
        for idx, metric_name in enumerate(metrics_names):
            rgb_vals = rgb_array[:, idx]
            all_vals = all_array[:, idx]
            
            print(f"\n{metric_name}:")
            print(f"  RGB ({self.rgb_channels}ch):     Mean={rgb_vals.mean():.4f}, Std={rgb_vals.std():.4f}, "
                  f"Min={rgb_vals.min():.4f}, Max={rgb_vals.max():.4f}")
            print(f"  All Channels ({self.all_channels}ch): Mean={all_vals.mean():.4f}, Std={all_vals.std():.4f}, "
                  f"Min={all_vals.min():.4f}, Max={all_vals.max():.4f}")
            
            diff = all_vals.mean() - rgb_vals.mean()
            print(f"  Difference:   {diff:+.4f} {'(All Channels better)' if diff > 0 else '(RGB better)'}")
        print("="*60)
        
        plt.close()


def load_data_from_directory(image_dir: str, 
                             mask_dir: str,
                             limit: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load images and masks from directories
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        limit: Maximum number of images to load
        
    Returns:
        images, masks, filenames
    """
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    # Get all image files
    image_files = sorted(list(image_path.glob('*.png')) + 
                        list(image_path.glob('*.jpg')) + 
                        list(image_path.glob('*.jpeg')))
    
    if limit:
        image_files = image_files[:limit]
    
    images = []
    masks = []
    names = []
    
    print(f"Loading {len(image_files)} images...")
    
    for img_file in image_files:
        # Load image
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find corresponding mask
        mask_file = mask_path / img_file.name
        if not mask_file.exists():
            # Try alternative extensions
            mask_file = mask_path / (img_file.stem + '.png')
        
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)  # Binarize
            
            images.append(img)
            masks.append(mask)
            names.append(img_file.stem)
        else:
            print(f"Warning: No mask found for {img_file.name}")
    
    print(f"Loaded {len(images)} image-mask pairs")
    return images, masks, names


def main():
    """Main execution function"""
    
   # Configuration - UPDATE THESE PATHS!
    RGB_MODEL_PATH = "./experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth"  
    ALL_CHANNELS_MODEL_PATH = "./experiments/results/baseline/deeplabv3plus/all/checkpoints/best_model.pth"  
    
    IMAGE_DIR = "./dataset/test/images"  
    MASK_DIR = "./dataset/test/masks"    
    
    OUTPUT_DIR = "./experiment/visualizations/deeplabv3plus"
    
    # How many samples to visualize (set to None for all)
    NUM_SAMPLES = None
    
    print("="*60)
    print("DeepLabv3+ Model Comparison Visualization")
    print("="*60)
    
    # Initialize visualizer
    visualizer = SegmentationVisualizer(
        rgb_model_path=RGB_MODEL_PATH,
        all_channels_model_path=ALL_CHANNELS_MODEL_PATH
    )
    
    # Load test data
    print(f"\nLoading test data from:")
    print(f"  Images: {IMAGE_DIR}")
    print(f"  Masks:  {MASK_DIR}")
    
    images, masks, names = load_data_from_directory(
        IMAGE_DIR, 
        MASK_DIR, 
        limit=NUM_SAMPLES
    )
    
    if len(images) == 0:
        print("ERROR: No images loaded! Please check your paths.")
        return
    
    # Visualize batch
    print(f"\nGenerating visualizations for {len(images)} samples...")
    visualizer.visualize_batch(images, masks, OUTPUT_DIR, names)
    
    print("\n" + "="*60)
    print(f"Visualization complete! Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
