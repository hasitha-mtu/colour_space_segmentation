"""
Test Model Performance on Test Dataset
======================================
Comprehensive evaluation script for trained segmentation models
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import argparse
import sys

# Try to import custom model
try:
    from models.deeplabv3plus import DeepLabV3Plus
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    CUSTOM_MODEL_AVAILABLE = False
    print("Warning: Custom DeepLabV3Plus not found. Only standard models will work.")


class ModelTester:
    """Test segmentation model on test dataset"""
    
    def __init__(
        self,
        model_path: str,
        test_image_dir: str,
        test_mask_dir: str,
        feature_config: str = 'rgb',
        device: str = None,
        save_predictions: bool = False,
        output_dir: str = None
    ):
        """
        Initialize model tester
        
        Args:
            model_path: Path to trained model checkpoint
            test_image_dir: Directory containing test images
            test_mask_dir: Directory containing test masks
            feature_config: 'rgb', 'luminance', 'chrominance', or 'all'
            device: Device to use (auto-detect if None)
            save_predictions: Whether to save prediction masks
            output_dir: Directory to save results
        """
        self.model_path = Path(model_path)
        self.test_image_dir = Path(test_image_dir)
        self.test_mask_dir = Path(test_mask_dir)
        self.feature_config = feature_config
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir) if output_dir else Path('test_results')
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*70}")
        print(f"Model Performance Testing")
        print(f"{'='*70}")
        print(f"Model: {self.model_path.name}")
        print(f"Feature config: {feature_config}")
        print(f"Device: {self.device}")
        print(f"Test images: {test_image_dir}")
        print(f"Test masks: {test_mask_dir}")
        print(f"{'='*70}\n")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Get input channels
        self.in_channels = self._get_input_channels()
        print(f"Model input channels: {self.in_channels}")
        
        # Create output directory
        if self.save_predictions:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Predictions will be saved to: {self.output_dir}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        print("Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
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
        
        # Check for channel adapter architecture
        has_channel_adapter = 'channel_adapter.0.weight' in new_state_dict
        
        if has_channel_adapter:
            # Custom DeepLabV3Plus with channel adapter
            if not CUSTOM_MODEL_AVAILABLE:
                raise ImportError("Custom DeepLabV3Plus required but not found!")
            
            in_channels = new_state_dict['channel_adapter.0.weight'].shape[1]
            print(f"  Detected channel adapter architecture: {in_channels} → 3 channels")
            
            model = DeepLabV3Plus(
                in_channels=in_channels,
                num_classes=1,
                pretrained=False
            )
        else:
            # Standard torchvision DeepLabv3
            from torchvision.models.segmentation import deeplabv3_resnet50
            
            conv1_key = 'backbone.conv1.weight'
            if conv1_key in new_state_dict:
                in_channels = new_state_dict[conv1_key].shape[1]
            else:
                in_channels = 3  # Default
            
            print(f"  Detected standard architecture: {in_channels} input channels")
            
            model = deeplabv3_resnet50(num_classes=1, weights=None)
            
            if in_channels != 3:
                model.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing and len(missing) > 5:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected and len(unexpected) > 5:
            print(f"  Warning: {len(unexpected)} unexpected keys")
        
        model = model.to(self.device)
        print(f"  Model loaded successfully!")
        
        return model
    
    def _get_input_channels(self) -> int:
        """Get number of input channels expected by model"""
        if hasattr(self.model, 'in_channels'):
            return self.model.in_channels
        elif hasattr(self.model, 'channel_adapter') and self.model.channel_adapter is not None:
            return self.model.channel_adapter[0].in_channels
        else:
            return self.model.backbone.conv1.weight.shape[1]
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract multi-channel features from RGB image
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            features: Feature array (H, W, C) where C depends on config
        """
        img = image.astype(np.float32) / 255.0
        
        if self.feature_config == 'rgb' or self.feature_config == 'luminance':
            # Just RGB
            return img
        
        # Extract color space features
        rgb = img.copy()
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        ycbcr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
        
        if self.feature_config == 'chrominance':
            # 7 channels: HSV (3) + LAB (2) + YCbCr (2)
            features = np.stack([
                hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
                lab[:, :, 1], lab[:, :, 2],
                ycbcr[:, :, 1], ycbcr[:, :, 2],
            ], axis=-1)
        elif self.feature_config == 'all':
            # 10 channels: RGB (3) + HSV (3) + LAB (2) + YCbCr (2)
            features = np.stack([
                rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
                hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
                lab[:, :, 1], lab[:, :, 2],
                ycbcr[:, :, 1], ycbcr[:, :, 2],
            ], axis=-1)
        else:
            raise ValueError(f"Unknown feature config: {self.feature_config}")
        
        return features
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on an image
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            pred_binary: Binary prediction (H, W)
            pred_prob: Probability map (H, W)
        """
        # Extract features based on config
        if self.in_channels == 3:
            input_data = image.astype(np.float32) / 255.0
        else:
            input_data = self.extract_features(image)
        
        # Ensure correct number of channels
        if input_data.shape[-1] != self.in_channels:
            if self.in_channels == 3:
                input_data = input_data[:, :, :3]
            else:
                raise ValueError(f"Feature extraction produced {input_data.shape[-1]} channels, "
                               f"but model expects {self.in_channels}")
        
        # Convert to tensor
        input_tensor = torch.from_numpy(input_data.transpose(2, 0, 1)).float()
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            output = output['out']
        
        # Check if sigmoid already applied (custom DeepLabV3Plus)
        has_channel_adapter = hasattr(self.model, 'channel_adapter')
        if has_channel_adapter:
            pred_prob = output.squeeze().cpu().numpy()  # Already has sigmoid
        else:
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        pred_binary = (pred_prob > 0.5).astype(np.uint8)
        
        return pred_binary, pred_prob
    
    def calculate_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate segmentation metrics
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Dictionary of metrics
        """
        pred_flat = pred.flatten().astype(bool)
        gt_flat = gt.flatten().astype(bool)
        
        # Intersection and Union
        intersection = np.logical_and(pred_flat, gt_flat).sum()
        union = np.logical_or(pred_flat, gt_flat).sum()
        
        # True/False Positives/Negatives
        tp = intersection
        fp = np.logical_and(pred_flat, ~gt_flat).sum()
        fn = np.logical_and(~pred_flat, gt_flat).sum()
        tn = np.logical_and(~pred_flat, ~gt_flat).sum()
        
        # Metrics
        epsilon = 1e-7
        iou = intersection / (union + epsilon)
        dice = 2 * intersection / (pred_flat.sum() + gt_flat.sum() + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)
        
        return {
            'IoU': float(iou),
            'Dice': float(dice),
            'F1': float(f1),
            'Precision': float(precision),
            'Recall': float(recall),
            'Accuracy': float(accuracy),
            'Specificity': float(specificity),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        }
    
    def load_test_data(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Load all test images and masks
        
        Returns:
            List of (image, mask, filename) tuples
        """
        print("\nLoading test data...")
        
        # Get all image files
        image_files = sorted(
            list(self.test_image_dir.glob('*.png')) + 
            list(self.test_image_dir.glob('*.jpg')) + 
            list(self.test_image_dir.glob('*.jpeg'))
        )
        
        data = []
        missing_masks = []
        
        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find corresponding mask
            mask_file = self.test_mask_dir / img_file.name
            if not mask_file.exists():
                mask_file = self.test_mask_dir / (img_file.stem + '.png')
            
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.uint8)
                
                data.append((img, mask, img_file.stem))
            else:
                missing_masks.append(img_file.name)
        
        if missing_masks:
            print(f"  Warning: {len(missing_masks)} images without masks")
        
        print(f"  Loaded {len(data)} test samples")
        return data
    
    def test(self) -> Dict:
        """
        Run testing on all test samples
        
        Returns:
            Dictionary containing all results and statistics
        """
        # Load test data
        test_data = self.load_test_data()
        
        if len(test_data) == 0:
            raise ValueError("No test data found!")
        
        # Run inference on all samples
        print("\nRunning inference...")
        all_metrics = []
        sample_results = []
        
        for img, mask, filename in tqdm(test_data, desc="Testing"):
            # Predict
            pred_binary, pred_prob = self.predict(img)
            
            # Calculate metrics
            metrics = self.calculate_metrics(pred_binary, mask)
            metrics['filename'] = filename
            all_metrics.append(metrics)
            
            # Save prediction if requested
            if self.save_predictions:
                pred_save_path = self.output_dir / 'predictions' / f"{filename}_pred.png"
                pred_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pred_save_path), pred_binary * 255)
                
                # Save probability map
                prob_save_path = self.output_dir / 'probabilities' / f"{filename}_prob.png"
                prob_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(prob_save_path), (pred_prob * 255).astype(np.uint8))
            
            sample_results.append({
                'filename': filename,
                'metrics': metrics
            })
        
        # Calculate statistics
        print("\nCalculating statistics...")
        metric_names = ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'Accuracy', 'Specificity']
        
        statistics = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            statistics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        # Aggregate confusion matrix
        total_tp = sum(m['TP'] for m in all_metrics)
        total_fp = sum(m['FP'] for m in all_metrics)
        total_fn = sum(m['FN'] for m in all_metrics)
        total_tn = sum(m['TN'] for m in all_metrics)
        
        results = {
            'model_path': str(self.model_path),
            'feature_config': self.feature_config,
            'num_samples': len(test_data),
            'statistics': statistics,
            'confusion_matrix': {
                'TP': int(total_tp),
                'FP': int(total_fp),
                'FN': int(total_fn),
                'TN': int(total_tn)
            },
            'sample_results': sample_results
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Model: {results['model_path']}")
        print(f"Feature config: {results['feature_config']}")
        print(f"Number of samples: {results['num_samples']}")
        print("="*70)
        
        print("\nPER-METRIC STATISTICS:")
        print("-"*70)
        
        stats = results['statistics']
        for metric_name in ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'Accuracy', 'Specificity']:
            s = stats[metric_name]
            print(f"\n{metric_name}:")
            print(f"  Mean:   {s['mean']:.4f} ± {s['std']:.4f}")
            print(f"  Median: {s['median']:.4f}")
            print(f"  Range:  [{s['min']:.4f}, {s['max']:.4f}]")
            print(f"  IQR:    [{s['q25']:.4f}, {s['q75']:.4f}]")
        
        print("\n" + "-"*70)
        print("AGGREGATE CONFUSION MATRIX:")
        print("-"*70)
        cm = results['confusion_matrix']
        print(f"  True Positives:  {cm['TP']:,}")
        print(f"  False Positives: {cm['FP']:,}")
        print(f"  False Negatives: {cm['FN']:,}")
        print(f"  True Negatives:  {cm['TN']:,}")
        
        total = cm['TP'] + cm['FP'] + cm['FN'] + cm['TN']
        print(f"  Total pixels:    {total:,}")
        
        print("\n" + "="*70)
    
    def save_results(self, results: Dict, filename: str = 'test_results.json'):
        """Save results to JSON file"""
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test model performance on test dataset')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_image_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--test_mask_dir', type=str, required=True,
                       help='Directory containing test masks')
    
    # Optional arguments
    parser.add_argument('--feature_config', type=str, default='rgb',
                       choices=['rgb', 'luminance', 'chrominance', 'all'],
                       help='Feature configuration used during training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create tester
    tester = ModelTester(
        model_path=args.model_path,
        test_image_dir=args.test_image_dir,
        test_mask_dir=args.test_mask_dir,
        feature_config=args.feature_config,
        device=args.device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Run testing
    results = tester.test()
    
    # Print and save results
    tester.print_results(results)
    tester.save_results(results)
    
    print("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
