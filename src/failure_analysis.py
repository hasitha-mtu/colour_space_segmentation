"""
Failure Mode Analysis and Qualitative Visualization
====================================================
Analyzes where and why models fail:

1. Per-scene difficulty analysis (shadow level, vegetation density)
2. Qualitative prediction visualizations (best/worst cases)
3. Error pattern identification
4. Boundary accuracy analysis

Critical for paper discussion section:
- What conditions cause failures?
- Do engineered features help in specific scenarios?
- Where do foundation models excel vs. struggle?
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_dataloaders
from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.hybrid_dinov2 import HybridDINOv2
from src.models.sam_models import SAMEncoderDecoder, SAMFineTuned


class SceneDifficultyAnalyzer:
    """Analyze scene difficulty characteristics"""
    
    @staticmethod
    def estimate_shadow_level(image: np.ndarray) -> float:
        """
        Estimate percentage of image in shadow
        
        Uses V channel (brightness) from HSV
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            Shadow percentage (0-1)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        
        # Pixels with V < 100 (out of 255) considered shadowed
        shadow_threshold = 100
        shadow_pixels = (v_channel < shadow_threshold).sum()
        
        return shadow_pixels / v_channel.size
    
    @staticmethod
    def estimate_vegetation_density(image: np.ndarray) -> float:
        """
        Estimate vegetation density
        
        Uses hue and saturation to identify green pixels
        
        Args:
            image: RGB image (H, W, 3)
        
        Returns:
            Vegetation percentage (0-1)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        
        # Green hue range (approximately 35-85 in OpenCV's 0-180 scale)
        # High saturation indicates vivid green (vegetation)
        green_mask = ((h_channel >= 35) & (h_channel <= 85) & (s_channel > 50))
        
        return green_mask.sum() / green_mask.size
    
    @staticmethod
    def compute_water_coverage(mask: np.ndarray) -> float:
        """
        Compute water coverage in mask
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            Water coverage (0-1)
        """
        return mask.sum() / mask.size
    
    @staticmethod
    def categorize_difficulty(shadow_level: float, veg_density: float, 
                              water_coverage: float) -> str:
        """
        Categorize scene difficulty
        
        Args:
            shadow_level: Shadow percentage
            veg_density: Vegetation density
            water_coverage: Water coverage
        
        Returns:
            Difficulty category: 'easy', 'moderate', 'hard', 'extreme'
        """
        difficulty_score = 0
        
        # Shadow penalty
        if shadow_level > 0.7:
            difficulty_score += 3
        elif shadow_level > 0.5:
            difficulty_score += 2
        elif shadow_level > 0.3:
            difficulty_score += 1
        
        # Vegetation penalty
        if veg_density > 0.6:
            difficulty_score += 2
        elif veg_density > 0.4:
            difficulty_score += 1
        
        # Class imbalance penalty (very small or very large water)
        if water_coverage < 0.02 or water_coverage > 0.8:
            difficulty_score += 1
        
        # Categorize
        if difficulty_score >= 5:
            return 'extreme'
        elif difficulty_score >= 3:
            return 'hard'
        elif difficulty_score >= 1:
            return 'moderate'
        else:
            return 'easy'


class FailureModeAnalyzer:
    """Analyze model failures and generate qualitative visualizations"""
    
    def __init__(
        self,
        data_dir: str = 'data',
        output_dir: str = 'failure_analysis',
        device: str = None
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.difficulty_analyzer = SceneDifficultyAnalyzer()
        
        print(f"Failure Mode Analyzer initialized")
        print(f"  Output directory: {self.output_dir}")
    
    def _load_model(self, model_path: Path, model_config: Dict) -> torch.nn.Module:
        """Load trained model"""
        
        model_type = model_config['type']
        feature_config = model_config['feature_config']
        
        # Determine input channels
        channel_map = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
        in_channels = channel_map[feature_config]
        
        # Create model
        if model_type == 'unet':
            model = UNet(in_channels=in_channels, num_classes=1)
        elif model_type == 'deeplabv3plus':
            model = DeepLabV3Plus(in_channels=in_channels, num_classes=1, pretrained=False)
        elif model_type == 'dinov2':
            model = HybridDINOv2(
                in_channels=in_channels,
                num_classes=1,
                dino_model='dinov2_vitb14',
                freeze_dino=True
            )
        elif model_type == 'sam_encoder':
            model = SAMEncoderDecoder(
                sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
                model_type='vit_b'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def analyze_per_scene_performance(
        self,
        model_path: Path,
        model_config: Dict,
        n_samples: int = 100
    ) -> Dict:
        """
        Analyze performance stratified by scene difficulty
        
        Key Question: Do engineered features help in challenging scenarios?
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration dict
            n_samples: Number of samples to analyze
        
        Returns:
            Dictionary with per-difficulty-level performance
        """
        model = self._load_model(model_path, model_config)
        
        # Get dataloader
        _, val_loader = get_dataloaders(
            data_root=self.data_dir,
            feature_config=model_config['feature_config'],
            batch_size=1,  # Process one at a time for scene analysis
            num_workers=2,
            image_size=(512, 512),
            train_split=0.8,
            seed=42,
            normalize=model_config.get('normalize', True)
        )
        
        # Accumulators by difficulty
        results_by_difficulty = {
            'easy': {'ious': [], 'samples': []},
            'moderate': {'ious': [], 'samples': []},
            'hard': {'ious': [], 'samples': []},
            'extreme': {'ious': [], 'samples': []}
        }
        
        scene_metadata = []
        
        print(f"\nAnalyzing {model_config['name']}...")
        
        with torch.no_grad():
            for idx, (image, mask) in enumerate(tqdm(val_loader, desc="  Processing", total=n_samples)):
                if idx >= n_samples:
                    break
                
                # Convert to numpy for analysis
                image_np = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # Handle different channel counts
                if image_np.shape[2] == 3:
                    image_rgb = image_np
                else:
                    # For multi-channel, just use first 3 channels as proxy
                    image_rgb = image_np[:, :, :3]
                
                mask_np = mask[0, 0].numpy()
                
                # Analyze scene difficulty
                shadow_level = self.difficulty_analyzer.estimate_shadow_level(image_rgb)
                veg_density = self.difficulty_analyzer.estimate_vegetation_density(image_rgb)
                water_coverage = self.difficulty_analyzer.compute_water_coverage(mask_np)
                
                difficulty = self.difficulty_analyzer.categorize_difficulty(
                    shadow_level, veg_density, water_coverage
                )
                
                # Predict
                image_tensor = image.to(self.device)
                prediction = model(image_tensor)
                pred_np = prediction[0, 0].cpu().numpy()
                
                # Compute IoU
                pred_binary = (pred_np >= 0.5).astype(bool)
                mask_binary = mask_np.astype(bool)
                
                intersection = np.logical_and(pred_binary, mask_binary).sum()
                union = np.logical_or(pred_binary, mask_binary).sum()
                iou = intersection / union if union > 0 else 1.0
                
                # Store
                results_by_difficulty[difficulty]['ious'].append(iou)
                results_by_difficulty[difficulty]['samples'].append(idx)
                
                scene_metadata.append({
                    'idx': idx,
                    'difficulty': difficulty,
                    'shadow_level': shadow_level,
                    'veg_density': veg_density,
                    'water_coverage': water_coverage,
                    'iou': iou
                })
        
        # Compute statistics
        summary = {}
        for difficulty in ['easy', 'moderate', 'hard', 'extreme']:
            ious = results_by_difficulty[difficulty]['ious']
            if ious:
                summary[difficulty] = {
                    'mean_iou': np.mean(ious),
                    'std_iou': np.std(ious),
                    'n_samples': len(ious)
                }
            else:
                summary[difficulty] = {
                    'mean_iou': 0.0,
                    'std_iou': 0.0,
                    'n_samples': 0
                }
        
        return {
            'summary': summary,
            'metadata': scene_metadata,
            'model': model_config['name']
        }
    
    def visualize_predictions(
        self,
        model_path: Path,
        model_config: Dict,
        n_visualize: int = 6,
        mode: str = 'best_worst'
    ) -> None:
        """
        Generate qualitative prediction visualizations
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration
            n_visualize: Number of examples to visualize
            mode: 'best_worst' (3 best + 3 worst) or 'random'
        """
        model = self._load_model(model_path, model_config)
        
        # Get dataloader
        _, val_loader = get_dataloaders(
            data_root=self.data_dir,
            feature_config=model_config['feature_config'],
            batch_size=1,
            num_workers=2,
            image_size=(512, 512),
            train_split=0.8,
            seed=42,
            normalize=model_config.get('normalize', True)
        )
        
        # Collect predictions with IoUs
        predictions = []
        
        print(f"\nGenerating predictions for {model_config['name']}...")
        
        with torch.no_grad():
            for idx, (image, mask) in enumerate(tqdm(val_loader, desc="  Predicting")):
                if idx >= 50:  # Limit to first 50 for speed
                    break
                
                # Predict
                image_tensor = image.to(self.device)
                prediction = model(image_tensor)
                
                # Compute IoU
                pred_np = prediction[0, 0].cpu().numpy()
                mask_np = mask[0, 0].numpy()
                
                pred_binary = (pred_np >= 0.5).astype(bool)
                mask_binary = mask_np.astype(bool)
                
                intersection = np.logical_and(pred_binary, mask_binary).sum()
                union = np.logical_or(pred_binary, mask_binary).sum()
                iou = intersection / union if union > 0 else 1.0
                
                # Get RGB image for visualization
                image_np = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if image_np.shape[2] > 3:
                    image_rgb = image_np[:, :, :3]  # Take first 3 channels
                else:
                    image_rgb = image_np
                
                predictions.append({
                    'idx': idx,
                    'image': image_rgb,
                    'mask': mask_np,
                    'prediction': pred_np,
                    'iou': iou
                })
        
        # Select examples
        if mode == 'best_worst':
            # Sort by IoU
            predictions_sorted = sorted(predictions, key=lambda x: x['iou'])
            
            # Take 3 worst and 3 best
            worst = predictions_sorted[:n_visualize//2]
            best = predictions_sorted[-(n_visualize//2):]
            selected = worst + best
            
        else:  # random
            selected = np.random.choice(predictions, n_visualize, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(n_visualize, 4, figsize=(16, 4*n_visualize))
        
        if n_visualize == 1:
            axes = axes.reshape(1, -1)
        
        for i, pred_data in enumerate(selected):
            # Column 1: Original image
            axes[i, 0].imshow(pred_data['image'])
            axes[i, 0].set_title(f"Input (IoU: {pred_data['iou']:.3f})")
            axes[i, 0].axis('off')
            
            # Column 2: Ground truth
            axes[i, 1].imshow(pred_data['mask'], cmap='binary')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Column 3: Prediction
            axes[i, 2].imshow(pred_data['prediction'], cmap='binary', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Column 4: Error map
            error_map = np.abs(pred_data['mask'] - (pred_data['prediction'] > 0.5).astype(float))
            axes[i, 3].imshow(error_map, cmap='hot')
            axes[i, 3].set_title('Error (white = incorrect)')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{model_config['name']}_predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization: {output_path}")
    
    def compare_models_on_difficult_scenes(
        self,
        model_configs: List[Dict],
        difficulty: str = 'extreme'
    ) -> None:
        """
        Compare multiple models on difficult scenes
        
        Key Question: Which model type handles extreme conditions best?
        
        Args:
            model_configs: List of model configuration dicts
            difficulty: Difficulty level to analyze
        """
        print(f"\n{'='*70}")
        print(f"COMPARING MODELS ON {difficulty.upper()} SCENES")
        print(f"{'='*70}")
        
        results = []
        
        for config in model_configs:
            analysis = self.analyze_per_scene_performance(
                model_path=Path(config['checkpoint']),
                model_config=config,
                n_samples=100
            )
            
            if difficulty in analysis['summary']:
                results.append({
                    'model': config['name'],
                    'mean_iou': analysis['summary'][difficulty]['mean_iou'],
                    'std_iou': analysis['summary'][difficulty]['std_iou'],
                    'n_samples': analysis['summary'][difficulty]['n_samples']
                })
        
        # Create comparison plot
        if results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = [r['model'] for r in results]
            means = [r['mean_iou'] for r in results]
            stds = [r['std_iou'] for r in results]
            
            x = np.arange(len(models))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
            
            # Color bars by model type
            colors = []
            for model in models:
                if 'RGB' in model or 'rgb' in model:
                    colors.append('#2ecc71')  # Green for RGB
                else:
                    colors.append('#3498db')  # Blue for others
            
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(f'IoU on {difficulty.capitalize()} Scenes')
            ax.set_title(f'Performance Comparison: {difficulty.capitalize()} Difficulty')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='RGB'),
                Patch(facecolor='#3498db', label='Engineered Features')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            output_path = self.output_dir / f"comparison_{difficulty}_scenes.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved comparison: {output_path}")
            
            # Print results
            print("\nRESULTS:")
            print("-" * 70)
            for r in results:
                print(f"{r['model']:30} | IoU: {r['mean_iou']:.4f} ± {r['std_iou']:.4f} | "
                      f"n={r['n_samples']}")
            print("="*70)


def main():
    """Main failure analysis script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze failure modes')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='failure_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FailureModeAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Example model configurations
    # NOTE: User needs to update these paths based on their actual trained models
    model_configs = [
        {
            'name': 'UNet-RGB',
            'type': 'unet',
            'feature_config': 'rgb',
            'checkpoint': 'experiments/results/baseline/unet/rgb/checkpoints/best_model.pth',
            'normalize': True
        },
        {
            'name': 'UNet-All',
            'type': 'unet',
            'feature_config': 'all',
            'checkpoint': 'experiments/results/baseline/unet/all/checkpoints/best_model.pth',
            'normalize': True
        },
        {
            'name': 'DINOv2-RGB',
            'type': 'dinov2',
            'feature_config': 'rgb',
            'checkpoint': 'experiments/results/dinov2/rgb-frozen/checkpoints/best_model.pth',
            'normalize': False
        }
    ]
    
    print("\n" + "="*70)
    print("FAILURE MODE ANALYSIS")
    print("="*70)
    print("\nThis analysis will:")
    print("1. Evaluate models on scenes stratified by difficulty")
    print("2. Generate qualitative visualizations")
    print("3. Compare RGB vs. engineered features on challenging scenes")
    print("="*70)
    
    # Check which models exist
    available_models = []
    for config in model_configs:
        if Path(config['checkpoint']).exists():
            available_models.append(config)
            print(f"✓ Found: {config['name']}")
        else:
            print(f"✗ Missing: {config['name']}")
    
    if not available_models:
        print("\n✗ No trained models found. Please train models first.")
        return 1
    
    # Run analyses
    for config in available_models:
        # Per-scene performance
        print(f"\n{'='*70}")
        print(f"ANALYZING: {config['name']}")
        print(f"{'='*70}")
        
        analysis = analyzer.analyze_per_scene_performance(
            model_path=Path(config['checkpoint']),
            model_config=config,
            n_samples=50
        )
        
        # Print summary
        print("\nPerformance by Scene Difficulty:")
        print("-" * 70)
        for difficulty in ['easy', 'moderate', 'hard', 'extreme']:
            stats = analysis['summary'][difficulty]
            print(f"{difficulty.capitalize():12} | "
                  f"IoU: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f} | "
                  f"n={stats['n_samples']}")
        
        # Generate visualizations
        analyzer.visualize_predictions(
            model_path=Path(config['checkpoint']),
            model_config=config,
            n_visualize=6,
            mode='best_worst'
        )
    
    # Compare models on extreme scenes
    if len(available_models) >= 2:
        analyzer.compare_models_on_difficult_scenes(
            model_configs=available_models,
            difficulty='extreme'
        )
    
    print("\n✓ Failure mode analysis complete!")
    print(f"  Results saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
