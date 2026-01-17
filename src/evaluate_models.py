"""
Model Evaluation Script
=======================
Comprehensive evaluation of all trained models:
- Baseline CNNs (UNet, DeepLabv3+) with different feature configurations
- Foundation models (DINOv2, SAM)

Research Questions:
1. Do engineered color space features outperform RGB?
2. Do foundation models change color space dependencies?
3. What are computational tradeoffs for deployment?

Metrics computed:
- IoU (Intersection over Union)
- Dice Score
- F1 Score, Precision, Recall
- Accuracy, Specificity
- Boundary IoU (for edge accuracy)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.hybrid_dinov2 import HybridDINOv2
from src.models.sam_models import SAMEncoderDecoder, SAMFineTuned
from src.data.dataset import get_dataloaders
from sklearn.metrics import confusion_matrix
import cv2


class SegmentationMetrics:
    """Compute comprehensive segmentation metrics"""
    
    @staticmethod
    def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5):
        """Compute confusion matrix for binary segmentation"""
        pred_binary = (pred >= threshold).astype(np.uint8).flatten()
        target_binary = target.astype(np.uint8).flatten()
        
        tn, fp, fn, tp = confusion_matrix(target_binary, pred_binary, labels=[0, 1]).ravel()
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Intersection over Union"""
        pred_binary = (pred >= threshold).astype(bool)
        target_binary = target.astype(bool)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Dice coefficient (F1 score for segmentation)"""
        pred_binary = (pred >= threshold).astype(bool)
        target_binary = target.astype(bool)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        pred_sum = pred_binary.sum()
        target_sum = target_binary.sum()
        
        if pred_sum + target_sum == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2 * intersection / (pred_sum + target_sum)
    
    @staticmethod
    def compute_metrics_from_confusion(cm: Dict[str, int]) -> Dict[str, float]:
        """Compute all metrics from confusion matrix"""
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1
        }
    
    @staticmethod
    def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, 
                            threshold: float = 0.5, dilation: int = 2) -> float:
        """
        Boundary IoU - measures accuracy at object boundaries
        
        More sensitive to edge quality than standard IoU
        """
        pred_binary = (pred >= threshold).astype(np.uint8)
        target_binary = target.astype(np.uint8)
        
        # Compute boundaries using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        pred_dilated = cv2.dilate(pred_binary, kernel, iterations=dilation)
        pred_eroded = cv2.erode(pred_binary, kernel, iterations=dilation)
        pred_boundary = pred_dilated - pred_eroded
        
        target_dilated = cv2.dilate(target_binary, kernel, iterations=dilation)
        target_eroded = cv2.erode(target_binary, kernel, iterations=dilation)
        target_boundary = target_dilated - target_eroded
        
        # Compute IoU on boundaries
        intersection = np.logical_and(pred_boundary, target_boundary).sum()
        union = np.logical_or(pred_boundary, target_boundary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union


class ModelEvaluator:
    """Evaluate trained segmentation models"""
    
    def __init__(
        self,
        results_dir: str = 'experiments/results',
        data_dir: str = 'data',
        device: str = None,
        batch_size: int = 4,
        image_size: int = 512
    ):
        self.results_dir = Path(results_dir)
        self.data_dir = data_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.image_size = image_size
        
        print(f"Model Evaluator initialized")
        print(f"  Device: {self.device}")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Data directory: {self.data_dir}")
    
    def _load_model(
        self,
        model_type: str,
        feature_config: str,
        checkpoint_path: Path,
        **model_kwargs
    ) -> nn.Module:
        """Load trained model from checkpoint"""
        
        # Determine input channels
        channel_map = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
        in_channels = channel_map[feature_config]
        
        # Create model
        if model_type == 'unet':
            model = UNet(
                in_channels=in_channels,
                num_classes=1,
                base_channels=model_kwargs.get('base_channels', 64)
            )
        elif model_type == 'deeplabv3plus':
            model = DeepLabV3Plus(
                in_channels=in_channels,
                num_classes=1,
                pretrained=False  # Don't load ImageNet weights, we have trained weights
            )
        elif model_type == 'dinov2':
            model = HybridDINOv2(
                in_channels=in_channels,
                num_classes=1,
                dino_model=model_kwargs.get('dino_model', 'dinov2_vitb14'),
                freeze_dino=model_kwargs.get('freeze_dino', True),
                fusion_type=model_kwargs.get('fusion_type', 'simple')
            )
        elif model_type == 'sam_encoder':
            model = SAMEncoderDecoder(
                sam_checkpoint=model_kwargs.get('sam_checkpoint', 'checkpoints/sam_vit_b_01ec64.pth'),
                model_type=model_kwargs.get('sam_model_type', 'vit_b'),
                freeze_encoder=True
            )
        elif model_type == 'sam_finetuned':
            model = SAMFineTuned(
                sam_checkpoint=model_kwargs.get('sam_checkpoint', 'checkpoints/sam_vit_b_01ec64.pth'),
                model_type=model_kwargs.get('sam_model_type', 'vit_b')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            state_dict_to_load = None
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Checkpoint saved with metadata (epoch, optimizer, etc.)
                    state_dict_to_load = checkpoint['model_state_dict']
                    print(f"✓ Loaded checkpoint (with metadata): {checkpoint_path.name}")
                elif 'state_dict' in checkpoint:
                    # Alternative format
                    state_dict_to_load = checkpoint['state_dict']
                    print(f"✓ Loaded checkpoint (with state_dict): {checkpoint_path.name}")
                else:
                    # Assume it's a raw state dict
                    state_dict_to_load = checkpoint
                    print(f"✓ Loaded checkpoint (raw): {checkpoint_path.name}")
            else:
                # Fallback - try loading directly
                state_dict_to_load = checkpoint
                print(f"✓ Loaded checkpoint: {checkpoint_path.name}")
            
            # Load with strict=False to handle architecture mismatches
            # (e.g., aux_classifier present in trained model but not needed for inference)
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
                
                if unexpected_keys:
                    print(f"  ⚠ Ignoring unexpected keys: {len(unexpected_keys)} keys (e.g., aux_classifier)")
                if missing_keys:
                    print(f"  ⚠ Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
                    
            except Exception as e:
                print(f"  ✗ Error loading state dict: {e}")
                raise
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _evaluate_single_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """Evaluate single model on dataset"""
        
        metrics = SegmentationMetrics()
        
        # Accumulators
        all_ious = []
        all_dices = []
        all_boundary_ious = []
        all_cms = []
        inference_times = []
        
        print(f"\nEvaluating {model_name}...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"  Processing")):
                try:
                    # Handle flexible unpacking (dataloader might return 2 or 3+ values)
                    if isinstance(batch_data, (list, tuple)):
                        if len(batch_data) == 2:
                            images, masks = batch_data
                        elif len(batch_data) >= 3:
                            # If dataloader returns (images, masks, metadata), just take first 2
                            images, masks = batch_data[0], batch_data[1]
                        else:
                            raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
                    else:
                        raise ValueError(f"Unexpected batch data type: {type(batch_data)}")
                    
                    images = images.to(self.device)
                    masks_np = masks.numpy()
                    
                    # Time inference
                    start_time = time.time()
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    outputs = model(images)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        # DeepLabv3+ might return dict with 'out' and 'aux'
                        outputs = outputs['out']
                    elif isinstance(outputs, (tuple, list)):
                        # If model returns tuple/list, take first element
                        outputs = outputs[0]
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time / images.shape[0])  # Per image
                    
                    # Move to CPU
                    outputs_np = outputs.cpu().numpy()
                    
                    # Compute metrics for each image in batch
                    for i in range(images.shape[0]):
                        pred = outputs_np[i, 0]  # Remove channel dim
                        target = masks_np[i, 0]
                        
                        # IoU and Dice
                        iou = metrics.compute_iou(pred, target)
                        dice = metrics.compute_dice(pred, target)
                        boundary_iou = metrics.compute_boundary_iou(pred, target)
                        
                        all_ious.append(iou)
                        all_dices.append(dice)
                        all_boundary_ious.append(boundary_iou)
                        
                        # Confusion matrix
                        cm = metrics.compute_confusion_matrix(pred, target)
                        all_cms.append(cm)
                
                except Exception as e:
                    print(f"\n✗ Error in batch {batch_idx}:")
                    print(f"  Batch data type: {type(batch_data)}")
                    if isinstance(batch_data, (list, tuple)):
                        print(f"  Batch data length: {len(batch_data)}")
                        for idx, item in enumerate(batch_data):
                            print(f"  Item {idx}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        # Aggregate confusion matrices
        total_cm = {
            'TP': sum(cm['TP'] for cm in all_cms),
            'TN': sum(cm['TN'] for cm in all_cms),
            'FP': sum(cm['FP'] for cm in all_cms),
            'FN': sum(cm['FN'] for cm in all_cms)
        }
        
        # Compute metrics from aggregated confusion matrix
        cm_metrics = metrics.compute_metrics_from_confusion(total_cm)
        
        # Compile results
        results = {
            'model': model_name,
            'iou_mean': np.mean(all_ious),
            'iou_std': np.std(all_ious),
            'dice_mean': np.mean(all_dices),
            'dice_std': np.std(all_dices),
            'boundary_iou_mean': np.mean(all_boundary_ious),
            'boundary_iou_std': np.std(all_boundary_ious),
            'accuracy': cm_metrics['accuracy'],
            'precision': cm_metrics['precision'],
            'recall': cm_metrics['recall'],
            'specificity': cm_metrics['specificity'],
            'f1': cm_metrics['f1'],
            'inference_time_mean_ms': np.mean(inference_times) * 1000,
            'inference_time_std_ms': np.std(inference_times) * 1000,
            'n_samples': len(all_ious)
        }
        
        return results
    
    def evaluate_all_models(
        self,
        output_file: str = 'model_evaluation_results.csv'
    ) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Returns:
            DataFrame with evaluation results for all models
        """
        
        all_results = []
        
        # ============================================================
        # 1. BASELINE MODELS: UNet
        # ============================================================
        print("\n" + "="*70)
        print("EVALUATING BASELINE: UNet")
        print("="*70)
        
        for feature_config in ['rgb', 'luminance', 'chrominance', 'all']:
            checkpoint_dir = self.results_dir / 'baseline' / 'unet' / feature_config / 'checkpoints'
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            if not checkpoint_path.exists():
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                continue
            
            # Load model
            try:
                model = self._load_model(
                    model_type='unet',
                    feature_config=feature_config,
                    checkpoint_path=checkpoint_path
                )
                
                # Get appropriate dataloader
                dataloaders_result = get_dataloaders(
                    data_root=self.data_dir,
                    feature_config=feature_config,
                    batch_size=self.batch_size,
                    num_workers=4,
                    image_size=(self.image_size, self.image_size),
                    train_split=0.8,
                    seed=42,
                    normalize=True
                )
                # Extract val_loader (handles 2 or 3 return values)
                val_loader = dataloaders_result[1] if isinstance(dataloaders_result, (list, tuple)) and len(dataloaders_result) >= 2 else dataloaders_result
                
                # Evaluate
                model_name = f"UNet-{feature_config}"
                results = self._evaluate_single_model(model, val_loader, model_name)
                results['architecture'] = 'UNet'
                results['feature_config'] = feature_config
                results['model_family'] = 'Baseline CNN'
                
                all_results.append(results)
                
            except Exception as e:
                print(f"✗ Error evaluating UNet-{feature_config}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ============================================================
        # 2. BASELINE MODELS: DeepLabv3+
        # ============================================================
        print("\n" + "="*70)
        print("EVALUATING BASELINE: DeepLabv3+")
        print("="*70)
        
        for feature_config in ['rgb', 'luminance', 'chrominance', 'all']:
            checkpoint_dir = self.results_dir / 'baseline' / 'deeplabv3plus' / feature_config / 'checkpoints'
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            if not checkpoint_path.exists():
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                model = self._load_model(
                    model_type='deeplabv3plus',
                    feature_config=feature_config,
                    checkpoint_path=checkpoint_path
                )
                
                dataloaders_result = get_dataloaders(
                    data_root=self.data_dir,
                    feature_config=feature_config,
                    batch_size=self.batch_size,
                    num_workers=4,
                    image_size=(self.image_size, self.image_size),
                    train_split=0.8,
                    seed=42,
                    normalize=True
                )
                val_loader = dataloaders_result[1] if isinstance(dataloaders_result, (list, tuple)) and len(dataloaders_result) >= 2 else dataloaders_result
                
                model_name = f"DeepLabv3+-{feature_config}"
                results = self._evaluate_single_model(model, val_loader, model_name)
                results['architecture'] = 'DeepLabv3+'
                results['feature_config'] = feature_config
                results['model_family'] = 'Baseline CNN'
                
                all_results.append(results)
                
            except Exception as e:
                print(f"✗ Error evaluating DeepLabv3+-{feature_config}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ============================================================
        # 3. FOUNDATION MODEL: DINOv2
        # ============================================================
        print("\n" + "="*70)
        print("EVALUATING FOUNDATION MODEL: DINOv2")
        print("="*70)
        
        for feature_config in ['rgb', 'all']:
            # Path structure: dinov2/rgb/frozen/checkpoints or dinov2/all/frozen/checkpoints
            checkpoint_dir = self.results_dir / 'dinov2' / feature_config / 'frozen' / 'checkpoints'
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            if not checkpoint_path.exists():
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                model = self._load_model(
                    model_type='dinov2',
                    feature_config=feature_config,
                    checkpoint_path=checkpoint_path,
                    dino_model='dinov2_vitb14',
                    freeze_dino=True
                )
                
                dataloaders_result = get_dataloaders(
                    data_root=self.data_dir,
                    feature_config=feature_config,
                    batch_size=self.batch_size,
                    num_workers=4,
                    image_size=(448, 448),  # DINOv2 uses 448
                    train_split=0.8,
                    seed=42,
                    normalize=False  # DINOv2 handles normalization
                )
                val_loader = dataloaders_result[1] if isinstance(dataloaders_result, (list, tuple)) and len(dataloaders_result) >= 2 else dataloaders_result
                
                model_name = f"DINOv2-{feature_config}"
                results = self._evaluate_single_model(model, val_loader, model_name)
                results['architecture'] = 'Hybrid DINOv2'
                results['feature_config'] = feature_config
                results['model_family'] = 'Foundation Model (Self-supervised)'
                
                all_results.append(results)
                
            except Exception as e:
                print(f"✗ Error evaluating DINOv2-{feature_config}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ============================================================
        # 4. FOUNDATION MODEL: SAM
        # ============================================================
        print("\n" + "="*70)
        print("EVALUATING FOUNDATION MODEL: SAM")
        print("="*70)
        
        sam_configs = [
            ('encoder', 'vit_b', 'sam_encoder'),
            ('finetuned', 'vit_b', 'sam_finetuned')
        ]
        
        for sam_type, model_variant, model_type in sam_configs:
            # Path structure: sam/encoder/vit_b/checkpoints or sam/finetuned/vit_b/checkpoints
            checkpoint_dir = self.results_dir / 'sam' / sam_type / model_variant / 'checkpoints'
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            
            if not checkpoint_path.exists():
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                model = self._load_model(
                    model_type=model_type,
                    feature_config='rgb',  # SAM only works with RGB
                    checkpoint_path=checkpoint_path,
                    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
                    sam_model_type=model_variant
                )
                
                dataloaders_result = get_dataloaders(
                    data_root=self.data_dir,
                    feature_config='rgb',
                    batch_size=self.batch_size,
                    num_workers=4,
                    image_size=(self.image_size, self.image_size),
                    train_split=0.8,
                    seed=42,
                    normalize=False  # SAM handles normalization
                )
                val_loader = dataloaders_result[1] if isinstance(dataloaders_result, (list, tuple)) and len(dataloaders_result) >= 2 else dataloaders_result
                
                model_name = f"SAM-{sam_type}-{model_variant}"
                results = self._evaluate_single_model(model, val_loader, model_name)
                results['architecture'] = 'SAM' if sam_type == 'finetuned' else 'SAM Encoder + CNN'
                results['feature_config'] = 'rgb'
                results['model_family'] = 'Foundation Model (Segmentation-specific)'
                
                all_results.append(results)
                
            except Exception as e:
                print(f"✗ Error evaluating SAM-{sam_type}-{model_variant}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ============================================================
        # COMPILE RESULTS
        # ============================================================
        print("\n" + "="*70)
        print("COMPILING RESULTS")
        print("="*70)
        
        if not all_results:
            print("✗ No models were successfully evaluated!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        
        # Reorder columns
        column_order = [
            'model', 'architecture', 'feature_config', 'model_family',
            'iou_mean', 'iou_std', 'dice_mean', 'dice_std',
            'boundary_iou_mean', 'boundary_iou_std',
            'f1', 'precision', 'recall', 'accuracy', 'specificity',
            'inference_time_mean_ms', 'inference_time_std_ms',
            'n_samples'
        ]
        df = df[column_order]
        
        # Sort by IoU (descending)
        df = df.sort_values('iou_mean', ascending=False)
        
        # Save to CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(df[['model', 'iou_mean', 'dice_mean', 'f1', 'inference_time_mean_ms']].to_string(index=False))
        
        return df


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate all trained models')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                       help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_file', type=str, default='model_evaluation_results.csv',
                       help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Run evaluation
    results_df = evaluator.evaluate_all_models(output_file=args.output_file)
    
    if results_df.empty:
        print("\n✗ Evaluation failed - no results generated")
        return 1
    
    print("\n✓ Evaluation complete!")
    return 0


if __name__ == '__main__':
    exit(main())
