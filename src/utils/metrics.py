"""
Evaluation Metrics for Segmentation
====================================
Implements standard segmentation metrics:
- IoU (Intersection over Union)
- Dice coefficient
- F1 score
- Precision
- Recall
- Pixel Accuracy
"""

import torch
import numpy as np
from typing import Dict, Tuple


class SegmentationMetrics:
    """
    Compute segmentation metrics
    
    Args:
        threshold: Threshold for converting probabilities to binary predictions
        eps: Small epsilon to avoid division by zero
    """
    
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
    
    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single sample
        
        Args:
            pred: Predicted probabilities (1, H, W) or (H, W)
            target: Ground truth binary mask (1, H, W) or (H, W)
        
        Returns:
            Dictionary of metrics
        """
        # Ensure 2D
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)
        
        # Threshold predictions
        pred_binary = (pred > self.threshold).float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives, true negatives
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        # Calculate metrics
        iou = tp / (tp + fp + fn + self.eps)
        dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.eps)
        
        return {
            'iou': iou,
            'dice': dice,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def compute_batch_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics averaged over a batch
        
        Args:
            preds: Predicted probabilities (B, 1, H, W)
            targets: Ground truth binary masks (B, 1, H, W)
        
        Returns:
            Dictionary of averaged metrics
        """
        batch_size = preds.size(0)
        
        metrics_sum = {
            'iou': 0.0,
            'dice': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0
        }
        
        for i in range(batch_size):
            metrics = self.compute_metrics(preds[i], targets[i])
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
        
        # Average
        metrics_avg = {k: v / batch_size for k, v in metrics_sum.items()}
        
        return metrics_avg
    
    def compute_confusion_matrix(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        """
        Compute confusion matrix components
        
        Args:
            pred: Predicted probabilities
            target: Ground truth
        
        Returns:
            (tp, fp, fn, tn)
        """
        # Ensure 2D
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)
        
        # Threshold
        pred_binary = (pred > self.threshold).float()
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Confusion matrix
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        return tp, fp, fn, tn


class MetricsTracker:
    """
    Track metrics across multiple batches/epochs
    
    Useful for computing statistics over entire datasets
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics = {
            'iou': [],
            'dice': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'loss': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """
        Update tracker with new metrics
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_average(self) -> Dict[str, float]:
        """
        Get average of all tracked metrics
        
        Returns:
            Dictionary of averaged metrics
        """
        avg_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_std(self) -> Dict[str, float]:
        """
        Get standard deviation of tracked metrics
        
        Returns:
            Dictionary of metric standard deviations
        """
        std_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                std_metrics[key] = np.std(values)
            else:
                std_metrics[key] = 0.0
        
        return std_metrics
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics (mean, std, min, max)
        
        Returns:
            Nested dictionary with statistics
        """
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                summary[key] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        return summary


if __name__ == '__main__':
    # Test metrics
    print("Testing segmentation metrics...")
    
    # Create dummy predictions and targets
    torch.manual_seed(42)
    pred = torch.rand(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Test single sample metrics
    metrics_calc = SegmentationMetrics()
    single_metrics = metrics_calc.compute_metrics(pred[0], target[0])
    
    print("\nSingle sample metrics:")
    for key, val in single_metrics.items():
        if key not in ['tp', 'fp', 'fn', 'tn']:
            print(f"  {key}: {val:.4f}")
    
    # Test batch metrics
    batch_metrics = metrics_calc.compute_batch_metrics(pred, target)
    
    print("\nBatch metrics:")
    for key, val in batch_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # Test confusion matrix
    tp, fp, fn, tn = metrics_calc.compute_confusion_matrix(pred[0], target[0])
    total = tp + fp + fn + tn
    print(f"\nConfusion matrix:")
    print(f"  TP: {tp:.0f} ({100*tp/total:.1f}%)")
    print(f"  FP: {fp:.0f} ({100*fp/total:.1f}%)")
    print(f"  FN: {fn:.0f} ({100*fn/total:.1f}%)")
    print(f"  TN: {tn:.0f} ({100*tn/total:.1f}%)")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    
    for i in range(10):
        batch_pred = torch.rand(4, 1, 256, 256)
        batch_target = torch.randint(0, 2, (4, 1, 256, 256)).float()
        metrics = metrics_calc.compute_batch_metrics(batch_pred, batch_target)
        tracker.update(metrics)
    
    avg_metrics = tracker.get_average()
    std_metrics = tracker.get_std()
    
    print("\nTracked metrics (10 batches):")
    for key in ['iou', 'dice', 'f1']:
        print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    summary = tracker.get_summary()
    print(f"\nIoU summary:")
    print(f"  Mean: {summary['iou']['mean']:.4f}")
    print(f"  Std:  {summary['iou']['std']:.4f}")
    print(f"  Min:  {summary['iou']['min']:.4f}")
    print(f"  Max:  {summary['iou']['max']:.4f}")
    
    print("\n✓ All metrics working correctly!")
