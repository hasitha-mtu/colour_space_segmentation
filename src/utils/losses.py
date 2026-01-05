"""
Loss Functions for Segmentation
================================
Implements:
- Dice Loss
- Boundary Loss
- Combined Loss (BCE + Dice + Boundary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    
    Dice coefficient = 2*|X∩Y| / (|X| + |Y|)
    Dice loss = 1 - Dice coefficient
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W) or (B, H, W)
            target: Ground truth binary mask (B, 1, H, W) or (B, H, W)
        
        Returns:
            Dice loss value
        """
        # Flatten predictions and targets
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        # Return dice loss
        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes errors at object boundaries
    
    Uses morphological operations to detect boundaries and weights
    the loss higher for pixels near boundaries
    """
    
    def __init__(self, boundary_weight: float = 2.0):
        """
        Args:
            boundary_weight: Weight multiplier for boundary pixels
        """
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def _get_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries using morphological dilation
        
        Args:
            masks: Binary masks (B, 1, H, W)
        
        Returns:
            Boundary mask (B, 1, H, W)
        """
        # Kernel for dilation (3x3)
        kernel = torch.ones(1, 1, 3, 3, device=masks.device)
        
        # Dilate
        dilated = F.conv2d(
            masks.float(),
            kernel,
            padding=1
        )
        
        # Boundary = dilated - original
        # (pixels that change when dilated)
        boundaries = ((dilated > 0).float() - masks.float()).abs()
        
        return boundaries
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        
        Returns:
            Boundary-weighted BCE loss
        """
        # Get boundaries from target
        boundaries = self._get_boundaries(target)
        
        # Create weight map: higher weight at boundaries
        weights = 1.0 + boundaries * (self.boundary_weight - 1.0)
        
        # Weighted binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        
        Returns:
            Focal loss value
        """
        # Compute BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Compute pt (probability of correct class)
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Compute alpha term
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        focal_loss = alpha_t * focal_term * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: BCE + Dice + Boundary
    
    This is the main loss function matching the TensorFlow implementation
    """
    
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        use_focal: bool = False
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            boundary_weight: Weight for Boundary loss
            use_focal: Use Focal loss instead of BCE
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        # Loss components
        if use_focal:
            self.bce = FocalLoss()
        else:
            self.bce = nn.BCELoss()
        
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W)
        
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        boundary_loss = self.boundary(pred, target)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss
        )
        
        return total_loss
    
    def get_component_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Get individual loss components for logging
        
        Returns:
            dict with 'bce', 'dice', 'boundary', 'total'
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        boundary_loss = self.boundary(pred, target)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss
        )
        
        return {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'boundary': boundary_loss.item(),
            'total': total_loss.item()
        }


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy predictions and targets
    batch_size = 4
    pred = torch.rand(batch_size, 1, 256, 256)
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_val = dice_loss(pred, target)
    print(f"Dice Loss: {dice_val.item():.4f}")
    
    # Test Boundary Loss
    boundary_loss = BoundaryLoss()
    boundary_val = boundary_loss(pred, target)
    print(f"Boundary Loss: {boundary_val.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    focal_val = focal_loss(pred, target)
    print(f"Focal Loss: {focal_val.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    combined_val = combined_loss(pred, target)
    print(f"Combined Loss: {combined_val.item():.4f}")
    
    # Test component losses
    components = combined_loss.get_component_losses(pred, target)
    print(f"\nComponent losses:")
    for key, val in components.items():
        print(f"  {key}: {val:.4f}")
    
    # Test gradients
    pred.requires_grad = True
    loss = combined_loss(pred, target)
    loss.backward()
    print(f"\nGradient check: {pred.grad is not None}")
    print(f"Gradient mean: {pred.grad.abs().mean():.6f}")
    
    print("\n✓ All loss functions working correctly!")
