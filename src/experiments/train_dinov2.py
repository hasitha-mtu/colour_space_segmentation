"""
Train Hybrid CNN-DINOv2 Model
==============================
Trains the hybrid architecture combining CNN multi-channel processing
with DINOv2 foundation model features.

This is a key contribution of the research, comparing self-supervised
vision pretraining (DINOv2) vs segmentation-specific pretraining (SAM).
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hybrid_dinov2 import HybridDINOv2
from src.data.dataset import get_dataloaders
from src.utils.losses import CombinedLoss
from src.training.trainer import Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")


def train_dinov2(
    feature_config: str = 'all',
    dino_model: str = 'dinov2_vitb14',
    freeze_dino: bool = True,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: int = 448,  # Must be divisible by 14
    cnn_base_channels: int = 64,
    use_wandb: bool = False,
    data_dir: str = 'data',
    output_dir: str = 'experiments/results/dinov2',
    device: str = None
):
    """
    Train Hybrid DINOv2 model
    
    Args:
        feature_config: 'rgb', 'luminance', 'chrominance', or 'all'
        dino_model: DINOv2 variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
        freeze_dino: Freeze DINOv2 encoder weights
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        image_size: Input size (should be divisible by 14 for DINOv2)
        cnn_base_channels: Base channels for CNN branch
        use_wandb: Use W&B logging
        data_dir: Data directory
        output_dir: Output directory
        device: Device to use
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"Training Hybrid CNN-DINOv2")
    print(f"{'='*70}")
    print(f"Feature config: {feature_config}")
    print(f"DINOv2 model: {dino_model}")
    print(f"Freeze DINOv2: {freeze_dino}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # Determine channels
    channel_map = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
    in_channels = channel_map[feature_config]
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project='uav-water-segmentation',
            name=f'dinov2_{feature_config}_{"frozen" if freeze_dino else "tuned"}',
            config={
                'model': 'hybrid_dinov2',
                'feature_config': feature_config,
                'in_channels': in_channels,
                'dino_model': dino_model,
                'freeze_dino': freeze_dino,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'image_size': image_size
            }
        )
    
    # Create output directory
    output_path = Path(output_dir) / feature_config / ('frozen' if freeze_dino else 'tuned')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        data_root=data_dir,
        feature_config=feature_config,
        batch_size=batch_size,
        num_workers=4,
        image_size=(image_size, image_size),
        train_split=0.8,
        normalize=False,  # DINOv2 handles normalization internally
        seed=42
    )
    
    # Create model
    model = HybridDINOv2(
        in_channels=in_channels,
        num_classes=1,
        dino_model=dino_model,
        freeze_dino=freeze_dino,
        cnn_base_channels=cnn_base_channels
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"  Input channels: {in_channels}")
    print(f"  DINOv2: {dino_model}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.1f}%")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB\n")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.5
    )
    
    # Higher LR for trainable parts if DINOv2 is frozen
    if freeze_dino:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4
        )
    else:
        # Different LRs for DINOv2 vs CNN
        optimizer = AdamW([
            {'params': model.dino.parameters(), 'lr': lr / 10},  # Lower LR for foundation model
            {'params': model.cnn_encoder.parameters(), 'lr': lr},
            {'params': model.fusion.parameters(), 'lr': lr},
            {'params': model.decoder.parameters(), 'lr': lr}
        ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr / 100
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_wandb=use_wandb and WANDB_AVAILABLE,
        checkpoint_dir=str(output_path / 'checkpoints'),
        max_grad_norm=1.0,
        early_stopping_patience=20
    )
    
    # Train
    trainer.fit(num_epochs=epochs)
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pth')
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n✓ Training complete!")
    print(f"  Best IoU: {trainer.best_iou:.4f}")
    print(f"  Checkpoints saved to: {output_path / 'checkpoints'}")
    
    return model, trainer.train_history, trainer.val_history


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-DINOv2 model')
    
    # Model arguments
    parser.add_argument('--feature_config', type=str, default='all',
                       choices=['rgb', 'luminance', 'chrominance', 'all'],
                       help='Feature configuration')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2 model variant')
    parser.add_argument('--freeze_dino', action='store_true',
                       help='Freeze DINOv2 encoder weights')
    parser.add_argument('--cnn_base_channels', type=int, default=64,
                       help='Base channels for CNN branch')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=448,
                       help='Input image size (should be divisible by 14)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/results/dinov2',
                       help='Output directory')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Train model
    train_dinov2(**vars(args))


if __name__ == '__main__':
    main()
