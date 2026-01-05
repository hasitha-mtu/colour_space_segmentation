"""
Train Baseline CNN Models (UNet, DeepLabv3+)
=============================================
This script trains baseline segmentation models with different feature configurations
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus
from src.data.dataset import get_dataloaders
from src.utils.losses import CombinedLoss
from src.training.trainer import Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")


def train_baseline(
    model_type: str = 'unet',
    feature_config: str = 'rgb',
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: int = 512,
    base_channels: int = 64,
    use_wandb: bool = False,
    data_dir: str = 'data',
    output_dir: str = 'experiments/results/baseline',
    device: str = None
):
    """
    Train baseline segmentation model
    
    Args:
        model_type: 'unet' or 'deeplabv3plus'
        feature_config: 'rgb', 'luminance', 'chrominance', or 'all'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        image_size: Input image size
        base_channels: Base number of channels for UNet
        use_wandb: Use Weights & Biases logging
        data_dir: Data directory
        output_dir: Output directory for checkpoints
        device: Device to use (auto-detect if None)
    """
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"Training Baseline Model: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Feature config: {feature_config}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # Determine input channels
    channel_map = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
    in_channels = channel_map[feature_config]
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project='uav-water-segmentation',
            name=f'{model_type}_{feature_config}',
            config={
                'model_type': model_type,
                'feature_config': feature_config,
                'in_channels': in_channels,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'image_size': image_size,
                'base_channels': base_channels
            }
        )
    
    # Create output directory
    output_path = Path(output_dir) / model_type / feature_config
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders with automatic train/val split
    train_loader, val_loader = get_dataloaders(
        data_root=data_dir,
        feature_config=feature_config,
        batch_size=batch_size,
        num_workers=4,
        image_size=(image_size, image_size),
        train_split=0.8,  # 80% train, 20% val
        seed=42,
        normalize=True
    )
    
    # Create model
    if model_type == 'unet':
        model = UNet(
            in_channels=in_channels,
            num_classes=1,
            base_channels=base_channels
        )
    elif model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(
            in_channels=in_channels,
            num_classes=1,
            pretrained=True,
            freeze_backbone=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"  Type: {model_type}")
    print(f"  Input channels: {in_channels}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB\n")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.5
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    
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
    parser = argparse.ArgumentParser(description='Train baseline segmentation model')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'deeplabv3plus'],
                       help='Model architecture')
    parser.add_argument('--feature_config', type=str, default='rgb',
                       choices=['rgb', 'luminance', 'chrominance', 'all'],
                       help='Feature configuration')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels for UNet')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/results/baseline',
                       help='Output directory')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Train model
    train_baseline(**vars(args))


if __name__ == '__main__':
    main()
