"""
Train SAM Models
================
Trains SAM-based architectures:
1. SAM Encoder + CNN Decoder
2. Fine-tuned SAM (frozen encoder, trainable decoder)

Compares segmentation-specific pretraining (SAM) vs 
self-supervised vision pretraining (DINOv2).
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sam_models import SAMEncoderDecoder, SAMFineTuned
from src.data.dataset import get_dataloaders
from src.utils.losses import CombinedLoss
from src.training.trainer import Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")


def train_sam(
    sam_type: str = 'encoder',
    sam_checkpoint: str = 'checkpoints/sam_vit_b_01ec64.pth',
    model_type: str = 'vit_b',
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: int = 512,
    use_wandb: bool = False,
    data_dir: str = 'data',
    output_dir: str = 'experiments/results/sam',
    device: str = None
):
    """
    Train SAM-based model
    
    Args:
        sam_type: 'encoder' (SAM encoder + CNN decoder) or 'finetuned' (fine-tuned SAM)
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        image_size: Input size
        use_wandb: Use W&B logging
        data_dir: Data directory
        output_dir: Output directory
        device: Device to use
    
    Note: SAM only works with RGB input (3 channels)
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"Training SAM Model: {sam_type}")
    print(f"{'='*70}")
    print(f"SAM type: {sam_type}")
    print(f"Model: {model_type}")
    print(f"Checkpoint: {sam_checkpoint}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # Check if checkpoint exists
    if not Path(sam_checkpoint).exists():
        print(f"❌ SAM checkpoint not found: {sam_checkpoint}")
        print("\nDownload SAM checkpoint with:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/")
        return None, None, None
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project='uav-water-segmentation',
            name=f'sam_{sam_type}_{model_type}',
            config={
                'model': f'sam_{sam_type}',
                'model_type': model_type,
                'feature_config': 'rgb',  # SAM requires RGB
                'in_channels': 3,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'image_size': image_size
            }
        )
    
    # Create output directory
    output_path = Path(output_dir) / sam_type / model_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders (RGB only for SAM)
    train_loader, val_loader = get_dataloaders(
        data_root=data_dir,
        feature_config='rgb',  # SAM requires RGB
        batch_size=batch_size,
        num_workers=4,
        image_size=(image_size, image_size),
        train_split=0.8,
        normalize=False,  # SAM handles normalization
        seed=42
    )
    
    # Create model
    if sam_type == 'encoder':
        model = SAMEncoderDecoder(
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            freeze_encoder=True,
            decoder_channels=256
        )
    elif sam_type == 'finetuned':
        model = SAMFineTuned(
            sam_checkpoint=sam_checkpoint,
            model_type=model_type
        )
    else:
        raise ValueError(f"Unknown sam_type: {sam_type}. Use 'encoder' or 'finetuned'")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(f"  SAM type: {sam_type}")
    print(f"  Model: {model_type}")
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
    
    # Optimizer - only trainable parameters
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    parser = argparse.ArgumentParser(description='Train SAM-based models')
    
    # Model arguments
    parser.add_argument('--sam_type', type=str, default='encoder',
                       choices=['encoder', 'finetuned'],
                       help='SAM architecture type')
    parser.add_argument('--sam_checkpoint', type=str,
                       default='checkpoints/sam_vit_b_01ec64.pth',
                       help='Path to SAM checkpoint')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/results/sam',
                       help='Output directory')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Train model
    train_sam(**vars(args))


if __name__ == '__main__':
    main()
