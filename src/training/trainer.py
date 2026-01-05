"""
Generic Trainer for Segmentation Models
========================================
Handles training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional
from tqdm import tqdm
from pathlib import Path
import json
import time
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Handle imports for both package and direct execution
try:
    from ..utils.metrics import SegmentationMetrics, MetricsTracker
    from ..utils.losses import CombinedLoss
except ImportError:
    # If relative import fails, try absolute import
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.metrics import SegmentationMetrics, MetricsTracker
    from src.utils.losses import CombinedLoss


class Trainer:
    """
    Generic trainer for segmentation models
    
    Features:
    - Training and validation loops
    - Automatic checkpointing
    - W&B logging support
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[_LRScheduler] = None,
        use_wandb: bool = False,
        checkpoint_dir: str = 'checkpoints',
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 20
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            use_wandb: Use Weights & Biases logging
            checkpoint_dir: Directory for saving checkpoints
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.metrics_calculator = SegmentationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_iou = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # History
        self.train_history = []
        self.val_history = []
        
        # Timing
        self.start_time = None
        self.epoch_times = []
        
        if self.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not available, logging disabled")
            self.use_wandb = False
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        tracker = MetricsTracker()
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch:03d} [Train]',
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            
            with torch.no_grad():
                batch_metrics = self.metrics_calculator.compute_batch_metrics(
                    outputs, masks
                )
                tracker.update(batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': batch_metrics['iou']
            })
        
        # Compute epoch metrics
        num_batches = len(self.train_loader)
        epoch_metrics = tracker.get_average()
        epoch_metrics['loss'] = running_loss / num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        tracker = MetricsTracker()
        
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch:03d} [Val]  ',
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Metrics
            running_loss += loss.item()
            batch_metrics = self.metrics_calculator.compute_batch_metrics(
                outputs, masks
            )
            tracker.update(batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': batch_metrics['iou']
            })
        
        # Compute epoch metrics
        num_batches = len(self.val_loader)
        epoch_metrics = tracker.get_average()
        epoch_metrics['loss'] = running_loss / num_batches
        
        return epoch_metrics
    
    def fit(self, num_epochs: int):
        """
        Train for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Record epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                }
                wandb.log(log_dict)
            
            # Save history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)
            
            # Check for improvement
            if val_metrics['iou'] > self.best_iou:
                self.best_iou = val_metrics['iou']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best IoU: {self.best_iou:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1:03d}.pth')
        
        # Training complete
        total_time = time.time() - self.start_time
        self._print_training_summary(total_time)
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        
        # Save training history
        self.save_history()
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Print epoch summary"""
        print(f"\nEpoch {epoch:03d} ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
    
    def _print_training_summary(self, total_time: float):
        """Print training summary"""
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average epoch time: {sum(self.epoch_times)/len(self.epoch_times):.1f}s")
        print(f"Best IoU: {self.best_iou:.4f} (epoch {self.best_epoch})")
        print(f"Final IoU: {self.val_history[-1]['iou']:.4f}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # PyTorch 2.6+ requires weights_only=False for checkpoints with metadata
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_iou = checkpoint['best_iou']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best IoU: {self.best_iou:.4f} (epoch {self.best_epoch})")
    
    def save_history(self):
        """Save training history to JSON"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch
        }
        
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Saved training history to {self.checkpoint_dir / 'training_history.json'}")


if __name__ == '__main__':
    # Test trainer with dummy model
    print("Testing trainer...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.conv(x))
    
    # Create dummy data - proper format
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'image': torch.rand(3, 128, 128),
                'mask': torch.randint(0, 2, (1, 128, 128)).float(),
                'image_path': f'dummy_{idx}.png',
                'mask_path': f'mask_{idx}.png'
            }
    
    train_dataset = DummyDataset(20)
    val_dataset = DummyDataset(5)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create trainer
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu',
        use_wandb=False,
        checkpoint_dir='/home/claude/pytorch_river_seg/test_checkpoints'
    )
    
    # Train for 3 epochs
    trainer.fit(num_epochs=3)
    
    # Test checkpoint loading
    trainer.load_checkpoint('best_model.pth')
    
    print("\n✓ Trainer test passed!")
