"""
End-to-End Test Script
======================
Creates dummy data and runs a quick training test to verify the pipeline works
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.feature_extraction import FeatureExtractor
from src.data.dataset import RiverSegmentationDataset, get_dataloaders
from src.models.unet import UNet
from src.utils.losses import CombinedLoss
from src.utils.metrics import SegmentationMetrics
from src.training.trainer import Trainer


def create_dummy_data(num_images=20, img_size=512):
    """Create dummy dataset for testing"""
    
    print("Creating dummy dataset...")
    
    # Create simple structure: data_root/images/ and data_root/masks/
    base_dir = Path('test_data')
    (base_dir / 'images').mkdir(parents=True, exist_ok=True)
    (base_dir / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Create images and masks
    for i in range(num_images):
        # Random RGB image
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Random mask with some structure
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        radius = np.random.randint(img_size // 4, img_size // 3)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Add some noise
        noise = np.random.randint(0, 2, (img_size, img_size), dtype=np.uint8) * 255
        mask = cv2.addWeighted(mask, 0.8, noise, 0.2, 0)
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Save
        img_path = base_dir / 'images' / f'img_{i:03d}.png'
        mask_path = base_dir / 'masks' / f'img_{i:03d}.png'
        
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)
    
    print(f"✓ Created {num_images} images in simplified structure")
    print(f"  {base_dir}/images/ and {base_dir}/masks/")
    return base_dir


def test_feature_extraction():
    """Test feature extraction"""
    print("\n--- Testing Feature Extraction ---")
    
    extractor = FeatureExtractor()
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test all extraction methods
    luminance = extractor.extract_luminance(test_img)
    chrominance = extractor.extract_chrominance(test_img)
    all_features = extractor.extract_all_features(test_img)
    
    assert luminance.shape == (256, 256, 3), "Luminance shape mismatch"
    assert chrominance.shape == (256, 256, 7), "Chrominance shape mismatch"
    assert all_features.shape == (256, 256, 10), "All features shape mismatch"
    
    # Test normalization
    normalized = extractor.normalize_features(all_features)
    assert normalized.min() >= 0 and normalized.max() <= 1, "Normalization failed"
    
    print("✓ Feature extraction working correctly")


def test_dataset(data_dir):
    """Test dataset loading"""
    print("\n--- Testing Dataset ---")
    
    for config in ['rgb', 'luminance', 'chrominance', 'all']:
        dataset = RiverSegmentationDataset(
            data_root=str(data_dir),
            feature_config=config,
            image_size=(256, 256),
            augment=False
        )
        
        sample = dataset[0]
        expected_channels = {'rgb': 3, 'luminance': 3, 'chrominance': 7, 'all': 10}
        
        assert sample['image'].shape[0] == expected_channels[config], f"{config} channel mismatch"
        assert sample['mask'].shape == (1, 256, 256), "Mask shape mismatch"
        
        print(f"  ✓ {config}: {sample['image'].shape}")


def test_model():
    """Test model forward pass"""
    print("\n--- Testing Model ---")
    
    for config, channels in [('RGB', 3), ('All', 10)]:
        model = UNet(in_channels=channels, num_classes=1)
        x = torch.rand(2, channels, 256, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1, 256, 256), "Output shape mismatch"
        assert output.min() >= 0 and output.max() <= 1, "Output range incorrect"
        
        print(f"  ✓ {config} ({channels}ch): {output.shape}")


def test_loss_and_metrics():
    """Test loss functions and metrics"""
    print("\n--- Testing Loss & Metrics ---")
    
    pred = torch.rand(4, 1, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()
    
    # Test loss
    criterion = CombinedLoss()
    loss = criterion(pred, target)
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    # Test metrics
    metrics_calc = SegmentationMetrics()
    metrics = metrics_calc.compute_batch_metrics(pred, target)
    
    assert 0 <= metrics['iou'] <= 1, "IoU out of range"
    assert 0 <= metrics['dice'] <= 1, "Dice out of range"
    
    print(f"  ✓ IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")


def test_training(data_dir):
    """Test training loop"""
    print("\n--- Testing Training Loop ---")
    
    # Create dataloaders with automatic split
    train_loader, val_loader = get_dataloaders(
        data_root=str(data_dir),
        feature_config='rgb',
        batch_size=4,
        num_workers=0,
        image_size=(256, 256),
        train_split=0.8,
        seed=42
    )
    
    # Create model
    model = UNet(in_channels=3, num_classes=1, base_channels=32)  # Smaller for speed
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CombinedLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu',
        use_wandb=False,
        checkpoint_dir='test_checkpoints',
        early_stopping_patience=5
    )
    
    # Train for 3 epochs
    print("\n  Training for 3 epochs (this may take a minute)...")
    trainer.fit(num_epochs=3)
    
    # Verify checkpoint exists
    assert (Path('test_checkpoints') / 'best_model.pth').exists(), "Checkpoint not saved"
    print("  ✓ Checkpoint saved")
    
    # Test loading
    trainer.load_checkpoint('best_model.pth')
    print("  ✓ Checkpoint loaded")
    
    return trainer


def main():
    """Run all tests"""
    
    print("\n" + "="*70)
    print("RUNNING END-TO-END TESTS")
    print("="*70)
    
    try:
        # Test 1: Feature extraction
        test_feature_extraction()
        
        # Test 2: Create dummy data
        data_dir = create_dummy_data(num_images=20, img_size=256)
        
        # Test 3: Dataset
        test_dataset(data_dir)
        
        # Test 4: Model
        test_model()
        
        # Test 5: Loss and metrics
        test_loss_and_metrics()
        
        # Test 6: Training loop
        trainer = test_training(data_dir)
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print(f"\nFinal Results:")
        print(f"  Best IoU: {trainer.best_iou:.4f}")
        print(f"  Best Epoch: {trainer.best_epoch}")
        print(f"\nThe pipeline is ready to use!")
        print(f"Clean up test files with: rm -rf test_data test_checkpoints")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
