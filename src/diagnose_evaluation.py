"""
Diagnostic Script for Evaluation Issues
========================================
Tests dataloader and models to identify issues before running full evaluation.

Run this BEFORE running evaluate_models.py to catch issues early.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("DIAGNOSTIC TESTS FOR EVALUATION")
print("="*70)

# ============================================================================
# TEST 1: Check get_dataloaders return format
# ============================================================================
print("\n" + "─"*70)
print("TEST 1: Checking get_dataloaders return format")
print("─"*70)

try:
    from src.data.dataset import get_dataloaders
    
    print("✓ Imported get_dataloaders")
    
    # Test with minimal parameters
    result = get_dataloaders(
        data_root='dataset/test',
        feature_config='rgb',
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        image_size=(512, 512),
        train_split=0.8,
        seed=42
    )
    
    print(f"\nget_dataloaders returns: {type(result)}")
    
    if isinstance(result, (list, tuple)):
        print(f"Length: {len(result)}")
        for i, item in enumerate(result):
            print(f"  [{i}]: {type(item)}")
            if hasattr(item, '__len__'):
                print(f"       Dataset size: {len(item.dataset) if hasattr(item, 'dataset') else 'unknown'}")
        
        # Extract validation loader
        if len(result) >= 2:
            val_loader = result[1]
            print(f"\n✓ Validation loader extracted successfully")
        else:
            print(f"\n✗ ERROR: Expected at least 2 return values, got {len(result)}")
            sys.exit(1)
    else:
        val_loader = result
        print(f"✓ Single dataloader returned")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    batch_data = next(iter(val_loader))
    
    print(f"Batch data type: {type(batch_data)}")
    
    if isinstance(batch_data, dict):
        print(f"Batch data is a dictionary with keys: {list(batch_data.keys())}")
        
        # Extract images and masks from dict
        if 'image' in batch_data and 'mask' in batch_data:
            images = batch_data['image']
            masks = batch_data['mask']
            print(f"\n✓ Successfully extracted from dict")
            print(f"  Images key: 'image'")
            print(f"  Masks key: 'mask'")
        elif 'images' in batch_data and 'masks' in batch_data:
            images = batch_data['images']
            masks = batch_data['masks']
            print(f"\n✓ Successfully extracted from dict")
            print(f"  Images key: 'images'")
            print(f"  Masks key: 'masks'")
        else:
            print(f"\n✗ ERROR: Unknown dict keys. Expected 'image'/'mask' or 'images'/'masks'")
            print(f"  Available keys: {list(batch_data.keys())}")
            sys.exit(1)
        
        print(f"\nImages: shape={images.shape}, dtype={images.dtype}")
        print(f"Masks: shape={masks.shape}, dtype={masks.dtype}")
        
    elif isinstance(batch_data, (list, tuple)):
        print(f"Batch data length: {len(batch_data)}")
        for i, item in enumerate(batch_data):
            if hasattr(item, 'shape'):
                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  [{i}]: type={type(item)}")
        
        # Try to unpack
        if len(batch_data) == 2:
            images, masks = batch_data
            print(f"\n✓ Successfully unpacked 2 values")
        elif len(batch_data) == 3:
            images, masks, metadata = batch_data
            print(f"\n✓ Successfully unpacked 3 values")
            print(f"  Metadata type: {type(metadata)}")
        else:
            print(f"\n⚠ Unusual number of values: {len(batch_data)}")
            images, masks = batch_data[0], batch_data[1]
            print(f"  Taking first 2 values as images and masks")
        
        print(f"\nImages: shape={images.shape}, dtype={images.dtype}")
        print(f"Masks: shape={masks.shape}, dtype={masks.dtype}")
        
    else:
        print(f"✗ ERROR: Expected tuple/list/dict from dataloader, got {type(batch_data)}")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ ERROR in dataloader test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: Check model forward pass with actual batch
# ============================================================================
print("\n" + "─"*70)
print("TEST 2: Checking model forward pass")
print("─"*70)

try:
    from src.models.unet import UNet
    
    # Create UNet model
    model = UNet(in_channels=images.shape[1], num_classes=1)
    model.eval()
    
    print(f"✓ Created UNet model")
    print(f"  Input channels: {images.shape[1]}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\nForward pass successful!")
    print(f"  Output type: {type(outputs)}")
    
    if isinstance(outputs, (tuple, list)):
        print(f"  ✗ WARNING: Model returns {len(outputs)} values")
        print(f"  This may cause unpacking errors")
        for i, out in enumerate(outputs):
            if hasattr(out, 'shape'):
                print(f"  Output[{i}]: shape={out.shape}")
        # Take first output
        outputs = outputs[0]
    elif isinstance(outputs, dict):
        print(f"  ✗ WARNING: Model returns dict with keys: {list(outputs.keys())}")
        print(f"  Taking 'out' key")
        outputs = outputs['out']
    
    print(f"  Output shape: {outputs.shape}")
    print(f"  Output dtype: {outputs.dtype}")
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    # Verify shape matches expected
    expected_shape = (images.shape[0], 1, images.shape[2], images.shape[3])
    if outputs.shape == expected_shape:
        print(f"  ✓ Output shape matches expected: {expected_shape}")
    else:
        print(f"  ✗ WARNING: Output shape {outputs.shape} != expected {expected_shape}")

except Exception as e:
    print(f"\n✗ ERROR in model test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Check DeepLabv3+ forward pass
# ============================================================================
print("\n" + "─"*70)
print("TEST 3: Checking DeepLabv3+ forward pass")
print("─"*70)

try:
    from src.models.deeplabv3plus import DeepLabV3Plus
    
    # Create DeepLabv3+ model
    model = DeepLabV3Plus(in_channels=images.shape[1], num_classes=1, pretrained=False)
    model.eval()
    
    print(f"✓ Created DeepLabv3+ model")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\nForward pass successful!")
    print(f"  Output type: {type(outputs)}")
    
    if isinstance(outputs, (tuple, list)):
        print(f"  ✗ WARNING: Model returns {len(outputs)} values")
        outputs = outputs[0]
    elif isinstance(outputs, dict):
        print(f"  ⚠ Model returns dict (normal for DeepLabv3+): {list(outputs.keys())}")
        # The DeepLabV3Plus wrapper should handle this, but let's check
        outputs = outputs['out'] if 'out' in outputs else outputs
    
    print(f"  Output shape: {outputs.shape}")
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

except Exception as e:
    print(f"\n✗ ERROR in DeepLabv3+ test: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is optional

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

print("\n✓ All critical tests passed!")
print("\nYour setup:")
print(f"  - Dataloader returns: {len(result)} values")
if isinstance(batch_data, dict):
    print(f"  - Batch format: Dictionary with keys {list(batch_data.keys())}")
else:
    print(f"  - Batch contains: {len(batch_data)} items")
print(f"  - Images shape: {images.shape}")
print(f"  - Masks shape: {masks.shape}")
print(f"  - UNet output: {outputs.shape}")

print("\nRecommendations:")
if isinstance(batch_data, dict):
    print(f"  ✓ Dataloader returns dictionary format")
    print(f"    Keys used: 'image' and 'mask'")
    print(f"    This is NORMAL and handled correctly")
elif len(batch_data) > 2:
    print(f"  ⚠ Your dataloader returns {len(batch_data)} values")
    print(f"    Evaluation script will use first 2 (images, masks)")
    print(f"    This is NORMAL and handled correctly")
else:
    print(f"  ✓ Dataloader returns standard 2 values (images, masks)")

print("\nYou can now run: python evaluate_models.py")
