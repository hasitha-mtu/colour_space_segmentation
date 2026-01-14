"""
Quick checkpoint inspection script
Use this to verify what's inside your model checkpoints before running visualizations
"""

import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect a PyTorch checkpoint file"""
    print(f"\n{'='*70}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*70}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check top-level keys
        print("\n1. TOP-LEVEL KEYS:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"   - {key}: dict with {len(checkpoint[key])} items")
                elif isinstance(checkpoint[key], torch.Tensor):
                    print(f"   - {key}: Tensor {checkpoint[key].shape}")
                else:
                    print(f"   - {key}: {type(checkpoint[key])}")
        else:
            print("   Checkpoint is not a dict!")
        
        # Extract state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n   Using 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\n   Using 'state_dict' key")
        elif isinstance(checkpoint, dict) and 'backbone.conv1.weight' in checkpoint:
            state_dict = checkpoint
            print("\n   Using direct state dict")
        elif isinstance(checkpoint, dict):
            # Try to find state dict with 'model.' prefix
            first_key = list(checkpoint.keys())[0] if checkpoint else None
            if first_key and 'model.' in first_key:
                state_dict = checkpoint
                print("\n   Using direct state dict (with 'model.' prefix)")
            else:
                print("\n   ERROR: Cannot find state dict!")
                return
        else:
            print("\n   ERROR: Unsupported checkpoint format!")
            return
        
        # Check for 'model.' prefix
        has_prefix = False
        sample_keys = list(state_dict.keys())[:5]
        for key in sample_keys:
            if key.startswith('model.'):
                has_prefix = True
                break
        
        print(f"\n2. STATE DICT INFO:")
        print(f"   Total parameters: {len(state_dict)}")
        print(f"   Has 'model.' prefix: {has_prefix}")
        
        # Check for channel adapter
        has_channel_adapter = False
        channel_adapter_keys = [
            'channel_adapter.0.weight',
            'model.channel_adapter.0.weight'
        ]
        
        for key in channel_adapter_keys:
            if key in state_dict:
                has_channel_adapter = True
                adapter_weight = state_dict[key]
                print(f"\n   ✓ HAS CHANNEL ADAPTER!")
                print(f"     Key: {key}")
                print(f"     Shape: {adapter_weight.shape}")
                print(f"     Architecture: {adapter_weight.shape[1]} channels → {adapter_weight.shape[0]} channels")
                break
        
        if not has_channel_adapter:
            print(f"\n   No channel adapter detected")
        
        # Check first conv layer
        conv1_keys = [
            'backbone.conv1.weight',
            'model.backbone.conv1.weight',
            'model.model.backbone.conv1.weight',
            'conv1.weight',
            'model.conv1.weight'
        ]
        
        conv1_weight = None
        conv1_key_found = None
        for key in conv1_keys:
            if key in state_dict:
                conv1_weight = state_dict[key]
                conv1_key_found = key
                break
        
        print(f"\n3. INPUT LAYER INFO:")
        
        if has_channel_adapter:
            # Get the adapter weight
            adapter_key = None
            for key in channel_adapter_keys:
                if key in state_dict:
                    adapter_key = key
                    break
            
            if adapter_key:
                adapter_weight = state_dict[adapter_key]
                in_channels = adapter_weight.shape[1]
                out_channels = adapter_weight.shape[0]
                
                print(f"   Architecture: CHANNEL ADAPTER")
                print(f"   Adapter layer: {adapter_key}")
                print(f"   Shape: {adapter_weight.shape}")
                print(f"   ✓ TRUE INPUT CHANNELS: {in_channels}")
                print(f"   → Projects to {out_channels} channels before backbone")
                
                if in_channels == 3:
                    print(f"   → This is an RGB model (3 channels)")
                elif in_channels == 10:
                    print(f"   → This is a multi-channel model (10 channels)")
                else:
                    print(f"   → This uses {in_channels} input channels")
        
        elif conv1_weight is not None:
            in_channels = conv1_weight.shape[1]
            print(f"   First conv layer: {conv1_key_found}")
            print(f"   Shape: {conv1_weight.shape}")
            print(f"   ✓ INPUT CHANNELS: {in_channels}")
            
            if in_channels == 3:
                print(f"   → This is an RGB model (3 channels)")
            elif in_channels == 10:
                print(f"   → This is a multi-channel model (10 channels)")
            else:
                print(f"   → This uses {in_channels} input channels")
        else:
            print(f"   ✗ Cannot find first conv layer!")
            print(f"   Available keys (first 10):")
            for key in list(state_dict.keys())[:10]:
                print(f"      - {key}")
        
        # Check for aux_classifier
        has_aux = any('aux_classifier' in k for k in state_dict.keys())
        print(f"\n4. AUXILIARY CLASSIFIER:")
        print(f"   Has aux_classifier: {has_aux}")
        
        # Sample keys
        print(f"\n5. SAMPLE KEYS (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"   {i+1}. {key}: {shape}")
        
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR loading checkpoint: {e}\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\nUsage: python inspect_checkpoint.py <checkpoint1.pth> [checkpoint2.pth] [...]")
        print("\nExample:")
        print("  python inspect_checkpoint.py model_rgb.pth")
        print("  python inspect_checkpoint.py model_rgb.pth model_all_channels.pth")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("PYTORCH CHECKPOINT INSPECTOR")
    print("="*70)
    
    for checkpoint_path in sys.argv[1:]:
        if not Path(checkpoint_path).exists():
            print(f"\n✗ File not found: {checkpoint_path}")
            continue
        
        inspect_checkpoint(checkpoint_path)
    
    print("\nDone!\n")


if __name__ == "__main__":
    main()
