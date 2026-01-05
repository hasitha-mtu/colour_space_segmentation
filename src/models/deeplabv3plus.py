"""
DeepLabv3+ with Channel Adaptation
===================================
Supports multi-channel inputs (3, 7, 10 channels) with pretrained ResNet50 backbone

For non-RGB inputs, uses a learnable 1x1 convolution to project to 3 channels
before the pretrained backbone, then uses standard DeepLabv3+ architecture.

Reference: Chen et al. "Encoder-Decoder with Atrous Separable Convolution for 
           Semantic Image Segmentation" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ with channel adaptation for multi-channel inputs
    
    Architecture:
    - For RGB (3ch): Uses pretrained ResNet50 directly
    - For other (7ch, 10ch): Projects to 3ch → Pretrained ResNet50
    
    Args:
        in_channels: Number of input channels (3, 7, or 10)
        num_classes: Number of output classes (1 for binary segmentation)
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone weights (fine-tune decoder only)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Channel adapter for non-RGB inputs
        if in_channels != 3:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
            print(f"Added channel adapter: {in_channels} → 3 channels")
        else:
            self.channel_adapter = None
        
        # Load DeepLabv3+ with ResNet50 backbone
        if pretrained:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=weights)
            print("Loaded ImageNet pretrained ResNet50 backbone")
        else:
            self.model = deeplabv3_resnet50(weights=None)
            print("Using randomly initialized ResNet50 backbone")
        
        # Replace classifier head for binary segmentation
        # DeepLabv3 has classifier and aux_classifier
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        if hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Add sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
        # Optionally freeze backbone
        if freeze_backbone and pretrained:
            self._freeze_backbone()
            print("Froze backbone weights (training decoder only)")
    
    def _freeze_backbone(self):
        """Freeze backbone (ResNet50) weights"""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) where C can be 3, 7, or 10
        
        Returns:
            Output segmentation mask (B, 1, H, W) with sigmoid activation
        """
        original_size = x.shape[2:]
        
        # Adapt channels if needed
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        
        # Forward through DeepLabv3+
        output = self.model(x)
        
        # DeepLabv3 returns dict with 'out' and optionally 'aux'
        if isinstance(output, dict):
            x = output['out']
        else:
            x = output
        
        # Apply sigmoid
        x = self.sigmoid(x)
        
        # Resize to original input size if needed
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x


if __name__ == '__main__':
    # Test DeepLabv3+ with different input configurations
    print("Testing DeepLabv3+ with different input channels...\n")
    
    configs = [
        ('RGB', 3),
        ('Luminance', 3),
        ('Chrominance', 7),
        ('All features', 10)
    ]
    
    batch_size = 2
    img_size = 512
    
    for name, channels in configs:
        print(f"{'='*70}")
        print(f"Testing: {name} ({channels} channels)")
        print(f"{'='*70}")
        
        # Create model
        model = DeepLabV3Plus(
            in_channels=channels,
            num_classes=1,
            pretrained=(channels == 3),  # Only use pretrained for RGB
            freeze_backbone=False
        )
        
        # Create dummy input
        x = torch.rand(batch_size, channels, img_size, img_size)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Check output
        print(f"\nInput shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB\n")
    
    # Test gradient flow
    print(f"{'='*70}")
    print("Testing gradient flow")
    print(f"{'='*70}")
    
    model = DeepLabV3Plus(in_channels=3, num_classes=1, pretrained=False)
    x = torch.rand(1, 3, 256, 256, requires_grad=True)
    target = torch.rand(1, 1, 256, 256)
    
    output = model(x)
    loss = F.binary_cross_entropy(output, target)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient: {x.grad is not None}")
    print(f"Backbone gradient: {model.model.backbone.conv1.weight.grad is not None}")
    
    # Test with frozen backbone
    print(f"\n{'='*70}")
    print("Testing with frozen backbone")
    print(f"{'='*70}")
    
    model = DeepLabV3Plus(
        in_channels=10,
        num_classes=1,
        pretrained=True,
        freeze_backbone=True
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
    print(f"Percentage trainable: {100 * trainable / (trainable + frozen):.1f}%")
    
    print("\n✓ DeepLabv3+ tests passed!")
