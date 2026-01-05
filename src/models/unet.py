"""
UNet Architecture for Binary Segmentation
==========================================
Classic UNet with encoder-decoder structure and skip connections

Reference: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv => BN => ReLU) * 2
    
    This is the basic building block of UNet
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map from previous layer
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (pad if necessary)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for binary segmentation with configurable input channels
    
    Args:
        in_channels: Number of input channels (3 for RGB, 10 for all features)
        num_classes: Number of output classes (1 for binary segmentation)
        base_channels: Number of channels in first layer (default 64)
        bilinear: Use bilinear upsampling instead of transposed convolution
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output segmentation mask (B, 1, H, W)
        """
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        return logits


if __name__ == '__main__':
    # Test UNet
    print("Testing UNet...")
    
    # Test with different input configurations
    configs = [
        ('RGB', 3),
        ('Luminance', 3),
        ('Chrominance', 7),
        ('All features', 10)
    ]
    
    batch_size = 2
    img_size = 512
    
    for name, channels in configs:
        print(f"\n--- {name} ({channels} channels) ---")
        
        # Create model
        model = UNet(in_channels=channels, num_classes=1)
        
        # Create dummy input
        x = torch.rand(batch_size, channels, img_size, img_size)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Memory estimate
        param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        print(f"Model size: ~{param_size_mb:.1f} MB")
    
    # Test gradient flow
    print("\n--- Testing gradient flow ---")
    model = UNet(in_channels=3, num_classes=1)
    x = torch.rand(1, 3, 256, 256, requires_grad=True)
    target = torch.rand(1, 1, 256, 256)
    
    output = model(x)
    loss = F.binary_cross_entropy(output, target)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient: {x.grad is not None}")
    print(f"First layer gradient: {model.inc.double_conv[0].weight.grad is not None}")
    
    print("\n✓ UNet test passed!")
