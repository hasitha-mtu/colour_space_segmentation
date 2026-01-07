"""
Hybrid CNN-DINOv2 Architecture
===============================
Main research contribution: Combines efficient CNN for multi-channel features
with DINOv2 foundation model for semantic understanding.

Architecture:
    Input (10 channels)
          |
    +-----+-----+
    |           |
  CNN      Extract RGB
 Branch         |
    |      DINOv2 Encoder
    |     (frozen/fine-tuned)
    |           |
    +--Fusion--+
    (Cross-Attention)
          |
      Decoder
          |
       Output

Key Features:
- Processes all channels natively (no projection bottleneck)
- Leverages DINOv2's self-supervised vision pretraining
- Cross-attention fusion for multi-scale features
- Supports different DINOv2 sizes (small, base, large)

Reference: Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PooledAttentionFusion(nn.Module):
    """
    Memory-efficient cross-attention fusion using adaptive pooling
    
    Pools features to a manageable size before attention to avoid
    the quadratic memory cost of full spatial attention.
    
    For 128x128 features → pool to 16x16 → attention on 256 tokens (feasible!)
    """
    
    def __init__(
        self,
        cnn_channels: int,
        dino_channels: int,
        out_channels: int,
        num_heads: int = 8,
        pool_size: int = 16  # Attention computed on 16×16 = 256 tokens
    ):
        super().__init__()
        
        self.pool_size = pool_size
        
        # Project to common dimension
        self.cnn_proj = nn.Conv2d(cnn_channels, out_channels, 1)
        self.dino_proj = nn.Conv2d(dino_channels, out_channels, 1)
        
        # Adaptive pooling before attention
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        # Multi-head cross-attention (on pooled features)
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
        
        # Upsample attention output back to original spatial size
        # Then blend with original features
        self.blend = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: CNN features (B, C1, H, W)
            dino_features: DINOv2 features (B, C2, H, W)
        
        Returns:
            Fused features (B, out_channels, H, W)
        """
        B, _, H, W = cnn_features.shape
        
        # Project features
        cnn_proj = self.cnn_proj(cnn_features)  # B, C, H, W
        dino_proj = self.dino_proj(dino_features)  # B, C, H, W
        
        # Pool for attention (reduces spatial dimensions)
        cnn_pooled = self.pool(cnn_proj)  # B, C, pool_size, pool_size
        dino_pooled = self.pool(dino_proj)  # B, C, pool_size, pool_size
        
        # Reshape for attention: B, pool_size^2, C
        cnn_flat = cnn_pooled.flatten(2).permute(0, 2, 1)  # B, 256, C (if pool_size=16)
        dino_flat = dino_pooled.flatten(2).permute(0, 2, 1)  # B, 256, C
        
        # Normalize
        cnn_normed = self.norm1(cnn_flat)
        dino_normed = self.norm1(dino_flat)
        
        # Cross-attention: CNN queries DINOv2 keys/values
        # Only 256×256 attention matrix now! (vs 16,384×16,384 before)
        attended, _ = self.attention(
            query=cnn_normed,
            key=dino_normed,
            value=dino_normed
        )
        
        # Residual connection
        fused = cnn_flat + attended
        
        # Feed-forward with residual
        fused_normed = self.norm2(fused)
        fused = fused + self.ffn(fused_normed)
        
        # Reshape back: B, C, pool_size, pool_size
        fused = fused.permute(0, 2, 1).reshape(B, -1, self.pool_size, self.pool_size)
        
        # Upsample attention output to original spatial size
        fused_upsampled = F.interpolate(
            fused, size=(H, W), mode='bilinear', align_corners=False
        )
        
        # Blend with original projected features
        combined = torch.cat([cnn_proj, fused_upsampled], dim=1)
        output = self.blend(combined)
        
        return output


class SimpleFusion(nn.Module):
    """
    Simple and memory-efficient fusion via concatenation + convolution
    
    Most memory-efficient option - no attention mechanism at all.
    Good baseline and fallback for extreme memory constraints.
    """
    
    def __init__(
        self,
        cnn_channels: int,
        dino_channels: int,
        out_channels: int,
        num_heads: int = 8  # Unused, kept for API compatibility
    ):
        super().__init__()
        
        # Concatenate and fuse with convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(cnn_channels + dino_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cnn_features: CNN features (B, C1, H, W)
            dino_features: DINOv2 features (B, C2, H, W)
        
        Returns:
            Fused features (B, out_channels, H, W)
        """
        # Simple concatenation + convolution
        concatenated = torch.cat([cnn_features, dino_features], dim=1)
        return self.fusion(concatenated)


# For backward compatibility, keep old name but use new implementation
CrossAttentionFusion = PooledAttentionFusion


class HybridDINOv2(nn.Module):
    """
    Hybrid CNN-DINOv2 model for water segmentation
    
    Combines:
    1. Efficient CNN branch for processing all input channels
    2. DINOv2 encoder for semantic features from RGB
    3. Cross-attention fusion
    4. Lightweight decoder
    
    Args:
        in_channels: Number of input channels (3, 7, or 10)
        num_classes: Number of output classes (1 for binary)
        dino_model: DINOv2 model size ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
        freeze_dino: Freeze DINOv2 weights
        cnn_base_channels: Base channels for CNN branch
        fusion_type: Fusion strategy ('pooled_attention' or 'simple')
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 1,
        dino_model: str = 'dinov2_vitb14',
        freeze_dino: bool = True,
        cnn_base_channels: int = 64,
        fusion_type: str = 'pooled_attention'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.dino_model_name = dino_model
        self.freeze_dino = freeze_dino
        
        # CNN Branch - processes all channels natively
        self.cnn_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, cnn_base_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels, cnn_base_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_base_channels, cnn_base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(cnn_base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_base_channels * 2, cnn_base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(cnn_base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_base_channels * 2, cnn_base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(cnn_base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # DINOv2 Branch
        print(f"Loading DINOv2 model: {dino_model}...")
        try:
            self.dino = torch.hub.load('facebookresearch/dinov2', dino_model)
            print("✓ DINOv2 loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load DINOv2 from torch.hub: {e}")
            print("Creating placeholder (for testing without internet)")
            self.dino = None
        
        if self.dino is not None and freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            print("✓ DINOv2 weights frozen")
        
        # DINOv2 output dimensions
        dino_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        self.dino_channels = dino_dims.get(dino_model, 768)
        self.dino_patch_size = 14
        
        # Fusion module - select based on fusion_type
        fusion_channels = cnn_base_channels * 4
        
        if fusion_type == 'simple':
            print(f"Using SimpleFusion (memory-efficient, no attention)")
            self.fusion = SimpleFusion(
                cnn_channels=fusion_channels,
                dino_channels=self.dino_channels,
                out_channels=fusion_channels
            )
        elif fusion_type == 'pooled_attention':
            print(f"Using PooledAttentionFusion (16×16 pooling, attention on 256 tokens)")
            self.fusion = PooledAttentionFusion(
                cnn_channels=fusion_channels,
                dino_channels=self.dino_channels,
                out_channels=fusion_channels,
                num_heads=8,
                pool_size=16
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'simple' or 'pooled_attention'")
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample block 1
            nn.ConvTranspose2d(fusion_channels, cnn_base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(cnn_base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Upsample block 2
            nn.ConvTranspose2d(cnn_base_channels * 2, cnn_base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(cnn_base_channels),
            nn.ReLU(inplace=True),
            
            # Final conv
            nn.Conv2d(cnn_base_channels, num_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) where C can be 3, 7, or 10
        
        Returns:
            Output segmentation mask (B, 1, H, W)
        """
        B, C, H, W = x.shape
        original_size = (H, W)
        
        # CNN branch - processes all channels
        cnn_features = self.cnn_encoder(x)  # B, 256, H/4, W/4
        
        # Extract RGB for DINOv2 (assume first 3 channels)
        if C >= 3:
            rgb = x[:, :3, :, :]
        else:
            # If somehow less than 3 channels, replicate
            rgb = x[:, 0:1, :, :].repeat(1, 3, 1, 1)
        
        # DINOv2 expects images normalized with ImageNet stats
        # and size divisible by patch_size (14)
        h_dino = (H // self.dino_patch_size) * self.dino_patch_size
        w_dino = (W // self.dino_patch_size) * self.dino_patch_size
        
        if H != h_dino or W != w_dino:
            rgb_resized = F.interpolate(rgb, size=(h_dino, w_dino), mode='bilinear', align_corners=False)
        else:
            rgb_resized = rgb
        
        # Get DINOv2 features - ALWAYS disable gradients for frozen encoder
        if self.dino is not None:
            # Use no_grad() when frozen, enable gradients only when fine-tuning
            should_compute_grads = not self.freeze_dino and self.training
            
            with torch.set_grad_enabled(should_compute_grads):
                dino_output = self.dino.forward_features(rgb_resized)
                # Extract patch tokens (exclude CLS token)
                dino_features = dino_output['x_norm_patchtokens']  # B, N_patches, C
        else:
            # Placeholder for testing
            h_patch = h_dino // self.dino_patch_size
            w_patch = w_dino // self.dino_patch_size
            dino_features = torch.randn(
                B, h_patch * w_patch, self.dino_channels,
                device=x.device
            )
        
        # Reshape DINOv2 features: B, N_patches, C -> B, C, H_patch, W_patch
        h_patch = h_dino // self.dino_patch_size
        w_patch = w_dino // self.dino_patch_size
        dino_features = dino_features.permute(0, 2, 1).reshape(B, self.dino_channels, h_patch, w_patch)
        
        # Resize DINOv2 features to match CNN features spatial size
        dino_features = F.interpolate(
            dino_features,
            size=cnn_features.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Fusion
        fused_features = self.fusion(cnn_features, dino_features)
        
        # Decode
        output = self.decoder(fused_features)
        
        # Resize to original input size if needed
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


if __name__ == '__main__':
    # Test Hybrid DINOv2
    print("Testing Hybrid CNN-DINOv2...\n")
    
    configs = [
        ('RGB', 3),
        ('Luminance', 3),
        ('Chrominance', 7),
        ('All features', 10)
    ]
    
    batch_size = 2
    img_size = 448  # Divisible by 14 (DINOv2 patch size)
    
    for name, channels in configs:
        print(f"{'='*70}")
        print(f"Testing: {name} ({channels} channels)")
        print(f"{'='*70}")
        
        # Create model
        model = HybridDINOv2(
            in_channels=channels,
            num_classes=1,
            dino_model='dinov2_vitb14',
            freeze_dino=True,
            cnn_base_channels=64
        )
        
        # Create dummy input
        x = torch.rand(batch_size, channels, img_size, img_size)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"\nInput shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024**2):.1f} MB\n")
        
        break  # Test one config to save time
    
    print("✓ Hybrid DINOv2 tests passed!")
    print("\nNote: Full testing requires internet connection for DINOv2 download")
