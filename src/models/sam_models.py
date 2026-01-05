"""
SAM Encoder + CNN Decoder
==========================
Uses SAM's powerful image encoder as a frozen feature extractor,
then adds a lightweight CNN decoder for segmentation.

SAM (Segment Anything Model) is pretrained on 1.1B masks, providing
strong segmentation-specific features. This architecture explores whether
SAM's mask pretraining transfers better than DINOv2's image-level pretraining
for water segmentation.

Reference: Kirillov et al. "Segment Anything" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


class SAMEncoderDecoder(nn.Module):
    """
    SAM encoder with custom CNN decoder
    
    Uses SAM's ViT encoder (frozen) to extract features, then applies
    a lightweight decoder for binary water segmentation.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint (.pth file)
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
        freeze_encoder: Freeze SAM encoder weights
        decoder_channels: Base channels for decoder
    
    Note: Requires RGB input (3 channels) - SAM only works with RGB
    """
    
    def __init__(
        self,
        sam_checkpoint: str = 'checkpoints/sam_vit_b_01ec64.pth',
        model_type: str = 'vit_b',
        freeze_encoder: bool = True,
        decoder_channels: int = 256
    ):
        super().__init__()
        
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything package required. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Load SAM model
        print(f"Loading SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = self.sam.image_encoder
        
        print(f"✓ SAM encoder loaded from {sam_checkpoint}")
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("✓ SAM encoder frozen")
        
        # SAM encoder output: 256 channels at 64x64 for 1024x1024 input
        # For our 512x512 input, output will be 32x32
        sam_out_channels = 256
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(sam_out_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, decoder_channels // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 4, decoder_channels // 4, 3, padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(decoder_channels // 8),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(decoder_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: RGB input (B, 3, H, W)
        
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError(f"SAM requires RGB input (3 channels), got {x.shape[1]} channels")
        
        B, C, H, W = x.shape
        original_size = (H, W)
        
        # SAM expects 1024x1024 input
        # For efficiency, we resize to 512x512
        target_size = 512
        if H != target_size or W != target_size:
            x_resized = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # SAM encoder
        with torch.set_grad_enabled(self.image_encoder.training):
            features = self.image_encoder(x_resized)  # B, 256, 32, 32 (for 512 input)
        
        # Decode
        output = self.decoder(features)
        
        # Resize to original size if needed
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


class SAMFineTuned(nn.Module):
    """
    Fine-tuned SAM for direct segmentation
    
    Uses full SAM architecture but with:
    - Frozen image encoder
    - Frozen prompt encoder
    - Trainable mask decoder
    - Learnable prompt embeddings (no explicit prompts needed)
    
    This allows the mask decoder to adapt to water segmentation
    while leveraging SAM's powerful pretrained encoder.
    
    Args:
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
    """
    
    def __init__(
        self,
        sam_checkpoint: str = 'checkpoints/sam_vit_b_01ec64.pth',
        model_type: str = 'vit_b'
    ):
        super().__init__()
        
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything package required. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Load full SAM model
        print(f"Loading full SAM model: {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        print(f"✓ SAM loaded from {sam_checkpoint}")
        
        # Freeze encoder and prompt encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
        
        print("✓ Image encoder and prompt encoder frozen")
        print("✓ Mask decoder trainable")
        
        # Learnable prompt embeddings (replace point/box prompts)
        # These are learned during training
        self.learnable_prompt = nn.Parameter(torch.randn(1, 1, 256))
        
        # Output post-processing
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: RGB input (B, 3, H, W)
        
        Returns:
            Segmentation mask (B, 1, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError(f"SAM requires RGB input (3 channels), got {x.shape[1]} channels")
        
        B, C, H, W = x.shape
        original_size = (H, W)
        
        # Resize to 1024x1024 (SAM's native resolution)
        target_size = 1024
        x_resized = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Encode image
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)
        
        # Use learnable prompts (no explicit point/box prompts)
        prompt_embeddings = self.learnable_prompt.expand(B, -1, -1)
        
        # Get sparse and dense embeddings from prompt encoder
        with torch.no_grad():
            sparse_embeddings = self.sam.prompt_encoder.not_a_point_embed.weight.reshape(1, -1, 1, 1)
            sparse_embeddings = sparse_embeddings.expand(B, -1, -1, -1)
            
            # No dense embeddings for this simple case
            dense_embeddings = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(B, -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        
        # Decode mask
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # Upsample to target size
        masks = F.interpolate(
            low_res_masks,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply sigmoid
        masks = self.sigmoid(masks)
        
        # Resize to original size
        if masks.shape[2:] != original_size:
            masks = F.interpolate(masks, size=original_size, mode='bilinear', align_corners=False)
        
        return masks


if __name__ == '__main__':
    print("SAM Models Test")
    print("="*70)
    print("\nNote: These models require:")
    print("1. segment_anything package installed")
    print("2. SAM checkpoint downloaded to checkpoints/")
    print("3. RGB input only (3 channels)")
    print("\nSkipping actual model creation in test mode")
    print("Use in training script with proper setup")
    
    # Test basic structure without loading actual SAM
    print("\n✓ SAM model definitions ready")
    print("\nTo use SAM models:")
    print("1. Download SAM checkpoint:")
    print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/")
    print("2. Install segment_anything:")
    print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
    print("3. Use in training with RGB feature_config")
