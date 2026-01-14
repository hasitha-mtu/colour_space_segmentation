"""
Quick visualization script for rapid inspection of DeepLabv3+ predictions
Simpler and faster than the comprehensive comparison script
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import sys

def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract 10-channel features"""
    img = image.astype(np.float32) / 255.0
    rgb = img.copy()
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    ycbcr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    
    features = np.stack([
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
        lab[:, :, 1], lab[:, :, 2],
        ycbcr[:, :, 1], ycbcr[:, :, 2],
    ], axis=-1)
    
    return features

@torch.no_grad()
def predict(image, model, use_all_channels=False, device='cuda'):
    """Run inference"""
    if use_all_channels:
        input_data = extract_features(image)
    else:
        input_data = image.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(input_data.transpose(2, 0, 1)).float()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    output = model(input_tensor)
    if isinstance(output, dict):
        output = output['out']
    
    pred = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_binary = (pred > 0.5).astype(np.uint8)
    
    return pred_binary, pred

def quick_visualize(image, gt, rgb_pred, all_pred, save_path=None):
    """Create simple side-by-side visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt, cmap='Blues')
    axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_pred, cmap='Blues')
    axes[0, 2].set_title('RGB Model', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    overlay_rgb = image.copy()
    overlay_rgb[rgb_pred == 1] = [0, 255, 0]
    overlay_rgb[gt == 1] = [255, 0, 0]
    overlap = np.logical_and(rgb_pred, gt)
    overlay_rgb[overlap] = [255, 255, 0]
    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title('RGB Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(all_pred, cmap='Blues')
    axes[1, 1].set_title('All Channels Model', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    overlay_all = image.copy()
    overlay_all[all_pred == 1] = [0, 255, 0]
    overlay_all[gt == 1] = [255, 0, 0]
    overlap = np.logical_and(all_pred, gt)
    overlay_all[overlap] = [255, 255, 0]
    axes[1, 2].imshow(overlay_all)
    axes[1, 2].set_title('All Channels Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()

def main():
    """Quick visualization"""
    if len(sys.argv) < 5:
        print("Usage: python quick_visualize.py <rgb_model> <all_model> <image> <mask>")
        print("Example: python quick_visualize.py model_rgb.pth model_all.pth test.png mask.png")
        sys.exit(1)
    
    rgb_model_path = sys.argv[1]
    all_model_path = sys.argv[2]
    image_path = sys.argv[3]
    mask_path = sys.argv[4]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    from torchvision.models.segmentation import deeplabv3_resnet50
    
    print("Loading RGB model...")
    rgb_model = deeplabv3_resnet50(num_classes=1, weights=None)
    checkpoint = torch.load(rgb_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        rgb_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        rgb_model.load_state_dict(checkpoint['state_dict'])
    else:
        rgb_model.load_state_dict(checkpoint)
    rgb_model = rgb_model.to(device).eval()
    
    print("Loading All Channels model...")
    all_model = deeplabv3_resnet50(num_classes=1, weights=None)
    all_model.backbone.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
    checkpoint = torch.load(all_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        all_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        all_model.load_state_dict(checkpoint['state_dict'])
    else:
        all_model.load_state_dict(checkpoint)
    all_model = all_model.to(device).eval()
    
    # Load image and mask
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Loading mask: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    
    # Get predictions
    print("Running inference...")
    rgb_pred, _ = predict(image, rgb_model, use_all_channels=False, device=device)
    all_pred, _ = predict(image, all_model, use_all_channels=True, device=device)
    
    # Calculate metrics
    rgb_iou = np.logical_and(rgb_pred, mask).sum() / np.logical_or(rgb_pred, mask).sum()
    all_iou = np.logical_and(all_pred, mask).sum() / np.logical_or(all_pred, mask).sum()
    
    print(f"\nResults:")
    print(f"  RGB Model IoU:          {rgb_iou:.4f}")
    print(f"  All Channels Model IoU: {all_iou:.4f}")
    print(f"  Difference:             {all_iou - rgb_iou:+.4f}")
    
    # Visualize
    output_name = Path(image_path).stem + "_quick_comparison.png"
    quick_visualize(image, mask, rgb_pred, all_pred, save_path=output_name)

if __name__ == "__main__":
    main()
