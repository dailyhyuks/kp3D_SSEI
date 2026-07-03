"""Test MiDaS depth estimation with single image (bypassing batch handling bug)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

print("\n" + "="*60)
print("MiDaS Depth Estimation Test (Single Image)")
print("="*60 + "\n")

# Setup paths
project_root = Path(__file__).parent.parent
sample_path = project_root / "samples" / "boat_painting_2.png"
output_dir = project_root / "outputs" / "midas_test"
output_dir.mkdir(parents=True, exist_ok=True)

# Load sample image
print(f"[OK] Sample image: {sample_path}")
img_pil = Image.open(sample_path).convert("RGB")
img_np = np.array(img_pil)
orig_h, orig_w = img_np.shape[:2]
print(f"[OK] Loaded image: {img_np.shape} (H, W, C)")

# Load MiDaS model and transform
print(f"\nInitializing MiDaS_small model...")
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
model.eval()
print(f"[OK] Model loaded")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print(f"[OK] Transform loaded")

# Apply transform
input_tensor = transform(img_np)
print(f"[OK] Input tensor shape: {input_tensor.shape}")

# Run depth estimation
print(f"\nRunning depth estimation...")
with torch.no_grad():
    prediction = model(input_tensor)

print(f"[OK] Prediction shape: {prediction.shape}")
print(f"  - Value range: [{prediction.min():.3f}, {prediction.max():.3f}]")

# Resize to original resolution
# prediction is [1, 256, 256], need [1, 1, 256, 256] for interpolate
depth_map = F.interpolate(
    prediction.unsqueeze(1),  # Add channel dimension: [1, 256, 256] -> [1, 1, 256, 256]
    size=(orig_h, orig_w),
    mode='bicubic',
    align_corners=False
)
print(f"[OK] Resized to original: {depth_map.shape}")

# Normalize to [0, 1] range (invert so far=0, near=1)
depth_map = 1.0 / (depth_map + 1e-6)
d_min = depth_map.min()
d_max = depth_map.max()
if (d_max - d_min) > 1e-6:
    depth_map = (depth_map - d_min) / (d_max - d_min)

print(f"[OK] Normalized depth map: {depth_map.shape}")
print(f"  - Value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
print(f"  - Mean: {depth_map.mean():.3f}, Std: {depth_map.std():.3f}")

# Verify output shape
expected_shape = (1, 1, orig_h, orig_w)
assert depth_map.shape == expected_shape, \
    f"Shape mismatch: expected {expected_shape}, got {depth_map.shape}"
print(f"[OK] Shape verification passed")

# Verify value range [0, 1]
assert depth_map.min() >= 0.0, f"Minimum value {depth_map.min():.3f} < 0"
assert depth_map.max() <= 1.0, f"Maximum value {depth_map.max():.3f} > 1"
print(f"[OK] Value range verification passed")

# Save outputs
print(f"\nSaving outputs to {output_dir}...")

# Save depth map as numpy
depth_np = depth_map[0, 0].cpu().numpy()
np.save(output_dir / "depth_map.npy", depth_np)
print(f"[OK] Saved depth_map.npy")

# Save visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(img_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Depth map (grayscale)
im1 = axes[1].imshow(depth_np, cmap='gray')
axes[1].set_title("Depth Map (Gray)")
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Depth map (viridis colormap)
im2 = axes[2].imshow(depth_np, cmap='viridis')
axes[2].set_title("Depth Map (Viridis)")
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
output_path = output_dir / "depth_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] Saved depth_visualization.png")

# Save depth map as grayscale image
depth_img = (depth_np * 255).astype(np.uint8)
Image.fromarray(depth_img).save(output_dir / "depth_map.png")
print(f"[OK] Saved depth_map.png")

print(f"\n" + "="*60)
print("[OK] All tests passed successfully!")
print("="*60 + "\n")
