"""Diagnostic test for MiDaS depth estimation."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from kp3d.modules.shade.midas import MiDaSDepthEstimator
from kp3d.modules.shade.base import ShadeConfig


def test_midas_depth_estimation():
    """Test MiDaS depth estimation on sample image."""

    # Setup paths
    project_root = Path(__file__).parent.parent
    sample_path = project_root / "samples" / "boat_painting_2.png"
    output_dir = project_root / "outputs" / "midas_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MiDaS Depth Estimation Diagnostic Test")
    print(f"{'='*60}\n")

    # Check if sample image exists
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
    print(f"[OK] Sample image found: {sample_path}")

    # Load image
    img_pil = Image.open(sample_path).convert("RGB")
    img_np = np.array(img_pil)
    print(f"[OK] Loaded image: {img_np.shape} (H, W, C)")

    # Convert to tensor (B, C, H, W) in range [0, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    print(f"[OK] Converted to tensor: {img_tensor.shape}, range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

    # Initialize MiDaS with fastest model
    print(f"\nInitializing MiDaS_small model...")
    config = ShadeConfig(depth_model="MiDaS_small")
    estimator = MiDaSDepthEstimator(config=config)
    print(f"[OK] Model loaded on device: {estimator.device}")
    print(f"[OK] Model initialized: {estimator._initialized}")

    # Run depth estimation
    print(f"\nRunning depth estimation...")
    with torch.no_grad():
        output = estimator(img_tensor)

    depth_map = output.result
    print(f"[OK] Depth map computed: {depth_map.shape}")
    print(f"  - Value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"  - Mean: {depth_map.mean():.3f}, Std: {depth_map.std():.3f}")

    # Verify output shape
    expected_shape = (1, 1, img_np.shape[0], img_np.shape[1])
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

    # Print metadata
    print(f"\nMetadata:")
    for key, value in output.metadata.items():
        print(f"  - {key}: {value}")

    print(f"\n{'='*60}")
    print("[OK] All tests passed successfully!")
    print(f"{'='*60}\n")

    return True


if __name__ == "__main__":
    test_midas_depth_estimation()
