"""Diagnostic test for RealESRGAN super-resolution module.

Tests the RealESRGAN x2 model with a small test image to verify:
1. Model can be loaded
2. Image processing works
3. Output is 2x the input size
4. Output is saved correctly

DEPENDENCY INSTALLATION ISSUE (as of 2026-01-26):
=================================================
BasicSR 1.4.2 has a bug in setup.py that prevents installation with Python 3.14.
This is due to a KeyError: '__version__' in the version parsing code.

WORKAROUND OPTIONS:
1. Use Python 3.11 or 3.12 (recommended)
2. Wait for BasicSR update with Python 3.14 support
3. Manually patch BasicSR setup.py (advanced)

Once dependencies are installed, this test will verify:
- RealESRGAN model loading
- Image upscaling (2x)
- Output correctness
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from PIL import Image

from kp3d.modules.superres.base import SuperResConfig, ScaleFactor


def create_test_image(size: int = 256) -> torch.Tensor:
    """Create a test image with a simple pattern.

    Args:
        size: Size of the square test image.

    Returns:
        torch.Tensor of shape (1, 3, size, size) with values in [0, 1].
    """
    # Create a gradient pattern for better visual verification
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Horizontal gradient in R channel
    img[:, :, 0] = np.linspace(0, 1, size)[None, :]

    # Vertical gradient in G channel
    img[:, :, 1] = np.linspace(0, 1, size)[:, None]

    # Checkerboard pattern in B channel
    checker_size = 32
    for i in range(0, size, checker_size):
        for j in range(0, size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                img[i:i+checker_size, j:j+checker_size, 2] = 0.8

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def save_tensor_as_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor as an image file.

    Args:
        tensor: Image tensor of shape (1, 3, H, W) or (3, H, W).
        path: Output path for the image.
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Convert to numpy (H, W, C)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()

    # Clip and convert to uint8
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # Save with PIL
    Image.fromarray(img_np).save(path)
    print(f"Saved image to: {path}")


def test_realesrgan_x2():
    """Test RealESRGAN x2 model."""
    print("=" * 60)
    print("RealESRGAN Super-Resolution Diagnostic Test")
    print("=" * 60)

    python_version = sys.version_info
    print(f"\nPython version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Create output directory
    output_dir = project_root / "outputs" / "superres_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Check dependencies
    print("\n1. Checking dependencies...")
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        print("   [PASS] realesrgan and basicsr are installed")
    except ImportError as e:
        print(f"   [FAIL] Missing dependency: {e}")
        print("\n   INSTALLATION ISSUE:")
        print("   BasicSR 1.4.2 has compatibility issues with Python 3.14")
        print("   The package fails to build with KeyError: '__version__'")
        print()
        print("   RECOMMENDED SOLUTION:")
        print("   1. Use Python 3.11 or 3.12 environment")
        print("   2. Install dependencies:")
        print("      pip install realesrgan basicsr")
        print()
        print("   ALTERNATIVE (if stuck on Python 3.14):")
        print("   Wait for BasicSR package update with Python 3.14 support")
        print()
        return False

    # Import RealESRGAN module
    try:
        from kp3d.modules.superres.real_esrgan import RealESRGANModule
    except ImportError as e:
        print(f"   [FAIL] Failed to import RealESRGAN module: {e}")
        return False

    # Check CUDA availability
    print("\n2. Checking CUDA...")
    if torch.cuda.is_available():
        print(f"   [PASS] CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("   [WARN] CUDA not available, using CPU (slower)")
        device = torch.device("cpu")

    # Create test image
    print("\n3. Creating test image (256x256)...")
    test_image = create_test_image(256)
    print(f"   [PASS] Test image shape: {test_image.shape}")

    # Save input image
    input_path = output_dir / "input_256x256.png"
    save_tensor_as_image(test_image, input_path)

    # Initialize RealESRGAN
    print("\n4. Initializing RealESRGAN x2 model...")
    try:
        config = SuperResConfig(
            model_name="RealESRGAN_x2plus",
            scale=ScaleFactor.X2,
            tile_size=256,
            tile_overlap=10,
            denoise_strength=0.0,  # No denoising for diagnostic
        )

        model = RealESRGANModule(
            config=config,
            device=device,
            half_precision=False,  # Use FP32 for compatibility
        )
        print("   [PASS] Model initialized successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Scale factor: {config.scale.value}x")
        print(f"   Tile size: {config.tile_size}")

    except Exception as e:
        print(f"   [FAIL] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Process image
    print("\n5. Processing image with RealESRGAN...")
    try:
        # Move image to device
        test_image = test_image.to(device)

        # Run inference
        output = model.forward(
            test_image,
            scale=ScaleFactor.X2,
            denoise=False,
        )

        result = output.result
        metadata = output.metadata

        print("   [PASS] Processing completed")
        print(f"   Processing time: {metadata['processing_time']:.2f}s")
        print(f"   Input size: {metadata['original_size']}")
        print(f"   Output size: {metadata['output_size']}")

    except Exception as e:
        print(f"   [FAIL] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify output size
    print("\n6. Verifying output dimensions...")
    input_h, input_w = test_image.shape[2:]
    output_h, output_w = result.shape[2:]

    expected_h = input_h * 2
    expected_w = input_w * 2

    if output_h == expected_h and output_w == expected_w:
        print(f"   [PASS] Output size correct: {output_w}x{output_h} (2x scale)")
    else:
        print(f"   [FAIL] Output size incorrect:")
        print(f"     Expected: {expected_w}x{expected_h}")
        print(f"     Got: {output_w}x{output_h}")
        return False

    # Save output image
    print("\n7. Saving output image...")
    output_path = output_dir / "output_512x512.png"
    save_tensor_as_image(result, output_path)

    # Create comparison image
    print("\n8. Creating comparison image...")
    try:
        # Upscale input with nearest neighbor for comparison
        input_upscaled = torch.nn.functional.interpolate(
            test_image,
            scale_factor=2,
            mode="nearest",
        )

        # Create side-by-side comparison
        comparison = torch.cat([input_upscaled, result], dim=3)  # Concatenate along width
        comparison_path = output_dir / "comparison_nearest_vs_esrgan.png"
        save_tensor_as_image(comparison, comparison_path)

    except Exception as e:
        print(f"   [WARN] Comparison creation failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST PASSED [SUCCESS]")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles:")
    print(f"  - input_256x256.png (input image)")
    print(f"  - output_512x512.png (2x upscaled)")
    print(f"  - comparison_nearest_vs_esrgan.png (side-by-side)")
    print("\nModel details:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    return True


if __name__ == "__main__":
    try:
        success = test_realesrgan_x2()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
