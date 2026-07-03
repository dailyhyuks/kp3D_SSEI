"""Diagnostic test for HED edge detector weights loading."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from kp3d.modules.edge.hed import HEDEdgeDetector, HEDNetwork


def test_hed_weights_cache_location():
    """Test that the cache location is accessible."""
    cache_dir = Path.home() / ".cache" / "kp3d" / "hed"
    weights_file = cache_dir / "network-bsds500.pytorch"

    print("\n=== HED Weights Cache Diagnostic ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Cache dir exists: {cache_dir.exists()}")
    print(f"Weights file path: {weights_file}")
    print(f"Weights file exists: {weights_file.exists()}")

    if weights_file.exists():
        file_size = weights_file.stat().st_size
        print(f"Weights file size: {file_size:,} bytes ({file_size / (1024**2):.2f} MB)")

    return weights_file


def test_hed_network_creation():
    """Test that HED network can be created."""
    print("\n=== HED Network Creation ===")
    try:
        network = HEDNetwork()
        param_count = sum(p.numel() for p in network.parameters())
        print(f"Network created successfully")
        print(f"Total parameters: {param_count:,}")
        return True
    except Exception as e:
        print(f"Failed to create network: {e}")
        return False


def test_hed_weights_loading():
    """Test loading weights into HED network."""
    print("\n=== HED Weights Loading ===")
    weights_file = Path.home() / ".cache" / "kp3d" / "hed" / "network-bsds500.pytorch"

    if not weights_file.exists():
        print("Weights file does not exist, cannot test loading")
        return False

    try:
        state_dict = torch.load(weights_file, map_location='cpu', weights_only=False)

        print(f"State dict loaded, keys: {len(state_dict)}")
        print(f"First few keys: {list(state_dict.keys())[:5]}")

        # Test key mapping by creating a detector and using its load_weights method
        detector = HEDEdgeDetector()

        if detector._fallback_to_canny:
            print("Failed: Detector fell back to Canny")
            return False

        print("Weights loaded successfully into network via detector")
        return True

    except Exception as e:
        print(f"Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hed_detector_initialization():
    """Test HED detector initialization."""
    print("\n=== HED Detector Initialization ===")
    try:
        detector = HEDEdgeDetector()

        print(f"Detector name: {detector.name}")
        print(f"Fallback to Canny: {detector._fallback_to_canny}")
        print(f"Network initialized: {detector.network is not None}")
        print(f"Device: {detector.device}")

        if detector._fallback_to_canny:
            print("WARNING: Detector is using Canny fallback!")
            return False
        else:
            print("SUCCESS: HED network properly initialized")
            return True

    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hed_inference():
    """Test HED inference on a sample image."""
    print("\n=== HED Inference Test ===")
    try:
        detector = HEDEdgeDetector()

        if detector._fallback_to_canny:
            print("Skipping inference test - detector is in fallback mode")
            return False

        # Create a simple test image (gradient)
        test_image = torch.zeros(1, 3, 256, 256)
        for i in range(256):
            test_image[0, :, i, :] = i / 255.0

        print(f"Test image shape: {test_image.shape}")
        print(f"Test image range: [{test_image.min():.3f}, {test_image.max():.3f}]")

        # Run inference
        output = detector.forward(test_image)

        print(f"Output shape: {output.result.shape}")
        print(f"Output range: [{output.result.min():.3f}, {output.result.max():.3f}]")
        print(f"Method used: {output.metadata.get('method', 'unknown')}")
        print(f"Processing time: {output.metadata.get('processing_time', 0):.3f}s")

        if output.metadata.get('method') == 'hed':
            print("SUCCESS: HED inference working correctly")
            return True
        else:
            print("WARNING: Not using HED method")
            return False

    except Exception as e:
        print(f"Failed inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hed_weights_download():
    """Test downloading HED weights if they don't exist."""
    print("\n=== HED Weights Download Test ===")

    weights_file = Path.home() / ".cache" / "kp3d" / "hed" / "network-bsds500.pytorch"

    if weights_file.exists():
        print("Weights already exist, skipping download test")
        return True

    try:
        detector = HEDEdgeDetector()

        # Check if download was attempted
        if weights_file.exists():
            print("Weights downloaded successfully")
            return True
        else:
            print("Weights download failed or not attempted")
            return False

    except Exception as e:
        print(f"Download test failed: {e}")
        return False


def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("HED Edge Detector Diagnostic Suite")
    print("=" * 60)

    results = {}

    # Test 1: Cache location
    test_hed_weights_cache_location()

    # Test 2: Network creation
    results['network_creation'] = test_hed_network_creation()

    # Test 3: Weights loading
    results['weights_loading'] = test_hed_weights_loading()

    # Test 4: Detector initialization
    results['detector_init'] = test_hed_detector_initialization()

    # Test 5: Inference
    results['inference'] = test_hed_inference()

    # Test 6: Download (if needed)
    results['download'] = test_hed_weights_download()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED - HED is working correctly")
    else:
        print("OVERALL: SOME TESTS FAILED - HED may not be working properly")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
