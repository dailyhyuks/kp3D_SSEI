"""Integration tests for HED edge detection module."""

import pytest
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kp3d.modules.edge.hed import HEDEdgeDetector, HEDNetwork
from kp3d.modules.edge.base import EdgeConfig


@pytest.fixture(scope="module")
def hed_detector():
    """Create HED detector instance (shared across tests)."""
    detector = HEDEdgeDetector()
    return detector


@pytest.fixture(scope="module")
def sample_image_path():
    """Get path to sample image."""
    project_root = Path(__file__).parent.parent
    sample_path = project_root / "samples" / "boat_painting_2.png"
    if not sample_path.exists():
        pytest.skip(f"Sample image not found: {sample_path}")
    return sample_path


@pytest.fixture(scope="module")
def sample_image_tensor(sample_image_path):
    """Load sample image as tensor."""
    img_pil = Image.open(sample_image_path).convert("RGB")
    img_np = np.array(img_pil)
    # Convert to tensor (B, C, H, W) in range [0, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor


def test_hed_loads_weights(hed_detector):
    """Test that HED detector successfully loads weights without fallback."""
    assert hed_detector is not None, "HED detector failed to initialize"
    assert not hed_detector._fallback_to_canny, "HED detector fell back to Canny (weights not loaded)"
    assert hed_detector.network is not None, "HED network is None"
    assert hed_detector._initialized, "HED detector not marked as initialized"

    # Verify network has expected structure
    param_count = sum(p.numel() for p in hed_detector.network.parameters())
    assert param_count > 0, "Network has no parameters"

    # Expected parameter count for HED network (~14.7M parameters)
    expected_params = 14_000_000  # Approximate
    assert param_count > expected_params, f"Parameter count too low: {param_count}"


def test_hed_output_shape(hed_detector, sample_image_tensor):
    """Test that HED output has correct shape matching input."""
    batch_size, channels, height, width = sample_image_tensor.shape

    output = hed_detector.forward(sample_image_tensor)

    # Check result shape matches input spatial dimensions
    assert output.result.shape == (batch_size, 1, height, width), \
        f"Expected shape ({batch_size}, 1, {height}, {width}), got {output.result.shape}"

    # Check binary edges shape
    assert "edges_binary" in output.intermediate, "Missing edges_binary in intermediate results"
    assert output.intermediate["edges_binary"].shape == (batch_size, 1, height, width), \
        "Binary edges shape mismatch"


def test_hed_not_fallback_to_canny(hed_detector, sample_image_tensor):
    """Test that HED uses actual HED network, not Canny fallback."""
    output = hed_detector.forward(sample_image_tensor)

    # Check metadata confirms HED method
    assert "method" in output.metadata, "Missing method in metadata"
    assert output.metadata["method"] == "hed", \
        f"Expected method 'hed', got '{output.metadata['method']}'"

    # Verify network parameters are reported
    assert "network_params" in output.metadata, "Missing network_params in metadata"
    assert output.metadata["network_params"] > 0, "Network parameter count is zero"


def test_hed_on_sample_image(hed_detector, sample_image_tensor, sample_image_path):
    """Test HED edge detection on sample image with comprehensive checks."""
    print(f"\nTesting HED on: {sample_image_path.name}")
    print(f"Image shape: {sample_image_tensor.shape}")

    # Run edge detection
    output = hed_detector.forward(sample_image_tensor, threshold=0.5)

    # Check output value range [0, 1] for probability map
    assert output.result.min() >= 0.0, f"Min value {output.result.min():.3f} < 0"
    assert output.result.max() <= 1.0, f"Max value {output.result.max():.3f} > 1"

    # Check that edges were actually detected (not all zeros)
    edge_density = output.result.mean().item()
    assert edge_density > 0.0, "Edge map is all zeros"
    assert edge_density < 1.0, "Edge map is all ones"

    # Check binary edges are actually binary
    binary_edges = output.intermediate["edges_binary"]
    unique_values = torch.unique(binary_edges)
    assert len(unique_values) <= 2, f"Binary edges have {len(unique_values)} unique values (expected 2)"
    assert all(v in [0.0, 1.0] for v in unique_values), "Binary edges contain non-binary values"

    # Check metadata
    assert output.metadata["threshold"] == 0.5, "Threshold metadata mismatch"
    assert "processing_time" in output.metadata, "Missing processing_time"
    assert output.metadata["processing_time"] > 0, "Processing time is zero"

    # Performance check: should complete reasonably fast (< 5 seconds for typical image)
    assert output.metadata["processing_time"] < 5.0, \
        f"Processing took {output.metadata['processing_time']:.2f}s (too slow)"

    print(f"Edge density: {edge_density:.3f}")
    print(f"Processing time: {output.metadata['processing_time']:.3f}s")
    print(f"Value range: [{output.result.min():.3f}, {output.result.max():.3f}]")


def test_hed_batch_processing(hed_detector, sample_image_tensor):
    """Test HED can process batches correctly."""
    # Create a batch of 2 identical images
    batch = torch.cat([sample_image_tensor, sample_image_tensor], dim=0)
    assert batch.shape[0] == 2, "Failed to create batch"

    output = hed_detector.forward(batch)

    # Check output has correct batch size
    assert output.result.shape[0] == 2, f"Expected batch size 2, got {output.result.shape[0]}"

    # Results should be identical for identical inputs
    diff = torch.abs(output.result[0] - output.result[1]).max()
    assert diff < 1e-5, f"Identical inputs produced different outputs (max diff: {diff})"


def test_hed_different_thresholds(hed_detector, sample_image_tensor):
    """Test HED with different threshold values."""
    thresholds = [0.3, 0.5, 0.7]
    binary_densities = []

    for threshold in thresholds:
        output = hed_detector.forward(sample_image_tensor, threshold=threshold)
        binary_density = output.intermediate["edges_binary"].mean().item()
        binary_densities.append(binary_density)

    # Higher threshold should result in fewer edges (lower density)
    assert binary_densities[0] > binary_densities[1] > binary_densities[2], \
        "Binary edge density should decrease with higher threshold"


def test_hed_grayscale_input(hed_detector, sample_image_tensor):
    """Test HED with grayscale input (should convert to RGB internally)."""
    # Convert to grayscale
    grayscale = sample_image_tensor.mean(dim=1, keepdim=True)
    assert grayscale.shape[1] == 1, "Failed to create grayscale image"

    # Should handle grayscale input
    output = hed_detector.forward(grayscale)

    # Check output shape is valid
    batch_size, _, height, width = grayscale.shape
    assert output.result.shape == (batch_size, 1, height, width), "Shape mismatch for grayscale input"

    # Check method is still HED
    assert output.metadata["method"] == "hed", "Method changed for grayscale input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
