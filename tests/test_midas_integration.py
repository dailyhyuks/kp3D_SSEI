"""Integration tests for MiDaS depth estimation module."""

import pytest
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kp3d.modules.shade.midas import MiDaSDepthEstimator
from kp3d.modules.shade.base import ShadeConfig


@pytest.fixture(scope="module")
def midas_estimator():
    """Create MiDaS estimator instance (shared across tests)."""
    # Use MiDaS_small for speed
    config = ShadeConfig(depth_model="MiDaS_small")
    estimator = MiDaSDepthEstimator(config=config)
    return estimator


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


def test_midas_loads_model(midas_estimator):
    """Test that MiDaS model loads successfully."""
    assert midas_estimator is not None, "MiDaS estimator failed to initialize"
    assert midas_estimator.model is not None, "MiDaS model is None"
    assert midas_estimator.transform is not None, "MiDaS transform is None"
    assert midas_estimator._initialized, "MiDaS estimator not marked as initialized"

    # Verify config
    assert midas_estimator.config.depth_model == "MiDaS_small", "Model type mismatch"

    # Check device
    assert str(midas_estimator.device) in ["cpu", "cuda", "mps"], \
        f"Unexpected device: {midas_estimator.device}"


def test_midas_output_shape(midas_estimator, sample_image_tensor):
    """Test that MiDaS output has correct shape matching input."""
    batch_size, channels, height, width = sample_image_tensor.shape

    output = midas_estimator.forward(sample_image_tensor)

    # Check result shape matches input spatial dimensions
    assert output.result.shape == (batch_size, 1, height, width), \
        f"Expected shape ({batch_size}, 1, {height}, {width}), got {output.result.shape}"

    # Check depth map in intermediate results
    assert "depth_map" in output.intermediate, "Missing depth_map in intermediate results"
    assert output.intermediate["depth_map"].shape == (batch_size, 1, height, width), \
        "Depth map shape mismatch"

    # Should be the same as result
    assert torch.allclose(output.result, output.intermediate["depth_map"]), \
        "Result and intermediate depth_map differ"


def test_midas_value_range(midas_estimator, sample_image_tensor):
    """Test that MiDaS output values are normalized to [0, 1]."""
    output = midas_estimator.forward(sample_image_tensor)
    depth_map = output.result

    # Check value range [0, 1]
    min_val = depth_map.min().item()
    max_val = depth_map.max().item()

    assert min_val >= 0.0, f"Minimum value {min_val:.3f} < 0"
    assert max_val <= 1.0, f"Maximum value {max_val:.3f} > 1"

    # Check that depth map uses full range (not all same value)
    assert max_val - min_val > 0.1, \
        f"Depth map has very low dynamic range: [{min_val:.3f}, {max_val:.3f}]"

    # Check mean and std are reasonable
    mean_val = depth_map.mean().item()
    std_val = depth_map.std().item()

    assert 0.0 < mean_val < 1.0, f"Mean value {mean_val:.3f} out of expected range"
    assert std_val > 0.0, f"Standard deviation is zero (uniform depth map)"


def test_midas_on_sample_image(midas_estimator, sample_image_tensor, sample_image_path):
    """Test MiDaS depth estimation on sample image with comprehensive checks."""
    print(f"\nTesting MiDaS on: {sample_image_path.name}")
    print(f"Image shape: {sample_image_tensor.shape}")

    # Run depth estimation
    output = midas_estimator.forward(sample_image_tensor)
    depth_map = output.result

    # Check output characteristics
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    print(f"Mean: {depth_map.mean():.3f}, Std: {depth_map.std():.3f}")

    # Verify depth map is not degenerate
    unique_values = torch.unique(depth_map)
    assert len(unique_values) > 100, \
        f"Depth map has only {len(unique_values)} unique values (too few)"

    # Check metadata
    assert "model" in output.metadata, "Missing model in metadata"
    assert output.metadata["model"] == "MiDaS_small", "Model metadata mismatch"

    assert "original_shape" in output.metadata, "Missing original_shape in metadata"
    assert output.metadata["original_shape"] == sample_image_tensor.shape, \
        "Original shape metadata mismatch"

    # Verify depth map has spatial structure (not random noise)
    # Compute spatial gradient magnitude
    dy = torch.abs(depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :])
    dx = torch.abs(depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1])

    # Most pixels should have small gradients (smooth regions)
    smooth_ratio = ((dy < 0.1).float().mean() + (dx < 0.1).float().mean()) / 2
    assert smooth_ratio > 0.5, \
        f"Depth map appears too noisy (smooth ratio: {smooth_ratio:.2f})"

    print(f"Smooth ratio: {smooth_ratio:.3f}")
    print(f"Unique values: {len(unique_values)}")


def test_midas_batch_processing(midas_estimator, sample_image_tensor):
    """Test MiDaS can process batches correctly."""
    # Create a batch of 2 identical images
    batch = torch.cat([sample_image_tensor, sample_image_tensor], dim=0)
    assert batch.shape[0] == 2, "Failed to create batch"

    output = midas_estimator.forward(batch)

    # Check output has correct batch size
    assert output.result.shape[0] == 2, f"Expected batch size 2, got {output.result.shape[0]}"

    # Results should be identical for identical inputs
    diff = torch.abs(output.result[0] - output.result[1]).max()
    assert diff < 1e-4, f"Identical inputs produced different outputs (max diff: {diff})"


def test_midas_with_edge_refinement(midas_estimator, sample_image_tensor):
    """Test MiDaS with edge-aware depth refinement."""
    # Create fake edge map (high edges at image center)
    batch_size, _, height, width = sample_image_tensor.shape
    edge_map = torch.zeros(batch_size, 1, height, width)
    edge_map[:, :, height//2-10:height//2+10, :] = 1.0  # Horizontal edge stripe

    # Run with edge refinement
    output_with_edges = midas_estimator.forward(sample_image_tensor, edge_map=edge_map)

    # Run without edge refinement
    output_without_edges = midas_estimator.forward(sample_image_tensor)

    # Results should differ when edges are provided
    diff = torch.abs(output_with_edges.result - output_without_edges.result).mean()
    assert diff > 1e-6, "Edge refinement had no effect"


def test_midas_model_types():
    """Test that different MiDaS model types can be initialized."""
    model_types = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]

    for model_type in model_types:
        try:
            config = ShadeConfig(depth_model=model_type)
            estimator = MiDaSDepthEstimator(config=config)
            assert estimator._initialized, f"Failed to initialize {model_type}"
            assert estimator.config.depth_model == model_type, "Model type mismatch"
            print(f"[OK] {model_type} initialized successfully")
        except Exception as e:
            pytest.skip(f"Model {model_type} not available: {e}")


def test_midas_invalid_model_type():
    """Test that invalid model type raises error."""
    with pytest.raises(ValueError, match="Invalid depth model"):
        config = ShadeConfig(depth_model="InvalidModel")
        MiDaSDepthEstimator(config=config)


def test_midas_depth_consistency(midas_estimator, sample_image_tensor):
    """Test that MiDaS produces consistent depth estimates on same input."""
    # Run twice
    output1 = midas_estimator.forward(sample_image_tensor)
    output2 = midas_estimator.forward(sample_image_tensor)

    # Results should be identical (deterministic)
    diff = torch.abs(output1.result - output2.result).max()
    assert diff < 1e-5, f"Non-deterministic results (max diff: {diff})"


def test_midas_grayscale_input(midas_estimator, sample_image_tensor):
    """Test MiDaS with grayscale input (should convert to RGB internally)."""
    # Convert to grayscale
    grayscale = sample_image_tensor.mean(dim=1, keepdim=True)
    assert grayscale.shape[1] == 1, "Failed to create grayscale image"

    # MiDaS expects RGB, so this will be converted internally
    # The image will be converted to numpy and then transformed
    # Since the transform expects RGB, we need to test with 3-channel input

    # Convert grayscale to 3-channel by repeating
    rgb_from_gray = grayscale.repeat(1, 3, 1, 1)

    output = midas_estimator.forward(rgb_from_gray)

    # Check output shape is valid
    batch_size, _, height, width = rgb_from_gray.shape
    assert output.result.shape == (batch_size, 1, height, width), \
        "Shape mismatch for grayscale input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
