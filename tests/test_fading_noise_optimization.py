"""Test suite for FadingNoise optimization features.

Validates:
- Performance improvements
- Memory optimization
- Result consistency
- Configuration options
"""

import sys
import time
import torch
import numpy as np
import pytest

sys.path.insert(0, 'src')

from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer
from kp3d.modules.restoration.base import RestorationConfig


class TestFadingNoiseOptimization:
    """Test FadingNoise performance optimizations."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        torch.manual_seed(42)
        return torch.rand(1, 3, 256, 256)

    def test_store_intermediates_flag(self, test_image):
        """Test that store_intermediates flag controls output."""
        # With intermediates
        config1 = RestorationConfig(store_intermediates=True)
        restorer1 = FadingNoiseRestorer(config=config1)
        output1 = restorer1(test_image)

        assert len(output1.intermediate) > 0, "Should store intermediates"
        assert 'original' in output1.intermediate
        assert 'edge_mask' in output1.intermediate
        assert 'noise_mask' in output1.intermediate

        # Without intermediates
        config2 = RestorationConfig(store_intermediates=False)
        restorer2 = FadingNoiseRestorer(config=config2)
        output2 = restorer2(test_image)

        assert len(output2.intermediate) == 0, "Should not store intermediates"

    def test_fast_mode(self, test_image):
        """Test that fast_mode reduces processing time."""
        # Normal mode
        config1 = RestorationConfig(
            fast_mode=False,
            store_intermediates=False,
            multi_scale=True
        )
        restorer1 = FadingNoiseRestorer(config=config1)

        start = time.time()
        output1 = restorer1(test_image)
        time1 = time.time() - start

        # Fast mode
        config2 = RestorationConfig(
            fast_mode=True,
            store_intermediates=False,
            multi_scale=True
        )
        restorer2 = FadingNoiseRestorer(config=config2)

        start = time.time()
        output2 = restorer2(test_image)
        time2 = time.time() - start

        # Fast mode should be faster or comparable
        assert time2 <= time1 * 1.1, f"Fast mode should be faster: {time2:.3f}s vs {time1:.3f}s"

    def test_result_consistency(self, test_image):
        """Test that optimizations don't change results significantly."""
        configs = [
            RestorationConfig(fast_mode=False, store_intermediates=True),
            RestorationConfig(fast_mode=False, store_intermediates=False),
            RestorationConfig(fast_mode=True, store_intermediates=False),
        ]

        results = []
        for config in configs:
            restorer = FadingNoiseRestorer(config=config)
            output = restorer(test_image)
            results.append(output.result)

        # Check shape consistency
        for result in results:
            assert result.shape == results[0].shape

        # Check numerical consistency (allow small differences in fast mode)
        # Normal modes should be identical
        diff_01 = torch.abs(results[0] - results[1]).max().item()
        assert diff_01 < 1e-6, f"Optimized result differs: {diff_01}"

        # Fast mode allows slightly different results
        diff_02 = torch.abs(results[0] - results[2]).mean().item()
        assert diff_02 < 0.01, f"Fast mode result too different: {diff_02}"

    def test_memory_optimization(self, test_image):
        """Test that memory optimization reduces allocations."""
        import gc

        # Force garbage collection
        gc.collect()

        # With intermediates
        config1 = RestorationConfig(store_intermediates=True)
        restorer1 = FadingNoiseRestorer(config=config1)
        output1 = restorer1(test_image)

        # Estimate memory usage from intermediate count
        memory1 = len(output1.intermediate)

        # Without intermediates
        config2 = RestorationConfig(store_intermediates=False)
        restorer2 = FadingNoiseRestorer(config=config2)
        output2 = restorer2(test_image)

        memory2 = len(output2.intermediate)

        assert memory2 < memory1, "Should use less memory without intermediates"
        assert memory2 == 0, "Should have zero intermediates"

    def test_config_options(self, test_image):
        """Test that all config options work."""
        config = RestorationConfig(
            fast_mode=True,
            store_intermediates=False,
            multi_scale=True,
            adaptive_threshold=True,
            window_sizes=(7, 15),
            inpaint_mode="opencv",
            preserve_texture=False
        )

        restorer = FadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert 'noise_spots_detected' in output.metadata
        assert 'processing_time' in output.metadata

    def test_backward_compatibility(self, test_image):
        """Test that default behavior is unchanged."""
        # Default config
        config = RestorationConfig()

        # Should have these defaults
        assert config.store_intermediates == True, "Default should store intermediates"
        assert config.fast_mode == False, "Default should not use fast mode"
        assert config.multi_scale == True, "Default should use multi-scale"

        restorer = FadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert len(output.intermediate) > 0, "Default should store intermediates"

    def test_performance_regression(self, test_image):
        """Test that optimizations don't cause regression."""
        config = RestorationConfig(
            fast_mode=False,
            store_intermediates=False,
            multi_scale=True
        )
        restorer = FadingNoiseRestorer(config=config)

        # Multiple runs to check stability
        times = []
        for _ in range(3):
            start = time.time()
            output = restorer(test_image)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Processing time should be stable (low variance)
        assert std_time / avg_time < 0.2, f"Unstable performance: {std_time/avg_time:.2%}"

        # Should complete in reasonable time
        assert avg_time < 2.0, f"Too slow: {avg_time:.3f}s"

    def test_vectorization(self, test_image):
        """Test that vectorized operations work correctly."""
        config = RestorationConfig(
            inpaint_mode="color_aware",  # Uses vectorized inpainting
            store_intermediates=False
        )
        restorer = FadingNoiseRestorer(config=config)

        output = restorer(test_image)

        # Should complete without errors
        assert output.result.shape == test_image.shape
        assert torch.isfinite(output.result).all(), "Result should be finite"

    def test_metadata_consistency(self, test_image):
        """Test that metadata is consistent across modes."""
        configs = [
            RestorationConfig(fast_mode=False, store_intermediates=True),
            RestorationConfig(fast_mode=False, store_intermediates=False),
        ]

        metadatas = []
        for config in configs:
            restorer = FadingNoiseRestorer(config=config)
            output = restorer(test_image)
            metadatas.append(output.metadata)

        # Key metadata should be present
        for metadata in metadatas:
            assert 'method' in metadata
            assert 'processing_time' in metadata
            assert 'noise_spots_detected' in metadata
            assert 'multi_scale' in metadata

        # Noise detection should be consistent
        assert metadatas[0]['noise_spots_detected'] == metadatas[1]['noise_spots_detected']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
