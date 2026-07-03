"""Test suite for v7 restoration module improvements.

Validates:
- EnhancedFadingNoiseRestorer: adaptive windows, PatchMatch, noise classification
- EnhancedFrequencyAwareRestorer: continuous sigma
- EnhancedGridPatternRestorer: rotated FFT
- NoiseShapeClassifier: 4-type classification
- Backward compatibility with existing restorers
"""

import sys
import time

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, 'src')

from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer
from kp3d.modules.restoration.frequency_aware import FrequencyAwareRestorer
from kp3d.modules.restoration.grid_pattern import GridPatternRestorer
from kp3d.modules.restoration.enhanced_fading_noise import EnhancedFadingNoiseRestorer
from kp3d.modules.restoration.enhanced_frequency_aware import EnhancedFrequencyAwareRestorer
from kp3d.modules.restoration.enhanced_grid_pattern import EnhancedGridPatternRestorer
from kp3d.modules.restoration.noise_classifier import NoiseShapeClassifier
from kp3d.modules.restoration.adaptive_windows import (
    AdaptiveWindowDetector,
    ContinuousSigmaSelector,
    compute_adaptive_windows,
    compute_continuous_sigma_map,
    adaptive_gaussian_blur,
)
from kp3d.modules.restoration.patchmatch_inpaint import RestorationPatchMatch
from kp3d.modules.restoration import RestorationModule


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def test_image():
    """Create test image tensor (1, 3, 256, 256)."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 256, 256)


@pytest.fixture
def test_image_np():
    """Create test image numpy (256, 256, 3) uint8."""
    np.random.seed(42)
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def noise_mask():
    """Create a synthetic noise mask with various blob shapes."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Dust-like: small circles
    cv2.circle(mask, (50, 50), 3, 255, -1)
    cv2.circle(mask, (80, 80), 4, 255, -1)
    cv2.circle(mask, (110, 60), 2, 255, -1)
    # Crack-like: thin lines
    cv2.line(mask, (150, 30), (150, 80), 255, 1)
    cv2.line(mask, (180, 40), (200, 90), 255, 1)
    # Stain-like: large irregular blob
    cv2.ellipse(mask, (100, 180), (20, 15), 30, 0, 360, 255, -1)
    # Mold-like: fuzzy cluster
    for _ in range(10):
        cx = 200 + np.random.randint(-10, 10)
        cy = 200 + np.random.randint(-10, 10)
        r = np.random.randint(2, 5)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


# =========================================================================
# TestAdaptiveWindows
# =========================================================================

class TestAdaptiveWindows:
    """Test adaptive window detector."""

    def test_default_windows(self):
        """Default windows for standard image size."""
        windows = compute_adaptive_windows((1024, 1024))
        assert len(windows) >= 3
        assert all(w % 2 == 1 for w in windows), "All windows must be odd"

    def test_small_image_fewer_windows(self):
        """Smaller images should get fewer windows."""
        small = compute_adaptive_windows((256, 256))
        large = compute_adaptive_windows((2048, 2048))
        assert len(small) <= len(large)

    def test_high_noise_larger_windows(self):
        """High noise scale should produce larger windows."""
        low_noise = compute_adaptive_windows((512, 512), noise_scale=0.1)
        high_noise = compute_adaptive_windows((512, 512), noise_scale=0.8)
        assert max(high_noise) >= max(low_noise)

    def test_continuous_sigma_map(self):
        """Sigma map should have correct range."""
        edge = np.random.rand(100, 100).astype(np.float32)
        sigma_map = compute_continuous_sigma_map(edge, min_sigma=1.0, max_sigma=8.0)
        assert sigma_map.min() >= 1.0 - 0.01
        assert sigma_map.max() <= 8.0 + 0.01

    def test_continuous_sigma_edge_mapping(self):
        """Strong edges should get small sigma."""
        edge = np.zeros((100, 100), dtype=np.float32)
        edge[40:60, 40:60] = 1.0  # Strong edge region
        sigma_map = compute_continuous_sigma_map(edge, min_sigma=1.0, max_sigma=8.0)
        assert sigma_map[50, 50] < sigma_map[0, 0]

    def test_adaptive_gaussian_blur(self):
        """Adaptive blur should produce valid output."""
        image = np.random.rand(64, 64, 3).astype(np.float32) * 255
        sigma_map = np.random.rand(64, 64).astype(np.float32) * 5 + 1.5
        result = adaptive_gaussian_blur(image, sigma_map, num_levels=5)
        assert result.shape == image.shape
        assert np.isfinite(result).all()

    def test_sigma_selector_quantization(self):
        """Quantization should produce valid indices and weights."""
        selector = ContinuousSigmaSelector(1.0, 10.0, 10)
        sigma_map = np.random.rand(50, 50).astype(np.float32) * 9 + 1
        lower_idx, weights = selector.quantize_sigma_map(sigma_map)
        assert lower_idx.min() >= 0
        assert lower_idx.max() <= 8
        assert weights.min() >= 0
        assert weights.max() <= 1.0 + 1e-6


# =========================================================================
# TestNoiseClassifier
# =========================================================================

class TestNoiseClassifier:
    """Test noise shape classifier."""

    def test_classify_dust(self):
        """Dust blobs should be classified as dust."""
        classifier = NoiseShapeClassifier()
        features = {
            "circularity": 0.85,
            "aspect_ratio": 1.2,
            "solidity": 0.9,
            "fractal_dim": 1.05,
            "area": 30,
        }
        result = classifier.classify_single(features)
        assert result == "dust"

    def test_classify_crack(self):
        """Crack patterns should be classified as crack."""
        classifier = NoiseShapeClassifier()
        features = {
            "circularity": 0.1,
            "aspect_ratio": 5.0,
            "solidity": 0.4,
            "fractal_dim": 1.4,
            "area": 100,
        }
        result = classifier.classify_single(features)
        assert result == "crack"

    def test_classify_mask(self, noise_mask):
        """Should classify all blobs in a mask."""
        classifier = NoiseShapeClassifier()
        label_types, label_features = classifier.classify_mask(noise_mask)
        assert len(label_types) > 0
        assert all(t in ("dust", "crack", "stain", "mold") for t in label_types.values())

    def test_strategy_mapping(self):
        """Each type should map to a valid strategy."""
        classifier = NoiseShapeClassifier()
        for noise_type in ("dust", "crack", "stain", "mold"):
            strategy = classifier.get_strategy(noise_type)
            assert strategy in ("patchmatch", "ns", "color_aware", "telea")

    def test_custom_strategy_map(self):
        """Custom strategy map should override defaults."""
        custom = {"dust": "ns", "crack": "patchmatch", "stain": "ns", "mold": "ns"}
        classifier = NoiseShapeClassifier(strategy_map=custom)
        assert classifier.get_strategy("dust") == "ns"

    def test_classification_summary(self, noise_mask):
        """Summary should count all types."""
        classifier = NoiseShapeClassifier()
        label_types, _ = classifier.classify_mask(noise_mask)
        summary = classifier.get_classification_summary(label_types)
        assert sum(summary.values()) == len(label_types)

    def test_strategy_masks(self, noise_mask):
        """Strategy masks should cover all classified blobs."""
        classifier = NoiseShapeClassifier()
        label_types, _ = classifier.classify_mask(noise_mask)
        strategy_masks = classifier.create_strategy_masks(noise_mask, label_types)
        assert len(strategy_masks) > 0
        # All masks should be binary
        for smask in strategy_masks.values():
            assert set(np.unique(smask)).issubset({0, 255})


# =========================================================================
# TestPatchMatch
# =========================================================================

class TestPatchMatch:
    """Test restoration PatchMatch inpainter."""

    def test_basic_inpaint(self, test_image_np):
        """Basic inpainting should work."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 10, 255, -1)

        pm = RestorationPatchMatch(patch_size=7, iterations=3, search_samples=50)
        result = pm.inpaint(test_image_np, mask)
        assert result.shape == test_image_np.shape
        assert result.dtype == np.uint8

    def test_empty_mask(self, test_image_np):
        """Empty mask should return copy of original."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        pm = RestorationPatchMatch()
        result = pm.inpaint(test_image_np, mask)
        np.testing.assert_array_equal(result, test_image_np)

    def test_texture_preservation(self, test_image_np):
        """Texture preservation mode should work without errors."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 5, 255, -1)

        pm = RestorationPatchMatch(preserve_texture=True, texture_sigma=2.0)
        result = pm.inpaint(test_image_np, mask)
        assert result.shape == test_image_np.shape


# =========================================================================
# TestEnhancedFadingNoise
# =========================================================================

class TestEnhancedFadingNoise:
    """Test EnhancedFadingNoiseRestorer."""

    def test_default_backward_compat(self, test_image):
        """Default config should behave like parent."""
        config = RestorationConfig()  # All v7 features disabled
        restorer = EnhancedFadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert 'method' in output.metadata
        assert output.metadata['method'] == 'enhanced_fading_noise'

    def test_adaptive_windows(self, test_image):
        """Adaptive windows should produce valid output."""
        config = RestorationConfig(
            use_adaptive_windows=True,
            noise_scale_estimate=0.5,
            store_intermediates=False,
        )
        restorer = EnhancedFadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert output.metadata.get('v7_adaptive_windows') is True

    def test_patchmatch_inpaint_mode(self, test_image):
        """PatchMatch inpaint mode should work."""
        config = RestorationConfig(
            inpaint_mode="patchmatch",
            store_intermediates=False,
        )
        restorer = EnhancedFadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert torch.isfinite(output.result).all()

    def test_noise_classification(self, test_image):
        """Noise classification should produce classification metadata."""
        config = RestorationConfig(
            use_noise_classification=True,
            store_intermediates=False,
        )
        restorer = EnhancedFadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert output.metadata.get('v7_noise_classification') is True

    def test_all_v7_features(self, test_image):
        """All v7 features combined should work."""
        config = RestorationConfig(
            use_adaptive_windows=True,
            noise_scale_estimate=0.3,
            use_noise_classification=True,
            inpaint_mode="patchmatch",
            store_intermediates=True,
        )
        restorer = EnhancedFadingNoiseRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert len(output.intermediate) > 0


# =========================================================================
# TestEnhancedFrequencyAware
# =========================================================================

class TestEnhancedFrequencyAware:
    """Test EnhancedFrequencyAwareRestorer."""

    def test_default_backward_compat(self, test_image):
        """Default config should work like parent."""
        config = RestorationConfig(store_intermediates=False)
        restorer = EnhancedFrequencyAwareRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert 'method' in output.metadata

    def test_continuous_sigma(self, test_image):
        """Continuous sigma should produce valid output."""
        config = RestorationConfig(
            continuous_sigma=True,
            sigma_levels=10,
            sigma_min=1.5,
            sigma_max=10.0,
            store_intermediates=False,
        )
        restorer = EnhancedFrequencyAwareRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert output.metadata.get('v7_continuous_sigma') is True
        assert output.metadata.get('v7_sigma_levels') == 10

    def test_uses_enhanced_fading(self, test_image):
        """Should use EnhancedFadingNoiseRestorer internally."""
        config = RestorationConfig(store_intermediates=False)
        restorer = EnhancedFrequencyAwareRestorer(config=config)
        assert isinstance(restorer.fading_restorer, EnhancedFadingNoiseRestorer)


# =========================================================================
# TestEnhancedGridPattern
# =========================================================================

class TestEnhancedGridPattern:
    """Test EnhancedGridPatternRestorer."""

    def test_default_backward_compat(self, test_image):
        """Default config should work like parent."""
        config = RestorationConfig(store_intermediates=False)
        restorer = EnhancedGridPatternRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert 'method' in output.metadata

    def test_auto_angle_detection(self, test_image):
        """Auto angle detection should produce valid output."""
        config = RestorationConfig(
            fft_auto_angle=True,
            fft_angle_tolerance=5.0,
            store_intermediates=False,
        )
        restorer = EnhancedGridPatternRestorer(config=config)
        output = restorer(test_image)

        assert output.result.shape == test_image.shape
        assert output.metadata.get('v7_auto_angle') is True

    def test_angle_detection_method(self):
        """Angle detection should return valid angles."""
        config = RestorationConfig(fft_auto_angle=True)
        restorer = EnhancedGridPatternRestorer(config=config)

        # Create synthetic grid image
        img = np.zeros((128, 128), dtype=np.uint8)
        for i in range(0, 128, 8):
            img[i, :] = 200  # Horizontal lines
            img[:, i] = 200  # Vertical lines

        angles = restorer.detect_grid_angles(img)
        assert len(angles) >= 1
        assert all(0 <= a <= 180 for a in angles)

    def test_radial_frequency_detection(self):
        """Radial frequency detection should find peaks."""
        config = RestorationConfig(fft_auto_angle=True)
        restorer = EnhancedGridPatternRestorer(config=config)

        # Create synthetic grid
        img = np.zeros((128, 128), dtype=np.float32)
        for i in range(0, 128, 8):
            img[i, :] = 200
            img[:, i] = 200

        freqs, pairs = restorer.detect_grid_frequencies_radial(img, angles=[0, 90])
        assert isinstance(freqs, list)
        assert isinstance(pairs, list)


# =========================================================================
# TestBackwardCompatibility
# =========================================================================

class TestBackwardCompatibility:
    """Ensure v7 classes maintain backward compatibility."""

    def test_same_interface_fading(self, test_image):
        """EnhancedFadingNoise should have same interface as FadingNoise."""
        config = RestorationConfig(store_intermediates=False)
        original = FadingNoiseRestorer(config=config)
        enhanced = EnhancedFadingNoiseRestorer(config=config)

        out_orig = original(test_image)
        out_enh = enhanced(test_image)

        assert out_orig.result.shape == out_enh.result.shape
        # Results should be very similar when v7 is disabled
        diff = torch.abs(out_orig.result - out_enh.result).mean().item()
        assert diff < 0.01, f"Results differ too much: {diff}"

    def test_same_interface_grid(self, test_image):
        """EnhancedGridPattern should have same interface as GridPattern."""
        config = RestorationConfig(store_intermediates=False)
        original = GridPatternRestorer(config=config)
        enhanced = EnhancedGridPatternRestorer(config=config)

        out_orig = original(test_image)
        out_enh = enhanced(test_image)

        assert out_orig.result.shape == out_enh.result.shape

    def test_restoration_module_dispatch(self, test_image):
        """RestorationModule should dispatch to enhanced restorers."""
        for method in ("enhanced_fading_noise", "enhanced_frequency_aware", "enhanced_grid_pattern"):
            config = RestorationConfig(store_intermediates=False)
            module = RestorationModule(method=method, config=config)
            output = module(test_image)
            assert output.result.shape == test_image.shape

    def test_config_defaults_unchanged(self):
        """v7 config defaults should not affect existing behavior."""
        config = RestorationConfig()
        assert config.use_adaptive_windows is False
        assert config.continuous_sigma is False
        assert config.use_noise_classification is False
        assert config.fft_auto_angle is False

    def test_original_restorers_unchanged(self, test_image):
        """Original restorers should still work normally."""
        for method in ("fading_noise", "frequency_aware", "grid_pattern"):
            config = RestorationConfig(store_intermediates=False)
            module = RestorationModule(method=method, config=config)
            output = module(test_image)
            assert output.result.shape == test_image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
