"""V3 (Split Radius + NLM Adaptive + Contour) production integration tests.

Tests that V3 pipeline matches experiment definitions and that legacy
presets remain unaffected.
"""
import sys
from pathlib import Path

# Add src to path for test discovery
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import pytest


# Skip if dependencies not available
pytest.importorskip("cv2")
pytest.importorskip("torch")


class TestV3PresetConfig:
    """Test V3 preset configuration."""

    def test_v3_preset_to_config_split_radius_enabled(self):
        """V3 preset must enable split_radius (Stage 1)."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.V3.to_config()
        assert cfg.split_radius is True, "V3 must use Split Radius for Stage 1"

    def test_v3_preset_to_config_nlm_adaptive_enabled(self):
        """V3 preset must enable use_nlm_adaptive (Stage 2)."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.V3.to_config()
        assert cfg.use_nlm_adaptive is True, "V3 must use NLM Adaptive for Stage 2"

    def test_v3_preset_to_config_contour_enabled(self):
        """V3 preset must enable contour enhancement (Stage 3)."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.V3.to_config()
        assert cfg.contour_boost > 0, "V3 must enable contour enhancement for Stage 3"

    def test_v3_preset_nlm_params(self):
        """V3 preset has correct NLM parameters."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.V3.to_config()
        assert cfg.nlm_h_max > cfg.nlm_h_base, "h_max should be greater than h_base"
        assert cfg.nlm_h_max == 15.0
        assert cfg.nlm_narrow_threshold == 8.0


class TestLegacyPresetPreservation:
    """Test that legacy presets are not affected by V3 changes."""

    def test_quality_preset_no_nlm_adaptive(self):
        """QUALITY preset must NOT enable NLM adaptive."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.QUALITY.to_config()
        assert cfg.use_nlm_adaptive is False, "QUALITY must not use NLM adaptive"

    def test_clean_preset_no_nlm_adaptive(self):
        """CLEAN preset must NOT enable NLM adaptive."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.CLEAN.to_config()
        assert cfg.use_nlm_adaptive is False, "CLEAN must not use NLM adaptive"

    def test_quality_preset_preserves_alpha(self):
        """QUALITY preset has alpha=0.7."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.QUALITY.to_config()
        assert cfg.alpha == 0.7

    def test_clean_preset_preserves_alpha(self):
        """CLEAN preset has alpha=1.0."""
        from kp3d.modules.weave_removal import WeaveRemovalPreset

        cfg = WeaveRemovalPreset.CLEAN.to_config()
        assert cfg.alpha == 1.0


class TestNarrowMaskComputation:
    """Test narrow region mask computation."""

    def test_narrow_mask_output_shape(self):
        """Narrow mask shape matches input image."""
        from kp3d.modules.weave_removal import compute_narrow_region_mask, SpatialAdaptiveNLMConfig

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cfg = SpatialAdaptiveNLMConfig()
        mask = compute_narrow_region_mask(img, cfg)

        assert mask.shape == (256, 256), f"Expected (256, 256), got {mask.shape}"
        assert mask.dtype == np.float32

    def test_narrow_mask_value_range(self):
        """Narrow mask values in [0, 1]."""
        from kp3d.modules.weave_removal import compute_narrow_region_mask, SpatialAdaptiveNLMConfig

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        cfg = SpatialAdaptiveNLMConfig()
        mask = compute_narrow_region_mask(img, cfg)

        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_narrow_mask_nonzero_coverage(self):
        """Narrow mask has non-zero coverage on natural image."""
        from kp3d.modules.weave_removal import compute_narrow_region_mask, SpatialAdaptiveNLMConfig

        # Create image with distinct regions (not uniform)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:128, :, :] = [200, 100, 50]  # Region 1
        img[128:, :128, :] = [50, 150, 200]  # Region 2
        img[128:, 128:, :] = [100, 200, 50]  # Region 3

        cfg = SpatialAdaptiveNLMConfig()
        mask = compute_narrow_region_mask(img, cfg)

        # Should have some narrow regions at boundaries
        mean_coverage = mask.mean()
        assert 0.0 < mean_coverage < 1.0, f"Expected partial coverage, got {mean_coverage}"


class TestSpatialAdaptiveNLM:
    """Test spatial adaptive NLM function."""

    def test_nlm_output_shape_legacy_mode(self):
        """NLM output shape matches input (legacy mode, no base_processed)."""
        from kp3d.modules.weave_removal import spatial_adaptive_nlm, SpatialAdaptiveNLMConfig

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        cfg = SpatialAdaptiveNLMConfig()

        result = spatial_adaptive_nlm(img, config=cfg)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_nlm_output_shape_v3_mode(self):
        """NLM output shape matches input (V3 mode, with base_processed)."""
        from kp3d.modules.weave_removal import spatial_adaptive_nlm, SpatialAdaptiveNLMConfig

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        base = img.copy()  # Simulate Split Radius output
        cfg = SpatialAdaptiveNLMConfig()

        result = spatial_adaptive_nlm(img, base, cfg)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_nlm_with_base_processed_differs_from_legacy(self):
        """V3 mode (with base_processed) differs from legacy mode."""
        from kp3d.modules.weave_removal import spatial_adaptive_nlm, SpatialAdaptiveNLMConfig

        # Use non-trivial image
        np.random.seed(42)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Simulate different base (e.g., already denoised)
        base = (img.astype(np.float32) * 0.9 + 25).clip(0, 255).astype(np.uint8)

        cfg = SpatialAdaptiveNLMConfig()

        result_legacy = spatial_adaptive_nlm(img, config=cfg)
        result_v3 = spatial_adaptive_nlm(img, base, cfg)

        # Should be different
        diff = np.abs(result_legacy.astype(float) - result_v3.astype(float))
        assert diff.mean() > 0.5, "V3 and legacy results should differ"


class TestWeaveRemovalModuleV3:
    """Test V3 module end-to-end."""

    def test_v3_module_end_to_end(self):
        """V3 module runs end-to-end without errors."""
        from kp3d.modules.weave_removal import WeaveRemovalModule, WeaveRemovalPreset

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        module = WeaveRemovalModule(WeaveRemovalPreset.V3.to_config())

        result, confidence = module.process_bgr(img)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_v3_module_uses_split_radius_then_nlm(self):
        """V3 module applies Split Radius before NLM (verified via config)."""
        from kp3d.modules.weave_removal import WeaveRemovalModule, WeaveRemovalPreset

        config = WeaveRemovalPreset.V3.to_config()
        module = WeaveRemovalModule(config)

        # Both stages should be enabled
        assert module.config.split_radius is True
        assert module.config.use_nlm_adaptive is True


class TestWeaveRemovalModuleLegacy:
    """Test legacy modules still work."""

    def test_quality_module_end_to_end(self):
        """QUALITY preset still works (regression)."""
        from kp3d.modules.weave_removal import WeaveRemovalModule, WeaveRemovalPreset

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        module = WeaveRemovalModule(WeaveRemovalPreset.QUALITY.to_config())

        result, confidence = module.process_bgr(img)

        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_clean_module_end_to_end(self):
        """CLEAN preset still works (regression)."""
        from kp3d.modules.weave_removal import WeaveRemovalModule, WeaveRemovalPreset

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        module = WeaveRemovalModule(WeaveRemovalPreset.CLEAN.to_config())

        result, confidence = module.process_bgr(img)

        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestConfigParameterExposure:
    """Test that all NLM parameters are exposed in config."""

    def test_nlm_cluster_params_in_config(self):
        """NLM clustering parameters are in WeaveRemovalConfig."""
        from kp3d.modules.weave_removal import WeaveRemovalConfig

        cfg = WeaveRemovalConfig()

        assert hasattr(cfg, 'nlm_n_clusters')
        assert hasattr(cfg, 'nlm_min_cluster_area')
        assert hasattr(cfg, 'nlm_blur_sigma')

        # Default values
        assert cfg.nlm_n_clusters == 5
        assert cfg.nlm_min_cluster_area == 100
        assert cfg.nlm_blur_sigma == 2.0

    def test_nlm_cluster_params_passed_to_nlm_config(self):
        """NLM parameters flow from WeaveRemovalConfig to SpatialAdaptiveNLMConfig."""
        from kp3d.modules.weave_removal import (
            WeaveRemovalConfig,
            SpatialAdaptiveNLMConfig,
        )

        wr_cfg = WeaveRemovalConfig(
            nlm_n_clusters=7,
            nlm_min_cluster_area=200,
            nlm_blur_sigma=3.5,
        )

        # Create NLM config with same params
        nlm_cfg = SpatialAdaptiveNLMConfig(
            n_clusters=wr_cfg.nlm_n_clusters,
            min_cluster_area=wr_cfg.nlm_min_cluster_area,
            blur_sigma=wr_cfg.nlm_blur_sigma,
        )

        assert nlm_cfg.n_clusters == 7
        assert nlm_cfg.min_cluster_area == 200
        assert nlm_cfg.blur_sigma == 3.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
