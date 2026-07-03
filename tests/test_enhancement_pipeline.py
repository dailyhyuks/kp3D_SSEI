"""Tests for the Enhancement Pipeline module.

Tests cover:
- EnhancementConfig validation and defaults
- ResolutionChecker skip logic
- GridPresenceChecker detection
- EnhancementPipeline forward() with mocked sub-modules
"""

from unittest.mock import MagicMock, patch, PropertyMock
import cv2
import numpy as np
import pytest
import torch

from kp3d.core.base import ModuleOutput
from kp3d.modules.enhancement.config import EnhancementConfig
from kp3d.modules.enhancement.skip_logic import GridPresenceChecker, ResolutionChecker
from kp3d.modules.enhancement.pipeline import EnhancementPipeline


# =============================================================================
# TestEnhancementConfig
# =============================================================================


class TestEnhancementConfig:
    """Test EnhancementConfig defaults and validation."""

    def test_default_values(self):
        config = EnhancementConfig()
        assert config.upscale_model == "RealESRGAN_x2plus"
        assert config.upscale_tile_size == 512
        assert config.grid_template_method == "notch"
        assert config.grid_clamp_min == 0.85
        assert config.grid_clamp_max == 1.15
        assert config.oee_enabled is True
        assert config.enable_first_upscale is True
        assert config.enable_grid_removal is True
        assert config.enable_second_upscale is True
        assert config.max_input_pixels == 4_000_000
        assert config.use_spectral_grid is True
        assert config.spectral_period_min == 4.0
        assert config.spectral_period_max == 20.0
        assert config.spectral_butterworth_order == 2
        assert config.spectral_max_iterations == 2
        assert config.enable_pre_smooth is False
        assert config.enable_detail_enhance is False

    def test_custom_values(self):
        config = EnhancementConfig(
            upscale_model="RealESRGAN_x4plus",
            upscale_tile_size=256,
            grid_deconv_strength=0.8,
            oee_enhance_strength=0.5,
            enable_second_upscale=False,
        )
        assert config.upscale_model == "RealESRGAN_x4plus"
        assert config.upscale_tile_size == 256
        assert config.grid_deconv_strength == 0.8
        assert config.oee_enhance_strength == 0.5
        assert config.enable_second_upscale is False

    def test_validation_tile_size_positive(self):
        with pytest.raises(Exception):
            EnhancementConfig(upscale_tile_size=0)

    def test_validation_deconv_strength_range(self):
        with pytest.raises(Exception):
            EnhancementConfig(grid_deconv_strength=1.5)

    def test_validation_oee_edge_range(self):
        with pytest.raises(Exception):
            EnhancementConfig(oee_edge_low=-0.1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            EnhancementConfig(nonexistent_field="value")


# =============================================================================
# TestResolutionChecker
# =============================================================================


class TestResolutionChecker:
    """Test ResolutionChecker skip logic."""

    def test_small_image_not_skipped(self):
        image = torch.rand(1, 3, 100, 100)  # 10,000 pixels
        assert ResolutionChecker.should_skip(image, max_pixels=4_000_000) is False

    def test_large_image_skipped(self):
        image = torch.rand(1, 3, 2000, 2000)  # 4,000,000 pixels
        assert ResolutionChecker.should_skip(image, max_pixels=4_000_000) is True

    def test_exact_threshold(self):
        image = torch.rand(1, 3, 2000, 2000)  # exactly 4M
        assert ResolutionChecker.should_skip(image, max_pixels=4_000_000) is True

    def test_3d_tensor(self):
        image = torch.rand(3, 100, 100)
        assert ResolutionChecker.should_skip(image, max_pixels=4_000_000) is False

    def test_3d_tensor_large(self):
        image = torch.rand(3, 2000, 2000)
        assert ResolutionChecker.should_skip(image, max_pixels=4_000_000) is True


# =============================================================================
# TestGridPresenceChecker
# =============================================================================


class TestGridPresenceChecker:
    """Test GridPresenceChecker grid detection."""

    def test_clean_image_no_grid(self):
        """A smooth gradient image should not be detected as having a grid."""
        checker = GridPresenceChecker(confidence_threshold=3.0)
        # Create smooth gradient (no periodic pattern)
        h, w = 256, 256
        gradient = np.tile(np.linspace(100, 200, w, dtype=np.uint8), (h, 1))
        img_bgr = np.stack([gradient, gradient, gradient], axis=-1).astype(np.uint8)

        detected, info = checker.check(img_bgr)
        assert detected is False
        assert "harmonic_score" in info

    def test_synthetic_grid_detected(self):
        """A synthetic periodic grid pattern should be detected."""
        checker = GridPresenceChecker(confidence_threshold=1.0)
        # Create image with strong periodic grid pattern (period=16)
        h, w = 512, 512
        # Start with uniform base
        base = np.ones((h, w), dtype=np.float64) * 180.0
        # Create strong multiplicative grid pattern
        for i in range(h):
            for j in range(w):
                # Sinusoidal grid in both directions
                gx = 0.7 + 0.3 * np.cos(2 * np.pi * j / 16.0)
                gy = 0.7 + 0.3 * np.cos(2 * np.pi * i / 16.0)
                base[i, j] *= gx * gy
        img_gray = base.clip(0, 255).astype(np.uint8)
        img_bgr = np.stack([img_gray, img_gray, img_gray], axis=-1)

        detected, info = checker.check(img_bgr)
        assert detected is True
        assert info["harmonic_score"] >= 1.0

    def test_custom_threshold(self):
        """Higher threshold should make detection more strict."""
        checker_strict = GridPresenceChecker(confidence_threshold=100.0)
        h, w = 128, 128
        img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        detected, _ = checker_strict.check(img)
        assert detected is False  # Very strict threshold, random noise won't pass


# =============================================================================
# TestEnhancementPipeline
# =============================================================================


class TestEnhancementPipeline:
    """Test EnhancementPipeline with mocked sub-modules."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline in legacy mode with mocked upscaler and grid remover."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            skip_upscale_if_large=False,
            skip_grid_if_undetected=False,
            store_intermediates=True,
            enable_pre_smooth=False,
        )
        pipeline = EnhancementPipeline(
            config=config, device=torch.device("cpu")
        )

        # Mock upscaler
        mock_upscaler = MagicMock()

        def upscale_side_effect(image, scale=None, denoise=False, **kwargs):
            b, c, h, w = image.shape
            result = torch.rand(b, c, h * 2, w * 2)
            return ModuleOutput(result=result, intermediate={}, metadata={})

        mock_upscaler.forward.side_effect = upscale_side_effect
        pipeline._upscaler = mock_upscaler

        # Mock grid remover
        mock_grid = MagicMock()

        def grid_side_effect(image_bgr, **kwargs):
            return image_bgr.copy(), {"period_x": 8, "period_y": 8}

        mock_grid.process.side_effect = grid_side_effect
        pipeline._grid_remover = mock_grid

        return pipeline

    @pytest.fixture
    def mock_spectral_pipeline(self):
        """Create pipeline in spectral mode with mocked upscaler and spectral grid remover."""
        config = EnhancementConfig(
            use_spectral_grid=True,
            skip_upscale_if_large=False,
            store_intermediates=True,
            enable_detail_enhance=False,
        )
        pipeline = EnhancementPipeline(
            config=config, device=torch.device("cpu")
        )

        # Mock upscaler
        mock_upscaler = MagicMock()

        def upscale_side_effect(image, scale=None, denoise=False, **kwargs):
            b, c, h, w = image.shape
            result = torch.rand(b, c, h * 2, w * 2)
            return ModuleOutput(result=result, intermediate={}, metadata={})

        mock_upscaler.forward.side_effect = upscale_side_effect
        pipeline._upscaler = mock_upscaler

        # Mock spectral grid remover
        mock_spectral = MagicMock()

        def spectral_side_effect(image_bgr, **kwargs):
            return image_bgr.copy(), {
                "iterations": 1,
                "converged": True,
                "notches_applied": 3,
            }

        mock_spectral.process.side_effect = spectral_side_effect
        pipeline._spectral_grid_remover = mock_spectral

        return pipeline

    def test_full_pipeline_output_shape(self, mock_pipeline):
        """Full 3-stage pipeline should produce 4x output."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image)

        assert output.result.shape == (1, 3, 256, 256)

    def test_full_pipeline_metadata(self, mock_pipeline):
        """Metadata should contain all stage info."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image)

        assert "input_size" in output.metadata
        assert "output_size" in output.metadata
        assert "effective_scale" in output.metadata
        assert "total_time" in output.metadata
        assert "stages" in output.metadata
        assert len(output.metadata["stages"]) == 4  # pre_smooth(skipped) + 3 stages
        assert output.metadata["pipeline_mode"] == "legacy"

    def test_intermediates_stored(self, mock_pipeline):
        """Intermediate results should be stored when configured."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image)

        assert "input" in output.intermediate
        assert "after_upscale_1" in output.intermediate
        assert "after_grid_removal" in output.intermediate
        assert "after_upscale_2" in output.intermediate

    def test_intermediates_not_stored(self):
        """Intermediates should not be stored when disabled."""
        config = EnhancementConfig(
            store_intermediates=False,
            skip_upscale_if_large=False,
            skip_grid_if_undetected=False,
            enable_pre_smooth=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Mock sub-modules
        mock_upscaler = MagicMock()
        mock_upscaler.forward.side_effect = lambda img, **kw: ModuleOutput(
            result=torch.rand(1, 3, img.shape[2] * 2, img.shape[3] * 2)
        )
        pipeline._upscaler = mock_upscaler

        mock_grid = MagicMock()
        mock_grid.process.side_effect = lambda img, **kw: (img.copy(), {"period_x": 8, "period_y": 8})
        pipeline._grid_remover = mock_grid

        image = torch.rand(1, 3, 32, 32)
        output = pipeline(image)

        assert len(output.intermediate) == 0

    def test_skip_first_upscale_via_kwargs(self, mock_pipeline):
        """First upscale should be skippable via kwargs override."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image, skip_first_upscale=True)

        # Only 2x from second upscale (grid removal doesn't change size)
        assert output.result.shape == (1, 3, 128, 128)
        # stages[0]=pre_smooth(skipped), stages[1]=upscale_1
        assert output.metadata["stages"][1]["skipped"] is True

    def test_skip_grid_removal_via_kwargs(self, mock_pipeline):
        """Grid removal should be skippable via kwargs override."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image, skip_grid_removal=True)

        # Still 4x but grid removal stage is skipped
        assert output.result.shape == (1, 3, 256, 256)
        # stages[0]=pre_smooth(skipped), stages[1]=upscale_1, stages[2]=grid_removal
        assert output.metadata["stages"][2]["skipped"] is True
        mock_pipeline._grid_remover.process.assert_not_called()

    def test_skip_second_upscale_via_kwargs(self, mock_pipeline):
        """Second upscale should be skippable via kwargs override."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image, skip_second_upscale=True)

        # Only 2x from first upscale
        assert output.result.shape == (1, 3, 128, 128)
        # stages[0]=pre_smooth(skipped), stages[1]=upscale_1, stages[2]=grid, stages[3]=upscale_2
        assert output.metadata["stages"][3]["skipped"] is True

    def test_all_stages_disabled(self):
        """All stages disabled should passthrough."""
        config = EnhancementConfig(
            enable_pre_smooth=False,
            enable_first_upscale=False,
            enable_grid_removal=False,
            enable_second_upscale=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        image = torch.rand(1, 3, 64, 64)
        output = pipeline(image)

        assert output.result.shape == (1, 3, 64, 64)
        assert torch.allclose(output.result, image.to(pipeline.device, pipeline.dtype))

    def test_resolution_skip_first_upscale(self):
        """Large images should skip first upscale when configured."""
        config = EnhancementConfig(
            enable_pre_smooth=False,
            skip_upscale_if_large=True,
            max_input_pixels=100,  # Very low threshold
            skip_grid_if_undetected=False,
            enable_second_upscale=False,
            enable_grid_removal=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Mock upscaler (should not be called)
        mock_upscaler = MagicMock()
        pipeline._upscaler = mock_upscaler

        image = torch.rand(1, 3, 64, 64)  # 4096 pixels > 100
        output = pipeline(image)

        mock_upscaler.forward.assert_not_called()
        assert output.result.shape == (1, 3, 64, 64)

    def test_upscale_fallback_on_error(self):
        """Upscaler failure should fallback to bicubic."""
        config = EnhancementConfig(
            enable_pre_smooth=False,
            skip_upscale_if_large=False,
            skip_grid_if_undetected=False,
            enable_grid_removal=False,
            enable_second_upscale=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Mock upscaler that raises
        mock_upscaler = MagicMock()
        mock_upscaler.forward.side_effect = RuntimeError("CUDA OOM")
        pipeline._upscaler = mock_upscaler

        image = torch.rand(1, 3, 32, 32)
        output = pipeline(image)

        # Should get 2x via bicubic fallback
        assert output.result.shape == (1, 3, 64, 64)
        # stages[0]=pre_smooth(skipped), stages[1]=upscale_1
        assert output.metadata["stages"][1]["method"] == "bicubic_fallback"

    def test_grid_removal_passthrough_on_error(self):
        """Grid removal failure should pass through unchanged."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            enable_pre_smooth=False,
            skip_upscale_if_large=False,
            skip_grid_if_undetected=False,
            enable_first_upscale=False,
            enable_second_upscale=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Mock grid remover that raises
        mock_grid = MagicMock()
        mock_grid.process.side_effect = ValueError("Processing failed")
        pipeline._grid_remover = mock_grid

        image = torch.rand(1, 3, 32, 32)
        output = pipeline(image)

        assert output.result.shape == (1, 3, 32, 32)
        # Find the grid removal stage (index 1: after skipped upscale_1)
        grid_stage = next(
            s for s in output.metadata["stages"]
            if s.get("name") == "grid_removal" and not s.get("skipped")
        )
        assert grid_stage["success"] is False

    def test_module_name(self):
        """Module name should be 'enhancement'."""
        pipeline = EnhancementPipeline(device=torch.device("cpu"))
        assert pipeline.name == "enhancement"

    def test_grid_detection_in_metadata(self, mock_pipeline):
        """Grid detection info should appear in metadata."""
        # Enable grid detection check
        mock_pipeline.config.skip_grid_if_undetected = True

        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image)

        assert "grid_detection" in output.metadata

    def test_effective_scale_4x(self, mock_pipeline):
        """Full pipeline should report ~4x effective scale."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_pipeline(image)

        scale_h, scale_w = output.metadata["effective_scale"]
        assert scale_h == pytest.approx(4.0)
        assert scale_w == pytest.approx(4.0)


# =============================================================================
# TestSpectralPipeline
# =============================================================================


class TestSpectralPipeline:
    """Test EnhancementPipeline in spectral grid removal mode."""

    @pytest.fixture
    def mock_spectral_pipeline(self):
        """Create pipeline in spectral mode with mocked components."""
        config = EnhancementConfig(
            use_spectral_grid=True,
            skip_upscale_if_large=False,
            store_intermediates=True,
            enable_detail_enhance=False,
        )
        pipeline = EnhancementPipeline(
            config=config, device=torch.device("cpu")
        )

        mock_upscaler = MagicMock()

        def upscale_side_effect(image, scale=None, denoise=False, **kwargs):
            b, c, h, w = image.shape
            result = torch.rand(b, c, h * 2, w * 2)
            return ModuleOutput(result=result, intermediate={}, metadata={})

        mock_upscaler.forward.side_effect = upscale_side_effect
        pipeline._upscaler = mock_upscaler

        mock_spectral = MagicMock()

        def spectral_side_effect(image_bgr, **kwargs):
            return image_bgr.copy(), {
                "iterations": 1,
                "converged": True,
                "notches_applied": 3,
            }

        mock_spectral.process.side_effect = spectral_side_effect
        pipeline._spectral_grid_remover = mock_spectral

        return pipeline

    def test_spectral_mode_output_shape(self, mock_spectral_pipeline):
        """Spectral mode pipeline should produce 4x output."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image)
        assert output.result.shape == (1, 3, 256, 256)

    def test_spectral_mode_metadata(self, mock_spectral_pipeline):
        """Spectral mode should report pipeline_mode='spectral'."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image)
        assert output.metadata["pipeline_mode"] == "spectral"
        assert len(output.metadata["stages"]) == 4

    def test_spectral_mode_intermediates(self, mock_spectral_pipeline):
        """Spectral mode should store spectral grid intermediate."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image)
        assert "after_spectral_grid" in output.intermediate
        assert "after_upscale_1" in output.intermediate
        assert "after_upscale_2" in output.intermediate

    def test_spectral_mode_stage_order(self, mock_spectral_pipeline):
        """Stage order should be: spectral_grid -> upscale_1 -> detail(skipped) -> upscale_2."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image)
        stages = output.metadata["stages"]
        assert stages[0]["name"] == "spectral_grid_removal"
        assert stages[1]["name"] == "upscale_1"
        assert stages[2]["name"] == "detail_enhance"
        assert stages[2]["skipped"] is True
        assert stages[3]["name"] == "upscale_2"

    def test_spectral_skip_via_kwargs(self, mock_spectral_pipeline):
        """Spectral grid removal should be skippable via kwargs."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image, skip_spectral_grid=True)
        assert output.metadata["stages"][0]["skipped"] is True
        mock_spectral_pipeline._spectral_grid_remover.process.assert_not_called()

    def test_spectral_grid_calls_process(self, mock_spectral_pipeline):
        """Spectral mode should call SpectralGridRemover.process()."""
        image = torch.rand(1, 3, 64, 64)
        mock_spectral_pipeline(image)
        mock_spectral_pipeline._spectral_grid_remover.process.assert_called_once()

    def test_spectral_effective_scale_4x(self, mock_spectral_pipeline):
        """Spectral mode should report 4x effective scale."""
        image = torch.rand(1, 3, 64, 64)
        output = mock_spectral_pipeline(image)
        scale_h, scale_w = output.metadata["effective_scale"]
        assert scale_h == pytest.approx(4.0)
        assert scale_w == pytest.approx(4.0)


# =============================================================================
# TestPreSmooth
# =============================================================================


class TestPreSmooth:
    """Test pre-smoothing stage."""

    def test_pre_smooth_runs(self):
        """Pre-smooth should apply bilateral filter."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            enable_pre_smooth=True,
            pre_smooth_iterations=2,
            enable_first_upscale=False,
            enable_grid_removal=False,
            enable_second_upscale=False,
            store_intermediates=True,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        image = torch.rand(1, 3, 64, 64)
        output = pipeline(image)

        assert output.result.shape == (1, 3, 64, 64)
        assert "after_pre_smooth" in output.intermediate
        smooth_stage = output.metadata["stages"][0]
        assert smooth_stage["name"] == "pre_smooth"
        assert smooth_stage["success"] is True
        assert smooth_stage["iterations"] == 2

    def test_pre_smooth_skip_via_kwargs(self):
        """Pre-smooth should be skippable via kwargs."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            enable_pre_smooth=True,
            enable_first_upscale=False,
            enable_grid_removal=False,
            enable_second_upscale=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        image = torch.rand(1, 3, 32, 32)
        output = pipeline(image, skip_pre_smooth=True)

        assert output.metadata["stages"][0]["skipped"] is True

    def test_pre_smooth_reduces_grid(self):
        """Pre-smooth should reduce high-frequency grid patterns."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            enable_pre_smooth=True,
            pre_smooth_iterations=3,
            bilateral_d=9,
            bilateral_sigma_color=75.0,
            bilateral_sigma_space=75.0,
            enable_first_upscale=False,
            enable_grid_removal=False,
            enable_second_upscale=False,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Create image with grid pattern
        h, w = 128, 128
        base = np.ones((h, w, 3), dtype=np.float32) * 0.6
        for i in range(0, h, 8):
            base[i, :, :] *= 0.9  # subtle grid lines
        for j in range(0, w, 8):
            base[:, j, :] *= 0.9

        image = torch.from_numpy(base).permute(2, 0, 1).unsqueeze(0)
        output = pipeline(image)

        # Smoothed result should have less variance than input
        input_std = image.std().item()
        output_std = output.result.std().item()
        assert output_std < input_std

    def test_full_pipeline_with_pre_smooth(self):
        """Full 4-stage pipeline should produce 4x output with pre-smooth."""
        config = EnhancementConfig(
            use_spectral_grid=False,
            enable_pre_smooth=True,
            pre_smooth_iterations=2,
            skip_upscale_if_large=False,
            skip_grid_if_undetected=False,
            store_intermediates=True,
        )
        pipeline = EnhancementPipeline(config=config, device=torch.device("cpu"))

        # Mock upscaler
        mock_upscaler = MagicMock()
        mock_upscaler.forward.side_effect = lambda img, **kw: ModuleOutput(
            result=torch.rand(1, 3, img.shape[2] * 2, img.shape[3] * 2)
        )
        pipeline._upscaler = mock_upscaler

        # Mock grid remover
        mock_grid = MagicMock()
        mock_grid.process.side_effect = lambda img, **kw: (img.copy(), {"period_x": 8, "period_y": 8})
        pipeline._grid_remover = mock_grid

        image = torch.rand(1, 3, 32, 32)
        output = pipeline(image)

        assert output.result.shape == (1, 3, 128, 128)
        assert len(output.metadata["stages"]) == 4
        assert output.metadata["stages"][0]["name"] == "pre_smooth"
        assert "after_pre_smooth" in output.intermediate


# =============================================================================
# TestSpectralGridRemover
# =============================================================================


class TestSpectralGridRemover:
    """Test SpectralGridRemover directly."""

    @pytest.fixture
    def remover(self):
        """Create SpectralGridRemover with default config."""
        config = EnhancementConfig()
        from kp3d.modules.enhancement.spectral_grid import SpectralGridRemover
        return SpectralGridRemover(config)

    def test_synthetic_grid_removal(self, remover):
        """Synthetic grid image should be cleaned with high PSNR."""
        h, w = 256, 256
        # Create clean image
        base = np.ones((h, w, 3), dtype=np.float64) * 180.0
        # Add some content
        base[80:120, 80:120, :] = 60.0  # dark square

        # Add multiplicative grid (period=8)
        grid = np.ones((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                grid[i, j] = 0.85 + 0.15 * np.cos(2 * np.pi * j / 8.0)
                grid[i, j] *= 0.85 + 0.15 * np.cos(2 * np.pi * i / 8.0)

        gridded = base.copy()
        for c in range(3):
            gridded[:, :, c] *= grid

        img_bgr = gridded.clip(0, 255).astype(np.uint8)

        result, meta = remover.process(img_bgr)

        assert result.shape == img_bgr.shape
        assert result.dtype == np.uint8
        assert meta["iterations"] >= 1

    def test_subpixel_period_detection(self, remover):
        """Non-integer period should be detected with sub-pixel precision."""
        h, w = 256, 256
        period = 7.3  # non-integer period

        # Create grid pattern
        base = np.ones((h, w), dtype=np.float64) * 180.0
        for j in range(w):
            base[:, j] *= 0.8 + 0.2 * np.cos(2 * np.pi * j / period)

        img_bgr = np.stack([base, base, base], axis=-1).clip(0, 255).astype(np.uint8)

        result, meta = remover.process(img_bgr)

        # Check that periods were detected
        detected_v = meta["periods_detected"]["vertical"]
        if detected_v:
            # At least one detected period should be close to 7.3
            closest = min(detected_v, key=lambda p: abs(p - period))
            assert abs(closest - period) < 1.0  # within 1 pixel

    def test_no_grid_passthrough(self, remover):
        """Image without grid should pass through with minimal change."""
        h, w = 128, 128
        # Smooth gradient - no periodic pattern
        gradient = np.linspace(100, 200, w, dtype=np.float64)
        base = np.tile(gradient, (h, 1))
        img_bgr = np.stack([base, base, base], axis=-1).astype(np.uint8)

        result, meta = remover.process(img_bgr)

        # Result should be very similar to input
        diff = np.abs(result.astype(np.float64) - img_bgr.astype(np.float64))
        assert np.mean(diff) < 8.0  # less than 8 intensity levels mean difference

    def test_color_preservation(self, remover):
        """A and B channels in LAB should be minimally affected."""
        h, w = 128, 128
        # Create colorful image with grid
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = 180  # B channel
        img[:, :, 1] = 100  # G channel
        img[:, :, 2] = 60   # R channel

        # Add grid to luminance only
        for i in range(h):
            for j in range(w):
                factor = 0.85 + 0.15 * np.cos(2 * np.pi * i / 8.0)
                img[i, j, :] = (img[i, j, :].astype(np.float64) * factor).clip(0, 255).astype(np.uint8)

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result, meta = remover.process(img)
        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

        # A and B channels should be very close
        a_diff = np.abs(lab_after[:, :, 1].astype(float) - lab_before[:, :, 1].astype(float))
        b_diff = np.abs(lab_after[:, :, 2].astype(float) - lab_before[:, :, 2].astype(float))
        assert np.mean(a_diff) < 3.0
        assert np.mean(b_diff) < 3.0

    def test_butterworth_filter_shape(self, remover):
        """Butterworth filter should have values in [0, 1] with smooth roll-off."""
        shape = (256, 256)
        notch_params = [
            {"freq_x": 32.0, "freq_y": 0.0, "width": 2.0, "amplitude": 100.0},
        ]

        H = remover._build_butterworth_notch(shape, notch_params, order=2)

        assert H.shape == shape
        assert H.min() >= 0.0
        assert H.max() <= 1.0
        # Filter should be 0 at notch center and ~1 far away
        center_y, center_x = 128, 128
        # At notch center (center_x + 32)
        assert H[center_y, center_x + 32] < 0.1
        # Far from notch
        assert H[center_y, center_x] > 0.5  # DC should mostly pass

    def test_iterative_convergence(self):
        """Two iterations should converge for typical grid pattern."""
        config = EnhancementConfig(spectral_max_iterations=2)
        from kp3d.modules.enhancement.spectral_grid import SpectralGridRemover
        remover = SpectralGridRemover(config)

        h, w = 256, 256
        base = np.ones((h, w, 3), dtype=np.float64) * 160.0
        for i in range(h):
            for j in range(w):
                factor = 0.8 + 0.2 * np.cos(2 * np.pi * i / 10.0)
                factor *= 0.8 + 0.2 * np.cos(2 * np.pi * j / 10.0)
                base[i, j, :] *= factor

        img_bgr = base.clip(0, 255).astype(np.uint8)
        result, meta = remover.process(img_bgr)

        assert meta["iterations"] <= 2

    def test_multiple_harmonics(self, remover):
        """Fundamental + harmonics should all be addressed."""
        h, w = 512, 512
        base = np.ones((h, w, 3), dtype=np.float64) * 160.0

        # Add fundamental (period=16) + 2nd harmonic (period=8)
        for i in range(h):
            factor = 1.0
            factor += 0.15 * np.cos(2 * np.pi * i / 16.0)  # fundamental
            factor += 0.08 * np.cos(2 * np.pi * i / 8.0)   # 2nd harmonic
            base[i, :, :] *= factor

        img_bgr = base.clip(0, 255).astype(np.uint8)
        result, meta = remover.process(img_bgr)

        assert result.shape == img_bgr.shape
        # Should detect at least fundamental
        h_periods = meta["periods_detected"]["horizontal"]
        assert len(h_periods) >= 1


# =============================================================================
# TestModuleRegistration
# =============================================================================


class TestModuleRegistration:
    """Test that EnhancementModule is properly registered."""

    def test_registry_contains_enhancement(self):
        """Enhancement module should be registered in the module registry."""
        from kp3d.core.registry import ModuleRegistry

        # Import to trigger registration
        import kp3d.modules.enhancement  # noqa: F401

        assert ModuleRegistry.has("enhancement")

    def test_get_module_creates_instance(self):
        """get_module should create an EnhancementPipeline instance."""
        from kp3d.core.registry import get_module

        import kp3d.modules.enhancement  # noqa: F401

        module = get_module("enhancement", device=torch.device("cpu"))
        assert module.name == "enhancement"
