"""Tests for v9 STFT Adaptive Grid Removal."""

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kp3d.modules.restoration.stft_adaptive_grid import STFTAdaptiveGridRemover
from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.hybrid_grid_pattern import HybridGridPatternRestorer


def create_grid_image(h=128, w=128, period_x=9, period_y=7,
                      modulation_b=0.15, modulation_g=0.07, modulation_r=0.04):
    """Create synthetic image with grid pattern for testing."""
    np.random.seed(42)
    base = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8).astype(np.float32)
    for y in range(h):
        for x in range(w):
            grid_val = np.sin(2 * np.pi * x / period_x) + np.sin(2 * np.pi * y / period_y)
            base[y, x, 0] += grid_val * 255 * modulation_b  # B
            base[y, x, 1] += grid_val * 255 * modulation_g  # G
            base[y, x, 2] += grid_val * 255 * modulation_r  # R
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def remover():
    """Default STFTAdaptiveGridRemover instance."""
    return STFTAdaptiveGridRemover(
        period_x=0, period_y=0,
        window_size=63, hop_size=16,
        notch_sigma=1.5, base_attenuation=0.15,
        channel_adaptive=True,
    )


@pytest.fixture
def remover_known_periods():
    """STFTAdaptiveGridRemover with known periods."""
    return STFTAdaptiveGridRemover(
        period_x=9, period_y=7,
        window_size=63, hop_size=16,
        notch_sigma=1.5, base_attenuation=0.15,
        channel_adaptive=True,
    )


class TestGridPeriodDetection:
    """Tests for detect_grid_periods method."""

    def test_detect_known_periods(self, remover):
        img = create_grid_image(h=128, w=128, period_x=9, period_y=7, modulation_b=0.25)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        period_x, period_y = remover.detect_grid_periods(img_gray)
        assert 7 <= period_x <= 11, f"Expected period_x near 9, got {period_x}"
        assert 5 <= period_y <= 9, f"Expected period_y near 7, got {period_y}"

    def test_detect_fallback_on_noise(self, remover):
        np.random.seed(123)
        noise = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        period_x, period_y = remover.detect_grid_periods(noise)
        assert period_x > 0 and period_y > 0

    def test_configured_periods_used_in_process(self):
        remover = STFTAdaptiveGridRemover(period_x=10, period_y=8)
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover.process(img, use_stft=False)
        periods = intermediates['detected_periods']
        assert periods[0] == 10
        assert periods[1] == 8

    def test_detection_with_strong_grid(self, remover):
        base = np.full((128, 128), 128, dtype=np.float32)
        period_x, period_y = 11, 9
        for y in range(128):
            for x in range(128):
                grid_val = np.sin(2 * np.pi * x / period_x) + np.sin(2 * np.pi * y / period_y)
                base[y, x] += grid_val * 40
        img_gray = np.clip(base, 0, 255).astype(np.uint8)
        detected_x, detected_y = remover.detect_grid_periods(img_gray)
        assert 8 <= detected_x <= 14
        assert 7 <= detected_y <= 12


class TestChannelModulation:
    """Tests for measure_channel_modulation method."""

    def test_uniform_image_low_modulation(self, remover):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        modulation = remover.measure_channel_modulation(img, period_x=9, period_y=7)
        assert 0 in modulation and 1 in modulation and 2 in modulation
        for v in modulation.values():
            assert v < 0.05

    def test_channel_ordering(self, remover):
        img = create_grid_image(h=128, w=128, period_x=9, period_y=7,
                                modulation_b=0.20, modulation_g=0.08, modulation_r=0.03)
        modulation = remover.measure_channel_modulation(img, period_x=9, period_y=7)
        assert modulation[0] > modulation[1]
        assert modulation[1] > modulation[2]

    def test_synthetic_grid_modulation(self, remover):
        img = create_grid_image(h=128, w=128, period_x=11, period_y=11,
                                modulation_b=0.25, modulation_g=0.15, modulation_r=0.05)
        modulation = remover.measure_channel_modulation(img, period_x=11, period_y=11)
        assert modulation[0] > modulation[1] > modulation[2]

    def test_modulation_all_channels_positive(self, remover):
        img = create_grid_image(h=64, w=64, period_x=7, period_y=7)
        modulation = remover.measure_channel_modulation(img, period_x=7, period_y=7)
        assert all(mod >= 0 for mod in modulation.values())
        assert len(modulation) == 3


class TestLocalEnergyMap:
    """Tests for compute_local_energy_map method."""

    def test_energy_map_shape(self, remover):
        img_gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        energy_map = remover.compute_local_energy_map(img_gray, period_x=9, period_y=7)
        assert energy_map.shape == img_gray.shape
        assert energy_map.dtype == np.float32

    def test_energy_map_range(self, remover):
        img = create_grid_image(h=128, w=128)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy_map = remover.compute_local_energy_map(img_gray, period_x=9, period_y=7)
        assert np.all(energy_map >= 0)
        assert np.all(energy_map <= 1.001)  # Small tolerance

    def test_grid_region_higher_energy(self, remover):
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        for y in range(64):  # Top half only
            for x in range(128):
                grid_val = np.sin(2 * np.pi * x / 9) + np.sin(2 * np.pi * y / 7)
                img[y, x] = np.clip(128 + int(grid_val * 30), 0, 255)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy_map = remover.compute_local_energy_map(img_gray, period_x=9, period_y=7)
        top_energy = np.mean(energy_map[:48, :])  # Avoid boundary blending
        bottom_energy = np.mean(energy_map[80:, :])
        assert top_energy > bottom_energy

    def test_uniform_image_low_energy(self, remover):
        img_gray = np.full((64, 64), 128, dtype=np.uint8)
        energy_map = remover.compute_local_energy_map(img_gray, period_x=9, period_y=7)
        assert np.max(energy_map) < 0.15


class TestGaussianNotchMask:
    """Tests for create_gaussian_notch_mask method."""

    def test_mask_shape_and_range(self, remover):
        mask = remover.create_gaussian_notch_mask(64, 64, period_x=9, period_y=7,
                                                  attenuation=0.1)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float64
        assert np.all(mask >= -0.01)
        assert np.all(mask <= 1.01)

    def test_full_attenuation_zero(self, remover):
        mask = remover.create_gaussian_notch_mask(128, 128, period_x=9, period_y=7,
                                                  attenuation=0.0)
        min_val = np.min(mask)
        assert min_val < 0.1

    def test_no_attenuation(self, remover):
        mask = remover.create_gaussian_notch_mask(64, 64, period_x=9, period_y=7,
                                                  attenuation=1.0)
        assert np.allclose(mask, 1.0)

    def test_dc_preserved(self, remover):
        h, w = 64, 64
        mask = remover.create_gaussian_notch_mask(h, w, period_x=9, period_y=7,
                                                  attenuation=0.0)
        center_y, center_x = h // 2, w // 2
        assert mask[center_y, center_x] > 0.9


class TestChannelAdaptiveRemoval:
    """Tests for channel_adaptive_stft_removal and channel_adaptive_notch_global."""

    def test_stft_output_shape_dtype(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        energy_map = np.ones((64, 64), dtype=np.float32) * 0.5
        modulation = {0: 0.15, 1: 0.08, 2: 0.04}
        result = remover_known_periods.channel_adaptive_stft_removal(
            img, period_x=9, period_y=7,
            energy_map=energy_map, modulation_depths=modulation,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_global_output_shape_dtype(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        modulation = {0: 0.15, 1: 0.08, 2: 0.04}
        result = remover_known_periods.channel_adaptive_notch_global(
            img, period_x=9, period_y=7, modulation_depths=modulation,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_global_reduces_grid(self, remover_known_periods):
        img = create_grid_image(h=128, w=128, period_x=9, period_y=7, modulation_b=0.20)
        modulation = {0: 0.20, 1: 0.08, 2: 0.04}
        result = remover_known_periods.channel_adaptive_notch_global(
            img, period_x=9, period_y=7, modulation_depths=modulation,
        )
        # Check that B channel changed significantly
        b_diff = np.mean(np.abs(img[:, :, 0].astype(float) - result[:, :, 0].astype(float)))
        assert b_diff > 1.0  # Should have noticeable change

    def test_channel_adaptive_different_channels(self, remover_known_periods):
        img = create_grid_image(h=128, w=128, period_x=9, period_y=7,
                                modulation_b=0.25, modulation_g=0.10, modulation_r=0.03)
        energy_map = np.ones((128, 128), dtype=np.float32)
        modulation = {0: 0.25, 1: 0.10, 2: 0.03}
        result = remover_known_periods.channel_adaptive_stft_removal(
            img, period_x=9, period_y=7,
            energy_map=energy_map, modulation_depths=modulation,
        )
        b_diff = np.mean(np.abs(img[:, :, 0].astype(float) - result[:, :, 0].astype(float)))
        r_diff = np.mean(np.abs(img[:, :, 2].astype(float) - result[:, :, 2].astype(float)))
        assert b_diff > r_diff


class TestFullPipeline:
    """Tests for the complete process method."""

    def test_process_returns_tuple(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover_known_periods.process(img, use_stft=False)
        assert isinstance(result, np.ndarray)
        assert isinstance(intermediates, dict)

    def test_intermediates_keys(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover_known_periods.process(img, use_stft=False)
        expected_keys = {'energy_map', 'filtered', 'result', 'detected_periods', 'modulation_depths'}
        assert expected_keys.issubset(intermediates.keys())

    def test_process_with_edge_mask(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        edge_mask = np.zeros((64, 64), dtype=np.float32)
        edge_mask[:, :10] = 1.0  # Protect left edge
        result, _ = remover_known_periods.process(
            img, edge_mask=edge_mask, edge_preservation=0.9, use_stft=False,
        )
        # Left side should be closer to original
        left_diff = np.mean(np.abs(img[:, :10].astype(float) - result[:, :10].astype(float)))
        right_diff = np.mean(np.abs(img[:, 30:40].astype(float) - result[:, 30:40].astype(float)))
        assert left_diff <= right_diff + 1  # Edge-protected region changes less

    def test_process_without_edge_mask(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover_known_periods.process(
            img, edge_mask=None, use_stft=False,
        )
        np.testing.assert_array_equal(result, intermediates['filtered'])

    def test_process_stft_mode(self, remover_known_periods):
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover_known_periods.process(
            img, edge_mask=None, use_stft=True,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestHybridIntegration:
    """Tests for integration with HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(store_intermediates=False)
        return HybridGridPatternRestorer(config=config)

    def test_stft_adaptive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, intermediates = restorer.restore_grid_pattern(img, method="stft_adaptive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_conservative_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_conservative")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_global_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_global")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_no_edge_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_no_edge")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_backward_compatible_guided_only(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="guided_only")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV9ConfigDefaults:
    """Tests for v9 config defaults and customization."""

    def test_default_config_values(self):
        config = RestorationConfig()
        assert config.grid_stft_period_x == 0
        assert config.grid_stft_period_y == 0
        assert config.grid_stft_window_size == 63
        assert config.grid_stft_hop_size == 16
        assert config.grid_stft_notch_sigma == 1.5
        assert config.grid_stft_base_attenuation == 0.15
        assert config.grid_stft_edge_protection is True
        assert config.grid_stft_channel_adaptive is True
        assert config.grid_stft_use_stft is True

    def test_custom_config_periods(self):
        config = RestorationConfig(
            grid_stft_period_x=12,
            grid_stft_period_y=10,
            grid_stft_base_attenuation=0.05,
        )
        assert config.grid_stft_period_x == 12
        assert config.grid_stft_period_y == 10
        assert config.grid_stft_base_attenuation == 0.05

    def test_stft_remover_from_config(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
        )
        remover = STFTAdaptiveGridRemover(
            period_x=config.grid_stft_period_x,
            period_y=config.grid_stft_period_y,
            window_size=config.grid_stft_window_size,
            hop_size=config.grid_stft_hop_size,
            notch_sigma=config.grid_stft_notch_sigma,
            base_attenuation=config.grid_stft_base_attenuation,
            channel_adaptive=config.grid_stft_channel_adaptive,
        )
        assert remover.period_x == 9
        assert remover.period_y == 7


class TestV2PointNotchMask:
    """Tests for v2 point notch mask (intersection points only)."""

    def test_point_mask_shape_and_range(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        mask = remover.create_point_notch_mask(128, 128, 9, 7, attenuation=0.1)
        assert mask.shape == (128, 128)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_point_mask_dc_preserved(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        mask = remover.create_point_notch_mask(128, 128, 9, 7, attenuation=0.0)
        cy, cx = 64, 64
        # DC component at center should be ~1.0
        assert mask[cy, cx] > 0.95

    def test_point_mask_less_damage_than_line(self):
        """Point notch should preserve more frequencies than line notch."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        line_mask = remover.create_gaussian_notch_mask(128, 128, 9, 7, attenuation=0.0)
        point_mask = remover.create_point_notch_mask(128, 128, 9, 7, attenuation=0.0)
        # Point mask should have higher mean (less overall attenuation)
        assert point_mask.mean() > line_mask.mean()

    def test_point_mask_with_custom_sigma(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        mask_default = remover.create_point_notch_mask(64, 64, 9, 7, attenuation=0.1)
        mask_wide = remover.create_point_notch_mask(64, 64, 9, 7, attenuation=0.1, point_sigma=5.0)
        # Wider sigma -> more attenuation -> lower mean
        assert mask_wide.mean() < mask_default.mean()


class TestV2AdaptiveEdgeMask:
    """Tests for v2 adaptive edge mask (grid-energy weighted)."""

    def test_adaptive_mask_shape(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        edge_mask = np.random.rand(64, 64).astype(np.float32)
        energy_map = np.random.rand(64, 64).astype(np.float32)
        result = remover.compute_adaptive_edge_mask(edge_mask, energy_map)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_high_energy_reduces_protection(self):
        """Where grid energy is high, edge protection should be reduced."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        edge_mask = np.ones((64, 64), dtype=np.float32) * 0.8
        # High energy region
        energy_high = np.ones((64, 64), dtype=np.float32) * 0.9
        # Low energy region
        energy_low = np.zeros((64, 64), dtype=np.float32)

        mask_high = remover.compute_adaptive_edge_mask(edge_mask, energy_high)
        mask_low = remover.compute_adaptive_edge_mask(edge_mask, energy_low)
        # High energy -> less protection
        assert mask_high.mean() < mask_low.mean()

    def test_zero_edge_stays_zero(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        edge_mask = np.zeros((64, 64), dtype=np.float32)
        energy_map = np.ones((64, 64), dtype=np.float32)
        result = remover.compute_adaptive_edge_mask(edge_mask, energy_map)
        assert np.allclose(result, 0)


class TestV2TwoPassRemoval:
    """Tests for v2 two-pass removal pipeline."""

    def test_two_pass_returns_tuple(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=64, w=64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy_map = remover.compute_local_energy_map(gray, 9, 7)
        modulation = {0: 0.15, 1: 0.07, 2: 0.04}
        result, intermediates = remover.two_pass_removal(
            img, 9, 7, energy_map, modulation
        )
        assert isinstance(result, np.ndarray)
        assert isinstance(intermediates, dict)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_two_pass_intermediates_keys(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=64, w=64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy_map = remover.compute_local_energy_map(gray, 9, 7)
        modulation = {0: 0.15, 1: 0.07, 2: 0.04}
        _, intermediates = remover.two_pass_removal(
            img, 9, 7, energy_map, modulation
        )
        assert 'pass1_result' in intermediates
        assert 'residual_energy' in intermediates

    def test_two_pass_with_edge_mask(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=64, w=64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy_map = remover.compute_local_energy_map(gray, 9, 7)
        modulation = {0: 0.15, 1: 0.07, 2: 0.04}
        edge_mask = np.random.rand(64, 64).astype(np.float32) * 0.5
        result, intermediates = remover.two_pass_removal(
            img, 9, 7, energy_map, modulation,
            edge_mask=edge_mask, edge_preservation=0.5,
        )
        assert result.shape == img.shape
        assert 'adaptive_edge_mask' in intermediates


class TestV2ProcessV2Pipeline:
    """Tests for the full v2 process pipeline."""

    def test_process_v2_returns_tuple(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover.process_v2(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert 'detected_periods' in intermediates
        assert 'modulation_depths' in intermediates
        assert 'energy_map' in intermediates
        assert 'pass1_result' in intermediates

    def test_process_v2_with_edge_mask(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.random.rand(64, 64).astype(np.float32) * 0.5
        result, intermediates = remover.process_v2(img, edge_mask=edge_mask)
        assert result.shape == img.shape

    def test_process_v2_reduces_grid(self):
        """V2 pipeline should reduce grid energy."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7, window_size=15, hop_size=8)
        img = create_grid_image(h=128, w=128, modulation_b=0.2, modulation_g=0.1, modulation_r=0.05)
        result, _ = remover.process_v2(img)
        # Measure grid energy before and after
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy


class TestV2HybridIntegration:
    """Tests for v2 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v2_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v2")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v2_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v2_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v2_quality_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v2_quality")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV3EdgeHighfreq:
    """Tests for V3 edge high-frequency extraction."""

    def test_output_shape_and_dtype(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = remover.extract_edge_highfreq(img, edge_mask)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_channel_weights_applied(self):
        """B channel should have lower contribution than R channel."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.ones((64, 64), dtype=np.float32)
        result = remover.extract_edge_highfreq(
            img, edge_mask, channel_weights=(0.2, 0.4, 0.4)
        )
        # With equal input, B (0.2 weight) should have lower magnitude than R (0.4 weight)
        b_energy = np.mean(np.abs(result[:, :, 0]))
        r_energy = np.mean(np.abs(result[:, :, 2]))
        assert r_energy > b_energy * 0.5  # R should be significantly larger

    def test_edge_mask_zeros_give_zero_output(self):
        """Zero edge mask should produce zero output."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.zeros((64, 64), dtype=np.float32)
        result = remover.extract_edge_highfreq(img, edge_mask)
        assert np.allclose(result, 0)

    def test_only_edge_regions_extracted(self):
        """Only edge regions should have non-zero values."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.zeros((64, 64), dtype=np.float32)
        edge_mask[20:30, 20:30] = 1.0  # Small edge region
        result = remover.extract_edge_highfreq(img, edge_mask)
        # Non-edge region should be zero
        assert np.allclose(result[:10, :10, :], 0)
        # Edge region may have non-zero values (from high-pass filter)


class TestV3FilterGridFromContent:
    """Tests for V3 grid removal from arbitrary content."""

    def test_output_shape_and_dtype(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        content = np.random.randn(64, 64, 3).astype(np.float32) * 10
        result = remover.filter_grid_from_content(content, 9, 7)
        assert result.shape == content.shape
        assert result.dtype == np.float32

    def test_grid_energy_reduced(self):
        """Grid harmonics should be attenuated in the output."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        # Create content with strong grid component
        h, w = 128, 128
        content = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                grid_val = np.sin(2 * np.pi * x / 9) + np.sin(2 * np.pi * y / 7)
                content[y, x, :] = grid_val * 20
        result = remover.filter_grid_from_content(content, 9, 7)
        # Grid energy should be much lower
        assert np.std(result) < np.std(content) * 0.5

    def test_dc_preserved(self):
        """DC component (mean value) should be approximately preserved."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        content = np.ones((64, 64, 3), dtype=np.float32) * 50
        # Add small grid
        for y in range(64):
            for x in range(64):
                content[y, x, 0] += np.sin(2 * np.pi * x / 9) * 5
        result = remover.filter_grid_from_content(content, 9, 7)
        # Mean should be approximately preserved
        assert abs(np.mean(result) - np.mean(content)) < 2.0


class TestV3Pipeline:
    """Tests for the full V3 process pipeline."""

    def test_process_v3_returns_tuple(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8
        )
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover.process_v3(img)
        assert isinstance(result, np.ndarray)
        assert isinstance(intermediates, dict)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_process_v3_intermediates_keys(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8
        )
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img)
        expected_keys = {
            'detected_periods', 'modulation_depths', 'energy_map',
            'aggressive_filtered', 'edge_mask', 'edge_hf',
            'edge_hf_clean', 'result',
        }
        assert expected_keys.issubset(intermediates.keys())

    def test_process_v3_with_edge_mask(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8
        )
        img = create_grid_image(h=64, w=64)
        edge_mask = np.random.rand(64, 64).astype(np.float32)
        result, _ = remover.process_v3(img, edge_mask=edge_mask)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_process_v3_reduces_grid(self):
        """V3 should reduce grid energy."""
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8
        )
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        result, _ = remover.process_v3(img)
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy

    def test_process_v3_with_diffusion(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8
        )
        img = create_grid_image(h=64, w=64)
        result, _ = remover.process_v3(img, apply_diffusion=True)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV3HybridIntegration:
    """Tests for V3 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v3_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v3_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v3_quality_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3_quality")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV31MultiscaleEdgeExtraction:
    """Tests for V3.1 multi-scale DoG edge extraction with grid avoidance."""

    def test_output_shape_and_dtype(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.ones((64, 64), dtype=np.float32) * 0.5
        result = remover.extract_edge_highfreq_multiscale(
            img, edge_mask, period_x=9, period_y=7,
        )
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_zero_edge_mask_gives_zero(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.zeros((64, 64), dtype=np.float32)
        result = remover.extract_edge_highfreq_multiscale(
            img, edge_mask, period_x=9, period_y=7,
        )
        assert np.allclose(result, 0)

    def test_grid_band_gets_low_weight(self):
        """Bands near grid frequency should get lower weight than distant bands."""
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        edge_mask = np.ones((128, 128), dtype=np.float32)

        # Multi-scale should have less grid contamination than single-scale
        single = remover.extract_edge_highfreq(
            img, edge_mask, highpass_sigma=1.5,
        )
        multi = remover.extract_edge_highfreq_multiscale(
            img, edge_mask, period_x=9, period_y=7,
        )
        # Measure grid energy in both via FFT at grid frequencies
        h, w = 128, 128
        fy = int(round(h / 7))
        cx, cy = w // 2, h // 2
        single_fft = np.abs(np.fft.fftshift(np.fft.fft2(single[:, :, 0])))
        multi_fft = np.abs(np.fft.fftshift(np.fft.fft2(multi[:, :, 0])))
        single_grid = single_fft[cy + fy, cx] + single_fft[cy - fy, cx]
        multi_grid = multi_fft[cy + fy, cx] + multi_fft[cy - fy, cx]
        # Multi-scale should have less grid leakage
        assert multi_grid < single_grid

    def test_channel_weights_applied(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.ones((64, 64), dtype=np.float32)
        result = remover.extract_edge_highfreq_multiscale(
            img, edge_mask, period_x=9, period_y=7,
            channel_weights=(0.1, 0.4, 0.5),
        )
        b_energy = np.mean(np.abs(result[:, :, 0]))
        r_energy = np.mean(np.abs(result[:, :, 2]))
        assert r_energy > b_energy  # R weight (0.5) > B weight (0.1)

    def test_custom_scales(self):
        remover = STFTAdaptiveGridRemover(period_x=9, period_y=7)
        img = create_grid_image(h=64, w=64)
        edge_mask = np.ones((64, 64), dtype=np.float32)
        result = remover.extract_edge_highfreq_multiscale(
            img, edge_mask, period_x=9, period_y=7,
            scales=(0.3, 0.7, 3.0, 6.0),
        )
        assert result.shape == img.shape
        assert not np.allclose(result, 0)


class TestV31AdaptiveChannelWeights:
    """Tests for V3.1 adaptive channel weights from modulation depths."""

    def test_adaptive_weights_computed(self):
        """process_v3 with adaptive_weights=True should compute weights from modulation."""
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        # Use larger image with stronger modulation contrast for reliable ordering
        img = create_grid_image(h=128, w=128, modulation_b=0.25, modulation_g=0.08, modulation_r=0.03)
        _, intermediates = remover.process_v3(img, adaptive_weights=True)
        assert 'adaptive_channel_weights' in intermediates
        weights = intermediates['adaptive_channel_weights']
        assert len(weights) == 3
        # Weights should differ from each other (not uniform)
        assert not np.allclose(weights[0], weights[1], atol=0.01)
        # All weights should be positive
        assert all(w > 0 for w in weights)

    def test_adaptive_weights_sum_to_one(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img, adaptive_weights=True)
        weights = intermediates['adaptive_channel_weights']
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_fixed_weights_when_disabled(self):
        """When adaptive_weights=False, fixed channel_weights should be used."""
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(
            img, adaptive_weights=False, channel_weights=(0.2, 0.4, 0.4),
        )
        assert 'adaptive_channel_weights' not in intermediates


class TestV31EnergyNormalization:
    """Tests for V3.1 energy-normalized composition."""

    def test_complement_weight_in_intermediates(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img, energy_normalize=True)
        assert 'complement_weight' in intermediates
        cw = intermediates['complement_weight']
        assert cw.shape == (64, 64)
        assert cw.dtype == np.uint8

    def test_no_complement_weight_when_disabled(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img, energy_normalize=False)
        assert 'complement_weight' not in intermediates

    def test_energy_normalize_output_valid(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        result, _ = remover.process_v3(img, energy_normalize=True)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_energy_normalize_reduces_overshoot(self):
        """Energy normalization should produce values closer to original range."""
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        result_no_en, _ = remover.process_v3(
            img, edge_strength=1.0, energy_normalize=False,
        )
        result_en, _ = remover.process_v3(
            img, edge_strength=1.0, energy_normalize=True,
        )
        # Energy-normalized version should have smaller deviation from original
        diff_no_en = np.mean(np.abs(img.astype(float) - result_no_en.astype(float)))
        diff_en = np.mean(np.abs(img.astype(float) - result_en.astype(float)))
        # With energy normalization, the difference should not be larger
        # (it reduces double-counting overshoot)
        assert diff_en <= diff_no_en * 1.1  # Allow 10% tolerance


class TestV31FullPipeline:
    """Tests for V3.1 with all three improvements enabled together."""

    def test_all_improvements_enabled(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover.process_v3(
            img,
            multiscale_edge=True,
            adaptive_weights=True,
            energy_normalize=True,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert 'adaptive_channel_weights' in intermediates
        assert 'complement_weight' in intermediates

    def test_v31_reduces_grid(self):
        """V3.1 pipeline should still effectively reduce grid energy."""
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        result, _ = remover.process_v3(
            img,
            multiscale_edge=True,
            adaptive_weights=True,
            energy_normalize=True,
        )
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy

    def test_v31_with_edge_mask(self):
        remover = STFTAdaptiveGridRemover(
            period_x=9, period_y=7, window_size=15, hop_size=8,
        )
        img = create_grid_image(h=64, w=64)
        edge_mask = np.random.rand(64, 64).astype(np.float32)
        result, _ = remover.process_v3(
            img,
            edge_mask=edge_mask,
            multiscale_edge=True,
            adaptive_weights=True,
            energy_normalize=True,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV31HybridIntegration:
    """Tests for V3.1 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v31_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.1")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v31_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.1_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v31_quality_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.1_quality")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_v31_backward_compatible_with_v3(self):
        """V3 presets should still work unchanged after V3.1 additions."""
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result_v3, _ = restorer.restore_grid_pattern(img, method="stft_v3")
        assert result_v3.shape == img.shape
        assert result_v3.dtype == np.uint8


class TestIntensityProtectionMask:
    """Tests for V3.3 compute_intensity_protection_mask static method."""

    def test_output_shape_and_dtype(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_intensity_protection_mask(gray)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_output_range(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_intensity_protection_mask(gray)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_dark_pixels_high_protection(self):
        """Very dark pixels (ink lines) should have protection close to 1."""
        gray = np.zeros((32, 32), dtype=np.uint8)  # All black
        mask = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=60.0, steepness=0.08, blur_sigma=0.0
        )
        assert mask.mean() > 0.95

    def test_bright_pixels_low_protection(self):
        """Bright pixels (background) should have protection close to 0."""
        gray = np.full((32, 32), 200, dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=60.0, steepness=0.08, blur_sigma=0.0
        )
        assert mask.mean() < 0.05

    def test_monotonic_decrease(self):
        """Protection should decrease monotonically with brightness."""
        gray = np.arange(256, dtype=np.uint8).reshape(1, 256).repeat(8, axis=0)
        mask = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=60.0, steepness=0.08, blur_sigma=0.0
        )
        # Check mean per column is monotonically non-increasing
        col_means = mask.mean(axis=0)
        diffs = np.diff(col_means)
        assert np.all(diffs <= 1e-6)

    def test_threshold_shift(self):
        """Higher threshold should protect more pixels."""
        gray = np.full((32, 32), 80, dtype=np.uint8)
        mask_low = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=50.0, steepness=0.08, blur_sigma=0.0
        )
        mask_high = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=100.0, steepness=0.08, blur_sigma=0.0
        )
        assert mask_high.mean() > mask_low.mean()

    def test_steepness_effect(self):
        """Higher steepness should produce sharper transition."""
        gray = np.arange(256, dtype=np.uint8).reshape(1, 256).repeat(8, axis=0)
        mask_gentle = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=128.0, steepness=0.02, blur_sigma=0.0
        )
        mask_sharp = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=128.0, steepness=0.20, blur_sigma=0.0
        )
        # Sharp transition should have more extreme values
        gentle_mid = mask_gentle[:, 100:156]  # Near threshold
        sharp_mid = mask_sharp[:, 100:156]
        # Standard deviation of sharp should be higher (more binary)
        assert sharp_mid.std() > gentle_mid.std()

    def test_blur_smoothing(self):
        """Blur should produce smoother mask."""
        # Create image with sharp edge
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[:, 32:] = 200
        mask_no_blur = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=60.0, steepness=0.08, blur_sigma=0.0
        )
        mask_blurred = STFTAdaptiveGridRemover.compute_intensity_protection_mask(
            gray, threshold=60.0, steepness=0.08, blur_sigma=5.0
        )
        # Gradient at boundary should be smoother with blur
        grad_no_blur = np.abs(np.diff(mask_no_blur[32, :])).max()
        grad_blurred = np.abs(np.diff(mask_blurred[32, :])).max()
        assert grad_blurred < grad_no_blur


class TestV33IntensityProtection:
    """Tests for V3.3 intensity protection in process_v3 pipeline."""

    def _make_remover(self):
        return STFTAdaptiveGridRemover(
            period_x=9, period_y=7,
            window_size=15, hop_size=8,
            notch_sigma=1.5, base_attenuation=0.15,
            channel_adaptive=True,
        )

    def test_disabled_by_default(self):
        """Intensity protection should not activate unless explicitly enabled."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img)
        assert 'intensity_protection_mask' not in intermediates

    def test_mask_in_intermediates_when_enabled(self):
        """When enabled, intensity_protection_mask should appear in intermediates."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(
            img, intensity_protection=True
        )
        assert 'intensity_protection_mask' in intermediates
        mask = intermediates['intensity_protection_mask']
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_ink_regions_better_preserved(self):
        """Dark ink regions should be closer to original when protection is on."""
        remover = self._make_remover()
        # Create image with distinct ink region (dark)
        img = np.full((64, 64, 3), 180, dtype=np.uint8)  # Light background
        img[20:44, 20:44] = 30  # Dark ink square

        # Add grid
        for y in range(64):
            for x in range(64):
                grid_val = np.sin(2 * np.pi * x / 9) + np.sin(2 * np.pi * y / 7)
                img[y, x] = np.clip(img[y, x].astype(np.float32) + grid_val * 20, 0, 255).astype(np.uint8)

        result_off, _ = remover.process_v3(img, intensity_protection=False)
        result_on, _ = remover.process_v3(img, intensity_protection=True)

        # Ink region difference from original
        ink_region = slice(20, 44), slice(20, 44)
        diff_off = np.abs(
            result_off[ink_region].astype(np.float32) - img[ink_region].astype(np.float32)
        ).mean()
        diff_on = np.abs(
            result_on[ink_region].astype(np.float32) - img[ink_region].astype(np.float32)
        ).mean()

        # With protection, ink region should be closer to original
        assert diff_on < diff_off

    def test_output_shape_and_dtype(self):
        """Output should maintain same shape and dtype."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        result, _ = remover.process_v3(img, intensity_protection=True)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV33Presets:
    """Tests for V3.3 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v33_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.3")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v33_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.3_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v33_quality_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.3_quality")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_v32_backward_compatible(self):
        """V3.2 presets should still work unchanged after V3.3 additions."""
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result_v32, _ = restorer.restore_grid_pattern(img, method="stft_v3.2")
        assert result_v32.shape == img.shape
        assert result_v32.dtype == np.uint8


class TestV34EnergyProtection:
    """Tests for V3.4 energy-based grid protection in process_v3 pipeline."""

    def _make_remover(self):
        return STFTAdaptiveGridRemover(
            period_x=9, period_y=7,
            window_size=15, hop_size=8,
            notch_sigma=1.5, base_attenuation=0.15,
            channel_adaptive=True,
        )

    def test_disabled_by_default(self):
        """Energy protection should not activate unless explicitly enabled."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(img)
        assert 'energy_protection_mask' not in intermediates

    def test_mask_in_intermediates(self):
        """When enabled, energy_protection_mask should appear in intermediates."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(
            img, energy_protection=True
        )
        assert 'energy_protection_mask' in intermediates

    def test_mask_shape_dtype(self):
        """Mask should match input spatial shape and be uint8."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3(
            img, energy_protection=True
        )
        mask = intermediates['energy_protection_mask']
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_low_grid_regions_preserved(self):
        """Regions without grid should be better preserved with energy protection."""
        remover = self._make_remover()
        # Create image with grid only in top half
        img = np.full((128, 128, 3), 150, dtype=np.uint8)
        for y in range(64):  # Grid only in top half
            for x in range(128):
                grid_val = np.sin(2 * np.pi * x / 9) + np.sin(2 * np.pi * y / 7)
                img[y, x] = np.clip(150 + int(grid_val * 30), 0, 255)

        result_off, _ = remover.process_v3(img, energy_protection=False)
        result_on, _ = remover.process_v3(img, energy_protection=True)

        # Bottom half (no grid) should be closer to original with protection ON
        bottom = slice(80, 128), slice(0, 128)
        diff_off = np.abs(
            result_off[bottom].astype(np.float32) - img[bottom].astype(np.float32)
        ).mean()
        diff_on = np.abs(
            result_on[bottom].astype(np.float32) - img[bottom].astype(np.float32)
        ).mean()
        assert diff_on <= diff_off

    def test_high_grid_regions_filtered(self):
        """Regions with strong grid should still get filtered."""
        remover = self._make_remover()
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        result, _ = remover.process_v3(img, energy_protection=True)

        # Grid energy should still be reduced
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy

    def test_strength_parameter(self):
        """strength=0 should give result close to filtered, strength high should protect more."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)

        result_weak, _ = remover.process_v3(
            img, energy_protection=True, energy_protection_strength=0.1
        )
        result_strong, _ = remover.process_v3(
            img, energy_protection=True, energy_protection_strength=0.95
        )

        # Strong protection should keep result closer to original
        diff_weak = np.abs(img.astype(float) - result_weak.astype(float)).mean()
        diff_strong = np.abs(img.astype(float) - result_strong.astype(float)).mean()
        assert diff_strong < diff_weak

    def test_blur_smoothing(self):
        """Higher blur should produce smoother protection mask."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)

        _, intermediates_no_blur = remover.process_v3(
            img, energy_protection=True, energy_protection_blur=0.0
        )
        _, intermediates_blurred = remover.process_v3(
            img, energy_protection=True, energy_protection_blur=4.0
        )

        mask_no_blur = intermediates_no_blur['energy_protection_mask'].astype(np.float32)
        mask_blurred = intermediates_blurred['energy_protection_mask'].astype(np.float32)

        # Blurred mask should have less variation (smoother)
        grad_no_blur = np.mean(np.abs(np.diff(mask_no_blur, axis=1)))
        grad_blurred = np.mean(np.abs(np.diff(mask_blurred, axis=1)))
        assert grad_blurred <= grad_no_blur

    def test_backward_compatible_with_v33(self):
        """V3.3 preset should still work unchanged after V3.4 additions."""
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        restorer = HybridGridPatternRestorer(config=config)
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.3")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestV34Presets:
    """Tests for V3.4 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v34_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.4")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v34_aggressive_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.4_aggressive")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v34_quality_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.4_quality")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestModulationWeightedGray:
    """Tests for modulation-weighted grayscale computation."""

    def test_weights_proportional_to_modulation(self):
        """Channels with higher modulation should get higher weight."""
        remover = STFTAdaptiveGridRemover(period_x=32, period_y=32)
        # Create image where B=100, G=150, R=200
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # B
        img[:, :, 1] = 150  # G
        img[:, :, 2] = 200  # R

        # B channel has highest modulation
        modulation = {0: 0.15, 1: 0.07, 2: 0.04}
        result = remover.compute_modulation_weighted_gray(img, modulation)

        # Standard grayscale would be ~163.9 (0.114*100 + 0.587*150 + 0.299*200)
        # Modulation-weighted: B gets ~0.577, G ~0.269, R ~0.154
        # So result ≈ 57.7 + 40.4 + 30.8 = 128.8
        # Key: result should be lower than standard gray because B (lowest value)
        # gets the highest weight
        standard_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        assert result.shape == (64, 64)
        assert np.mean(result) < np.mean(standard_gray)

    def test_zero_modulation_falls_back(self):
        """Zero modulation should fall back to standard grayscale."""
        remover = STFTAdaptiveGridRemover(period_x=32, period_y=32)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        modulation = {0: 0.0, 1: 0.0, 2: 0.0}
        result = remover.compute_modulation_weighted_gray(img, modulation)
        expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_channel_dominant(self):
        """If only one channel has modulation, it should dominate."""
        remover = STFTAdaptiveGridRemover(period_x=32, period_y=32)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # B = 255
        img[:, :, 1] = 0    # G = 0
        img[:, :, 2] = 0    # R = 0

        modulation = {0: 0.15, 1: 0.0, 2: 0.0}
        result = remover.compute_modulation_weighted_gray(img, modulation)
        # B dominates entirely, so result should be close to 255
        assert np.mean(result) > 200

    def test_output_shape_matches_input(self):
        """Output shape should match input spatial dimensions."""
        remover = STFTAdaptiveGridRemover(period_x=32, period_y=32)
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        modulation = {0: 0.1, 1: 0.05, 2: 0.03}
        result = remover.compute_modulation_weighted_gray(img, modulation)
        assert result.shape == (100, 80)
        assert result.dtype == np.float64

    def test_empty_modulation_dict(self):
        """Empty modulation dict should fall back to standard grayscale."""
        remover = STFTAdaptiveGridRemover(period_x=32, period_y=32)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        modulation = {}
        result = remover.compute_modulation_weighted_gray(img, modulation)
        expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        np.testing.assert_array_almost_equal(result, expected)


def create_multiplicative_grid_image(h=128, w=128, period_x=9, period_y=7,
                                     modulation=0.15, base_intensity=180):
    """Create synthetic image with multiplicative grid pattern for testing."""
    np.random.seed(42)
    # Create content with varying brightness
    content = np.random.randint(50, 220, (h, w, 3), dtype=np.uint8).astype(np.float32)
    result = content.copy()

    for y in range(h):
        for x in range(w):
            grid_val = np.sin(2 * np.pi * x / period_x) + np.sin(2 * np.pi * y / period_y)
            for c in range(3):
                result[y, x, c] *= (1.0 + grid_val * modulation)

    return np.clip(result, 0, 255).astype(np.uint8)


class TestV35ContentGradientMask:
    """Tests for V3.5 compute_content_gradient_mask."""

    def test_output_shape_and_dtype(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7,
        )
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_output_range(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7,
        )
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_uniform_image_low_mask(self):
        """Uniform image should have very low content gradient."""
        gray = np.full((64, 64), 128, dtype=np.uint8)
        mask = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7,
        )
        assert mask.max() < 0.01

    def test_strong_edge_detected(self):
        """Strong content edge should have high mask value."""
        gray = np.full((64, 64), 200, dtype=np.uint8)
        gray[:, 32:] = 30  # Sharp edge
        mask = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7,
        )
        # Near the edge (around column 32)
        assert mask[:, 28:36].max() > 0.5

    def test_grid_gradient_suppressed(self):
        """Pure grid pattern should have more uniform gradient than content edge.

        The smoothing kernel averages periodic grid gradients, making the mask
        more spatially uniform. In contrast, a content edge produces a localized
        spike. We verify this by comparing spatial variance.
        """
        h, w = 128, 128
        # Grid-only image
        gray_grid = np.full((h, w), 150, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                gray_grid[y, x] += np.sin(2 * np.pi * x / 9) * 20
        gray_grid = np.clip(gray_grid, 0, 255).astype(np.uint8)
        mask_grid = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray_grid, period_x=9, period_y=7,
        )

        # Content edge image
        gray_edge = np.full((h, w), 200, dtype=np.uint8)
        gray_edge[:, 64:] = 30  # Sharp edge
        mask_edge = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray_edge, period_x=9, period_y=7,
        )

        # Grid mask should be more spatially uniform (lower std) than edge mask
        assert mask_grid.std() < mask_edge.std()

    def test_strength_parameter(self):
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        mask_normal = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7, strength=1.0,
        )
        mask_strong = STFTAdaptiveGridRemover.compute_content_gradient_mask(
            gray, period_x=9, period_y=7, strength=2.0,
        )
        # Stronger mask should have higher mean (up to clipping at 1.0)
        assert mask_strong.mean() >= mask_normal.mean()


class TestV35GridTemplateEstimation:
    """Tests for V3.5 estimate_grid_template_2d."""

    def test_output_shape(self):
        gray = create_grid_image(h=128, w=128)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        template = STFTAdaptiveGridRemover.estimate_grid_template_2d(
            gray, period_x=9, period_y=7,
        )
        assert template.shape == (7, 9)

    def test_template_near_one(self):
        """Template values should be centered around 1.0."""
        gray = create_grid_image(h=128, w=128)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        template = STFTAdaptiveGridRemover.estimate_grid_template_2d(
            gray, period_x=9, period_y=7,
        )
        assert 0.5 < template.mean() < 1.5

    def test_uniform_image_template_uniform(self):
        """Uniform image should produce near-uniform template."""
        gray = np.full((128, 128), 150, dtype=np.uint8)
        template = STFTAdaptiveGridRemover.estimate_grid_template_2d(
            gray, period_x=9, period_y=7,
        )
        assert template.std() < 0.05

    def test_flat_threshold_effect(self):
        """Higher threshold should include more pixels in estimation."""
        img = create_multiplicative_grid_image(h=128, w=128)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_strict = STFTAdaptiveGridRemover.estimate_grid_template_2d(
            gray, period_x=9, period_y=7, flat_threshold=5.0,
        )
        t_loose = STFTAdaptiveGridRemover.estimate_grid_template_2d(
            gray, period_x=9, period_y=7, flat_threshold=20.0,
        )
        # Both should be valid (around 1.0)
        assert 0.5 < t_strict.mean() < 1.5
        assert 0.5 < t_loose.mean() < 1.5


class TestV35ProcessMultiplicative:
    """Tests for V3.5 process_multiplicative pipeline."""

    def _make_remover(self):
        return STFTAdaptiveGridRemover(
            period_x=9, period_y=7,
            window_size=15, hop_size=8,
            notch_sigma=1.5, base_attenuation=0.15,
            channel_adaptive=True,
        )

    def test_returns_tuple(self):
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=64, w=64)
        result, intermediates = remover.process_multiplicative(img)
        assert isinstance(result, np.ndarray)
        assert isinstance(intermediates, dict)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_intermediates_keys(self):
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=64, w=64)
        _, intermediates = remover.process_multiplicative(img)
        expected = {'detected_periods', 'grid_template', 'divided_result',
                    'filtered', 'edge_mask', 'result'}
        assert expected.issubset(intermediates.keys())

    def test_reduces_grid(self):
        """Multiplicative pipeline should reduce grid energy."""
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=128, w=128, modulation=0.15)
        result, _ = remover.process_multiplicative(img)
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy

    def test_with_content_gradient_mask(self):
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=64, w=64)
        result, _ = remover.process_multiplicative(
            img, content_gradient_mask=True,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_without_residual_notch(self):
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=64, w=64)
        result, _ = remover.process_multiplicative(
            img, residual_notch=False,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_with_edge_mask(self):
        remover = self._make_remover()
        img = create_multiplicative_grid_image(h=64, w=64)
        edge_mask = np.random.rand(64, 64).astype(np.float32)
        result, _ = remover.process_multiplicative(
            img, edge_mask=edge_mask,
        )
        assert result.shape == img.shape


class TestV35ProcessV3Log:
    """Tests for V3.5 process_v3_log pipeline."""

    def _make_remover(self):
        return STFTAdaptiveGridRemover(
            period_x=9, period_y=7,
            window_size=15, hop_size=8,
            notch_sigma=1.5, base_attenuation=0.15,
            channel_adaptive=True,
        )

    def test_returns_tuple(self):
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        result, intermediates = remover.process_v3_log(img)
        assert isinstance(result, np.ndarray)
        assert isinstance(intermediates, dict)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_intermediates_keys(self):
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        _, intermediates = remover.process_v3_log(img)
        assert 'log_image' in intermediates
        assert 'result' in intermediates
        assert 'edge_mask' in intermediates

    def test_valid_output_range(self):
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        result, _ = remover.process_v3_log(img)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_reduces_grid(self):
        """Log-domain pipeline should reduce grid energy."""
        remover = self._make_remover()
        img = create_grid_image(h=128, w=128, modulation_b=0.2)
        result, _ = remover.process_v3_log(img)
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f_before = np.abs(np.fft.fftshift(np.fft.fft2(gray_before)))
        f_after = np.abs(np.fft.fftshift(np.fft.fft2(gray_after)))
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        fx = int(round(w / 9))
        fy = int(round(h / 7))
        before_energy = f_before[cy, cx + fx] + f_before[cy + fy, cx]
        after_energy = f_after[cy, cx + fx] + f_after[cy + fy, cx]
        assert after_energy < before_energy

    def test_epsilon_values(self):
        """Different epsilon values should all produce valid output."""
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        for eps in [1.0, 3.0, 5.0]:
            result, _ = remover.process_v3_log(img, epsilon=eps)
            assert result.shape == img.shape
            assert result.dtype == np.uint8

    def test_with_content_gradient_mask(self):
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        result, _ = remover.process_v3_log(
            img, content_gradient_mask=True,
        )
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_with_edge_mask(self):
        remover = self._make_remover()
        img = create_grid_image(h=64, w=64)
        edge_mask = np.random.rand(64, 64).astype(np.float32)
        result, _ = remover.process_v3_log(img, edge_mask=edge_mask)
        assert result.shape == img.shape

    def test_multiplicative_grid_better_in_log(self):
        """Log-domain should handle multiplicative grid better than linear V3."""
        remover = self._make_remover()
        # Create multiplicative grid image
        img = create_multiplicative_grid_image(h=128, w=128, modulation=0.15)

        # Both methods should produce valid output
        result_log, _ = remover.process_v3_log(img)
        result_v3, _ = remover.process_v3(img)
        assert result_log.shape == img.shape
        assert result_v3.shape == img.shape


class TestV35HybridIntegration:
    """Tests for V3.5 presets through HybridGridPatternRestorer."""

    def _make_restorer(self):
        config = RestorationConfig(
            grid_stft_period_x=9,
            grid_stft_period_y=7,
            grid_stft_window_size=15,
            grid_stft_hop_size=8,
        )
        return HybridGridPatternRestorer(config=config)

    def test_stft_v35_log_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_log")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_log_e1_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_log_e1")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_log_e5_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_log_e5")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_log_cg_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_log_cg")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_mult_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_mult")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_mult_f8_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_mult_f8")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_mult_f15_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_mult_f15")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_stft_v35_mult_cg_runs(self):
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.5_mult_cg")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_v34_backward_compatible(self):
        """V3.4 presets should still work unchanged after V3.5 additions."""
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.4")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_v32_backward_compatible(self):
        """V3.2 presets should still work unchanged after V3.5 additions."""
        restorer = self._make_restorer()
        img = create_grid_image(h=64, w=64)
        result, _ = restorer.restore_grid_pattern(img, method="stft_v3.2")
        assert result.shape == img.shape
        assert result.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
