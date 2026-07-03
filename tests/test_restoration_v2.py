"""Tests for Segment-Aware Restoration Module (restoration_v2).

Tests per-object restoration pipeline including:
- Config defaults and overrides
- Ink line detection and preservation
- Fading restoration (bilateral + neural fallback)
- Boundary blending / feathering
- Full orchestrator pipeline
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kp3d.modules.restoration_v2 import (
    SegmentAwareRestorer,
    SegmentAwareRestorationConfig,
    ObjectMask,
    RestorationResult,
)
from kp3d.modules.restoration_v2.utils import (
    CropRegion,
    crop_object_region,
    paste_crop_back,
    detect_ink_mask,
    compute_mask_area,
    normalize_mask_to_float,
)
from kp3d.modules.restoration_v2.fading_restorer import FadingRestorer
from kp3d.modules.restoration_v2.boundary_blender import BoundaryBlender
from kp3d.modules.restoration_v2.per_object_restorer import PerObjectRestorer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_image():
    """Create a 256x256 BGR test image with colored regions."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background: gray
    img[:, :] = (128, 128, 128)
    # Object 1 region: reddish (simulates faded pigment)
    img[50:150, 50:150] = (60, 80, 180)
    # Ink lines: very dark strokes within object
    img[70:72, 60:140] = (10, 10, 10)
    img[100:102, 60:140] = (10, 10, 10)
    img[60:140, 90:92] = (10, 10, 10)
    return img


@pytest.fixture
def object_mask():
    """Binary mask for object region (50:150, 50:150)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    return mask


@pytest.fixture
def small_mask():
    """Very small mask (below min area threshold)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:105, 100:105] = 255  # 25 pixels
    return mask


@pytest.fixture
def config():
    """Default config for testing."""
    return SegmentAwareRestorationConfig()


@pytest.fixture
def config_no_neural():
    """Config with bilateral only (no neural)."""
    return SegmentAwareRestorationConfig(fading_method="bilateral_only")


# ============================================================
# Test: Config
# ============================================================

class TestConfig:
    def test_default_values(self):
        cfg = SegmentAwareRestorationConfig()
        assert cfg.crop_padding_px == 32
        assert cfg.min_object_area_px == 500
        assert cfg.fading_method == "cq"
        # v12 CQ defaults
        assert cfg.cq_k_min == 35
        assert cfg.cq_k_max == 80
        assert cfg.cq_pre_filter == "rolling_guidance"
        assert cfg.cq_min_crop_size == 64
        # Bilateral fallback defaults
        assert cfg.bilateral_d == 15
        assert cfg.bilateral_iterations == 2
        assert cfg.use_guided_filter is True
        assert cfg.neural_strength == 0.4
        assert cfg.ink_l_threshold == 25.0
        assert cfg.ink_protection_strength == 0.85
        assert cfg.ink_morph_open_size == 3
        assert cfg.feather_radius_px == 5
        assert cfg.skip_background is True
        assert cfg.use_clahe is False
        assert cfg.clahe_clip_limit == 3.0
        assert cfg.clahe_grid_size == 4

    def test_custom_values(self):
        cfg = SegmentAwareRestorationConfig(
            crop_padding_px=16,
            neural_strength=0.5,
            feather_radius_px=5,
        )
        assert cfg.crop_padding_px == 16
        assert cfg.neural_strength == 0.5
        assert cfg.feather_radius_px == 5

    def test_config_from_dict(self):
        """Test creating config from dict (as used in pipeline integration)."""
        params = {"neural_strength": 0.3, "feather_radius_px": 7}
        cfg = SegmentAwareRestorationConfig(**params)
        assert cfg.neural_strength == 0.3
        assert cfg.feather_radius_px == 7
        # Other values remain default
        assert cfg.bilateral_d == 15


# ============================================================
# Test: Utils
# ============================================================

class TestUtils:
    def test_crop_object_region(self, sample_image, object_mask):
        crop = crop_object_region(sample_image, object_mask, padding=10)
        assert crop.x1 == 40
        assert crop.y1 == 40
        assert crop.x2 == 160
        assert crop.y2 == 160
        assert crop.crop_image.shape == (120, 120, 3)
        assert crop.crop_mask.shape == (120, 120)

    def test_crop_with_edge_clipping(self):
        """Test crop near image edge doesn't exceed bounds."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:20, 0:20] = 255  # Top-left corner

        crop = crop_object_region(img, mask, padding=50)
        assert crop.x1 == 0
        assert crop.y1 == 0
        assert crop.x2 <= 100
        assert crop.y2 <= 100

    def test_crop_empty_mask_raises(self, sample_image):
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        with pytest.raises(ValueError, match="Empty mask"):
            crop_object_region(sample_image, empty_mask)

    def test_paste_crop_back(self, sample_image, object_mask):
        crop = crop_object_region(sample_image, object_mask, padding=10)
        # Create a "restored" version (just brighten)
        restored = np.clip(crop.crop_image.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        blend = normalize_mask_to_float(crop.crop_mask)

        original = sample_image.copy()
        result = paste_crop_back(sample_image, crop, restored, blend)

        # Pixels inside mask should differ from original
        assert not np.array_equal(result[80, 80], original[80, 80])
        # Pixels outside mask should be unchanged
        assert np.array_equal(result[10, 10], original[10, 10])

    def test_detect_ink_mask(self, sample_image, object_mask):
        ink = detect_ink_mask(sample_image, l_threshold=40.0, mask=object_mask)
        # Should detect the dark ink lines we put in
        assert ink.dtype == np.uint8
        assert np.any(ink[70:72, 60:140] > 0)  # Horizontal ink line
        assert np.any(ink[60:140, 90:92] > 0)  # Vertical ink line
        # Should not detect outside object mask
        assert np.all(ink[0:50, :] == 0)

    def test_compute_mask_area(self, object_mask, small_mask):
        assert compute_mask_area(object_mask) == 100 * 100  # 10000 pixels
        assert compute_mask_area(small_mask) == 5 * 5  # 25 pixels

    def test_normalize_mask_to_float(self, object_mask):
        result = normalize_mask_to_float(object_mask)
        assert result.dtype == np.float32
        assert result.max() == 1.0
        assert result.min() == 0.0


# ============================================================
# Test: BoundaryBlender
# ============================================================

class TestBoundaryBlender:
    def test_feathered_mask_shape(self, config, object_mask):
        blender = BoundaryBlender(config)
        feathered = blender.create_feathered_mask(object_mask)
        assert feathered.shape == object_mask.shape
        assert feathered.dtype == np.float32
        assert feathered.max() <= 1.0
        assert feathered.min() >= 0.0

    def test_feathered_mask_has_gradient(self, config, object_mask):
        blender = BoundaryBlender(config)
        feathered = blender.create_feathered_mask(object_mask)
        # Should have intermediate values at boundary (not just 0/1)
        unique_vals = np.unique(feathered)
        assert len(unique_vals) > 2  # More than just 0 and 1

    def test_no_feathering_when_radius_zero(self, object_mask):
        cfg = SegmentAwareRestorationConfig(feather_radius_px=0)
        blender = BoundaryBlender(cfg)
        feathered = blender.create_feathered_mask(object_mask)
        # Should be hard mask (only 0.0 and 1.0)
        unique_vals = set(np.unique(feathered).tolist())
        assert unique_vals == {0.0, 1.0}

    def test_ink_protection_mask(self, config, sample_image, object_mask):
        blender = BoundaryBlender(config)
        ink = detect_ink_mask(sample_image, l_threshold=40.0, mask=object_mask)
        protection = blender.create_ink_protection_mask(ink, protection_strength=0.9)

        # Ink regions should have low blend values (close to 0 → keep original)
        ink_pixels = ink > 0
        if np.any(ink_pixels):
            assert protection[ink_pixels].mean() < 0.2

        # Non-ink regions should have high blend values (close to 1 → use restored)
        non_ink = (ink == 0) & (object_mask > 0)
        if np.any(non_ink):
            assert protection[non_ink].mean() > 0.9

    def test_composite_preserves_ink(self, config, sample_image, object_mask):
        blender = BoundaryBlender(config)
        ink = detect_ink_mask(sample_image, l_threshold=40.0, mask=object_mask)

        # Create obviously different "restored" image
        restored = np.ones_like(sample_image) * 200

        result = blender.composite_with_ink_protection(
            sample_image, restored, object_mask, ink
        )

        # Ink regions should be close to original (high protection)
        ink_mask_bool = ink > 0
        if np.any(ink_mask_bool):
            orig_ink_vals = sample_image[ink_mask_bool].astype(np.float32)
            result_ink_vals = result[ink_mask_bool].astype(np.float32)
            # SSIM-like check: ink pixels should be very similar to original
            diff = np.abs(orig_ink_vals - result_ink_vals).mean()
            assert diff < 30  # Small difference due to protection not being 100%


# ============================================================
# Test: FadingRestorer
# ============================================================

class TestFadingRestorer:
    def test_bilateral_only(self):
        """Test bilateral filtering modifies non-ink object pixels."""
        # Create image with noisy region that bilateral will clearly change
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add random noise in object area (bilateral will smooth this)
        rng = np.random.default_rng(42)
        img[30:100, 30:100] = rng.integers(100, 200, size=(70, 70, 3), dtype=np.uint8)

        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:100, 30:100] = 255

        cfg = SegmentAwareRestorationConfig(fading_method="bilateral_only")
        restorer = FadingRestorer(cfg)

        # Use empty ink mask (no ink) so bilateral applies to all of object
        ink = np.zeros_like(mask)
        result = restorer.restore(img, mask, ink)

        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # Bilateral on noisy region should produce different output
        assert not np.array_equal(result[30:100, 30:100], img[30:100, 30:100])

    def test_ink_regions_unchanged(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(fading_method="bilateral_only")
        restorer = FadingRestorer(cfg)

        ink = detect_ink_mask(sample_image, mask=object_mask)
        result = restorer.restore(sample_image, object_mask, ink)

        # Ink regions should be exactly original (process_mask excludes ink)
        ink_bool = ink > 0
        if np.any(ink_bool):
            np.testing.assert_array_equal(result[ink_bool], sample_image[ink_bool])

    def test_outside_mask_unchanged(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(fading_method="bilateral_only")
        restorer = FadingRestorer(cfg)

        ink = detect_ink_mask(sample_image, mask=object_mask)
        result = restorer.restore(sample_image, object_mask, ink)

        # Outside object mask should be unchanged
        outside = object_mask == 0
        np.testing.assert_array_equal(result[outside], sample_image[outside])


# ============================================================
# Test: PerObjectRestorer
# ============================================================

class TestPerObjectRestorer:
    def test_restore_object_basic(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(fading_method="bilateral_only")
        restorer = PerObjectRestorer(cfg)

        # Crop first (as orchestrator would)
        crop = crop_object_region(sample_image, object_mask, padding=cfg.crop_padding_px)
        result = restorer.restore_object(crop.crop_image, crop.crop_mask)

        assert result.shape == crop.crop_image.shape
        assert result.dtype == np.uint8

    def test_ink_preservation_ssim(self, sample_image, object_mask):
        """Verify ink lines are better preserved than non-ink regions."""
        cfg = SegmentAwareRestorationConfig(fading_method="bilateral_only")
        restorer = PerObjectRestorer(cfg)

        crop = crop_object_region(sample_image, object_mask, padding=cfg.crop_padding_px)
        result = restorer.restore_object(crop.crop_image, crop.crop_mask)

        # Detect ink in the crop (after morph cleanup, same as pipeline)
        ink_raw = detect_ink_mask(crop.crop_image, l_threshold=cfg.ink_l_threshold, mask=crop.crop_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.ink_morph_open_size, cfg.ink_morph_open_size))
        ink = cv2.morphologyEx(ink_raw, cv2.MORPH_OPEN, kernel)
        ink_bool = ink > 0
        non_ink_bool = (crop.crop_mask > 0) & ~ink_bool

        if np.any(ink_bool) and np.any(non_ink_bool):
            # Ink regions should change LESS than non-ink regions
            ink_mae = np.abs(crop.crop_image[ink_bool].astype(np.float32) - result[ink_bool].astype(np.float32)).mean()
            non_ink_mae = np.abs(crop.crop_image[non_ink_bool].astype(np.float32) - result[non_ink_bool].astype(np.float32)).mean()
            assert ink_mae < non_ink_mae, f"Ink changed more than non-ink: ink_mae={ink_mae:.1f}, non_ink_mae={non_ink_mae:.1f}"


# ============================================================
# Test: SegmentAwareRestorer (Orchestrator)
# ============================================================

class TestSegmentAwareRestorer:
    def test_restore_all_objects_basic(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            min_object_area_px=100,
        )
        restorer = SegmentAwareRestorer(config=cfg)

        masks = [ObjectMask(label="object1", mask=object_mask, is_background=False)]
        result = restorer.restore_all_objects(sample_image, masks)

        assert isinstance(result, RestorationResult)
        assert result.restored_image.shape == sample_image.shape
        assert result.objects_processed == 1
        assert result.objects_skipped == 0
        assert "object1" in result.per_object_times

    def test_skip_background(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            skip_background=True,
        )
        restorer = SegmentAwareRestorer(config=cfg)

        masks = [ObjectMask(label="bg", mask=object_mask, is_background=True)]
        result = restorer.restore_all_objects(sample_image, masks)

        assert result.objects_processed == 0
        assert result.objects_skipped == 1
        # Image should be unchanged
        np.testing.assert_array_equal(result.restored_image, sample_image)

    def test_skip_small_objects(self, sample_image, small_mask):
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            min_object_area_px=500,  # small_mask has only 25 pixels
        )
        restorer = SegmentAwareRestorer(config=cfg)

        masks = [ObjectMask(label="tiny", mask=small_mask, is_background=False)]
        result = restorer.restore_all_objects(sample_image, masks)

        assert result.objects_processed == 0
        assert result.objects_skipped == 1

    def test_multiple_objects(self, sample_image):
        """Test processing multiple objects independently."""
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            min_object_area_px=100,
        )
        restorer = SegmentAwareRestorer(config=cfg)

        # Create two non-overlapping object masks
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[20:80, 20:80] = 255

        mask2 = np.zeros((256, 256), dtype=np.uint8)
        mask2[150:220, 150:220] = 255

        masks = [
            ObjectMask(label="obj1", mask=mask1, is_background=False),
            ObjectMask(label="obj2", mask=mask2, is_background=False),
        ]
        result = restorer.restore_all_objects(sample_image, masks)

        assert result.objects_processed == 2
        assert result.objects_skipped == 0
        assert "obj1" in result.per_object_times
        assert "obj2" in result.per_object_times

    def test_result_metadata(self, sample_image, object_mask):
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            min_object_area_px=100,
        )
        restorer = SegmentAwareRestorer(config=cfg)

        masks = [ObjectMask(label="vase", mask=object_mask)]
        result = restorer.restore_all_objects(sample_image, masks)

        assert "total_time" in result.metadata
        assert "config" in result.metadata
        assert result.metadata["config"]["fading_method"] == "bilateral_only"

    def test_does_not_modify_outside_masks(self, sample_image, object_mask):
        """Pixels outside all object masks should remain unchanged."""
        cfg = SegmentAwareRestorationConfig(
            fading_method="bilateral_only",
            min_object_area_px=100,
            feather_radius_px=0,  # No feathering for clean test
        )
        restorer = SegmentAwareRestorer(config=cfg)

        masks = [ObjectMask(label="obj", mask=object_mask)]
        result = restorer.restore_all_objects(sample_image, masks)

        # Outside the mask (with some buffer for padding crop overlap)
        # Check far away from mask boundary
        np.testing.assert_array_equal(
            result.restored_image[0:30, 0:30],
            sample_image[0:30, 0:30],
        )


# ============================================================
# Test: Pipeline Integration (import check)
# ============================================================

class TestPipelineIntegration:
    def test_occlusion_config_has_fields(self):
        """Verify OcclusionConfig has the new fields."""
        from kp3d.modules.occlusion.base import OcclusionConfig
        cfg = OcclusionConfig()
        assert hasattr(cfg, 'use_segment_aware_restoration')
        assert hasattr(cfg, 'segment_aware_restoration_config')
        assert cfg.use_segment_aware_restoration is False
        assert cfg.segment_aware_restoration_config is None

    def test_occlusion_config_enable(self):
        """Verify OcclusionConfig can enable segment-aware restoration."""
        from kp3d.modules.occlusion.base import OcclusionConfig
        cfg = OcclusionConfig(
            use_segment_aware_restoration=True,
            segment_aware_restoration_config={"neural_strength": 0.3},
        )
        assert cfg.use_segment_aware_restoration is True
        assert cfg.segment_aware_restoration_config == {"neural_strength": 0.3}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
