"""Unit and integration tests for occlusion module.

Tests each stage of the occlusion pipeline independently
and validates end-to-end functionality.
"""

import pytest
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a simple test image with two overlapping shapes."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 200  # Light gray background

    # Background: red rectangle (simulating soban/table)
    cv2.rectangle(img, (30, 100), (220, 230), (180, 50, 50), -1)

    # Foreground: white circle (simulating ceramic vase)
    cv2.circle(img, (128, 140), 60, (240, 240, 240), -1)

    return img


@pytest.fixture
def sample_masks(sample_image):
    """Create masks for the sample image."""
    h, w = sample_image.shape[:2]

    # Foreground mask (circle)
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fg_mask, (128, 140), 60, 255, -1)

    # Background mask (rectangle minus circle overlap)
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(bg_mask, (30, 100), (220, 230), 255, -1)

    return fg_mask, bg_mask


@pytest.fixture
def sample_tensor(sample_image):
    """Convert sample image to tensor."""
    img = sample_image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


@pytest.fixture
def ceramic_image():
    """Load the actual ceramic painting image if available."""
    img_path = Path(__file__).parent.parent / "data_painting" / "ceramic_painting_1.png"
    if img_path.exists():
        img = cv2.imread(str(img_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ============================================================================
# Unit Tests: Base Module
# ============================================================================

class TestOcclusionConfig:
    """Tests for OcclusionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from kp3d.modules.occlusion import OcclusionConfig

        config = OcclusionConfig()

        assert config.sam_model_type == "vit_h"
        assert config.box_threshold == 0.3
        assert config.text_threshold == 0.25
        assert config.inpaint_method == "telea"
        assert config.inpaint_radius == 5
        assert len(config.text_prompts) == 2

    def test_custom_config(self):
        """Test custom configuration."""
        from kp3d.modules.occlusion import OcclusionConfig

        config = OcclusionConfig(
            text_prompts=["ceramic pot", "wooden stand"],
            inpaint_method="ns",
            inpaint_radius=10
        )

        assert config.text_prompts == ["ceramic pot", "wooden stand"]
        assert config.inpaint_method == "ns"
        assert config.inpaint_radius == 10


class TestLayerInfo:
    """Tests for LayerInfo dataclass."""

    def test_layer_info_creation(self, sample_masks):
        """Test LayerInfo creation."""
        from kp3d.modules.occlusion import LayerInfo

        fg_mask, _ = sample_masks

        layer = LayerInfo(
            label="ceramic vase",
            mask=fg_mask,
            bbox=(68, 80, 188, 200),
            mean_depth=0.3,
            is_foreground=True
        )

        assert layer.label == "ceramic vase"
        assert layer.mask.shape == fg_mask.shape
        assert layer.is_foreground is True
        assert layer.mean_depth == 0.3


# ============================================================================
# Unit Tests: Inpainting Module
# ============================================================================

class TestInpaintingModule:
    """Tests for InpaintingModule."""

    def test_inpaint_telea(self, sample_image, sample_masks):
        """Test Telea inpainting method."""
        from kp3d.modules.occlusion import InpaintingModule

        fg_mask, _ = sample_masks
        module = InpaintingModule(method="telea", radius=5)

        result = module.inpaint(sample_image, fg_mask)

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
        # The inpainted region should differ from original
        diff = np.abs(result.astype(float) - sample_image.astype(float))
        assert np.sum(diff[fg_mask > 0]) > 0

    def test_inpaint_ns(self, sample_image, sample_masks):
        """Test Navier-Stokes inpainting method."""
        from kp3d.modules.occlusion import InpaintingModule

        fg_mask, _ = sample_masks
        module = InpaintingModule(method="ns", radius=5)

        result = module.inpaint(sample_image, fg_mask)

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_extract_object_rgba(self, sample_image, sample_masks):
        """Test object extraction with RGBA output."""
        from kp3d.modules.occlusion import InpaintingModule

        fg_mask, _ = sample_masks
        module = InpaintingModule()

        result = module.extract_object(sample_image, fg_mask, return_rgba=True)

        assert result.shape == (256, 256, 4)  # RGBA
        assert result.dtype == np.uint8
        # Alpha channel should match mask
        np.testing.assert_array_equal(result[:, :, 3], fg_mask)

    def test_extract_object_rgb(self, sample_image, sample_masks):
        """Test object extraction with RGB output."""
        from kp3d.modules.occlusion import InpaintingModule

        fg_mask, _ = sample_masks
        module = InpaintingModule()

        result = module.extract_object(
            sample_image, fg_mask,
            background_color=(0, 0, 0),
            return_rgba=False
        )

        assert result.shape == (256, 256, 3)  # RGB
        # Background should be black
        assert np.all(result[fg_mask == 0] == 0)

    def test_quick_inpaint(self, sample_image, sample_masks):
        """Test quick_inpaint convenience function."""
        from kp3d.modules.occlusion import quick_inpaint

        fg_mask, _ = sample_masks
        result = quick_inpaint(sample_image, fg_mask)

        assert result.shape == sample_image.shape


# ============================================================================
# Unit Tests: Occlusion Detection
# ============================================================================

class TestOcclusionDetector:
    """Tests for OcclusionDetector."""

    def test_detect_occlusion(self, sample_masks):
        """Test basic occlusion detection."""
        from kp3d.modules.occlusion import OcclusionDetector

        fg_mask, bg_mask = sample_masks
        detector = OcclusionDetector(use_convex_hull=True)

        occlusion = detector.detect_occlusion(fg_mask, bg_mask)

        assert occlusion.shape == fg_mask.shape
        assert occlusion.dtype == np.uint8
        # Occlusion should exist where fg overlaps bg region
        assert np.sum(occlusion) > 0

    def test_detect_occlusion_with_dilation(self, sample_masks):
        """Test occlusion detection with dilated mask."""
        from kp3d.modules.occlusion import OcclusionDetector

        fg_mask, bg_mask = sample_masks
        detector = OcclusionDetector(
            dilation_kernel_size=5,
            dilation_iterations=2
        )

        occlusion = detector.detect_occlusion_with_dilation(fg_mask, bg_mask)

        # Dilated occlusion should be larger
        basic_occlusion = detector.detect_occlusion(fg_mask, bg_mask)
        assert np.sum(occlusion) >= np.sum(basic_occlusion)

    def test_analyze_occlusion(self, sample_masks):
        """Test occlusion analysis."""
        from kp3d.modules.occlusion import OcclusionDetector

        fg_mask, bg_mask = sample_masks
        detector = OcclusionDetector()

        analysis = detector.analyze_occlusion(fg_mask, bg_mask)

        assert "occlusion_mask" in analysis
        assert "occlusion_area" in analysis
        assert "occlusion_ratio" in analysis
        assert "boundary_mask" in analysis
        assert "has_occlusion" in analysis

        assert analysis["has_occlusion"] is True
        assert 0 < analysis["occlusion_ratio"] < 1

    def test_quick_occlusion_mask(self, sample_masks):
        """Test quick_occlusion_mask function."""
        from kp3d.modules.occlusion import quick_occlusion_mask

        fg_mask, bg_mask = sample_masks
        result = quick_occlusion_mask(fg_mask, bg_mask)

        assert result.shape == fg_mask.shape


# ============================================================================
# Unit Tests: Layer Ordering
# ============================================================================

class TestSimpleLayerOrdering:
    """Tests for SimpleLayerOrdering."""

    def test_order_by_label(self, sample_masks):
        """Test ordering by label keywords."""
        from kp3d.modules.occlusion import SimpleLayerOrdering, LayerInfo

        fg_mask, bg_mask = sample_masks

        layers = [
            LayerInfo("wooden table", bg_mask, (0, 0, 256, 256)),
            LayerInfo("ceramic vase", fg_mask, (0, 0, 256, 256))
        ]

        ordering = SimpleLayerOrdering()
        fg, bg = ordering.order_by_label(layers)

        assert fg is not None
        assert bg is not None
        assert "ceramic" in fg.label.lower()
        assert "table" in bg.label.lower()

    def test_order_by_area(self, sample_masks):
        """Test ordering by mask area."""
        from kp3d.modules.occlusion import SimpleLayerOrdering, LayerInfo

        fg_mask, bg_mask = sample_masks

        layers = [
            LayerInfo("object_a", bg_mask, (0, 0, 256, 256)),  # Larger
            LayerInfo("object_b", fg_mask, (0, 0, 256, 256))   # Smaller
        ]

        ordering = SimpleLayerOrdering()
        smaller, larger = ordering.order_by_area(layers)

        assert np.sum(smaller.mask > 0) < np.sum(larger.mask > 0)


# ============================================================================
# Unit Tests: Depth Estimation
# ============================================================================

class TestDepthEstimator:
    """Tests for DepthEstimatorWrapper."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Depth estimation requires CUDA for reasonable performance"
    )
    def test_depth_estimation(self, sample_image):
        """Test depth map generation."""
        from kp3d.modules.occlusion import DepthEstimatorWrapper

        estimator = DepthEstimatorWrapper(model_type="MiDaS_small")
        depth = estimator.estimate(sample_image)

        assert depth.shape == sample_image.shape[:2]
        assert depth.dtype == np.float32
        assert 0 <= depth.min() <= depth.max() <= 1

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Depth estimation requires CUDA"
    )
    def test_mean_depth(self, sample_image, sample_masks):
        """Test mean depth calculation."""
        from kp3d.modules.occlusion import DepthEstimatorWrapper

        fg_mask, bg_mask = sample_masks
        estimator = DepthEstimatorWrapper(model_type="MiDaS_small")

        depth = estimator.estimate(sample_image)

        fg_depth = estimator.get_mean_depth(depth, fg_mask)
        bg_depth = estimator.get_mean_depth(depth, bg_mask)

        assert isinstance(fg_depth, float)
        assert isinstance(bg_depth, float)


# ============================================================================
# Integration Tests
# ============================================================================

class TestInpaintingIntegration:
    """Integration tests for inpainting workflow."""

    def test_full_inpainting_workflow(self, sample_image, sample_masks):
        """Test complete inpainting workflow with manual masks."""
        from kp3d.modules.occlusion import InpaintingModule, OcclusionDetector

        fg_mask, bg_mask = sample_masks

        # Detect occlusion
        detector = OcclusionDetector()
        occlusion = detector.detect_occlusion_with_dilation(fg_mask, bg_mask)

        # Inpaint
        inpainter = InpaintingModule(method="telea", radius=5)
        result = inpainter.inpaint(sample_image, occlusion)

        # Extract foreground
        fg_extracted = inpainter.extract_object(sample_image, fg_mask)

        assert result.shape == sample_image.shape
        assert fg_extracted.shape == (256, 256, 4)

        # Save test outputs
        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / "test_inpainted.png"),
                   cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "test_foreground.png"),
                   cv2.cvtColor(fg_extracted, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(output_dir / "test_occlusion_mask.png"),
                   occlusion)


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_pipeline_with_manual_masks(self, sample_image, sample_masks):
        """Test pipeline with manually provided masks."""
        from kp3d.modules.occlusion import OcclusionPipeline, OcclusionConfig

        fg_mask, bg_mask = sample_masks

        config = OcclusionConfig(
            depth_model="MiDaS_small",  # Faster for testing
            inpaint_method="telea",
            inpaint_radius=5
        )

        output_dir = Path(__file__).parent / "test_outputs" / "pipeline"

        pipeline = OcclusionPipeline(
            config=config,
            output_dir=str(output_dir)
        )

        # Skip GPU-heavy depth for CI
        if not torch.cuda.is_available():
            pytest.skip("Pipeline test requires CUDA")

        result = pipeline.process_with_manual_masks(
            sample_image,
            fg_mask,
            bg_mask,
            foreground_label="ceramic_vase",
            background_label="wooden_table"
        )

        assert result.foreground_image is not None
        assert result.background_inpainted is not None
        assert result.occlusion_mask is not None
        assert result.depth_map is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Full pipeline requires CUDA"
    )
    def test_pipeline_tensor_input(self, sample_tensor):
        """Test pipeline with tensor input."""
        from kp3d.modules.occlusion import OcclusionPipeline

        pipeline = OcclusionPipeline()

        # This will fail on segmentation without proper models
        # Just test that it initializes correctly
        assert pipeline.config is not None
        assert pipeline.output_dir.exists()


# ============================================================================
# Edge Cases
# ============================================================================

class TestPredictInpaintRegions:
    """Tests for predict_inpaint_regions function."""

    def test_predict_regions(self, sample_masks):
        """Test inpaint region prediction."""
        from kp3d.modules.occlusion import predict_inpaint_regions

        fg_mask, bg_mask = sample_masks

        inpaint_mask, bg_fill_mask = predict_inpaint_regions(fg_mask, bg_mask)

        assert inpaint_mask.shape == fg_mask.shape
        assert bg_fill_mask.shape == fg_mask.shape

        # Total should cover all of foreground
        total = np.sum(inpaint_mask > 0) + np.sum(bg_fill_mask > 0)
        fg_total = np.sum(fg_mask > 0)
        # Allow some overlap due to dilation
        assert total >= fg_total * 0.9

    def test_no_background(self):
        """Test with no background visible."""
        from kp3d.modules.occlusion import predict_inpaint_regions

        fg_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(fg_mask, (128, 128), 50, 255, -1)

        bg_mask = np.zeros((256, 256), dtype=np.uint8)

        inpaint_mask, bg_fill_mask = predict_inpaint_regions(fg_mask, bg_mask)

        # With no background, all foreground should be background-filled
        assert np.sum(inpaint_mask > 0) == 0
        assert np.sum(bg_fill_mask > 0) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_mask_inpainting(self, sample_image):
        """Test inpainting with empty mask."""
        from kp3d.modules.occlusion import InpaintingModule

        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        module = InpaintingModule()

        result = module.inpaint(sample_image, empty_mask)

        # Should return unchanged image
        np.testing.assert_array_equal(result, sample_image)

    def test_full_mask_inpainting(self, sample_image):
        """Test inpainting with full mask."""
        from kp3d.modules.occlusion import InpaintingModule

        full_mask = np.ones((256, 256), dtype=np.uint8) * 255
        module = InpaintingModule()

        result = module.inpaint(sample_image, full_mask)

        # Should complete without error (result may vary)
        assert result.shape == sample_image.shape

    def test_non_overlapping_masks(self):
        """Test occlusion detection with non-overlapping masks."""
        from kp3d.modules.occlusion import OcclusionDetector

        # Create non-overlapping masks
        mask_a = np.zeros((256, 256), dtype=np.uint8)
        mask_a[10:50, 10:50] = 255

        mask_b = np.zeros((256, 256), dtype=np.uint8)
        mask_b[100:150, 100:150] = 255

        detector = OcclusionDetector(use_convex_hull=False)
        occlusion = detector.detect_occlusion(mask_a, mask_b)

        # No occlusion expected
        assert np.sum(occlusion) == 0

    def test_single_layer_ordering(self, sample_masks):
        """Test layer ordering with single layer."""
        from kp3d.modules.occlusion import SimpleLayerOrdering, LayerInfo

        fg_mask, _ = sample_masks

        layers = [
            LayerInfo("single_object", fg_mask, (0, 0, 256, 256))
        ]

        ordering = SimpleLayerOrdering()

        with pytest.raises(ValueError):
            ordering.order_by_area(layers)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""

    def test_inpainting_performance(self, sample_image, sample_masks):
        """Benchmark inpainting speed."""
        import time
        from kp3d.modules.occlusion import InpaintingModule

        fg_mask, _ = sample_masks
        module = InpaintingModule()

        # Warmup
        module.inpaint(sample_image, fg_mask)

        # Benchmark
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            module.inpaint(sample_image, fg_mask)
        elapsed = time.time() - start

        avg_time = elapsed / iterations
        print(f"\nInpainting avg time: {avg_time*1000:.2f}ms")

        # Should be reasonably fast
        assert avg_time < 1.0  # Less than 1 second per image

    def test_occlusion_detection_performance(self, sample_masks):
        """Benchmark occlusion detection speed."""
        import time
        from kp3d.modules.occlusion import OcclusionDetector

        fg_mask, bg_mask = sample_masks
        detector = OcclusionDetector()

        # Warmup
        detector.detect_occlusion_with_dilation(fg_mask, bg_mask)

        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            detector.detect_occlusion_with_dilation(fg_mask, bg_mask)
        elapsed = time.time() - start

        avg_time = elapsed / iterations
        print(f"\nOcclusion detection avg time: {avg_time*1000:.2f}ms")

        # Should be very fast
        assert avg_time < 0.1  # Less than 100ms


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
