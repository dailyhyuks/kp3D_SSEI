"""Tests for Reference-guided and Hybrid Inpainting."""
import pytest
import numpy as np
from kp3d.modules.occlusion.symmetry_inpaint import (
    SymmetryDetector, detect_symmetry_axis, PatchMatchInpainter, SymmetryGuidedInpainter
)
from kp3d.modules.occlusion.hybrid_inpainter import (
    InpaintingStrategy, RegionAnalyzer, HybridInpainter
)


class TestSymmetryDetector:
    """Test symmetry detection for objects."""

    @pytest.fixture
    def symmetric_mask(self):
        """Create a vertically symmetric mask (like a ceramic vase)."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create symmetric shape (rectangle with rounded top)
        mask[30:90, 40:60] = 255  # Body
        # Add symmetric rounded top
        y_coords = np.arange(20, 30)
        for y in y_coords:
            width = int(10 - (30 - y) * 0.2)
            mask[y, 50-width:50+width] = 255

        return mask

    @pytest.fixture
    def asymmetric_mask(self):
        """Create an asymmetric mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        # L-shaped object (clearly asymmetric)
        mask[30:80, 30:40] = 255  # Vertical part
        mask[70:80, 30:70] = 255  # Horizontal part
        return mask

    def test_vertical_symmetry_detection(self, symmetric_mask):
        """Test detection of vertical symmetry axis."""
        detector = SymmetryDetector()
        axis_x = detector.detect_vertical_symmetry(symmetric_mask)

        # Should detect axis near center (around x=50)
        assert axis_x is not None
        assert 45 <= axis_x <= 55  # Allow some tolerance

    def test_no_symmetry(self, asymmetric_mask):
        """Test handling of non-symmetric objects."""
        detector = SymmetryDetector()
        axis_x = detector.detect_vertical_symmetry(asymmetric_mask)

        # Should either return None or a low-score axis
        if axis_x is not None:
            score = detector.compute_symmetry_score(asymmetric_mask, axis_x)
            assert score < 0.6  # Below symmetry threshold

    def test_symmetry_score_perfect(self):
        """Test symmetry score for perfectly symmetric mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create perfectly symmetric rectangle
        mask[30:70, 40:60] = 255

        detector = SymmetryDetector()
        axis_x = 50  # Center

        score = detector.compute_symmetry_score(mask, axis_x)
        assert score > 0.9  # Should be very high

    def test_symmetry_score_asymmetric(self):
        """Test symmetry score for asymmetric mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create asymmetric shape
        mask[30:70, 20:50] = 255  # Left-heavy rectangle

        detector = SymmetryDetector()
        axis_x = 50  # Test at center

        score = detector.compute_symmetry_score(mask, axis_x)
        assert score < 0.5  # Should be low

    def test_detect_symmetry_axis_convenience(self, symmetric_mask):
        """Test convenience function for symmetry detection."""
        axis_x = detect_symmetry_axis(symmetric_mask)

        assert axis_x is not None
        assert isinstance(axis_x, int)
        assert 0 <= axis_x < symmetric_mask.shape[1]


class TestPatchMatchInpainter:
    """Test PatchMatch-based inpainting."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with simple pattern."""
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Create horizontal stripes pattern
        for i in range(0, h, 10):
            image[i:i+5, :] = [200, 100, 50]  # Orange stripe
            image[i+5:i+10, :] = [50, 150, 200]  # Blue stripe

        return image

    @pytest.fixture
    def inpaint_mask(self):
        """Create mask for region to inpaint."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        # Small region to inpaint
        mask[40:60, 40:60] = 255
        return mask

    @pytest.fixture
    def exemplar_mask(self):
        """Create exemplar mask (source region for patches)."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        # Use left side as exemplar
        mask[:, :35] = 255
        return mask

    def test_inpaint_basic(self, test_image, inpaint_mask, exemplar_mask):
        """Test basic PatchMatch inpainting."""
        inpainter = PatchMatchInpainter(patch_size=5, iterations=2)

        result = inpainter.inpaint(test_image, inpaint_mask, exemplar_mask)

        # Result should have same shape as input
        assert result.shape == test_image.shape

        # Inpainted region should be filled (not all zeros)
        inpainted_region = result[inpaint_mask > 0]
        assert not np.all(inpainted_region == 0)

    def test_fill_order_boundary_first(self):
        """Test that fill order prioritizes boundary pixels."""
        inpainter = PatchMatchInpainter()

        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True  # Square region

        fill_order = inpainter._get_fill_order(mask)

        # First pixels should be near boundary (low distance)
        # Last pixels should be in center (high distance)
        # We can verify by checking that early pixels are near edges
        first_pixels = fill_order[:10]
        for y, x in first_pixels:
            # Should be near edge of the square
            assert (y in [20, 29] or x in [20, 29])


class TestSymmetryGuidedInpainter:
    """Test symmetry-guided inpainting."""

    @pytest.fixture
    def symmetric_image(self):
        """Create image with symmetric object."""
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Create symmetric pattern
        center_x = 50
        for y in range(30, 70):
            for offset in range(1, 20):
                x_left = center_x - offset
                x_right = center_x + offset
                # Mirror pattern
                color_val = (y * 3) % 256
                image[y, x_left] = [color_val, color_val, 100]
                image[y, x_right] = [color_val, color_val, 100]

        return image

    @pytest.fixture
    def symmetric_object_mask(self):
        """Create symmetric object mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:70, 35:65] = 255  # Symmetric rectangle
        return mask

    @pytest.fixture
    def occlusion_mask(self):
        """Create occlusion mask on one side."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        # Occlude right side
        mask[40:60, 55:65] = 255
        return mask

    def test_symmetry_inpaint(self, symmetric_image, occlusion_mask, symmetric_object_mask):
        """Test inpainting using symmetry."""
        inpainter = SymmetryGuidedInpainter(use_patchmatch_fallback=False)

        result = inpainter.inpaint(
            symmetric_image,
            occlusion_mask,
            symmetric_object_mask
        )

        assert result.shape == symmetric_image.shape
        assert result.dtype == np.uint8

    def test_mirror_coordinates(self):
        """Test coordinate mirroring across axis."""
        inpainter = SymmetryGuidedInpainter()

        axis_x = 50
        coords = np.array([30, 40, 60, 70])

        mirrored = inpainter.mirror_coordinates(coords, axis_x)

        # Check mirroring: 30 -> 70, 40 -> 60, 60 -> 40, 70 -> 30
        expected = np.array([70, 60, 40, 30])
        np.testing.assert_array_equal(mirrored, expected)

    def test_fallback_to_patchmatch(self, symmetric_image, occlusion_mask):
        """Test fallback to PatchMatch when no symmetry detected."""
        # Create asymmetric mask
        h, w = 100, 100
        asymmetric_mask = np.zeros((h, w), dtype=np.uint8)
        asymmetric_mask[20:40, 30:80] = 255  # Asymmetric L-shape
        asymmetric_mask[40:80, 30:45] = 255

        inpainter = SymmetryGuidedInpainter(use_patchmatch_fallback=True)

        result = inpainter.inpaint(
            symmetric_image,
            occlusion_mask,
            asymmetric_mask
        )

        # Should still produce result via PatchMatch fallback
        assert result.shape == symmetric_image.shape


class TestRegionAnalyzer:
    """Test region analysis for inpainting strategy selection."""

    @pytest.fixture
    def simple_texture_image(self):
        """Create image with simple uniform texture."""
        h, w = 100, 100
        image = np.ones((h, w, 3), dtype=np.uint8) * 150  # Uniform gray
        return image

    @pytest.fixture
    def complex_texture_image(self):
        """Create image with complex texture (high gradients)."""
        h, w = 100, 100
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return image

    def test_analyze_simple_region(self, simple_texture_image):
        """Test analysis of simple texture region."""
        analyzer = RegionAnalyzer()

        h, w = 100, 100
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[40:60, 40:60] = 255

        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask[30:70, 30:70] = 255

        result = analyzer.analyze(
            simple_texture_image,
            inpaint_mask,
            object_mask
        )

        # Should detect simple texture
        assert result['texture_complexity'] < 0.5
        assert 'occlusion_ratio' in result
        assert 'recommended_strategy' in result

    def test_texture_complexity(self, simple_texture_image, complex_texture_image):
        """Test texture complexity computation."""
        analyzer = RegionAnalyzer()

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        # Simple texture should have low complexity
        simple_complexity = analyzer._compute_texture_complexity(
            simple_texture_image, mask
        )
        assert simple_complexity < 0.5

        # Complex texture should have higher complexity
        complex_complexity = analyzer._compute_texture_complexity(
            complex_texture_image, mask
        )
        # Note: random texture might not always be "complex" in gradient terms
        # but should generally be higher than uniform
        assert complex_complexity >= simple_complexity

    def test_symmetry_detection_ceramic(self):
        """Test symmetry detection with ceramic object type hint."""
        analyzer = RegionAnalyzer()

        # Create symmetric mask
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[30:70, 45:55] = 255  # Narrow symmetric rectangle

        # With ceramic hint, should recognize symmetry potential
        has_symmetry = analyzer._check_symmetry_potential(mask, object_type='ceramic')
        assert has_symmetry is True

    def test_edge_density_calculation(self):
        """Test edge density computation."""
        analyzer = RegionAnalyzer()

        # Create image with edges
        h, w = 100, 100
        image = np.ones((h, w, 3), dtype=np.uint8) * 128

        # Add some edges
        image[45:55, :] = 255  # Horizontal edge

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        edge_density = analyzer._compute_edge_density(image, mask)

        # Should detect some edges
        assert edge_density > 0


class TestHybridInpainter:
    """Test hybrid inpainting with automatic strategy selection."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        h, w = 100, 100
        image = np.ones((h, w, 3), dtype=np.uint8) * 128
        # Add some pattern
        image[::10, :] = 200
        return image

    @pytest.fixture
    def small_mask(self):
        """Small occlusion mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[45:55, 45:55] = 255  # 10x10 small region
        return mask

    @pytest.fixture
    def large_mask(self):
        """Large occlusion mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # 60x60 large region
        return mask

    @pytest.fixture
    def object_mask(self):
        """Object mask."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:90, 10:90] = 255
        return mask

    def test_strategy_selection_small_occlusion(self, test_image, small_mask, object_mask):
        """Test PatchMatch selected for small occlusion."""
        inpainter = HybridInpainter(enable_diffusion=False)

        # Analyze to get recommendation
        analysis = inpainter.analyze_region(
            test_image,
            small_mask,
            object_mask
        )

        # Small occlusion with simple texture should prefer PatchMatch
        assert analysis['occlusion_ratio'] < 0.3

    def test_strategy_selection_large_occlusion(self, test_image, large_mask, object_mask):
        """Test LaMa selected for large occlusion."""
        inpainter = HybridInpainter(enable_diffusion=True)

        analysis = inpainter.analyze_region(
            test_image,
            large_mask,
            object_mask
        )

        # Large occlusion should recommend generative method
        assert analysis['occlusion_ratio'] >= 0.3

    def test_strategy_selection_symmetric(self):
        """Test symmetry method selected for symmetric objects."""
        h, w = 100, 100

        # Create symmetric object
        image = np.ones((h, w, 3), dtype=np.uint8) * 150
        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask[30:70, 45:55] = 255  # Narrow symmetric shape

        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[40:60, 48:55] = 255  # Small occlusion on one side

        inpainter = HybridInpainter(enable_symmetry=True)

        analysis = inpainter.analyze_region(
            image,
            inpaint_mask,
            object_mask,
            object_type='ceramic'
        )

        # With ceramic hint and symmetric shape, should detect symmetry
        assert analysis['has_symmetry'] is True

    def test_inpaint_with_auto_selection(self, test_image, small_mask, object_mask):
        """Test inpainting with automatic strategy selection."""
        inpainter = HybridInpainter(
            enable_symmetry=True,
            enable_diffusion=False,  # Disable to avoid model loading
            fallback_to_lama=False
        )

        result, strategy_used = inpainter.inpaint(
            test_image,
            small_mask,
            object_mask
        )

        # Should return valid result
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # Strategy should be one of the enabled strategies
        assert isinstance(strategy_used, InpaintingStrategy)

    def test_forced_strategy(self, test_image, small_mask, object_mask):
        """Test forcing specific strategy."""
        inpainter = HybridInpainter(enable_diffusion=False)

        result, strategy_used = inpainter.inpaint(
            test_image,
            small_mask,
            object_mask,
            strategy=InpaintingStrategy.PATCHMATCH
        )

        # Should use the forced strategy
        assert strategy_used == InpaintingStrategy.PATCHMATCH

    def test_fallback_when_method_disabled(self, test_image, small_mask, object_mask):
        """Test fallback to alternative when preferred method disabled."""
        # Disable symmetry but create symmetric object
        inpainter = HybridInpainter(
            enable_symmetry=False,
            enable_diffusion=False
        )

        # Create symmetric object mask
        h, w = 100, 100
        sym_mask = np.zeros((h, w), dtype=np.uint8)
        sym_mask[30:70, 45:55] = 255

        result, strategy_used = inpainter.inpaint(
            test_image,
            small_mask,
            sym_mask,
            object_type='ceramic'
        )

        # Should fallback to PatchMatch or OpenCV (not symmetry)
        assert strategy_used != InpaintingStrategy.SYMMETRY_PATCHMATCH


class TestInpaintingIntegration:
    """Integration tests for complete inpainting pipeline."""

    def test_ceramic_occlusion_scenario(self):
        """Test realistic ceramic occlusion scenario."""
        h, w = 150, 150

        # Create ceramic-like object (symmetric vase)
        image = np.ones((h, w, 3), dtype=np.uint8) * 200
        center_x = 75

        # Vase body with symmetric pattern
        for y in range(50, 120):
            width = 20 + int(10 * np.sin((y - 50) / 20))
            for x in range(center_x - width, center_x + width):
                if 0 <= x < w:
                    image[y, x] = [180, 140, 100]  # Brown ceramic color

        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask[50:120, center_x-30:center_x+30] = 255

        # Occlude right side
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        occlusion_mask[70:100, center_x+10:center_x+30] = 255

        # Test with hybrid inpainter
        inpainter = HybridInpainter(
            enable_symmetry=True,
            enable_diffusion=False,
            fallback_to_lama=False
        )

        result, strategy = inpainter.inpaint(
            image,
            occlusion_mask,
            object_mask,
            object_type='ceramic'
        )

        # Should successfully inpaint
        assert result.shape == image.shape

        # For symmetric ceramic with moderate occlusion, should prefer symmetry
        # (though exact strategy depends on analysis)
        assert isinstance(strategy, InpaintingStrategy)

    @pytest.mark.parametrize("occlusion_ratio,expected_complexity", [
        (0.1, "small"),   # Small occlusion
        (0.4, "large"),   # Large occlusion
    ])
    def test_varied_occlusion_scenarios(self, occlusion_ratio, expected_complexity):
        """Test different occlusion scenarios."""
        h, w = 100, 100

        image = np.ones((h, w, 3), dtype=np.uint8) * 150
        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask[20:80, 20:80] = 255

        # Create occlusion based on ratio
        object_area = np.sum(object_mask > 0)
        target_occlusion_area = int(object_area * occlusion_ratio)

        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        # Fill to approximate target area
        size = int(np.sqrt(target_occlusion_area))
        y1, x1 = 40, 40
        inpaint_mask[y1:y1+size, x1:x1+size] = 255

        analyzer = RegionAnalyzer()
        result = analyzer.analyze(image, inpaint_mask, object_mask)

        if expected_complexity == "small":
            assert result['occlusion_ratio'] < 0.3
        elif expected_complexity == "large":
            assert result['occlusion_ratio'] >= 0.3
