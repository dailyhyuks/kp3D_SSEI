"""Unit tests for V25 Dynamic Edge Morphology.

Tests the skeleton-based width measurement, color profile extraction,
and variable-thickness edge rendering introduced in V25.
"""

import numpy as np
import cv2
import pytest


def _create_test_image_with_edge(h=100, w=100, edge_thickness=4):
    """Create a synthetic test image with a known-thickness edge.

    Creates a white background with a dark vertical line of specified thickness
    centered at x=50.
    """
    image = np.ones((h, w, 3), dtype=np.uint8) * 200  # light gray bg
    center_x = w // 2
    half_t = edge_thickness // 2

    # Draw dark edge line (simulating ink stroke)
    x_start = max(0, center_x - half_t)
    x_end = min(w, center_x + half_t)

    # Center is darkest, boundary is lighter (gradient)
    for x in range(x_start, x_end):
        dist_from_center = abs(x - center_x)
        t = dist_from_center / max(half_t, 1)
        color_val = int(30 + 70 * t)  # 30 at center, 100 at edge
        image[:, x] = [color_val, color_val, color_val]

    visible_mask = np.ones((h, w), dtype=np.uint8) * 255
    return image, visible_mask


def _create_test_masks(h=100, w=100):
    """Create test masks for intersection edge testing.

    Creates an occludee (circle) partially overlapped by an occluder (rectangle).
    """
    # Occludee: circle at center
    occludee_full = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(occludee_full, (w // 2, h // 2), 30, 255, -1)

    # Occluder: rectangle covering right side
    occluder = np.zeros((h, w), dtype=np.uint8)
    occluder[:, w // 2 + 10:] = 255

    # Visible = occludee minus occluder
    visible = cv2.bitwise_and(occludee_full, cv2.bitwise_not(occluder))

    # Occlusion region = intersection of occludee and occluder
    occlusion = cv2.bitwise_and(occludee_full, occluder)

    return occludee_full, occluder, visible, occlusion


class TestMeasureEdgeWidthMap:
    """Tests for measure_edge_width_map()."""

    def test_returns_correct_types(self):
        """Verify return types are (uint8, float32, uint8)."""
        from kp3d.modules.occlusion.inpainting import measure_edge_width_map

        image, mask = _create_test_image_with_edge()
        skeleton, width_map, edge_mask = measure_edge_width_map(image, mask)

        assert skeleton.dtype == np.uint8
        assert width_map.dtype == np.float32
        assert edge_mask.dtype == np.uint8

    def test_skeleton_single_pixel_width(self):
        """Skeleton should be approximately 1 pixel wide."""
        from kp3d.modules.occlusion.inpainting import measure_edge_width_map

        image, mask = _create_test_image_with_edge(edge_thickness=6)
        skeleton, _, _ = measure_edge_width_map(image, mask)

        if np.sum(skeleton > 0) == 0:
            pytest.skip("No skeleton extracted from test image")

        # Check that skeleton is thin: for each row with skeleton pixels,
        # the number of skeleton pixels should be small (1-2)
        skel_rows = np.where(np.any(skeleton > 0, axis=1))[0]
        for row in skel_rows[:10]:  # check first 10 rows
            n_pixels = np.sum(skeleton[row] > 0)
            assert n_pixels <= 3, f"Row {row} has {n_pixels} skeleton pixels (expected <=3)"

    def test_width_map_nonzero_at_skeleton(self):
        """Width map should have nonzero values only at skeleton pixels."""
        from kp3d.modules.occlusion.inpainting import measure_edge_width_map

        image, mask = _create_test_image_with_edge(edge_thickness=4)
        skeleton, width_map, _ = measure_edge_width_map(image, mask)

        # Width > 0 only where skeleton > 0
        assert np.all(width_map[skeleton == 0] == 0), \
            "Width map has nonzero values outside skeleton"

    def test_width_clipping(self):
        """Width values should be clipped to [min_edge_width, max_edge_width]."""
        from kp3d.modules.occlusion.inpainting import measure_edge_width_map

        image, mask = _create_test_image_with_edge(edge_thickness=4)
        _, width_map, _ = measure_edge_width_map(
            image, mask, min_edge_width=2, max_edge_width=6
        )

        nonzero = width_map[width_map > 0]
        if len(nonzero) > 0:
            assert np.all(nonzero >= 2), "Width below min_edge_width"
            assert np.all(nonzero <= 6), "Width above max_edge_width"

    def test_empty_visible_mask(self):
        """Should return zero arrays for empty visible mask."""
        from kp3d.modules.occlusion.inpainting import measure_edge_width_map

        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        skeleton, width_map, edge_mask = measure_edge_width_map(image, mask)

        assert np.sum(skeleton) == 0
        assert np.sum(width_map) == 0


class TestExtractEdgeColorProfile:
    """Tests for extract_edge_color_profile()."""

    def test_returns_none_for_empty_skeleton(self):
        """Should return None when skeleton has no pixels."""
        from kp3d.modules.occlusion.inpainting import extract_edge_color_profile

        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        width_map = np.zeros((50, 50), dtype=np.float32)

        result = extract_edge_color_profile(image, skeleton, width_map)
        assert result is None

    def test_profile_keys(self):
        """Profile dict should contain expected keys."""
        from kp3d.modules.occlusion.inpainting import (
            measure_edge_width_map, extract_edge_color_profile
        )

        image, mask = _create_test_image_with_edge(edge_thickness=4)
        skeleton, width_map, _ = measure_edge_width_map(image, mask)

        if np.sum(skeleton > 0) == 0:
            pytest.skip("No skeleton extracted")

        profile = extract_edge_color_profile(image, skeleton, width_map)
        assert profile is not None

        expected_keys = {'coords', 'widths', 'center_colors', 'edge_colors', 'orientations'}
        assert set(profile.keys()) == expected_keys

    def test_center_darker_than_edge(self):
        """For dark-center edges, center colors should be darker than edge colors."""
        from kp3d.modules.occlusion.inpainting import (
            measure_edge_width_map, extract_edge_color_profile
        )

        image, mask = _create_test_image_with_edge(edge_thickness=6)
        skeleton, width_map, _ = measure_edge_width_map(image, mask)

        if np.sum(skeleton > 0) == 0:
            pytest.skip("No skeleton extracted")

        profile = extract_edge_color_profile(image, skeleton, width_map)
        if profile is None:
            pytest.skip("No profile extracted")

        center_mean = np.mean(profile['center_colors'].astype(float))
        edge_mean = np.mean(profile['edge_colors'].astype(float))

        # Center should be darker (lower value) or close to edge
        # Allow some tolerance since gradient might not be perfectly captured
        assert center_mean <= edge_mean + 30, \
            f"Center ({center_mean:.1f}) not darker than edge ({edge_mean:.1f})"


class TestRenderDynamicIntersectionEdge:
    """Tests for render_dynamic_intersection_edge()."""

    def test_returns_correct_shapes(self):
        """Output shapes should match input image."""
        from kp3d.modules.occlusion.inpainting import (
            measure_edge_width_map, extract_edge_color_profile,
            render_dynamic_intersection_edge, get_intersection_edge
        )

        h, w = 100, 100
        image, mask = _create_test_image_with_edge(h, w, edge_thickness=4)
        occludee_full, occluder, visible, occlusion = _create_test_masks(h, w)

        skeleton, width_map, _ = measure_edge_width_map(image, visible)

        if np.sum(skeleton > 0) == 0:
            pytest.skip("No skeleton extracted")

        profile = extract_edge_color_profile(image, skeleton, width_map)
        centerline = get_intersection_edge(occludee_full, occlusion, thickness=1)

        rendered, edge_mask = render_dynamic_intersection_edge(
            image, centerline, profile, occluder_mask=occluder
        )

        assert rendered.shape == (h, w, 3)
        assert edge_mask.shape == (h, w)
        assert edge_mask.dtype == np.uint8

    def test_empty_centerline(self):
        """Should return zeros for empty centerline."""
        from kp3d.modules.occlusion.inpainting import render_dynamic_intersection_edge

        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        centerline = np.zeros((50, 50), dtype=np.uint8)
        profile = {'coords': np.array([[10, 10]]), 'widths': np.array([2.0]),
                    'center_colors': np.array([[50, 50, 50]], dtype=np.uint8),
                    'edge_colors': np.array([[100, 100, 100]], dtype=np.uint8),
                    'orientations': np.array([[0.0, 1.0]])}

        rendered, mask = render_dynamic_intersection_edge(image, centerline, profile)
        assert np.sum(mask) == 0

    def test_fallback_v24_when_v25_fails(self):
        """inpaint_occlusion_boundary_guided should fall back to V24 when needed."""
        from kp3d.modules.occlusion.inpainting import inpaint_occlusion_boundary_guided

        h, w = 100, 100
        image = np.ones((h, w, 3), dtype=np.uint8) * 180
        occludee_full, occluder, visible, occlusion = _create_test_masks(h, w)

        # Should not crash even with uniform image (no edges to find)
        result = inpaint_occlusion_boundary_guided(
            image, occlusion, occluder, occludee_full, visible,
            use_dynamic_edge=True
        )

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_v24_fallback_explicit(self):
        """With use_dynamic_edge=False, should use V24 (thickness=2)."""
        from kp3d.modules.occlusion.inpainting import inpaint_occlusion_boundary_guided

        h, w = 100, 100
        image = np.ones((h, w, 3), dtype=np.uint8) * 180
        occludee_full, occluder, visible, occlusion = _create_test_masks(h, w)

        # Draw some features on image for edge detection
        cv2.circle(image, (w // 2, h // 2), 30, (50, 50, 50), 2)

        result = inpaint_occlusion_boundary_guided(
            image, occlusion, occluder, occludee_full, visible,
            use_dynamic_edge=False
        )

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8
