"""Tests for Layered Occlusion Detection."""
import pytest
import numpy as np
from kp3d.modules.occlusion.occlusion_detection import (
    LayeredOcclusionResult,
    LayeredOcclusionDetector,
    OcclusionRelation,
    get_layer_priority,
    get_base_layer,
    masks_overlap,
    OcclusionDetector
)


class TestLayerPriority:
    """Test layer priority functions."""

    def test_get_layer_priority(self):
        """Test layer priority extraction."""
        assert get_layer_priority('object_2_1') == 3
        assert get_layer_priority('object_2_2') == 3
        assert get_layer_priority('object_2_3') == 3
        assert get_layer_priority('object_3') == 2
        assert get_layer_priority('object_1') == 1
        assert get_layer_priority('background') == 0

    def test_get_base_layer(self):
        """Test base layer extraction."""
        assert get_base_layer('object_2_1') == 'object_2'
        assert get_base_layer('object_2_2') == 'object_2'
        assert get_base_layer('object_2_3') == 'object_2'
        assert get_base_layer('object_3') == 'object_3'
        assert get_base_layer('object_1') == 'object_1'
        assert get_base_layer('background') == 'background'


class TestLayeredOcclusionDetector:
    """Test LayeredOcclusionDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = LayeredOcclusionDetector()
        assert detector.detector is not None

    def test_object_2_occludes_object_1(self):
        """Test object_2 occludes object_1."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[30, 30], [70, 30], [70, 70], [30, 70]]
            },
            {
                'label': 'object_1',
                'points': [[50, 50], [90, 50], [90, 90], [50, 90]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        assert len(result.occlusion_relations) == 1
        rel = result.occlusion_relations[0]
        assert rel.occluder_label == 'object_2_1'
        assert rel.occludee_label == 'object_1'
        assert rel.occlusion_ratio > 0

    def test_object_2_occludes_object_3(self):
        """Test object_2 occludes object_3."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[30, 30], [70, 30], [70, 70], [30, 70]]
            },
            {
                'label': 'object_3',
                'points': [[50, 50], [90, 50], [90, 90], [50, 90]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        assert len(result.occlusion_relations) == 1
        rel = result.occlusion_relations[0]
        assert rel.occluder_label == 'object_2_1'
        assert rel.occludee_label == 'object_3'
        assert rel.occlusion_ratio > 0

    def test_object_3_occludes_object_1(self):
        """Test object_3 occludes object_1."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_3',
                'points': [[30, 30], [70, 30], [70, 70], [30, 70]]
            },
            {
                'label': 'object_1',
                'points': [[50, 50], [90, 50], [90, 90], [50, 90]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        assert len(result.occlusion_relations) == 1
        rel = result.occlusion_relations[0]
        assert rel.occluder_label == 'object_3'
        assert rel.occludee_label == 'object_1'
        assert rel.occlusion_ratio > 0

    def test_same_layer_no_occlusion(self):
        """Test object_2_* objects don't occlude each other."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[20, 20], [50, 20], [50, 50], [20, 50]]
            },
            {
                'label': 'object_2_2',
                'points': [[40, 40], [70, 40], [70, 70], [40, 70]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        # No occlusion between same layer
        assert len(result.occlusion_relations) == 0

    def test_background_no_occlusion(self):
        """Test background doesn't participate in occlusion."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[30, 30], [70, 30], [70, 70], [30, 70]]
            },
            {
                'label': 'background',
                'points': [[10, 10], [90, 10], [90, 90], [10, 90]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        # No occlusion with background
        assert len(result.occlusion_relations) == 0
        # Background mask should be captured
        assert np.sum(result.background_mask) > 0

    def test_full_hierarchy(self):
        """Test full occlusion hierarchy: object_2 > object_3 > object_1."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[30, 30], [60, 30], [60, 60], [30, 60]]
            },
            {
                'label': 'object_3',
                'points': [[40, 40], [70, 40], [70, 70], [40, 70]]
            },
            {
                'label': 'object_1',
                'points': [[50, 50], [80, 50], [80, 80], [50, 80]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        # Expected relations:
        # object_2_1 -> object_3
        # object_2_1 -> object_1
        # object_3 -> object_1
        assert len(result.occlusion_relations) == 3

        occluder_occludee_pairs = [
            (rel.occluder_label, rel.occludee_label)
            for rel in result.occlusion_relations
        ]

        assert ('object_2_1', 'object_3') in occluder_occludee_pairs
        assert ('object_2_1', 'object_1') in occluder_occludee_pairs
        assert ('object_3', 'object_1') in occluder_occludee_pairs

    def test_multiple_object_2(self):
        """Test multiple object_2_* with object_1."""
        detector = LayeredOcclusionDetector()

        shapes = [
            {
                'label': 'object_2_1',
                'points': [[10, 30], [40, 30], [40, 60], [10, 60]]
            },
            {
                'label': 'object_2_2',
                'points': [[60, 30], [90, 30], [90, 60], [60, 60]]
            },
            {
                'label': 'object_1',
                'points': [[20, 40], [80, 40], [80, 80], [20, 80]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        # Both object_2_1 and object_2_2 should occlude object_1
        # No occlusion between object_2_1 and object_2_2
        occluder_occludee_pairs = [
            (rel.occluder_label, rel.occludee_label)
            for rel in result.occlusion_relations
        ]

        assert ('object_2_1', 'object_1') in occluder_occludee_pairs
        assert ('object_2_2', 'object_1') in occluder_occludee_pairs
        assert ('object_2_1', 'object_2_2') not in occluder_occludee_pairs
        assert ('object_2_2', 'object_2_1') not in occluder_occludee_pairs


class TestMasksOverlap:
    """Test masks_overlap utility function."""

    def test_overlapping_masks(self):
        """Test detection of overlapping masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)

        mask1[30:70, 30:70] = 255
        mask2[50:90, 50:90] = 255

        assert masks_overlap(mask1, mask2) == True

    def test_separate_masks(self):
        """Test non-overlapping masks return False."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)

        mask1[10:30, 10:30] = 255
        mask2[70:90, 70:90] = 255

        assert masks_overlap(mask1, mask2) == False

    def test_empty_masks(self):
        """Test empty masks don't overlap."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)

        assert masks_overlap(mask1, mask2) == False


class TestOcclusionDetector:
    """Test the base OcclusionDetector class."""

    def test_detect_occlusion(self):
        """Test basic occlusion detection."""
        detector = OcclusionDetector()

        fg = np.zeros((100, 100), dtype=np.uint8)
        bg = np.zeros((100, 100), dtype=np.uint8)

        fg[30:70, 30:70] = 255
        bg[50:90, 50:90] = 255

        occlusion_mask = detector.detect_occlusion(fg, bg)

        assert occlusion_mask.shape == (100, 100)
        assert np.sum(occlusion_mask > 0) > 0

    def test_analyze_occlusion(self):
        """Test occlusion analysis."""
        detector = OcclusionDetector()

        fg = np.zeros((100, 100), dtype=np.uint8)
        bg = np.zeros((100, 100), dtype=np.uint8)

        fg[30:70, 30:70] = 255
        bg[50:90, 50:90] = 255

        result = detector.analyze_occlusion(fg, bg)

        assert 'occlusion_mask' in result
        assert 'occlusion_area' in result
        assert 'occlusion_ratio' in result
        assert 'has_occlusion' in result
        assert result['has_occlusion'] == True


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_korean_painting_scenario(self):
        """Test scenario resembling Korean painting annotations."""
        detector = LayeredOcclusionDetector()

        # Soban (table) at bottom, ceramics on top
        shapes = [
            # Soban (object_1) - bottom layer
            {
                'label': 'object_1',
                'points': [[10, 50], [90, 50], [90, 95], [10, 95]]
            },
            # Ceramic 1 (object_2_1) - top layer
            {
                'label': 'object_2_1',
                'points': [[20, 20], [45, 20], [45, 70], [20, 70]]
            },
            # Ceramic 2 (object_2_2) - top layer
            {
                'label': 'object_2_2',
                'points': [[55, 25], [80, 25], [80, 75], [55, 75]]
            },
            # Background
            {
                'label': 'background',
                'points': [[0, 0], [100, 0], [100, 40], [0, 40]]
            }
        ]

        result = detector.detect_from_shapes(shapes, (100, 100))

        # Verify structure
        assert isinstance(result, LayeredOcclusionResult)
        assert 'object_1' in result.layer_masks
        assert 'object_2_1' in result.layer_masks
        assert 'object_2_2' in result.layer_masks

        # Verify occlusion relations
        occluder_occludee_pairs = [
            (rel.occluder_label, rel.occludee_label)
            for rel in result.occlusion_relations
        ]

        # Both ceramics should occlude the soban
        assert ('object_2_1', 'object_1') in occluder_occludee_pairs
        assert ('object_2_2', 'object_1') in occluder_occludee_pairs

        # Ceramics don't occlude each other
        assert ('object_2_1', 'object_2_2') not in occluder_occludee_pairs
        assert ('object_2_2', 'object_2_1') not in occluder_occludee_pairs

        # Background mask captured
        assert np.sum(result.background_mask) > 0
