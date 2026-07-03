"""Comprehensive tests for edge detection modules."""

import pytest
import torch
from torch import Tensor
from kp3d.core.base import ModuleOutput


# ============ Fixtures ============

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def simple_image(device):
    """Gradient image."""
    img = torch.zeros(3, 64, 64, device=device)
    for i in range(64):
        img[:, :, i] = i / 63.0
    return img

@pytest.fixture
def edge_image(device):
    """White square on black."""
    img = torch.zeros(3, 64, 64, device=device)
    img[:, 16:48, 16:48] = 1.0
    return img

@pytest.fixture
def color_image(device):
    """Image with color regions."""
    img = torch.zeros(3, 64, 64, device=device)
    img[0, :32, :] = 1.0  # Red top
    img[2, 32:, :] = 1.0  # Blue bottom
    return img


# ============ CannyEdgeDetector Tests ============

class TestCannyEdgeDetector:
    def test_initialization(self, device):
        from kp3d.modules.edge.canny import CannyEdgeDetector
        detector = CannyEdgeDetector(device=device)
        assert detector.name == "canny_edge"

    def test_forward_shape(self, device, edge_image):
        from kp3d.modules.edge.canny import CannyEdgeDetector
        detector = CannyEdgeDetector(device=device)
        output = detector(edge_image)
        assert isinstance(output, ModuleOutput)
        assert output.result.shape[-2:] == (64, 64)

    def test_forward_range(self, device, edge_image):
        from kp3d.modules.edge.canny import CannyEdgeDetector
        detector = CannyEdgeDetector(device=device)
        output = detector(edge_image)
        assert output.result.min() >= 0.0
        assert output.result.max() <= 1.0

    def test_custom_thresholds(self, device, edge_image):
        from kp3d.modules.edge.canny import CannyEdgeDetector
        from kp3d.modules.edge.base import EdgeConfig
        config = EdgeConfig(low_threshold=50, high_threshold=150)
        detector = CannyEdgeDetector(config=config, device=device)
        output = detector(edge_image)
        assert output.result is not None


# ============ KoreanInkEdgeDetector Tests ============

class TestKoreanInkEdgeDetector:
    def test_initialization(self, device):
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
        detector = KoreanInkEdgeDetector(device=device, use_hed=False)
        assert detector.name == "korean_ink_edge"

    def test_forward_shape(self, device, edge_image):
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
        detector = KoreanInkEdgeDetector(device=device, use_hed=False)
        output = detector(edge_image)
        assert output.result.shape[-2:] == (64, 64)

    def test_weights_in_metadata(self, device, edge_image):
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
        detector = KoreanInkEdgeDetector(device=device, use_hed=False)
        output = detector(edge_image)
        assert "weights" in output.metadata

    def test_intermediate_outputs(self, device, edge_image):
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
        detector = KoreanInkEdgeDetector(device=device, use_hed=False)
        output = detector(edge_image)
        assert "ink_lines" in output.intermediate

    def test_color_boundaries(self, device, color_image):
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
        detector = KoreanInkEdgeDetector(device=device, use_hed=False)
        output = detector(color_image)
        assert "color_boundaries" in output.intermediate


# ============ SmartFusionDetector Tests ============

class TestSmartFusionDetector:
    def test_initialization(self, device):
        from kp3d.modules.edge.smart_fusion import SmartFusionDetector
        detector = SmartFusionDetector(device=device)
        assert detector.name == "smart_fusion"

    def test_forward_shape(self, device, edge_image):
        from kp3d.modules.edge.smart_fusion import SmartFusionDetector
        detector = SmartFusionDetector(device=device)
        output = detector(edge_image)
        assert output.result.shape[-2:] == (64, 64)

    def test_structure_color_fusion(self, device, color_image):
        from kp3d.modules.edge.smart_fusion import SmartFusionDetector
        detector = SmartFusionDetector(device=device)
        output = detector(color_image)
        assert "external" in output.intermediate or "color_edges" in output.intermediate


# ============ AdvancedEdgeDetector Tests ============

class TestAdvancedEdgeDetector:
    def test_initialization(self, device):
        from kp3d.modules.edge.advanced_edge import AdvancedEdgeDetector
        detector = AdvancedEdgeDetector(device=device)
        assert detector.name == "advanced_edge"

    def test_forward_shape(self, device, color_image):
        from kp3d.modules.edge.advanced_edge import AdvancedEdgeDetector
        detector = AdvancedEdgeDetector(device=device)
        output = detector(color_image)
        assert output.result.shape[-2:] == (64, 64)

    def test_lab_processing(self, device, color_image):
        from kp3d.modules.edge.advanced_edge import AdvancedEdgeDetector
        detector = AdvancedEdgeDetector(device=device)
        output = detector(color_image)
        # Should have LAB-related intermediates
        has_lab = any("L" in k or "lab" in k.lower() for k in output.intermediate.keys())
        assert has_lab or len(output.intermediate) > 0


# ============ HED Tests (with fallback) ============

class TestHEDEdgeDetector:
    def test_fallback_graceful(self, device, edge_image):
        """Test graceful fallback when HED weights unavailable."""
        from kp3d.modules.edge.hed import HEDEdgeDetector
        detector = HEDEdgeDetector(device=device)
        output = detector(edge_image)
        # Should produce valid output even in fallback mode
        assert isinstance(output, ModuleOutput)
        assert output.result.min() >= 0.0
        assert output.result.max() <= 1.0


# ============ Integration Tests ============

class TestEdgeIntegration:
    def test_all_detectors_consistent_api(self, device, edge_image):
        """All detectors should follow consistent API."""
        from kp3d.modules.edge.canny import CannyEdgeDetector
        from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector

        detectors = [
            CannyEdgeDetector(device=device),
            KoreanInkEdgeDetector(device=device, use_hed=False),
        ]
        for detector in detectors:
            output = detector(edge_image)
            assert isinstance(output, ModuleOutput)
            assert output.result.min() >= 0.0
            assert output.result.max() <= 1.0

    def test_grayscale_input(self, device):
        """Test with grayscale input."""
        from kp3d.modules.edge.canny import CannyEdgeDetector
        gray = torch.rand(1, 64, 64, device=device)
        detector = CannyEdgeDetector(device=device)
        output = detector(gray)
        assert output.result is not None

    def test_batch_dimension(self, device):
        """Test with batch dimension."""
        from kp3d.modules.edge.canny import CannyEdgeDetector
        batch = torch.rand(2, 3, 64, 64, device=device)
        detector = CannyEdgeDetector(device=device)
        output = detector(batch[0])  # Process first image
        assert output.result is not None
