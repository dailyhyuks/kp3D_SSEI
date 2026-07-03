"""Integration tests for Pipeline orchestration."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from kp3d.core.base import BasePreprocessModule, ModuleOutput


class MockModule(BasePreprocessModule):
    """Mock preprocessing module for testing."""

    def __init__(self, device=None, name="mock", should_fail=False):
        super().__init__(device=device)
        self._name = name
        self._should_fail = should_fail
        self._initialized = True

    @property
    def name(self):
        return self._name

    def forward(self, image, **kwargs):
        if self._should_fail:
            raise RuntimeError(f"Mock failure in {self._name}")
        result = torch.clamp(image + 0.1, 0, 1)
        return ModuleOutput(
            result=result,
            intermediate={"input": image},
            metadata={"module": self._name}
        )

    def load_weights(self, path):
        pass


@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def sample_image():
    return torch.rand(3, 64, 64)

@pytest.fixture
def batch_images():
    return torch.rand(4, 3, 64, 64)


# ============ Pipeline Initialization Tests ============

class TestPipelineInitialization:
    def test_default_initialization(self):
        """Test Pipeline initializes with defaults."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        assert pipeline is not None

    def test_initialization_with_device(self, device):
        """Test Pipeline with explicit device."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline(device=device)
        assert pipeline.device == device


# ============ Pipeline Execution Tests ============

class TestPipelineExecution:
    def test_add_module(self):
        """Test adding modules to pipeline."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        mock = MockModule(name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        assert "test" in pipeline._modules

    def test_process_single_image(self, sample_image):
        """Test processing single image."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        mock = MockModule(name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        result = pipeline.run(sample_image)
        assert result.result.shape == sample_image.shape

    def test_process_with_intermediates(self, sample_image):
        """Test return_intermediates parameter."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        mock = MockModule(name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        result = pipeline.run(sample_image, return_intermediates=True)
        assert result.intermediate is not None

    def test_module_chaining(self, sample_image):
        """Test multiple modules in sequence."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.add_module(MockModule(name="mod1"), name="mod1", enabled=True)
        pipeline.add_module(MockModule(name="mod2"), name="mod2", enabled=True)
        result = pipeline.run(sample_image)
        # Each module adds 0.1, so output should be higher
        assert result.result.mean() > sample_image.mean()


# ============ Batch Processing Tests ============

class TestBatchProcessing:
    def test_batch_input(self, batch_images):
        """Test processing batch of images."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        mock = MockModule(name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        result = pipeline.run(batch_images)
        assert result.result.shape == batch_images.shape


# ============ Error Handling Tests ============

class TestErrorHandling:
    def test_module_failure(self, sample_image):
        """Test module failure propagation."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        failing = MockModule(name="failing", should_fail=True)
        pipeline.add_module(failing, name="failing", enabled=True)
        with pytest.raises(RuntimeError, match="Mock failure"):
            pipeline.run(sample_image)

    def test_invalid_input_dims(self):
        """Test invalid input dimensions."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        mock = MockModule(name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        with pytest.raises(ValueError):
            pipeline.run(torch.rand(64, 64))  # Missing channel dim

    def test_empty_pipeline(self, sample_image):
        """Test empty pipeline returns input."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        result = pipeline.run(sample_image)
        assert torch.allclose(result.result, sample_image)


# ============ Device Management Tests ============

class TestDeviceManagement:
    def test_pipeline_device(self, device):
        """Test pipeline uses correct device."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline(device=device)
        assert pipeline.device == device

    def test_output_on_same_device(self, device, sample_image):
        """Test output tensor is on same device as input."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline(device=device)
        mock = MockModule(device=device, name="test")
        pipeline.add_module(mock, name="test", enabled=True)
        input_tensor = sample_image.to(device)
        result = pipeline.run(input_tensor)
        assert result.result.device == device


# ============ Module Ordering Tests ============

class TestModuleOrdering:
    def test_module_order_preserved(self, sample_image):
        """Test modules execute in add order."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()

        call_order = []

        class OrderedMock(MockModule):
            def forward(self, image, **kwargs):
                call_order.append(self._name)
                return super().forward(image, **kwargs)

        pipeline.add_module(OrderedMock(name="first"), name="first", enabled=True)
        pipeline.add_module(OrderedMock(name="second"), name="second", enabled=True)
        pipeline.run(sample_image)

        assert call_order == ["first", "second"]


# ============ Integration with Real Modules (Mocked) ============

class TestRealModuleIntegration:
    def test_edge_shade_pipeline(self, sample_image):
        """Test edge -> shade pipeline."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.add_module(MockModule(name="edge"), name="edge", enabled=True)
        pipeline.add_module(MockModule(name="shade"), name="shade", enabled=True)
        result = pipeline.run(sample_image)
        assert result.result is not None
        assert result.result.shape == sample_image.shape

    def test_full_pipeline(self, sample_image):
        """Test superres -> edge -> shade full pipeline."""
        from kp3d.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.add_module(MockModule(name="superres"), name="superres", enabled=True)
        pipeline.add_module(MockModule(name="edge"), name="edge", enabled=True)
        pipeline.add_module(MockModule(name="shade"), name="shade", enabled=True)
        result = pipeline.run(sample_image, return_intermediates=True)
        assert "superres" in result.intermediate or len(result.intermediate) > 0
