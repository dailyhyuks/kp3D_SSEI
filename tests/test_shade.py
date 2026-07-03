"""Comprehensive tests for Shade Generation module."""

import pytest
import torch
from torch import Tensor
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import FrozenInstanceError

# Test fixtures
@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def sample_image():
    """Create a sample RGB image tensor."""
    return torch.rand(1, 3, 64, 64)

@pytest.fixture
def sample_depth_map():
    """Create a sample depth map."""
    return torch.rand(1, 1, 64, 64)

# ============ LightSource Tests ============

def test_light_source_defaults():
    """Test LightSource default values."""
    from kp3d.modules.shade.base import LightSource
    light = LightSource()
    assert light.direction == (0.0, 0.0, 1.0)
    assert light.intensity == 1.0
    assert light.ambient == 0.2
    assert light.color == (1.0, 1.0, 1.0)

def test_light_source_custom():
    """Test LightSource with custom values."""
    from kp3d.modules.shade.base import LightSource
    light = LightSource(
        direction=(1.0, 0.5, 0.5),
        intensity=0.8,
        ambient=0.3,
        color=(1.0, 0.9, 0.8)
    )
    assert light.direction == (1.0, 0.5, 0.5)
    assert light.intensity == 0.8

# ============ ShadeConfig Tests ============

def test_shade_config_defaults():
    """Test ShadeConfig default values."""
    from kp3d.modules.shade.base import ShadeConfig
    config = ShadeConfig()
    assert config.depth_model == "DPT_Hybrid"
    assert config.shade_intensity == 0.7
    assert len(config.light_sources) == 1

def test_shade_config_custom():
    """Test ShadeConfig with custom light sources."""
    from kp3d.modules.shade.base import ShadeConfig, LightSource
    lights = [
        LightSource(direction=(1, 0, 1)),
        LightSource(direction=(-1, 0, 1))
    ]
    config = ShadeConfig(light_sources=lights, shade_intensity=0.5)
    assert len(config.light_sources) == 2
    assert config.shade_intensity == 0.5

# ============ ShadeModule Tests ============

def test_shade_module_init():
    """Test ShadeModule initialization."""
    from kp3d.modules.shade import ShadeModule
    module = ShadeModule(
        target_illumination=0.6,
        preserve_details=True,
        gamma=1.2
    )
    assert module.target_illumination == 0.6
    assert module.gamma == 1.2
    assert module.name == "shade"

def test_shade_module_forward(sample_image):
    """Test ShadeModule forward pass."""
    from kp3d.modules.shade import ShadeModule
    from kp3d.core.base import ModuleOutput
    module = ShadeModule(target_illumination=0.5)
    output = module(sample_image)
    assert isinstance(output, ModuleOutput)
    assert output.result.shape == sample_image.shape
    assert output.result.min() >= 0.0
    assert output.result.max() <= 1.0

def test_shade_module_gamma_correction(sample_image):
    """Test gamma correction effect."""
    from kp3d.modules.shade import ShadeModule
    module_no_gamma = ShadeModule(gamma=1.0)
    module_gamma = ShadeModule(gamma=2.2)
    out1 = module_no_gamma(sample_image)
    out2 = module_gamma(sample_image)
    # Different gamma should produce different results
    assert not torch.allclose(out1.result, out2.result)

def test_shade_module_load_weights():
    """Test load_weights sets initialized flag."""
    from kp3d.modules.shade import ShadeModule
    module = ShadeModule()
    assert not module._initialized
    module.load_weights("dummy_path")
    assert module._initialized

# ============ LightingSimulator Tests ============

def test_lighting_compute_normals(sample_depth_map):
    """Test normal map computation from depth."""
    from kp3d.modules.shade.lighting import LightingSimulator
    simulator = LightingSimulator()
    normals = simulator.compute_normals(sample_depth_map)
    assert normals.shape == (1, 3, 64, 64)

def test_lighting_normals_normalized(sample_depth_map):
    """Test that normals are unit vectors."""
    from kp3d.modules.shade.lighting import LightingSimulator
    simulator = LightingSimulator()
    normals = simulator.compute_normals(sample_depth_map)
    norms = torch.norm(normals, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_lighting_apply_single_light(sample_image, sample_depth_map):
    """Test lighting with single light source."""
    from kp3d.modules.shade.lighting import LightingSimulator
    from kp3d.modules.shade.base import LightSource
    simulator = LightingSimulator()
    normals = simulator.compute_normals(sample_depth_map)
    light = LightSource(direction=(0, 0, 1))
    result = simulator.apply_lighting(sample_image, normals, [light])
    assert result.shape == sample_image.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_lighting_multiple_lights(sample_image, sample_depth_map):
    """Test lighting with multiple light sources."""
    from kp3d.modules.shade.lighting import LightingSimulator
    from kp3d.modules.shade.base import LightSource
    simulator = LightingSimulator()
    normals = simulator.compute_normals(sample_depth_map)
    lights = [
        LightSource(direction=(1, 0, 1)),
        LightSource(direction=(-1, 0, 1))
    ]
    result = simulator.apply_lighting(sample_image, normals, lights)
    assert result.shape == sample_image.shape

def test_lighting_shadows(sample_depth_map):
    """Test shadow generation."""
    from kp3d.modules.shade.lighting import LightingSimulator
    simulator = LightingSimulator()
    shadow = simulator.generate_shadows(sample_depth_map, (1, 0, 1))
    assert shadow.shape == sample_depth_map.shape
    assert shadow.min() >= 0.0
    assert shadow.max() <= 1.0

# ============ MiDaSDepthEstimator Tests ============

@patch('torch.hub.load')
def test_midas_init(mock_hub):
    """Test MiDaS initialization with mocked hub."""
    mock_model = MagicMock()
    mock_transform = MagicMock()
    mock_hub.side_effect = [mock_model, MagicMock(dpt_transform=mock_transform)]

    from kp3d.modules.shade.midas import MiDaSDepthEstimator
    from kp3d.modules.shade.base import ShadeConfig

    config = ShadeConfig(depth_model="DPT_Hybrid")
    estimator = MiDaSDepthEstimator(config=config)
    assert estimator._initialized

def test_midas_invalid_model():
    """Test invalid model type raises error."""
    from kp3d.modules.shade.midas import MiDaSDepthEstimator
    from kp3d.modules.shade.base import ShadeConfig

    config = ShadeConfig(depth_model="InvalidModel")
    with pytest.raises(ValueError):
        MiDaSDepthEstimator(config=config)

# ============ ShadeGeneratorModule Tests ============

@patch('torch.hub.load')
def test_shade_generator_init(mock_hub):
    """Test ShadeGeneratorModule initialization."""
    mock_model = MagicMock()
    mock_transform = MagicMock()
    mock_hub.side_effect = [mock_model, MagicMock(dpt_transform=mock_transform)]

    from kp3d.modules.shade.lighting import ShadeGeneratorModule
    module = ShadeGeneratorModule()
    assert module._initialized
    assert module.name == "shade_generator"
