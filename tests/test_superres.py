"""Comprehensive tests for SuperResolution modules."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from kp3d.core.base import ModuleOutput


# ============ Fixtures ============

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def sample_image():
    """Create a sample RGB image tensor."""
    return torch.rand(1, 3, 64, 64)


# ============ SuperResConfig Tests ============

def test_superres_config_defaults():
    """Test default configuration values."""
    from kp3d.modules.superres.base import SuperResConfig, ScaleFactor
    config = SuperResConfig()
    assert config.scale == ScaleFactor.X4
    assert config.tile_size == 512

def test_superres_config_custom():
    """Test custom configuration."""
    from kp3d.modules.superres.base import SuperResConfig, ScaleFactor
    config = SuperResConfig(scale=ScaleFactor.X2, tile_size=256)
    assert config.scale == ScaleFactor.X2
    assert config.tile_size == 256


# ============ BaseSuperResolution Tests ============

def test_base_superres_abstract():
    """Test base class is abstract."""
    from kp3d.modules.superres.base import BaseSuperResolution
    import abc
    assert hasattr(BaseSuperResolution, '__abstractmethods__')


# ============ RealESRGAN Tests (Mocked) ============

@pytest.fixture
def mock_realesrgan_deps():
    """Mock Real-ESRGAN dependencies."""
    with patch("kp3d.modules.superres.real_esrgan.REALESRGAN_AVAILABLE", True):
        mock_rrdb = MagicMock()
        mock_upsampler = MagicMock()

        def mock_enhance(img_bgr, outscale=4):
            h, w = img_bgr.shape[:2]
            output = (torch.rand(h * outscale, w * outscale, 3).numpy() * 255).astype('uint8')
            return output, None

        mock_upsampler.enhance = mock_enhance

        with patch("kp3d.modules.superres.real_esrgan.RRDBNet", return_value=mock_rrdb):
            with patch("kp3d.modules.superres.real_esrgan.RealESRGANer", return_value=mock_upsampler):
                yield mock_upsampler

def test_realesrgan_not_available():
    """Test error when Real-ESRGAN not installed."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    with patch("kp3d.modules.superres.real_esrgan.REALESRGAN_AVAILABLE", False):
        with pytest.raises(ImportError):
            RealESRGANModule()

def test_realesrgan_init_x4(mock_realesrgan_deps):
    """Test 4x initialization."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    from kp3d.modules.superres.base import ScaleFactor
    module = RealESRGANModule(scale=ScaleFactor.X4)
    assert module.scale == 4
    assert module._initialized

def test_realesrgan_init_x2(mock_realesrgan_deps):
    """Test 2x initialization."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    from kp3d.modules.superres.base import ScaleFactor
    module = RealESRGANModule(scale=ScaleFactor.X2)
    assert module.scale == 2

def test_realesrgan_forward_shape(mock_realesrgan_deps, sample_image):
    """Test output shape is correctly scaled."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule()
    output = module(sample_image)
    assert isinstance(output, ModuleOutput)
    # 4x scale: 64 -> 256
    assert output.result.shape == (1, 3, 256, 256)

def test_realesrgan_forward_batch(mock_realesrgan_deps):
    """Test batch processing."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule()
    batch = torch.rand(3, 3, 32, 32)
    output = module(batch)
    assert output.result.shape == (3, 3, 128, 128)

def test_realesrgan_forward_metadata(mock_realesrgan_deps, sample_image):
    """Test metadata is included."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule()
    output = module(sample_image)
    assert "scale" in output.metadata
    assert output.metadata["scale"] == 4

def test_realesrgan_name(mock_realesrgan_deps):
    """Test module name."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule()
    assert module.name == "real_esrgan"

def test_realesrgan_device_handling(mock_realesrgan_deps):
    """Test device parameter."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule(device=torch.device("cpu"))
    assert module.device == torch.device("cpu")


# ============ SuperResModule Wrapper Tests ============

def test_superres_module_registration():
    """Test module is registered."""
    from kp3d.core.registry import ModuleRegistry
    from kp3d.modules.superres import SuperResModule
    registry = ModuleRegistry()
    assert "superres" in registry._modules

def test_superres_module_wrapper(mock_realesrgan_deps):
    """Test wrapper inherits correctly."""
    from kp3d.modules.superres import SuperResModule
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = SuperResModule()
    assert isinstance(module, RealESRGANModule)


# ============ Error Handling Tests ============

def test_realesrgan_invalid_input(mock_realesrgan_deps):
    """Test error on invalid input."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    module = RealESRGANModule()
    with pytest.raises((ValueError, RuntimeError)):
        module(torch.rand(64, 64))  # Missing batch and channel dims
