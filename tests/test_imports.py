"""Test that all modules can be imported correctly."""

def test_core_imports():
    """Test core module imports."""
    from kp3d.core.base import BasePreprocessModule, ModuleOutput
    from kp3d.core.config import PipelineConfig
    from kp3d.core.registry import ModuleRegistry, register_module
    from kp3d.core.device import DeviceManager
    assert True


def test_module_imports():
    """Test preprocessing module imports."""
    from kp3d.modules import SuperResModule, EdgeModule, ShadeModule
    assert SuperResModule is not None
    assert EdgeModule is not None
    assert ShadeModule is not None


def test_base_module_imports():
    """Test base module classes."""
    from kp3d.modules.superres.base import BaseSuperResolution
    from kp3d.modules.edge.base import BaseEdgeDetection
    from kp3d.modules.shade.base import BaseShadeGeneration
    assert True


def test_implementation_imports():
    """Test specific implementation imports."""
    from kp3d.modules.superres.real_esrgan import RealESRGANModule
    from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
    from kp3d.modules.shade.lighting import ShadeGeneratorModule
    assert True


def test_pipeline_import():
    """Test pipeline import."""
    from kp3d.pipeline import Pipeline
    assert Pipeline is not None


def test_visualization_imports():
    """Test visualization utilities."""
    from kp3d.visualization import create_side_by_side, visualize_pipeline_results
    assert True
