"""Basic pipeline tests."""

import pytest
import torch
from kp3d.pipeline import Pipeline
from kp3d.core.config import PipelineConfig


def test_pipeline_initialization():
    """Test that pipeline initializes correctly."""
    config = PipelineConfig()
    # Disable modules that need weights for quick test
    config.superres.enabled = False
    config.shade.enabled = False

    pipeline = Pipeline(config=config)
    assert pipeline is not None


def test_pipeline_process_shape():
    """Test that output shape is correct."""
    config = PipelineConfig()
    config.superres.enabled = False
    config.shade.enabled = False

    pipeline = Pipeline(config=config)

    # Create dummy input
    dummy_input = torch.rand(3, 64, 64)
    result = pipeline.process(dummy_input)

    assert result.shape[0] == 3  # RGB output
