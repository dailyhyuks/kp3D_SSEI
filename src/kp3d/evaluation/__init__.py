"""Evaluation framework for Korean Painting 3D preprocessing pipeline.

Provides quantitative and qualitative evaluation of:
- Stage 1 Enhancement (Weave Removal via Spectral Interpolation)
- Stage 4 Inpainting (SSEI V25 PatchMatch + Dynamic Edge Morphology)
- E2E Pipeline Ablation Study

Uses existing metrics from kp3d.metrics (MetricsCalculator, InpaintingMetrics).
"""

from kp3d.evaluation.config import EvalConfig, load_config

__all__ = [
    "EvalConfig",
    "load_config",
]
