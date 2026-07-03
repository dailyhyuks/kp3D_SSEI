"""Evaluation experiments for the Korean Painting 3D pipeline.

- restoration_eval: Stage 1 Enhancement (Weave Removal) evaluation
- inpainting_eval: Stage 4 Inpainting (SSEI V25) evaluation
- e2e_ablation: End-to-End pipeline ablation study
"""

from kp3d.evaluation.experiments.restoration_eval import RestorationExperiment
from kp3d.evaluation.experiments.inpainting_eval import InpaintingExperiment
from kp3d.evaluation.experiments.e2e_ablation import AblationExperiment

__all__ = [
    "RestorationExperiment",
    "InpaintingExperiment",
    "AblationExperiment",
]
