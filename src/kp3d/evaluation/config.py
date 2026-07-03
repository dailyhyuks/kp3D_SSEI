"""Evaluation configuration (YAML-based).

Defines all settings for evaluation experiments including data paths,
baseline selections, and metric configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        data_dir: Path to annotation data (data_original_painting/data_anno/).
        image_dir: Path to raw images.
        output_dir: Path for results output.
        grid_period_x: Horizontal grid period (pixels) for synthetic grid.
        grid_period_y: Vertical grid period (pixels) for synthetic grid.
        occlusion_types: Synthetic occlusion mask types.
        enhancement_baselines: Baselines for Enhancement (Stage 1) eval.
        inpainting_baselines: Baselines for Inpainting (Stage 4) eval.
        enhancement_metrics: Metrics for Enhancement eval.
        inpainting_metrics: Metrics for Inpainting eval.
        weave_removal_preset: Preset for our WeaveRemoval method.
        max_images: Maximum images to process (0=all).
        dry_run: If True, only validate setup without full computation.
    """

    # Data paths
    data_dir: str = "data_original_painting/data_anno"
    image_dir: str = "data_original_painting/data_anno"
    output_dir: str = "evaluation_results"

    # Synthetic grid parameters (matching real scan characteristics)
    grid_period_x: int = 9
    grid_period_y: int = 7
    grid_modulation_b: float = 0.148
    grid_modulation_g: float = 0.074
    grid_modulation_r: float = 0.045

    # Synthetic occlusion settings
    occlusion_types: List[str] = field(
        default_factory=lambda: ["center_ellipse", "center_rect", "random_blob"]
    )

    # Enhancement baselines (Stage 1: grid/weave removal)
    enhancement_baselines: List[str] = field(
        default_factory=lambda: [
            "bilateral",
            "nlmeans",
            "median",
            "guided",
            "butterworth",
        ]
    )

    # Inpainting baselines (Stage 4)
    inpainting_baselines: List[str] = field(
        default_factory=lambda: ["opencv_telea", "opencv_ns", "lama"]
    )

    # Enhancement metrics
    enhancement_metrics: List[str] = field(
        default_factory=lambda: [
            "psnr",
            "ssim",
            "edge_preservation",
            "grid_energy",
            "band_snr",
            "naturalness",
        ]
    )

    # Inpainting metrics
    inpainting_metrics: List[str] = field(
        default_factory=lambda: ["psnr", "ssim", "lpips", "cor", "bs", "tc"]
    )

    # Our method config
    weave_removal_preset: str = "quality"  # "quality" or "clean"

    # Annotation-based evaluation
    use_annotation_masks: bool = False  # Use LabelMe annotation masks instead of synthetic

    # Execution settings
    max_images: int = 0  # 0 = all
    dry_run: bool = False


def load_config(yaml_path: Optional[str] = None) -> EvalConfig:
    """Load evaluation config from YAML file.

    Args:
        yaml_path: Path to YAML config. If None, returns defaults.

    Returns:
        EvalConfig instance.
    """
    if yaml_path is None:
        return EvalConfig()

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return EvalConfig()

    # Filter to only known fields using dataclass field names
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(EvalConfig)}
    return EvalConfig(**{k: v for k, v in data.items() if k in valid_fields})
