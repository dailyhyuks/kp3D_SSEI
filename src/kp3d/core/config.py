"""Configuration models using Pydantic for validation and serialization."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class SuperResConfig(BaseModel):
    """Configuration for super-resolution module."""

    enabled: bool = True
    model_name: str = Field(
        default="realesrgan",
        description="Super-resolution model to use",
    )
    scale: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Upscaling factor",
    )
    tile_size: int = Field(
        default=512,
        ge=64,
        description="Tile size for processing large images",
    )
    tile_overlap: int = Field(
        default=32,
        ge=0,
        description="Overlap between tiles",
    )
    half_precision: bool = Field(
        default=True,
        description="Use FP16 for faster inference",
    )
    denoise_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Denoising strength (0=off, 1=max)",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Custom checkpoint path (uses default if None)",
    )

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: int) -> int:
        """Ensure scale is a power of 2."""
        if v not in (1, 2, 4, 8):
            raise ValueError("Scale must be 1, 2, 4, or 8")
        return v


class EdgeConfig(BaseModel):
    """Configuration for edge enhancement module."""

    enabled: bool = True
    model_name: str = Field(
        default="hed",
        description="Edge detection model (hed, pidinet, etc.)",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Edge detection threshold",
    )
    nms: bool = Field(
        default=True,
        description="Apply non-maximum suppression",
    )
    enhance_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Edge enhancement strength",
    )
    line_refinement: bool = Field(
        default=True,
        description="Apply line refinement post-processing",
    )
    min_line_length: int = Field(
        default=10,
        ge=0,
        description="Minimum line length to keep",
    )
    checkpoint_path: Optional[str] = None


class RestorationConfig(BaseModel):
    """Configuration for restoration module."""

    enabled: bool = True
    method: str = Field(
        default="fading_noise",
        description="Restoration method",
    )
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Restoration strength",
    )
    preserve_edges: bool = Field(
        default=True,
        description="Try to preserve edges during restoration",
    )
    denoise_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Denoising strength",
    )
    checkpoint_path: Optional[str] = None


class EdgeResynthConfig(BaseModel):
    """Configuration for edge resynthesis module.

    Research Notes (contour_enhancement experiments):
        - Sobel achieves F1=0.908 (best) vs Canny F1=0.804 vs Scharr F1=0.869
        - Edge extraction BEFORE upscaling preserves contours better
        - See research/contour_enhancement/IDEA.md for details
    """

    enabled: bool = True
    edge_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for original edges",
    )
    current_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for current edges",
    )
    unsharp_strength: float = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="Unsharp masking strength",
    )
    unsharp_radius: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Unsharp masking radius",
    )
    blend_mode: str = Field(
        default="weighted",
        description="Edge blend mode: weighted, max, overlay, softlight",
    )
    preserve_original_edges: bool = Field(
        default=True,
        description="Whether to preserve original edges",
    )
    edge_detection_method: str = Field(
        default="sobel",
        description="Edge detection method: sobel (recommended), canny, scharr",
    )


class ShadeConfig(BaseModel):
    """Configuration for shade normalization module."""

    enabled: bool = True
    model_name: str = Field(
        default="shade_net",
        description="Shade normalization model",
    )
    target_illumination: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Target average illumination",
    )
    preserve_details: bool = Field(
        default=True,
        description="Preserve fine details during normalization",
    )
    color_correction: bool = Field(
        default=True,
        description="Apply color correction",
    )
    gamma: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Gamma correction value",
    )
    checkpoint_path: Optional[str] = None


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    output_dir: str = Field(
        default="outputs",
        description="Directory for output files",
    )
    save_intermediate: bool = Field(
        default=True,
        description="Save intermediate processing results",
    )
    format: Literal["png", "jpg", "tiff"] = Field(
        default="png",
        description="Output image format",
    )
    quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality (only for jpg format)",
    )
    create_visualization: bool = Field(
        default=True,
        description="Create comparison visualization",
    )
    visualization_layout: Literal["grid", "side_by_side", "overlay"] = Field(
        default="grid",
        description="Layout for comparison visualization",
    )


class PipelineConfig(BaseModel):
    """Main configuration for the preprocessing pipeline."""

    # Module configurations
    superres: SuperResConfig = Field(default_factory=SuperResConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    shade: ShadeConfig = Field(default_factory=ShadeConfig)
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    edge_resynth: EdgeResynthConfig = Field(default_factory=EdgeResynthConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Pipeline settings
    processing_order: List[str] = Field(
        default=["superres", "edge", "shade"],
        description="Order of processing modules",
    )
    device: str = Field(
        default="auto",
        description="Device for computation (auto, cpu, cuda, cuda:0, etc.)",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for processing",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loading workers",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging",
    )

    @field_validator("processing_order")
    @classmethod
    def validate_processing_order(cls, v: List[str]) -> List[str]:
        """Validate processing order contains valid module names."""
        valid_modules = {"superres", "edge", "shade", "restoration", "edge_resynth"}
        for module in v:
            if module not in valid_modules:
                raise ValueError(
                    f"Invalid module '{module}'. Must be one of: {valid_modules}"
                )
        return v

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            PipelineConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValidationError: If config is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path for output YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def get_enabled_modules(self) -> List[str]:
        """Get list of enabled modules in processing order.

        Returns:
            List of enabled module names.
        """
        enabled = []
        for module_name in self.processing_order:
            config = getattr(self, module_name)
            if config.enabled:
                enabled.append(module_name)
        return enabled


class AppSettings(BaseSettings):
    """Application-level settings from environment variables."""

    kp3d_config_path: str = Field(
        default="configs/default.yaml",
        description="Default configuration file path",
    )
    kp3d_cache_dir: str = Field(
        default=".cache/kp3d",
        description="Cache directory for model weights",
    )
    kp3d_log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    kp3d_device: str = Field(
        default="auto",
        description="Default compute device",
    )

    class Config:
        """Pydantic settings configuration."""

        env_prefix = ""
        case_sensitive = False
