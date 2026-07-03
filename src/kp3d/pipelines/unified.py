"""Unified Pipeline - Separation + Preprocessing for 3D reconstruction.

Combines SeparationPipeline (object separation + inpainting) with
PreprocessingPipeline (quality enhancement) to produce high-quality
layer images ready for 3D reconstruction.

Workflow:
    Input Image
        ↓
    [SeparationPipeline] → foreground, background_inpainted
        ↓
    [PreprocessingPipeline] → enhance each layer
        ↓
    Output: enhanced_foreground, enhanced_background
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torch import Tensor
from loguru import logger

from kp3d.core.config import PipelineConfig
from kp3d.core.device import DeviceManager
from kp3d.modules.occlusion.base import OcclusionConfig
from kp3d.modules.occlusion.pipeline import OcclusionPipeline
from kp3d.pipeline import Pipeline as PreprocessingPipeline


@dataclass
class UnifiedConfig:
    """Configuration for unified pipeline.

    Attributes:
        separation: Config for object separation (occlusion handling).
        preprocessing: Config for quality enhancement.
        process_foreground: Whether to enhance foreground layer.
        process_background: Whether to enhance background layer.
        output_dir: Directory for saving outputs.
    """
    separation: OcclusionConfig = field(default_factory=lambda: OcclusionConfig(
        segmentation_mode="auto",
        sam_model_type="vit_b",
        inpaint_method="controlnet",
        use_auto_refinement=False,
    ))
    preprocessing: Optional[PipelineConfig] = None
    process_foreground: bool = True
    process_background: bool = True
    output_dir: str = "outputs/unified"


@dataclass
class UnifiedOutput:
    """Output from unified pipeline.

    Attributes:
        foreground: Enhanced foreground image (RGBA or RGB).
        background: Enhanced background image (RGB).
        foreground_mask: Binary mask of foreground object.
        background_mask: Binary mask of background object.
        depth_map: Depth estimation map.
        metadata: Processing metadata and intermediate results.
    """
    foreground: np.ndarray
    background: np.ndarray
    foreground_mask: np.ndarray
    background_mask: np.ndarray
    depth_map: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedPipeline:
    """Unified pipeline combining separation and preprocessing.

    Orchestrates the full workflow from raw painting image to
    3D-ready layer images:
    1. Separate foreground/background using occlusion handling
    2. Inpaint hidden regions
    3. Enhance each layer with preprocessing pipeline
    4. Output high-quality layer images

    Example:
        >>> config = UnifiedConfig()
        >>> pipeline = UnifiedPipeline(config)
        >>> output = pipeline.process_file("painting.png")
        >>> cv2.imwrite("foreground.png", output.foreground)
        >>> cv2.imwrite("background.png", output.background)
    """

    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize unified pipeline.

        Args:
            config: Pipeline configuration.
            device: Compute device (auto-detected if None).
        """
        self.config = config or UnifiedConfig()

        # Setup device
        self._device_manager = DeviceManager()
        if device is None:
            self._device = self._device_manager.get_optimal_device()
        else:
            self._device = device
        self._device_manager.set_device(self._device)
        logger.info(f"UnifiedPipeline using device: {self._device}")

        # Initialize sub-pipelines
        self._separation_pipeline: Optional[OcclusionPipeline] = None
        self._preprocessing_pipeline: Optional[PreprocessingPipeline] = None

        # Lazy initialization (to save memory)
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of sub-pipelines."""
        if self._initialized:
            return

        logger.info("Initializing sub-pipelines...")

        # Initialize separation pipeline
        self._separation_pipeline = OcclusionPipeline(
            config=self.config.separation,
            output_dir=str(Path(self.config.output_dir) / "separation"),
        )

        # Initialize preprocessing pipeline (if config provided)
        if self.config.preprocessing is not None:
            self._preprocessing_pipeline = PreprocessingPipeline(
                config=self.config.preprocessing,
            )
        else:
            logger.info("No preprocessing config - skipping enhancement")

        self._initialized = True
        logger.info("Sub-pipelines initialized")

    def process(
        self,
        image: Union[Tensor, np.ndarray],
        save_intermediates: bool = False,
        **kwargs: Any,
    ) -> UnifiedOutput:
        """Process an image through the unified pipeline.

        Args:
            image: Input image as tensor (B, C, H, W) or numpy (H, W, C).
            save_intermediates: Save intermediate results.
            **kwargs: Additional arguments passed to sub-pipelines.

        Returns:
            UnifiedOutput with separated and enhanced layer images.
        """
        self._ensure_initialized()

        # Convert numpy to tensor if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            # HWC -> CHW -> BCHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self._device)

        # Stage 1: Separation
        logger.info("Stage 1: Object separation and inpainting")
        sep_output = self._separation_pipeline(
            image_tensor,
            save_intermediates=save_intermediates,
            **kwargs,
        )

        # Extract results from ModuleOutput
        # result = inpainted background (tensor)
        # intermediate contains foreground, masks, depth
        background = self._tensor_to_numpy(sep_output.result)
        fg_rgb = self._tensor_to_numpy(sep_output.intermediate["foreground"])
        fg_alpha = sep_output.intermediate["foreground_alpha"].squeeze().cpu().numpy()
        fg_mask = sep_output.intermediate["foreground_mask"].squeeze().cpu().numpy()
        bg_mask = sep_output.intermediate["background_mask"].squeeze().cpu().numpy()
        depth_map = sep_output.intermediate["depth_map"].squeeze().cpu().numpy()

        # Apply alpha mask to foreground (white background outside object)
        foreground = np.ones_like(fg_rgb) * 255
        alpha_mask = (fg_alpha > 0.5)
        foreground[alpha_mask] = fg_rgb[alpha_mask]

        # Stage 2: Preprocessing (optional)
        if self._preprocessing_pipeline is not None:
            logger.info("Stage 2: Quality enhancement")

            if self.config.process_foreground:
                logger.info("  Enhancing foreground...")
                foreground = self._enhance_layer(foreground, fg_mask)

            if self.config.process_background:
                logger.info("  Enhancing background...")
                background = self._enhance_layer(background, bg_mask)
        else:
            logger.info("Stage 2: Skipped (no preprocessing config)")

        # Build output
        output = UnifiedOutput(
            foreground=foreground,
            background=background,
            foreground_mask=fg_mask,
            background_mask=bg_mask,
            depth_map=depth_map,
            metadata={
                "separation_metadata": sep_output.metadata,
            },
        )

        # Save outputs
        if save_intermediates:
            self._save_outputs(output)

        return output

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy RGB image.

        Args:
            tensor: Image tensor (B, C, H, W) or (C, H, W).

        Returns:
            Numpy array (H, W, C) as uint8.
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        # CHW -> HWC
        arr = tensor.permute(1, 2, 0).cpu().numpy()
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr

    def _enhance_layer(
        self,
        layer: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Enhance a single layer using preprocessing pipeline.

        Args:
            layer: Layer image (H, W, C) RGB.
            mask: Binary mask of valid regions.

        Returns:
            Enhanced layer image.
        """
        # Convert to tensor
        layer_float = layer.astype(np.float32) / 255.0 if layer.max() > 1 else layer.astype(np.float32)
        layer_tensor = torch.from_numpy(layer_float).permute(2, 0, 1).unsqueeze(0)
        layer_tensor = layer_tensor.to(self._device)

        # Process
        result = self._preprocessing_pipeline.process(layer_tensor)

        # Convert back to numpy
        if isinstance(result, dict):
            # Get final output
            result = list(result.values())[-1].result

        result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result_np = (np.clip(result_np, 0, 1) * 255).astype(np.uint8)

        return result_np

    def _save_outputs(self, output: UnifiedOutput) -> None:
        """Save output images.

        Args:
            output: Pipeline output to save.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get output dimensions (may differ from mask if superres applied)
        out_h, out_w = output.foreground.shape[:2]
        mask_h, mask_w = output.foreground_mask.shape[:2]

        # Resize mask if needed (for superres compatibility)
        if (out_h, out_w) != (mask_h, mask_w):
            fg_mask_resized = cv2.resize(
                output.foreground_mask.astype(np.float32),
                (out_w, out_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            fg_mask_resized = output.foreground_mask

        # Save foreground (with alpha if mask available)
        fg_rgba = np.dstack([
            output.foreground,
            (fg_mask_resized * 255).astype(np.uint8),
        ])
        cv2.imwrite(
            str(output_dir / "foreground.png"),
            cv2.cvtColor(fg_rgba, cv2.COLOR_RGBA2BGRA),
        )

        # Save background
        cv2.imwrite(
            str(output_dir / "background.png"),
            cv2.cvtColor(output.background, cv2.COLOR_RGB2BGR),
        )

        # Save masks
        cv2.imwrite(
            str(output_dir / "foreground_mask.png"),
            (output.foreground_mask * 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(output_dir / "background_mask.png"),
            (output.background_mask * 255).astype(np.uint8),
        )

        # Save depth map (normalized)
        depth_norm = output.depth_map.copy()
        if depth_norm.max() > depth_norm.min():
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
        depth_vis = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_map.png"), depth_vis)

        logger.info(f"Saved outputs to: {output_dir}")

    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> UnifiedOutput:
        """Process an image file.

        Args:
            input_path: Path to input image.
            output_dir: Output directory (uses config default if None).
            **kwargs: Additional arguments.

        Returns:
            UnifiedOutput with processed layers.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Update output dir if specified
        if output_dir is not None:
            self.config.output_dir = str(output_dir)

        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info(f"Processing: {input_path.name} ({image_rgb.shape[1]}x{image_rgb.shape[0]})")

        return self.process(image_rgb, save_intermediates=True, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedPipeline("
            f"device={self._device}, "
            f"separation={self.config.separation.inpaint_method}, "
            f"preprocessing={'enabled' if self.config.preprocessing else 'disabled'})"
        )


__all__ = ["UnifiedPipeline", "UnifiedConfig", "UnifiedOutput"]
