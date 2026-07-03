"""Pipeline orchestration for Korean painting preprocessing.

Provides the main Pipeline class that orchestrates multiple preprocessing
modules in sequence with proper resource management.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from loguru import logger

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.config import PipelineConfig
from kp3d.core.device import DeviceManager
from kp3d.core.registry import ModuleRegistry


class Pipeline:
    """Main preprocessing pipeline orchestrator.

    Manages the execution of multiple preprocessing modules in sequence,
    handling device management, intermediate results, and error recovery.

    Supports edge preservation workflow:
    1. Extract edges from original image
    2. Run restoration/denoising
    3. Resynthesize original edges into restored image
    4. Upscale with super-resolution
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration object.
            config_path: Path to YAML configuration file.
                         Ignored if config is provided.
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()

        self._device_manager = DeviceManager()
        self._modules: Dict[str, BasePreprocessModule] = {}
        self._preserved_edges: Optional[Tensor] = None  # Store original edges
        self._setup_device()
        self._initialize_modules()

    def _setup_device(self) -> None:
        """Configure compute device based on config."""
        if self.config.device == "auto":
            self._device = self._device_manager.get_optimal_device()
        else:
            self._device = torch.device(self.config.device)

        self._device_manager.set_device(self._device)
        logger.info(f"Pipeline using device: {self._device}")

    def _initialize_modules(self) -> None:
        """Initialize preprocessing modules based on config."""
        enabled_modules = self.config.get_enabled_modules()

        for module_name in enabled_modules:
            module_config = getattr(self.config, module_name, None)
            if module_config is None:
                logger.warning(f"No config found for module '{module_name}', skipping")
                continue

            # Build kwargs for module initialization
            kwargs = {}

            # Add module-specific parameter mappings
            if module_name == "edge":
                kwargs["method"] = module_config.model_name
                kwargs["threshold"] = module_config.threshold
            elif module_name == "restoration":
                kwargs["method"] = getattr(module_config, "method", "fading_noise")
            elif module_name == "edge_resynth":
                # Edge resynth uses the edge module with resynth method
                from kp3d.modules.edge.resynthesizer import EdgeResynthesizer, EdgeResynthesizerConfig
                resynth_config = EdgeResynthesizerConfig(
                    edge_weight=module_config.edge_weight,
                    current_weight=module_config.current_weight,
                    unsharp_strength=module_config.unsharp_strength,
                    unsharp_radius=module_config.unsharp_radius,
                    blend_mode=module_config.blend_mode,
                    preserve_original_edges=module_config.preserve_original_edges,
                    edge_detection_method=getattr(module_config, "edge_detection_method", "sobel"),
                )
                module = EdgeResynthesizer(
                    device=self._device,
                    resynth_config=resynth_config,
                )
                self._modules[module_name] = module
                logger.info(f"Initialized module: {module_name} (edge method: {resynth_config.edge_detection_method})")
                continue

            try:
                module = ModuleRegistry.get(module_name, device=self._device, **kwargs)
                self._modules[module_name] = module
                logger.info(f"Initialized module: {module_name}")
            except KeyError:
                logger.warning(f"Module '{module_name}' not registered, skipping")

    @property
    def device(self) -> torch.device:
        """Get current compute device."""
        return self._device

    @property
    def modules(self) -> Dict[str, BasePreprocessModule]:
        """Get dictionary of active modules."""
        return self._modules.copy()

    def process(
        self,
        image: Tensor,
        modules: Optional[List[str]] = None,
        return_intermediate: bool = False,
        preserve_edges: bool = True,
        **kwargs: Any,
    ) -> Union[Tensor, Dict[str, ModuleOutput]]:
        """Process an image through the pipeline.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W).
            modules: Specific modules to run. Uses config order if None.
            return_intermediate: Return all intermediate results.
            preserve_edges: Extract and preserve original edges for resynthesis.
            **kwargs: Additional parameters passed to modules.

        Returns:
            Final processed tensor, or dict of all outputs if return_intermediate.
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self._device)
        current = image
        outputs: Dict[str, ModuleOutput] = {}

        # Determine which modules to run
        module_order = modules or self.config.get_enabled_modules()

        # Pre-extract edges if edge_resynth is in the pipeline
        if preserve_edges and "edge_resynth" in module_order:
            self._extract_and_preserve_edges(image)

        for module_name in module_order:
            if module_name not in self._modules:
                logger.warning(f"Module '{module_name}' not available, skipping")
                continue

            module = self._modules[module_name]
            logger.debug(f"Running module: {module_name}")

            try:
                # Special handling for edge_resynth - pass preserved edges
                if module_name == "edge_resynth" and self._preserved_edges is not None:
                    output = module(current, original_edges=self._preserved_edges, **kwargs)
                else:
                    output = module(current, **kwargs)

                outputs[module_name] = output
                current = output.result

                # Clear cache periodically
                self._device_manager.clear_cache()

            except Exception as e:
                logger.error(f"Error in module '{module_name}': {e}")
                raise

        if return_intermediate:
            return outputs

        return current.squeeze(0) if current.shape[0] == 1 else current

    def _extract_and_preserve_edges(self, image: Tensor) -> None:
        """Extract and store edges from the original image using Sobel.

        Uses Sobel edge detection which achieves F1=0.908 against pseudo GT,
        significantly better than Canny (F1=0.804).
        See research/contour_enhancement/IDEA.md for experiment details.

        Args:
            image: Original input image tensor.
        """
        import cv2
        import numpy as np

        # Convert tensor to numpy
        if image.dim() == 4:
            img = image[0]
        else:
            img = image

        img_np = img.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        # Convert to grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        # Use Sobel for edge detection (best method from experiments)
        # Sobel F1=0.908 vs Canny F1=0.804 against pseudo GT
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Convert back to tensor
        edges_tensor = torch.from_numpy(edges.astype(np.float32) / 255.0)
        edges_tensor = edges_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self._preserved_edges = edges_tensor.to(self._device)
        logger.debug("Preserved original edges using Sobel for resynthesis")

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Process an image file.

        Args:
            input_path: Path to input image.
            output_path: Path for output. Auto-generated if None.
            **kwargs: Additional parameters.

        Returns:
            Processed image tensor.
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load image
        image = Image.open(input_path).convert("RGB")
        tensor = TF.to_tensor(image)

        # Process
        result = self.process(tensor, **kwargs)

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            result_image = TF.to_pil_image(result.clamp(0, 1))
            result_image.save(output_path, quality=self.config.output.quality)
            logger.info(f"Saved output to: {output_path}")

        return result

    def __repr__(self) -> str:
        """String representation."""
        modules = ", ".join(self._modules.keys())
        return f"Pipeline(device={self._device}, modules=[{modules}])"


__all__ = ["Pipeline"]
