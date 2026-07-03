"""Base classes for preprocessing modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor


@dataclass
class ModuleOutput:
    """Standard output format for preprocessing modules.

    Attributes:
        result: The primary output tensor (processed image).
        intermediate: Dictionary of intermediate processing results for debugging/visualization.
        metadata: Dictionary of metadata about the processing (timing, parameters, etc.).
    """
    result: Tensor
    intermediate: Dict[str, Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModuleOutput":
        """Move all tensors to the specified device."""
        return ModuleOutput(
            result=self.result.to(device),
            intermediate={k: v.to(device) for k, v in self.intermediate.items()},
            metadata=self.metadata.copy(),
        )

    def detach(self) -> "ModuleOutput":
        """Detach all tensors from computation graph."""
        return ModuleOutput(
            result=self.result.detach(),
            intermediate={k: v.detach() for k, v in self.intermediate.items()},
            metadata=self.metadata.copy(),
        )

    def cpu(self) -> "ModuleOutput":
        """Move all tensors to CPU."""
        return self.to(torch.device("cpu"))


class BasePreprocessModule(ABC):
    """Abstract base class for all preprocessing modules.

    All preprocessing modules (super-resolution, edge enhancement, shade normalization)
    must inherit from this class and implement the required abstract methods.

    Attributes:
        device: The device to run computations on (CPU or CUDA).
        dtype: The data type for tensor computations.
        _initialized: Whether the module has been fully initialized with weights.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the preprocessing module.

        Args:
            device: Device for computations. Defaults to CUDA if available.
            dtype: Data type for tensor operations.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._initialized: bool = False

    @abstractmethod
    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Process a single image through the module.

        Args:
            image: Input image tensor of shape (C, H, W) or (B, C, H, W).
            **kwargs: Additional module-specific parameters.

        Returns:
            ModuleOutput containing the processed result and metadata.
        """
        ...

    def __call__(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Make the module callable.

        Validates input and delegates to forward().
        """
        image = self._validate_input(image)
        return self.forward(image, **kwargs)

    def _validate_input(self, image: Tensor) -> Tensor:
        """Validate and preprocess input tensor.

        Args:
            image: Input tensor.

        Returns:
            Validated tensor moved to correct device and dtype.

        Raises:
            ValueError: If input tensor has invalid dimensions.
        """
        if image.dim() not in (3, 4):
            raise ValueError(
                f"Expected 3D (C, H, W) or 4D (B, C, H, W) tensor, got {image.dim()}D"
            )

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        return image.to(device=self.device, dtype=self.dtype)

    @abstractmethod
    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint is incompatible.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the module's unique identifier name."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the module is fully initialized with weights."""
        return self._initialized

    def to(self, device: torch.device) -> "BasePreprocessModule":
        """Move module to the specified device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        self.device = device
        return self

    def half(self) -> "BasePreprocessModule":
        """Convert module to half precision (float16).

        Returns:
            Self for method chaining.
        """
        self.dtype = torch.float16
        return self

    def float(self) -> "BasePreprocessModule":
        """Convert module to full precision (float32).

        Returns:
            Self for method chaining.
        """
        self.dtype = torch.float32
        return self


class BatchProcessor:
    """Utility class for processing batches of images through a module.

    Handles batching, progress tracking, and memory management for
    processing large collections of images.
    """

    def __init__(
        self,
        module: BasePreprocessModule,
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Initialize the batch processor.

        Args:
            module: The preprocessing module to use.
            batch_size: Number of images to process at once.
            show_progress: Whether to display a progress bar.
        """
        self.module = module
        self.batch_size = batch_size
        self.show_progress = show_progress

    def process(
        self,
        images: Union[List[Tensor], Tensor],
        **kwargs: Any,
    ) -> List[ModuleOutput]:
        """Process a collection of images.

        Args:
            images: List of image tensors or a batched tensor.
            **kwargs: Additional parameters passed to the module.

        Returns:
            List of ModuleOutput for each processed image.
        """
        from tqdm import tqdm

        # Convert batched tensor to list
        if isinstance(images, Tensor):
            if images.dim() == 4:
                images = [images[i] for i in range(images.shape[0])]
            else:
                images = [images]

        results: List[ModuleOutput] = []
        iterator = tqdm(images, desc=f"Processing with {self.module.name}") if self.show_progress else images

        for image in iterator:
            output = self.module(image, **kwargs)
            results.append(output.cpu().detach())

            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
