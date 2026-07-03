"""Base classes and configuration for Super Resolution modules."""

from enum import Enum
from typing import Any, Optional

import torch
from pydantic import BaseModel, Field
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput


class ScaleFactor(Enum):
    """Supported upscaling factors."""
    X2 = 2
    X4 = 4


class SuperResConfig(BaseModel):
    """Configuration for super-resolution processing.

    Attributes:
        scale: Upscaling factor (2x or 4x).
        denoise_strength: Denoising strength (0.0 to 1.0).
        tile_size: Size of tiles for processing large images.
        tile_overlap: Overlap between tiles to avoid seam artifacts.
        preserve_ink_lines: Whether to preserve fine ink line details.
        model_name: Name of the model to use.
    """
    scale: ScaleFactor = ScaleFactor.X4
    denoise_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    tile_size: int = Field(default=512, gt=0)
    tile_overlap: int = Field(default=32, ge=0)
    preserve_ink_lines: bool = True
    model_name: str = "RealESRGAN_x4plus"


class BaseSuperResolution(BasePreprocessModule):
    """Abstract base class for super-resolution modules.

    Provides common functionality for upscaling images while preserving
    details characteristic of Korean traditional paintings.

    Attributes:
        config: Super-resolution configuration.
        scale: Current upscaling factor.
    """

    def __init__(
        self,
        config: Optional[SuperResConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """Initialize the super-resolution module.

        Args:
            config: Configuration object. If None, uses defaults.
            device: Computation device. Defaults to CUDA if available.
            dtype: Data type for tensor operations.
            **kwargs: Additional configuration overrides.
        """
        super().__init__(device=device, dtype=dtype)

        # Initialize config with overrides from kwargs
        self.config = config or SuperResConfig(**kwargs)
        self.scale = self.config.scale.value

    def _tile_image(
        self,
        image: Tensor,
        tile_size: int,
        overlap: int,
    ) -> tuple[list[Tensor], list[tuple[int, int, int, int]]]:
        """Split image into overlapping tiles for processing.

        Args:
            image: Input image tensor (B, C, H, W).
            tile_size: Size of each tile.
            overlap: Overlap between tiles.

        Returns:
            Tuple of (tiles, positions) where:
                - tiles: List of tile tensors
                - positions: List of (y1, y2, x1, x2) coordinates for each tile
        """
        b, c, h, w = image.shape
        stride = tile_size - overlap

        tiles = []
        positions = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1 = y
                y2 = min(y + tile_size, h)
                x1 = x
                x2 = min(x + tile_size, w)

                # Extract tile
                tile = image[:, :, y1:y2, x1:x2]

                # Pad if needed to maintain tile_size
                if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                    pad_h = tile_size - tile.shape[2]
                    pad_w = tile_size - tile.shape[3]
                    tile = torch.nn.functional.pad(
                        tile,
                        (0, pad_w, 0, pad_h),
                        mode='reflect'
                    )

                tiles.append(tile)
                positions.append((y1, y2, x1, x2))

        return tiles, positions

    def _merge_tiles(
        self,
        tiles: list[Tensor],
        positions: list[tuple[int, int, int, int]],
        output_shape: tuple[int, int, int, int],
        scale: int,
    ) -> Tensor:
        """Merge processed tiles back into a single image.

        Args:
            tiles: List of processed tile tensors.
            positions: List of original positions (y1, y2, x1, x2).
            output_shape: Shape of the output image (B, C, H, W).
            scale: Upscaling factor.

        Returns:
            Merged output tensor.
        """
        b, c, h_out, w_out = output_shape

        # Create output tensor and weight map for blending
        output = torch.zeros(output_shape, dtype=self.dtype, device=self.device)
        weight_map = torch.zeros(output_shape, dtype=self.dtype, device=self.device)

        # Create a linear blend weight for overlapping regions
        overlap = self.config.tile_overlap * scale

        for tile, (y1, y2, x1, x2) in zip(tiles, positions):
            # Scale positions
            sy1, sy2 = y1 * scale, y2 * scale
            sx1, sx2 = x1 * scale, x2 * scale

            # Get actual tile size (may be padded)
            tile_h, tile_w = sy2 - sy1, sx2 - sx1
            tile = tile[:, :, :tile_h, :tile_w]

            # Create blend weights (1.0 at center, fading to 0.5 at edges)
            tile_weight = torch.ones_like(tile)
            if overlap > 0:
                # Linear fade on edges
                for i in range(min(overlap, tile_h)):
                    alpha = (i + 1) / overlap
                    tile_weight[:, :, i, :] *= alpha
                    tile_weight[:, :, -(i+1), :] *= alpha
                for j in range(min(overlap, tile_w)):
                    alpha = (j + 1) / overlap
                    tile_weight[:, :, :, j] *= alpha
                    tile_weight[:, :, :, -(j+1)] *= alpha

            # Accumulate weighted tiles
            output[:, :, sy1:sy2, sx1:sx2] += tile * tile_weight
            weight_map[:, :, sy1:sy2, sx1:sx2] += tile_weight

        # Normalize by weight map
        output = output / (weight_map + 1e-8)

        return output

    def _denoise(self, image: Tensor, strength: float) -> Tensor:
        """Apply denoising to the image.

        Args:
            image: Input tensor.
            strength: Denoising strength (0.0 to 1.0).

        Returns:
            Denoised tensor.
        """
        if strength <= 0.0:
            return image

        # Simple bilateral filter approximation using Gaussian blur
        # Real implementation would use more sophisticated denoising
        kernel_size = int(3 + strength * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = strength * 2.0

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=self.dtype, device=self.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Apply separable convolution (channel-wise)
        num_channels = image.shape[1]

        # Expand kernel for each channel: [num_channels, 1, kernel_size, 1]
        kernel_v = g.view(1, 1, -1, 1).expand(num_channels, 1, -1, 1)
        image = torch.nn.functional.conv2d(
            image,
            kernel_v,
            padding=(kernel_size // 2, 0),
            groups=num_channels
        )

        kernel_h = g.view(1, 1, 1, -1).expand(num_channels, 1, 1, -1)
        image = torch.nn.functional.conv2d(
            image,
            kernel_h,
            padding=(0, kernel_size // 2),
            groups=num_channels
        )

        return image

    @property
    def name(self) -> str:
        """Return the module's unique identifier name."""
        return "superres_base"
