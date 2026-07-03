"""Utility functions for Korean painting preprocessing.

Provides common utilities for image I/O, tensor manipulation,
logging configuration, and other helper functions.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
            rotation="10 MB",
        )


def load_image(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> Tensor:
    """Load an image as a tensor.

    Args:
        path: Path to image file.
        size: Optional (width, height) to resize.
        grayscale: Convert to grayscale.

    Returns:
        Image tensor (C, H, W) with values in [0, 1].

    Raises:
        FileNotFoundError: If image doesn't exist.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mode = "L" if grayscale else "RGB"
    image = Image.open(path).convert(mode)

    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)

    return TF.to_tensor(image)


def save_image(
    tensor: Tensor,
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """Save a tensor as an image.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W).
        path: Output path.
        quality: JPEG quality (1-100).
    """
    import torchvision.transforms.functional as TF

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.clamp(0, 1)
    image = TF.to_pil_image(tensor)
    image.save(path, quality=quality)


def list_images(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp"),
    recursive: bool = False,
) -> List[Path]:
    """List image files in a directory.

    Args:
        directory: Directory to search.
        extensions: Valid image extensions.
        recursive: Search subdirectories.

    Returns:
        List of image file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    pattern = "**/*" if recursive else "*"
    images = []

    for ext in extensions:
        images.extend(directory.glob(f"{pattern}{ext}"))
        images.extend(directory.glob(f"{pattern}{ext.upper()}"))

    return sorted(set(images))


def ensure_tensor_batch(tensor: Tensor) -> Tensor:
    """Ensure tensor has batch dimension.

    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W).

    Returns:
        Tensor with batch dimension (B, C, H, W).
    """
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def remove_tensor_batch(tensor: Tensor) -> Tensor:
    """Remove batch dimension if size is 1.

    Args:
        tensor: Input tensor (B, C, H, W).

    Returns:
        Tensor (C, H, W) if batch size was 1, otherwise unchanged.
    """
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def get_image_size(tensor: Tensor) -> Tuple[int, int]:
    """Get (height, width) of image tensor.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W).

    Returns:
        Tuple of (height, width).
    """
    if tensor.dim() == 4:
        return (tensor.shape[2], tensor.shape[3])
    return (tensor.shape[1], tensor.shape[2])


def resize_tensor(
    tensor: Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> Tensor:
    """Resize image tensor.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W).
        size: Target (height, width).
        mode: Interpolation mode.

    Returns:
        Resized tensor.
    """
    needs_batch = tensor.dim() == 3
    if needs_batch:
        tensor = tensor.unsqueeze(0)

    result = torch.nn.functional.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )

    if needs_batch:
        result = result.squeeze(0)

    return result


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from .regression import (
    compute_ssim,
    compute_psnr,
    compare_against_golden,
    RegressionMetrics,
)

__all__ = [
    "setup_logging",
    "load_image",
    "save_image",
    "list_images",
    "ensure_tensor_batch",
    "remove_tensor_batch",
    "get_image_size",
    "resize_tensor",
    "set_seed",
    "compute_ssim",
    "compute_psnr",
    "compare_against_golden",
    "RegressionMetrics",
]
