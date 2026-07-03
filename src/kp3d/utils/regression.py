"""Visual regression testing utilities.

Provides image comparison metrics (SSIM, PSNR) and functions to compare
outputs against golden reference images for regression testing.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass
class RegressionMetrics:
    """Container for regression test metrics.

    Attributes:
        ssim: Structural Similarity Index (0-1, higher is better).
        psnr: Peak Signal-to-Noise Ratio in dB (higher is better).
        mse: Mean Squared Error (lower is better).
        passed: Whether the comparison passed thresholds.
        message: Optional message about the comparison.
    """

    ssim: float
    psnr: float
    mse: float
    passed: bool
    message: str = ""

    def __str__(self) -> str:
        """String representation of metrics."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Regression Test: {status}\n"
            f"  SSIM: {self.ssim:.4f}\n"
            f"  PSNR: {self.psnr:.2f} dB\n"
            f"  MSE:  {self.mse:.6f}\n"
            f"  {self.message}"
        )


def _ensure_numpy(image: Union[Tensor, np.ndarray, str, Path]) -> np.ndarray:
    """Convert various image types to numpy array.

    Args:
        image: Image as tensor, array, or path.

    Returns:
        Numpy array in range [0, 255], shape (H, W) or (H, W, C).
    """
    if isinstance(image, (str, Path)):
        # Load from file
        img = Image.open(image)
        arr = np.array(img)
    elif isinstance(image, Tensor):
        # Convert tensor to numpy
        if image.dim() == 4:
            image = image[0]  # Remove batch
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            image = image.permute(1, 2, 0)  # CHW to HWC
        arr = image.cpu().numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure range [0, 255]
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)

    # Squeeze single-channel dimension for consistency
    # (H, W, 1) -> (H, W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)

    return arr


def compute_ssim(
    image1: Union[Tensor, np.ndarray, str, Path],
    image2: Union[Tensor, np.ndarray, str, Path],
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 255.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    Implementation based on Wang et al., "Image Quality Assessment: From Error
    Visibility to Structural Similarity", IEEE TIP 2004.

    Args:
        image1: First image.
        image2: Second image.
        window_size: Size of the Gaussian window (must be odd).
        k1: Small constant for stability (typically 0.01).
        k2: Small constant for stability (typically 0.03).
        data_range: Dynamic range of the data (255 for uint8).

    Returns:
        SSIM value in range [0, 1], where 1 means identical images.

    Raises:
        ValueError: If images have different shapes.
    """
    # Convert to numpy arrays
    img1 = _ensure_numpy(image1).astype(np.float64)
    img2 = _ensure_numpy(image2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes must match: {img1.shape} vs {img2.shape}"
        )

    # Handle multi-channel by averaging SSIM across channels
    if img1.ndim == 3:
        ssim_values = []
        for c in range(img1.shape[2]):
            ssim_c = _compute_ssim_single_channel(
                img1[:, :, c],
                img2[:, :, c],
                window_size=window_size,
                k1=k1,
                k2=k2,
                data_range=data_range,
            )
            ssim_values.append(ssim_c)
        return float(np.mean(ssim_values))
    else:
        return _compute_ssim_single_channel(
            img1, img2, window_size, k1, k2, data_range
        )


def _compute_ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    k1: float,
    k2: float,
    data_range: float,
) -> float:
    """Compute SSIM for a single channel.

    Args:
        img1: First image (H, W).
        img2: Second image (H, W).
        window_size: Gaussian window size.
        k1, k2: Stability constants.
        data_range: Dynamic range.

    Returns:
        SSIM value.
    """
    from scipy.ndimage import gaussian_filter

    # Constants
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Gaussian window sigma
    sigma = 1.5

    # Compute local means
    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    # Compute local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    # Return mean SSIM
    return float(np.mean(ssim_map))


def compute_psnr(
    image1: Union[Tensor, np.ndarray, str, Path],
    image2: Union[Tensor, np.ndarray, str, Path],
    data_range: float = 255.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        image1: First image.
        image2: Second image.
        data_range: Maximum possible pixel value (255 for uint8).

    Returns:
        PSNR value in decibels (dB). Higher is better.
        Returns infinity if images are identical.

    Raises:
        ValueError: If images have different shapes.
    """
    # Convert to numpy arrays
    img1 = _ensure_numpy(image1).astype(np.float64)
    img2 = _ensure_numpy(image2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes must match: {img1.shape} vs {img2.shape}"
        )

    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")

    # PSNR formula
    psnr = 10 * np.log10((data_range ** 2) / mse)

    return float(psnr)


def compute_mse(
    image1: Union[Tensor, np.ndarray, str, Path],
    image2: Union[Tensor, np.ndarray, str, Path],
) -> float:
    """Compute Mean Squared Error (MSE) between two images.

    Args:
        image1: First image.
        image2: Second image.

    Returns:
        MSE value (lower is better).

    Raises:
        ValueError: If images have different shapes.
    """
    img1 = _ensure_numpy(image1).astype(np.float64)
    img2 = _ensure_numpy(image2).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes must match: {img1.shape} vs {img2.shape}"
        )

    mse = np.mean((img1 - img2) ** 2)
    return float(mse)


def compare_against_golden(
    test_image: Union[Tensor, np.ndarray, str, Path],
    golden_name: str,
    golden_dir: Union[str, Path] = "outputs/golden",
    ssim_threshold: float = 0.95,
    psnr_threshold: float = 30.0,
) -> RegressionMetrics:
    """Compare a test image against a golden reference.

    Args:
        test_image: Test image to compare.
        golden_name: Name of golden reference (e.g., "edge/boat_painting_2_hed.png").
        golden_dir: Directory containing golden references.
        ssim_threshold: Minimum SSIM to pass (0-1).
        psnr_threshold: Minimum PSNR to pass (dB).

    Returns:
        RegressionMetrics with comparison results.

    Raises:
        FileNotFoundError: If golden reference doesn't exist.
    """
    golden_dir = Path(golden_dir)
    golden_path = golden_dir / golden_name

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden reference not found: {golden_path}\n"
            f"Run generate_golden_references.py to create it."
        )

    # Load golden reference
    golden_image = _ensure_numpy(golden_path)

    # Compute metrics
    ssim = compute_ssim(test_image, golden_image)
    psnr = compute_psnr(test_image, golden_image)
    mse = compute_mse(test_image, golden_image)

    # Check thresholds
    ssim_pass = ssim >= ssim_threshold
    psnr_pass = psnr >= psnr_threshold
    passed = ssim_pass and psnr_pass

    # Generate message
    if passed:
        message = f"Test passed (SSIM >= {ssim_threshold}, PSNR >= {psnr_threshold})"
    else:
        reasons = []
        if not ssim_pass:
            reasons.append(f"SSIM {ssim:.4f} < {ssim_threshold}")
        if not psnr_pass:
            reasons.append(f"PSNR {psnr:.2f} < {psnr_threshold}")
        message = f"Test failed: {', '.join(reasons)}"

    return RegressionMetrics(
        ssim=ssim,
        psnr=psnr,
        mse=mse,
        passed=passed,
        message=message,
    )


def load_manifest(
    golden_dir: Union[str, Path] = "outputs/golden",
) -> dict:
    """Load golden reference manifest.

    Args:
        golden_dir: Directory containing manifest.json.

    Returns:
        Manifest dictionary.

    Raises:
        FileNotFoundError: If manifest doesn't exist.
    """
    manifest_path = Path(golden_dir) / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run generate_golden_references.py to create it."
        )

    with open(manifest_path, "r") as f:
        return json.load(f)


__all__ = [
    "RegressionMetrics",
    "compute_ssim",
    "compute_psnr",
    "compute_mse",
    "compare_against_golden",
    "load_manifest",
]
