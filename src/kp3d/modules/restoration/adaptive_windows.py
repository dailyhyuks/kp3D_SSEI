"""Adaptive window sizing and continuous sigma selection for v7 restoration.

Provides resolution-aware window computation and edge-guided continuous
sigma maps for improved frequency separation and outlier detection.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class AdaptiveWindowDetector:
    """Computes dynamic window sizes based on image resolution and noise scale."""

    BASE_WINDOWS = (7, 11, 15, 21)
    REFERENCE_SIZE = 1024  # Reference image dimension

    def compute(
        self,
        image_shape: Tuple[int, ...],
        noise_scale: Optional[float] = None,
    ) -> List[int]:
        """Compute adaptive window sizes.

        Args:
            image_shape: Image shape (H, W) or (H, W, C).
            noise_scale: Estimated noise scale (0-1). Higher = larger windows.

        Returns:
            List of odd window sizes.
        """
        h, w = image_shape[:2]
        max_dim = max(h, w)

        # Scale factor based on resolution relative to reference
        scale = max_dim / self.REFERENCE_SIZE

        # Adjust base windows by scale
        if scale < 0.5:
            # Small image: use fewer, smaller windows
            windows = [7, 11]
        elif scale < 1.0:
            windows = [7, 11, 15]
        else:
            windows = list(self.BASE_WINDOWS)

        # Adjust for noise scale if provided
        if noise_scale is not None:
            noise_scale = np.clip(noise_scale, 0.0, 1.0)
            if noise_scale > 0.5:
                # High noise: add larger windows
                max_win = int(21 + (noise_scale - 0.5) * 20)
                max_win = max_win if max_win % 2 == 1 else max_win + 1
                if max_win > windows[-1]:
                    windows.append(max_win)
            elif noise_scale < 0.2:
                # Low noise: keep only smaller windows
                windows = [w for w in windows if w <= 15]

        # Scale window sizes based on resolution
        if scale > 1.5:
            scaled = []
            for ws in windows:
                new_ws = int(ws * min(scale, 2.0))
                new_ws = new_ws if new_ws % 2 == 1 else new_ws + 1
                scaled.append(new_ws)
            windows = scaled

        # Ensure all windows are odd and within bounds
        windows = [max(3, w) for w in windows]
        windows = [w if w % 2 == 1 else w + 1 for w in windows]

        return sorted(set(windows))


class ContinuousSigmaSelector:
    """Generates continuous sigma maps based on edge magnitude.

    Near edges: small sigma (preserve detail)
    Away from edges: large sigma (smooth/restore)
    Uses quantized levels for computational efficiency.
    """

    def __init__(
        self,
        min_sigma: float = 1.5,
        max_sigma: float = 10.0,
        num_levels: int = 10,
    ):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_levels = num_levels
        self._sigma_levels = np.linspace(min_sigma, max_sigma, num_levels)

    @property
    def sigma_levels(self) -> np.ndarray:
        return self._sigma_levels

    def compute_sigma_map(self, edge_magnitude: np.ndarray) -> np.ndarray:
        """Compute per-pixel sigma from edge magnitude.

        Args:
            edge_magnitude: Edge strength map, float32, range [0, 1].
                High value = strong edge = small sigma.

        Returns:
            Sigma map (float32), same spatial dimensions.
        """
        # Invert: strong edges get small sigma
        # edge=1 -> sigma_min, edge=0 -> sigma_max
        sigma_map = self.max_sigma - edge_magnitude * (self.max_sigma - self.min_sigma)
        return sigma_map.astype(np.float32)

    def quantize_sigma_map(self, sigma_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize sigma map to discrete levels and compute interpolation weights.

        Args:
            sigma_map: Continuous sigma values.

        Returns:
            Tuple of (level_indices, interp_weights).
            level_indices: int array, index of lower sigma level.
            interp_weights: float32 array, weight for upper level (0-1).
        """
        # Normalize to [0, num_levels-1]
        normalized = (sigma_map - self.min_sigma) / (self.max_sigma - self.min_sigma + 1e-8)
        normalized = np.clip(normalized, 0, 1) * (self.num_levels - 1)

        lower_idx = np.floor(normalized).astype(np.int32)
        lower_idx = np.clip(lower_idx, 0, self.num_levels - 2)
        upper_weight = normalized - lower_idx.astype(np.float32)

        return lower_idx, upper_weight


def compute_adaptive_windows(
    image_shape: Tuple[int, ...],
    noise_scale: Optional[float] = None,
) -> List[int]:
    """Compute adaptive window sizes for the given image.

    Args:
        image_shape: Image shape (H, W) or (H, W, C).
        noise_scale: Estimated noise scale (0-1).

    Returns:
        List of odd window sizes.
    """
    detector = AdaptiveWindowDetector()
    return detector.compute(image_shape, noise_scale)


def compute_continuous_sigma_map(
    edge_magnitude: np.ndarray,
    min_sigma: float = 1.5,
    max_sigma: float = 10.0,
    levels: int = 10,
) -> np.ndarray:
    """Compute continuous sigma map from edge magnitude.

    Args:
        edge_magnitude: Edge strength (0-1 float32).
        min_sigma: Minimum sigma (near edges).
        max_sigma: Maximum sigma (away from edges).
        levels: Number of quantization levels.

    Returns:
        Per-pixel sigma map (float32).
    """
    selector = ContinuousSigmaSelector(min_sigma, max_sigma, levels)
    return selector.compute_sigma_map(edge_magnitude)


def adaptive_gaussian_blur(
    image: np.ndarray,
    sigma_map: np.ndarray,
    num_levels: int = 10,
    min_sigma: float = 1.5,
    max_sigma: float = 10.0,
) -> np.ndarray:
    """Apply per-pixel adaptive Gaussian blur using discrete levels.

    Pre-computes blurred images at each discrete sigma level, then
    interpolates between adjacent levels based on the sigma map.

    Args:
        image: Input image (float32), shape (H, W) or (H, W, C).
        sigma_map: Per-pixel sigma values (float32), shape (H, W).
        num_levels: Number of discrete sigma levels.
        min_sigma: Minimum sigma value.
        max_sigma: Maximum sigma value.

    Returns:
        Adaptively blurred image (float32).
    """
    selector = ContinuousSigmaSelector(min_sigma, max_sigma, num_levels)

    # Pre-compute blurred versions at each sigma level
    blurred_stack = []
    for sigma in selector.sigma_levels:
        ksize = int(np.ceil(sigma * 6)) | 1  # Ensure odd
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        blurred_stack.append(blurred)

    # Quantize sigma map
    lower_idx, upper_weight = selector.quantize_sigma_map(sigma_map)

    h, w = sigma_map.shape[:2]

    if image.ndim == 3:
        result = np.zeros_like(image)
        upper_weight_nd = upper_weight[:, :, np.newaxis]
        for level in range(num_levels - 1):
            mask = lower_idx == level
            if not np.any(mask):
                continue
            mask_nd = mask[:, :, np.newaxis]
            lower_val = blurred_stack[level]
            upper_val = blurred_stack[level + 1]
            blended = lower_val * (1 - upper_weight_nd) + upper_val * upper_weight_nd
            result = np.where(mask_nd, blended, result)
        # Handle pixels at max level
        mask = lower_idx == (num_levels - 1)
        if np.any(mask):
            mask_nd = mask[:, :, np.newaxis]
            result = np.where(mask_nd, blurred_stack[-1], result)
    else:
        result = np.zeros_like(image)
        for level in range(num_levels - 1):
            mask = lower_idx == level
            if not np.any(mask):
                continue
            lower_val = blurred_stack[level]
            upper_val = blurred_stack[level + 1]
            blended = lower_val * (1 - upper_weight) + upper_val * upper_weight
            result = np.where(mask, blended, result)
        mask = lower_idx == (num_levels - 1)
        if np.any(mask):
            result = np.where(mask, blurred_stack[-1], result)

    return result.astype(np.float32)
