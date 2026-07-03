"""
Edge-preserving processor for advanced edge preservation during grid pattern removal.

This module provides the EdgePreservingProcessor class which combines multiple
edge detection techniques (Difference of Gaussians, Laplacian of Gaussian,
Structure Tensor) with Perona-Malik anisotropic diffusion to preserve
brushstrokes and fine details while removing grid patterns from Korean paintings.

Key algorithms:
- DoG (Difference of Gaussians): Optimized for brushstroke detection
- LoG (Laplacian of Gaussian): Fine contour detection via zero-crossing
- Structure Tensor: Coherence-based edge strength measurement
- Perona-Malik diffusion: Edge-stopping smoothing that preserves high-contrast edges
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace


class EdgePreservingProcessor:
    """
    Advanced edge preservation processor for grid pattern removal in Korean paintings.

    This processor uses a combination of edge detection methods to create protection
    masks that preserve brushstrokes and fine artistic details while allowing
    grid patterns (low-contrast, repetitive structures) to be removed.

    Attributes:
        dog_sigma1: Smaller sigma for DoG edge detection (default: 1.0)
        dog_sigma2: Larger sigma for DoG edge detection (default: 2.0)
        diffusion_iterations: Number of anisotropic diffusion iterations (default: 10)
        diffusion_kappa: Edge-stopping parameter for diffusion (default: 30.0)
        diffusion_gamma: Update rate for diffusion (default: 0.1)
    """

    def __init__(
        self,
        dog_sigma1: float = 1.0,
        dog_sigma2: float = 2.0,
        diffusion_iterations: int = 10,
        diffusion_kappa: float = 30.0,
        diffusion_gamma: float = 0.1,
    ) -> None:
        """
        Initialize the EdgePreservingProcessor.

        Args:
            dog_sigma1: Smaller sigma for Difference of Gaussians (default: 1.0)
            dog_sigma2: Larger sigma for Difference of Gaussians (default: 2.0)
            diffusion_iterations: Number of Perona-Malik diffusion iterations (default: 10)
            diffusion_kappa: Gradient magnitude threshold for edge-stopping (default: 30.0)
            diffusion_gamma: Update rate per iteration (default: 0.1, should be <= 0.25)
        """
        self.dog_sigma1 = dog_sigma1
        self.dog_sigma2 = dog_sigma2
        self.diffusion_iterations = diffusion_iterations
        self.diffusion_kappa = diffusion_kappa
        self.diffusion_gamma = diffusion_gamma

    @staticmethod
    def _sigma_to_ksize(sigma: float) -> int:
        """
        Convert sigma to appropriate kernel size for Gaussian blur.

        The kernel size is computed as ceil(6*sigma) and ensured to be odd.

        Args:
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Odd integer kernel size
        """
        ksize = int(np.ceil(6 * sigma))
        if ksize % 2 == 0:
            ksize += 1
        return max(3, ksize)

    def difference_of_gaussians(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Compute Difference of Gaussians (DoG) edge detection.

        DoG approximates the Laplacian of Gaussian and is optimized for
        detecting brushstroke edges in paintings. It highlights regions
        where intensity changes occur at the scale defined by sigma1 and sigma2.

        Args:
            image_gray: Grayscale input image (float32 or uint8)

        Returns:
            DoG edge map normalized to [0, 1] as float32
        """
        # Ensure float32 for computation
        if image_gray.dtype != np.float32:
            image = image_gray.astype(np.float32) / 255.0
        else:
            image = image_gray.copy()

        # Compute kernel sizes from sigmas
        ksize1 = self._sigma_to_ksize(self.dog_sigma1)
        ksize2 = self._sigma_to_ksize(self.dog_sigma2)

        # Apply Gaussian blurs
        g1 = cv2.GaussianBlur(image, (ksize1, ksize1), self.dog_sigma1)
        g2 = cv2.GaussianBlur(image, (ksize2, ksize2), self.dog_sigma2)

        # Compute difference of Gaussians
        dog = g1 - g2

        # Normalize to [0, 1]
        dog_min = dog.min()
        dog_max = dog.max()
        if dog_max - dog_min > 1e-8:
            dog_edges = (dog - dog_min) / (dog_max - dog_min)
        else:
            dog_edges = np.zeros_like(dog)

        return dog_edges.astype(np.float32)

    def laplacian_of_gaussian(
        self, image_gray: np.ndarray, sigma: float = 1.5
    ) -> np.ndarray:
        """
        Compute Laplacian of Gaussian (LoG) edge detection with zero-crossing.

        LoG is used for fine contour detection. Zero-crossings in the LoG
        response indicate edge locations with sub-pixel accuracy.

        Args:
            image_gray: Grayscale input image (float32 or uint8)
            sigma: Sigma for the Gaussian component (default: 1.5)

        Returns:
            LoG edge map normalized to [0, 1] as float32
        """
        # Ensure float32 for computation
        if image_gray.dtype != np.float32:
            image = image_gray.astype(np.float32) / 255.0
        else:
            image = image_gray.copy()

        # Apply Laplacian of Gaussian using scipy
        log_response = gaussian_laplace(image, sigma=sigma)

        # Zero-crossing detection for edges
        # An edge exists where the sign changes between adjacent pixels
        h, w = log_response.shape

        # Detect zero crossings in horizontal and vertical directions
        zero_cross = np.zeros((h, w), dtype=np.float32)

        # Horizontal zero crossings
        sign_change_h = log_response[:, :-1] * log_response[:, 1:] < 0
        zero_cross[:, :-1] = np.maximum(
            zero_cross[:, :-1],
            sign_change_h.astype(np.float32)
            * np.abs(log_response[:, :-1] - log_response[:, 1:]),
        )

        # Vertical zero crossings
        sign_change_v = log_response[:-1, :] * log_response[1:, :] < 0
        zero_cross[:-1, :] = np.maximum(
            zero_cross[:-1, :],
            sign_change_v.astype(np.float32)
            * np.abs(log_response[:-1, :] - log_response[1:, :]),
        )

        # Normalize to [0, 1]
        zc_max = zero_cross.max()
        if zc_max > 1e-8:
            log_edges = zero_cross / zc_max
        else:
            log_edges = zero_cross

        return log_edges.astype(np.float32)

    def compute_structure_tensor_edges(
        self, image_gray: np.ndarray, rho: float = 3.0
    ) -> np.ndarray:
        """
        Compute structure tensor-based edge strength using coherence.

        The structure tensor captures local gradient distribution. The coherence
        measure (lambda1 - lambda2) / (lambda1 + lambda2) indicates how
        directional the local structure is - high coherence indicates edges.

        Args:
            image_gray: Grayscale input image (float32 or uint8)
            rho: Sigma for integration scale (Gaussian smoothing of tensor) (default: 3.0)

        Returns:
            Coherence map normalized to [0, 1] as float32
        """
        # Ensure float32 for computation
        if image_gray.dtype != np.float32:
            image = image_gray.astype(np.float32) / 255.0
        else:
            image = image_gray.copy()

        # Compute Sobel gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

        # Compute structure tensor components
        jxx = gx * gx
        jxy = gx * gy
        jyy = gy * gy

        # Smooth the structure tensor components (integration scale)
        ksize = self._sigma_to_ksize(rho)
        jxx = cv2.GaussianBlur(jxx, (ksize, ksize), rho)
        jxy = cv2.GaussianBlur(jxy, (ksize, ksize), rho)
        jyy = cv2.GaussianBlur(jyy, (ksize, ksize), rho)

        # Compute eigenvalues of the 2x2 structure tensor at each pixel
        # For 2x2 matrix [[jxx, jxy], [jxy, jyy]]:
        # lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
        trace = jxx + jyy
        det = jxx * jyy - jxy * jxy

        # Discriminant (ensure non-negative due to numerical errors)
        disc = np.maximum(trace * trace - 4 * det, 0)
        sqrt_disc = np.sqrt(disc)

        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2

        # Ensure lambda1 >= lambda2
        lambda1, lambda2 = np.maximum(lambda1, lambda2), np.minimum(lambda1, lambda2)

        # Compute coherence
        eps = 1e-8
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

        # Coherence is already in [0, 1] for positive eigenvalues
        coherence = np.clip(coherence, 0, 1)

        return coherence.astype(np.float32)

    def compute_edge_protection_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Compute combined edge protection mask from multiple edge detectors.

        Combines DoG, LoG, and Structure Tensor edge detection to create
        a comprehensive protection mask that identifies areas to preserve
        during grid pattern removal.

        Args:
            image_bgr: Input image in BGR format (uint8)

        Returns:
            Edge protection mask in [0, 1] as float32, where 1 = protect edge
        """
        # Convert to grayscale float32
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0

        # Compute individual edge maps
        dog = self.difference_of_gaussians(gray_float)
        log = self.laplacian_of_gaussian(gray_float)
        st = self.compute_structure_tensor_edges(gray_float)

        # Combine edge maps with weighted maximum
        # DoG: 0.4 weight (good for brushstrokes)
        # LoG: 0.3 weight (fine contours)
        # ST: 0.3 weight (coherent structures)
        combined = np.maximum(dog * 0.4, np.maximum(log * 0.3, st * 0.3))

        # Normalize
        c_max = combined.max()
        if c_max > 1e-8:
            combined = combined / c_max

        # Apply light Gaussian blur for smoother transitions
        ksize = 5
        combined = cv2.GaussianBlur(combined, (ksize, ksize), 1.0)

        # Clip to [0, 1]
        combined = np.clip(combined, 0, 1)

        return combined.astype(np.float32)

    def anisotropic_diffusion(
        self,
        image: np.ndarray,
        iterations: Optional[int] = None,
        kappa: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion.

        Anisotropic diffusion smooths homogeneous regions while preserving
        edges. The edge-stopping function prevents diffusion across strong
        gradients (edges) while allowing it in flat regions.

        This is ideal for removing grid patterns (low-contrast, repetitive)
        while preserving brushstrokes (high-contrast, artistic).

        Args:
            image: Input image (grayscale or color, any dtype)
            iterations: Number of diffusion iterations (default: self.diffusion_iterations)
            kappa: Gradient threshold for edge-stopping (default: self.diffusion_kappa)
            gamma: Update rate (default: self.diffusion_gamma, should be <= 0.25)

        Returns:
            Diffused image with same dtype as input
        """
        # Use default parameters if not provided
        if iterations is None:
            iterations = self.diffusion_iterations
        if kappa is None:
            kappa = self.diffusion_kappa
        if gamma is None:
            gamma = self.diffusion_gamma

        # Store original dtype for output
        original_dtype = image.dtype

        # Convert to float for computation
        if image.dtype == np.uint8:
            img = image.astype(np.float32)
        else:
            img = image.astype(np.float32).copy()

        # Handle multi-channel images by processing each channel
        if len(img.shape) == 3:
            channels = [
                self._anisotropic_diffusion_single(img[:, :, c], iterations, kappa, gamma)
                for c in range(img.shape[2])
            ]
            result = np.stack(channels, axis=2)
        else:
            result = self._anisotropic_diffusion_single(img, iterations, kappa, gamma)

        # Convert back to original dtype
        if original_dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = result.astype(original_dtype)

        return result

    def _anisotropic_diffusion_single(
        self, img: np.ndarray, iterations: int, kappa: float, gamma: float
    ) -> np.ndarray:
        """
        Apply Perona-Malik anisotropic diffusion to a single-channel image.

        Args:
            img: Single-channel float32 image
            iterations: Number of diffusion iterations
            kappa: Gradient threshold for edge-stopping
            gamma: Update rate per iteration

        Returns:
            Diffused single-channel image as float32
        """
        img = img.copy()
        h, w = img.shape

        for _ in range(iterations):
            # Compute 4-directional finite differences with padding
            # North: pixel above - current pixel
            d_north = np.zeros_like(img)
            d_north[1:, :] = img[:-1, :] - img[1:, :]

            # South: pixel below - current pixel
            d_south = np.zeros_like(img)
            d_south[:-1, :] = img[1:, :] - img[:-1, :]

            # East: pixel right - current pixel
            d_east = np.zeros_like(img)
            d_east[:, :-1] = img[:, 1:] - img[:, :-1]

            # West: pixel left - current pixel
            d_west = np.zeros_like(img)
            d_west[:, 1:] = img[:, :-1] - img[:, 1:]

            # Compute edge-stopping function (Perona-Malik function 1)
            # c = exp(-(|grad|/kappa)^2)
            c_north = np.exp(-((np.abs(d_north) / kappa) ** 2))
            c_south = np.exp(-((np.abs(d_south) / kappa) ** 2))
            c_east = np.exp(-((np.abs(d_east) / kappa) ** 2))
            c_west = np.exp(-((np.abs(d_west) / kappa) ** 2))

            # Update all pixels simultaneously
            img += gamma * (
                c_north * d_north + c_south * d_south + c_east * d_east + c_west * d_west
            )

        return img

    def edge_aware_grid_removal(
        self,
        image_bgr: np.ndarray,
        grid_removed_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blend grid-removed result with original using edge protection.

        In edge regions (high mask values), the original image is preserved
        to maintain brushstroke details. In non-edge regions (low mask values),
        the grid-removed result is used for clean output.

        Args:
            image_bgr: Original image in BGR format (uint8)
            grid_removed_bgr: Grid-removed image in BGR format (uint8)
            edge_mask: Optional pre-computed edge protection mask (float32 [0,1]).
                       If None, computed via compute_edge_protection_mask.

        Returns:
            Edge-protected result as uint8 BGR image
        """
        # Compute edge mask if not provided
        if edge_mask is None:
            edge_mask = self.compute_edge_protection_mask(image_bgr)

        # Ensure edge_mask is float32 in [0, 1]
        if edge_mask.dtype != np.float32:
            edge_mask = edge_mask.astype(np.float32)
        if edge_mask.max() > 1.0:
            edge_mask = edge_mask / 255.0

        # Expand mask to 3 channels for blending
        if len(edge_mask.shape) == 2:
            edge_mask_3ch = edge_mask[:, :, np.newaxis]
        else:
            edge_mask_3ch = edge_mask

        # Convert images to float for blending
        original_float = image_bgr.astype(np.float32)
        grid_removed_float = grid_removed_bgr.astype(np.float32)

        # Blend: edge regions keep original, non-edge regions use grid-removed
        # result = edge_mask * original + (1 - edge_mask) * grid_removed
        result = edge_mask_3ch * original_float + (1 - edge_mask_3ch) * grid_removed_float

        # Clip and convert to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


__all__ = ["EdgePreservingProcessor"]
