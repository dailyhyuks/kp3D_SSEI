"""Advanced edge detection with LAB fusion and morphological denoising."""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

# Try to import skimage for thinning, fallback to custom implementation
try:
    from skimage.morphology import thin as skimage_thin
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class AdvancedEdgeDetector(BaseEdgeDetection):
    """High-performance edge detector with LAB fusion.

    Features:
    1. LAB color space multi-channel edge detection
    2. Morphological denoising for clean edges
    3. Adaptive thresholding per channel
    4. Edge recovery and connection
    5. Skeleton thinning for crisp lines

    This detector is optimized for Korean traditional paintings where:
    - External silhouettes must be preserved accurately
    - Internal color boundaries (even subtle ones) should be detected
    - Noise from paper texture should be removed
    """

    def __init__(self, config: Optional[EdgeConfig] = None, **kwargs):
        """Initialize advanced edge detector.

        Args:
            config: Edge detection configuration.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(config=config, **kwargs)
        self._initialized = True

    @property
    def name(self) -> str:
        """Module name."""
        return "advanced_edge"

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights (not applicable for this detector).

        Args:
            checkpoint_path: Path to checkpoint (ignored).
        """
        self._initialized = True

    def _tensor_to_numpy_rgb(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to RGB numpy array.

        Args:
            tensor: Input tensor (B, C, H, W) or (C, H, W).

        Returns:
            RGB numpy array in HWC format, uint8 range [0, 255].
        """
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray) -> Tensor:
        """Convert numpy array to tensor.

        Args:
            arr: Input array (H, W) or (H, W, C).

        Returns:
            Tensor in CHW format, float32 range [0, 1].
        """
        arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[2] in [1, 3]:
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def detect_lab_edges(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect edges in LAB color space for each channel.

        LAB color space is perceptually uniform:
        - L: Lightness - captures structural edges
        - A: Green-Red axis - captures color boundaries
        - B: Blue-Yellow axis - captures color boundaries

        Args:
            image_rgb: RGB image in HWC format, uint8.

        Returns:
            Dictionary with keys 'L', 'A', 'B' (Canny edges) and
            'grad_L', 'grad_A', 'grad_B' (Sobel gradients).
        """
        # RGB to LAB conversion
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        edges = {}

        # L channel: Structural edges from brightness
        # Use adaptive thresholds based on image statistics
        L_blur = cv2.GaussianBlur(L, (3, 3), 0)
        L_median = np.median(L_blur)
        L_low = int(max(0, 0.5 * L_median))
        L_high = int(min(255, 1.2 * L_median))
        edges['L'] = cv2.Canny(L_blur, L_low, L_high)

        # A channel: Green-Red color boundaries
        A_blur = cv2.GaussianBlur(A, (5, 5), 0)
        A_median = np.median(A_blur)
        A_std = np.std(A_blur)
        A_low = int(max(0, A_median - 0.5 * A_std))
        A_high = int(min(255, A_median + 0.5 * A_std))
        edges['A'] = cv2.Canny(A_blur, max(10, A_low), max(30, A_high))

        # B channel: Blue-Yellow color boundaries
        B_blur = cv2.GaussianBlur(B, (5, 5), 0)
        B_median = np.median(B_blur)
        B_std = np.std(B_blur)
        B_low = int(max(0, B_median - 0.5 * B_std))
        B_high = int(min(255, B_median + 0.5 * B_std))
        edges['B'] = cv2.Canny(B_blur, max(10, B_low), max(30, B_high))

        # Sobel gradients for soft internal boundaries
        # These capture gradual transitions that Canny might miss
        sobel_L = cv2.Sobel(L_blur, cv2.CV_64F, 1, 1, ksize=3)
        sobel_A = cv2.Sobel(A_blur, cv2.CV_64F, 1, 1, ksize=3)
        sobel_B = cv2.Sobel(B_blur, cv2.CV_64F, 1, 1, ksize=3)

        # Gradient magnitudes
        grad_L = np.abs(sobel_L)
        grad_A = np.abs(sobel_A)
        grad_B = np.abs(sobel_B)

        # Normalize to 0-255
        def normalize_grad(g):
            if g.max() > 0:
                return (g / g.max() * 255).astype(np.uint8)
            return g.astype(np.uint8)

        edges['grad_L'] = normalize_grad(grad_L)
        edges['grad_A'] = normalize_grad(grad_A)
        edges['grad_B'] = normalize_grad(grad_B)

        # Laplacian of Gaussian for fine internal details
        # This captures textures and patterns that Canny/Sobel might miss
        log_kernel_size = 5
        log_sigma = 1.0
        L_log = cv2.GaussianBlur(L, (log_kernel_size, log_kernel_size), log_sigma)
        laplacian = cv2.Laplacian(L_log, cv2.CV_64F, ksize=3)
        laplacian_abs = np.abs(laplacian)
        edges['laplacian'] = normalize_grad(laplacian_abs)

        return edges

    def fuse_edges(
        self,
        edges: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Fuse multi-channel edges with weighted combination.

        Args:
            edges: Dictionary of edge maps from detect_lab_edges().
            weights: Optional custom weights for each channel.
                     If None, uses default weights optimized for
                     Korean paintings.

        Returns:
            Fused edge map (H, W), uint8 range [0, 255].
        """
        if weights is None:
            # Default weights optimized for Korean traditional paintings
            # Balance structural outlines with internal details
            weights = {
                'L': 0.30,       # Structural outlines (strongest)
                'A': 0.10,       # Red-Green color boundaries
                'B': 0.10,       # Blue-Yellow color boundaries
                'grad_L': 0.15,  # Soft brightness gradients
                'grad_A': 0.10,  # Soft color gradients
                'grad_B': 0.10,  # Soft color gradients
                'laplacian': 0.15,  # Fine internal details (patterns, textures)
            }

        h, w = edges['L'].shape
        fused = np.zeros((h, w), dtype=np.float32)

        for key, weight in weights.items():
            if key in edges:
                fused += weight * edges[key].astype(np.float32)

        # Clip and convert to uint8
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        return fused

    def morphological_denoise(
        self,
        edge_map: np.ndarray,
        min_size: int = 10,
        preserve_internal: bool = True
    ) -> np.ndarray:
        """Remove noise using morphological operations.

        Pipeline:
        1. Opening (erosion -> dilation): Removes small noise dots
        2. Closing (dilation -> erosion): Connects broken lines
        3. Connected component filtering: Removes tiny isolated regions

        Args:
            edge_map: Input edge map (H, W), uint8.
            min_size: Minimum connected component size to keep.
            preserve_internal: If True, use gentler denoising to preserve internal details.

        Returns:
            Denoised edge map (H, W), uint8.
        """
        # Binarize with adaptive threshold to include weak but meaningful edges
        threshold = 20 if preserve_internal else 30
        _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)

        # Opening: Remove small noise particles
        # Use smaller kernel when preserving internal details
        kernel_size = 2
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Closing: Connect nearby edge fragments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )

        # Use smaller min_size when preserving internal details
        effective_min_size = min_size // 2 if preserve_internal else min_size

        cleaned = np.zeros_like(closed)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= effective_min_size:
                cleaned[labels == i] = 255

        return cleaned

    def edge_recovery(
        self,
        edge_map: np.ndarray,
        original_edges: np.ndarray,
        threshold_strong: int = 80,
        threshold_medium: int = 50
    ) -> np.ndarray:
        """Recover important edges lost during denoising.

        Two-tier recovery:
        1. Strong edges (high confidence) - recovered anywhere near cleaned edges
        2. Medium edges - recovered only inside object regions

        Args:
            edge_map: Denoised edge map.
            original_edges: Original fused edge map before denoising.
            threshold_strong: Threshold for strong edge recovery.
            threshold_medium: Threshold for medium edge recovery.

        Returns:
            Edge map with recovered strong and internal edges.
        """
        # Tier 1: Recover strong edges near cleaned edges
        _, strong_edges = cv2.threshold(original_edges, threshold_strong, 255, cv2.THRESH_BINARY)

        # Dilate cleaned edges to create recovery zone
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_clean = cv2.dilate(edge_map, kernel_large, iterations=1)

        # Recover strong edges within recovery zone
        strong_recovery = cv2.bitwise_and(strong_edges, dilated_clean)

        # Tier 2: Recover medium-strength internal edges
        # These are typically internal details like patterns
        _, medium_edges = cv2.threshold(original_edges, threshold_medium, 255, cv2.THRESH_BINARY)

        # Create interior mask by finding regions enclosed by edges
        # Fill from the border to find background
        h, w = edge_map.shape
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        border_fill = edge_map.copy()
        cv2.floodFill(border_fill, mask, (0, 0), 255)

        # Interior is where the flood fill didn't reach
        interior_mask = (border_fill == 0).astype(np.uint8) * 255

        # Dilate interior slightly to include near-edge regions
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        interior_dilated = cv2.dilate(interior_mask, kernel_small, iterations=1)

        # Recover medium edges only inside the object
        medium_recovery = cv2.bitwise_and(medium_edges, interior_dilated)

        # Remove noise from medium recovery
        medium_recovery = cv2.morphologyEx(
            medium_recovery,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )

        # Combine all: original denoised + strong recovery + medium internal
        recovered = cv2.bitwise_or(edge_map, strong_recovery)
        recovered = cv2.bitwise_or(recovered, medium_recovery)

        return recovered

    def refine_edges(self, edge_map: np.ndarray) -> np.ndarray:
        """Refine edges to thin, crisp lines.

        Uses morphological thinning (skeletonization) to produce
        single-pixel-wide edges.

        Args:
            edge_map: Input edge map (H, W), uint8.

        Returns:
            Refined thin edge map (H, W), uint8.
        """
        # Try cv2.ximgproc.thinning if available (faster)
        if hasattr(cv2, 'ximgproc'):
            try:
                skeleton = cv2.ximgproc.thinning(edge_map)
                return skeleton
            except Exception:
                pass

        # Fallback: scipy morphological thinning
        skeleton = self._morphological_thinning(edge_map)
        return skeleton

    def _morphological_thinning(self, img: np.ndarray) -> np.ndarray:
        """Morphological thinning to produce skeleton.

        Uses skimage if available, otherwise falls back to
        Zhang-Suen thinning algorithm implementation.

        Args:
            img: Binary edge map.

        Returns:
            Skeletonized edge map.
        """
        # Convert to binary
        binary = img > 0

        if HAS_SKIMAGE:
            # Use skimage's thin function
            skeleton = skimage_thin(binary)
        else:
            # Fallback: Zhang-Suen thinning algorithm
            skeleton = self._zhang_suen_thinning(binary)

        return (skeleton * 255).astype(np.uint8)

    def _zhang_suen_thinning(self, binary: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning algorithm (vectorized).

        A classic thinning algorithm that preserves connectivity
        while reducing the binary image to a skeleton.
        Uses vectorized operations for better performance.

        Args:
            binary: Binary image (bool array).

        Returns:
            Thinned skeleton (bool array).
        """
        # Convert to uint8
        img = binary.astype(np.uint8)

        # Pad image
        padded = np.pad(img, 1, mode='constant', constant_values=0)
        skeleton = padded.copy()

        max_iterations = 100  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Extract 8-neighborhood using array slicing
            # P2-P9 in clockwise order starting from top
            P2 = skeleton[:-2, 1:-1]   # top
            P3 = skeleton[:-2, 2:]     # top-right
            P4 = skeleton[1:-1, 2:]    # right
            P5 = skeleton[2:, 2:]      # bottom-right
            P6 = skeleton[2:, 1:-1]    # bottom
            P7 = skeleton[2:, :-2]     # bottom-left
            P8 = skeleton[1:-1, :-2]   # left
            P9 = skeleton[:-2, :-2]    # top-left

            # Current pixel
            P1 = skeleton[1:-1, 1:-1]

            # B: Count of non-zero neighbors
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            # A: Count of 0->1 transitions in clockwise order
            A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
                ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
                ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
                ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
                ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
                ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
                ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
                ((P9 == 0) & (P2 == 1)).astype(np.uint8)

            # First sub-iteration conditions
            cond1 = (P1 == 1)
            cond2 = (B >= 2) & (B <= 6)
            cond3 = (A == 1)
            cond4 = (P2 == 0) | (P4 == 0) | (P6 == 0)
            cond5 = (P4 == 0) | (P6 == 0) | (P8 == 0)

            markers1 = cond1 & cond2 & cond3 & cond4 & cond5

            if not np.any(markers1):
                break

            # Apply first sub-iteration
            skeleton[1:-1, 1:-1][markers1] = 0

            # Re-extract neighborhoods after modification
            P2 = skeleton[:-2, 1:-1]
            P3 = skeleton[:-2, 2:]
            P4 = skeleton[1:-1, 2:]
            P5 = skeleton[2:, 2:]
            P6 = skeleton[2:, 1:-1]
            P7 = skeleton[2:, :-2]
            P8 = skeleton[1:-1, :-2]
            P9 = skeleton[:-2, :-2]
            P1 = skeleton[1:-1, 1:-1]

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
                ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
                ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
                ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
                ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
                ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
                ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
                ((P9 == 0) & (P2 == 1)).astype(np.uint8)

            # Second sub-iteration conditions
            cond1 = (P1 == 1)
            cond2 = (B >= 2) & (B <= 6)
            cond3 = (A == 1)
            cond4 = (P2 == 0) | (P4 == 0) | (P8 == 0)
            cond5 = (P2 == 0) | (P6 == 0) | (P8 == 0)

            markers2 = cond1 & cond2 & cond3 & cond4 & cond5

            if not np.any(markers2):
                break

            skeleton[1:-1, 1:-1][markers2] = 0

        # Remove padding
        result = skeleton[1:-1, 1:-1]
        return result > 0

    def detect_external_silhouette(self, image_rgb: np.ndarray) -> np.ndarray:
        """Detect strong external silhouette edges.

        Uses higher thresholds to capture only the strongest
        external contours of objects.

        Args:
            image_rgb: RGB image.

        Returns:
            External silhouette edge map.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Strong blur to remove internal details
        blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)

        # High thresholds for external edges only
        median = np.median(blurred)
        low = int(max(30, 0.8 * median))
        high = int(min(200, 1.5 * median))

        silhouette = cv2.Canny(blurred, low, high)

        # Dilate slightly to ensure continuity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        silhouette = cv2.dilate(silhouette, kernel, iterations=1)

        return silhouette

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Perform advanced edge detection.

        Full pipeline:
        1. LAB channel-wise edge detection
        2. Weighted channel fusion
        3. Morphological denoising
        4. Edge recovery
        5. Edge refinement (thinning)

        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W).
            **kwargs: Additional parameters.
                - min_size: Minimum component size for denoising (default 15).
                - weights: Custom fusion weights dictionary.
                - include_silhouette: Add external silhouette (default True).
                - silhouette_weight: Weight for silhouette (default 0.3).

        Returns:
            ModuleOutput with:
            - result: Final edge map tensor (B, 1, H, W)
            - intermediate: All intermediate edge maps
            - metadata: Processing information
        """
        start = time.time()

        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_results = []
        intermediates = {}

        # Extract parameters
        min_size = kwargs.get('min_size', 15)
        custom_weights = kwargs.get('weights', None)
        include_silhouette = kwargs.get('include_silhouette', True)
        silhouette_weight = kwargs.get('silhouette_weight', 0.3)

        for b in range(image.shape[0]):
            img_np = self._tensor_to_numpy_rgb(image[b])

            # 1. LAB channel-wise edge detection
            lab_edges = self.detect_lab_edges(img_np)

            # 2. External silhouette detection (optional)
            if include_silhouette:
                silhouette = self.detect_external_silhouette(img_np)
            else:
                silhouette = None

            # 3. Weighted fusion of LAB channels
            fused = self.fuse_edges(lab_edges, custom_weights)

            # 4. Blend with silhouette for stronger external edges
            if silhouette is not None:
                fused = cv2.addWeighted(
                    fused, 1.0 - silhouette_weight,
                    silhouette, silhouette_weight,
                    0
                )

            # 5. Morphological denoising
            denoised = self.morphological_denoise(fused, min_size=min_size)

            # 6. Edge recovery
            recovered = self.edge_recovery(denoised, fused)

            # 7. Edge refinement (thinning)
            refined = self.refine_edges(recovered)

            batch_results.append(refined)

            # Store intermediate results for first batch item
            if b == 0:
                intermediates = {
                    'L_edges': self._numpy_to_tensor(lab_edges['L']),
                    'A_edges': self._numpy_to_tensor(lab_edges['A']),
                    'B_edges': self._numpy_to_tensor(lab_edges['B']),
                    'grad_L': self._numpy_to_tensor(lab_edges['grad_L']),
                    'grad_A': self._numpy_to_tensor(lab_edges['grad_A']),
                    'grad_B': self._numpy_to_tensor(lab_edges['grad_B']),
                    'laplacian': self._numpy_to_tensor(lab_edges['laplacian']),
                    'fused': self._numpy_to_tensor(fused),
                    'denoised': self._numpy_to_tensor(denoised),
                    'recovered': self._numpy_to_tensor(recovered),
                }
                if silhouette is not None:
                    intermediates['silhouette'] = self._numpy_to_tensor(silhouette)

        # Stack batch results
        result_np = np.stack(batch_results, axis=0)
        # Add channel dimension: (B, H, W) -> (B, 1, H, W)
        result = torch.from_numpy(
            result_np[:, np.newaxis, :, :].astype(np.float32) / 255.0
        )
        result = result.to(device=self.device, dtype=self.dtype)

        elapsed = time.time() - start

        return ModuleOutput(
            result=result,
            intermediate=intermediates,
            metadata={
                'method': 'advanced_lab_fusion',
                'processing_time': elapsed,
                'stages': [
                    'lab_edges',
                    'silhouette',
                    'fusion',
                    'denoise',
                    'recovery',
                    'refine'
                ],
                'min_size': min_size,
                'include_silhouette': include_silhouette,
                'silhouette_weight': silhouette_weight,
            }
        )
