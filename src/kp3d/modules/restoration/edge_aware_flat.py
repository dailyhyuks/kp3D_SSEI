"""Edge-Aware Flat Color Restoration for Korean Traditional Paintings.

Korean traditional paintings have FLAT coloring with NO shadows or lighting gradients.
Key insight: detect object edges (ink lines) via LAB color difference, then flatten
colors inside each region. This removes ALL degradation (grid patterns, fading,
stains, noise) while preserving the essential structure.

Pipeline:
1. LAB ΔE Edge Detection - detect edges via perceptual color difference
2. Region Segmentation - connected components from inverted edge mask
3. Median Color Flattening - fill each region with its median LAB color

This is a pure algorithm class with no torch dependency.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import label


class EdgeAwareFlatProcessor:
    """Edge-aware flat color processor for Korean traditional painting restoration.

    Uses LAB color space ΔE (perceptual color difference) for edge detection,
    then flattens each region to its median color. This approach leverages the
    flat coloring style of traditional Korean paintings to remove degradation
    artifacts while preserving artistic intent.

    Attributes:
        delta_e_threshold: Minimum ΔE for edge detection (default 8.0).
        min_region_area: Minimum pixel area for a region (smaller merged).
        edge_dilate: Dilation radius for edge cleanup (pixels).
        blend_width: Width of edge zone blending (pixels).
        neighbor_window: Window size for neighborhood comparison.
    """

    def __init__(
        self,
        delta_e_high: float = 12.0,
        delta_e_low: float = 5.0,
        chrominance_sigma: float = 3.0,
        chrominance_threshold: float = 12.0,
        persistence_sigma: float = 3.0,
        confidence_threshold: float = 0.3,
        periodicity_threshold: float = 0.5,
        min_edge_length: int = 10,
        min_region_area: int = 20,
        edge_dilate: int = 1,
        blend_width: int = 2,
        pre_blur_sigma: float = 1.0,
        bilateral_iterations: int = 3,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
    ) -> None:
        self.delta_e_high = delta_e_high
        self.delta_e_low = delta_e_low
        self.chrominance_sigma = chrominance_sigma
        self.chrominance_threshold = chrominance_threshold
        self.persistence_sigma = persistence_sigma
        self.confidence_threshold = confidence_threshold
        self.periodicity_threshold = periodicity_threshold
        self.min_edge_length = min_edge_length
        self.min_region_area = min_region_area
        self.edge_dilate = edge_dilate
        self.blend_width = blend_width
        self.pre_blur_sigma = pre_blur_sigma
        self.bilateral_iterations = bilateral_iterations
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full edge-aware flat color restoration pipeline.

        Args:
            image_bgr: Input BGR image (uint8, shape HxWx3).

        Returns:
            Tuple of (result_bgr, intermediates_dict) where:
                - result_bgr: Processed BGR image (uint8)
                - intermediates_dict: Dictionary containing:
                    - edge_map: Raw binary edge detection (uint8)
                    - cleaned_edge_map: After morphological cleanup (uint8)
                    - label_map: Region labels (int32)
                    - n_regions: Number of regions found
                    - flattened: Result before edge blending (BGR uint8)
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 5 or w < 5:
            return image_bgr.copy(), self._empty_intermediates(h, w, image_bgr)

        # Pre-filter: bilateral to suppress grid/texture (reused for blending)
        bilateral = self._suppress_texture(image_bgr)

        # Stage 1: LAB ΔE Edge Detection (on bilateral result)
        edge_map = self._detect_edges_lab_delta_e(image_bgr, bilateral=bilateral)

        # Clean up edges with morphological operations
        cleaned_edge_map = self._cleanup_edges(edge_map)

        # Handle case where no edges detected
        if not np.any(cleaned_edge_map):
            label_map = np.ones((h, w), dtype=np.int32)
            n_regions = 1
            flattened = self._flatten_single_region(image_bgr)

            return flattened, {
                "edge_map": edge_map,
                "cleaned_edge_map": cleaned_edge_map,
                "label_map": label_map,
                "n_regions": n_regions,
                "flattened": flattened,
                "bilateral": bilateral,
            }

        # Stage 2: Region Segmentation
        label_map, n_regions = self._segment_regions(cleaned_edge_map)

        # Merge small regions
        label_map, n_regions = self._merge_small_regions(
            label_map, n_regions, image_bgr
        )

        # Stage 3: Median Color Flattening
        flattened = self._flatten_regions(image_bgr, label_map, n_regions)

        # Edge zone blending — use bilateral (grid-free) instead of original
        result = self._blend_edge_zones(flattened, bilateral, cleaned_edge_map)

        intermediates = {
            "edge_map": edge_map,
            "cleaned_edge_map": cleaned_edge_map,
            "label_map": label_map,
            "n_regions": n_regions,
            "flattened": flattened,
            "bilateral": bilateral,
        }

        return result, intermediates

    def _empty_intermediates(
        self, h: int, w: int, image_bgr: np.ndarray
    ) -> Dict[str, Any]:
        """Create empty intermediates dict for edge cases."""
        return {
            "edge_map": np.zeros((h, w), dtype=np.uint8),
            "cleaned_edge_map": np.zeros((h, w), dtype=np.uint8),
            "label_map": np.zeros((h, w), dtype=np.int32),
            "n_regions": 0,
            "flattened": image_bgr.copy(),
        }

    def _suppress_texture(self, image_bgr: np.ndarray) -> np.ndarray:
        """Suppress grid/texture patterns using iterative bilateral filtering.

        Bilateral filter preserves strong edges (object boundaries) while
        smoothing out repetitive texture (grid lines, fabric patterns, noise).
        Multiple iterations progressively remove texture while keeping edges sharp.

        Args:
            image_bgr: Input BGR image (uint8).

        Returns:
            Texture-suppressed BGR image (uint8).
        """
        filtered = image_bgr.copy()
        for _ in range(self.bilateral_iterations):
            filtered = cv2.bilateralFilter(
                filtered,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
        return filtered

    def _compute_delta_e_map(
        self, image_bgr: np.ndarray, bilateral: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ΔE gradient map in LAB color space.

        Args:
            image_bgr: Input BGR image (uint8).
            bilateral: Pre-computed bilateral filtered image.

        Returns:
            ΔE gradient map (float32).
        """
        smoothed = bilateral if bilateral is not None else self._suppress_texture(image_bgr)

        if self.pre_blur_sigma > 0:
            ksize = int(self.pre_blur_sigma * 4) | 1
            ksize = max(3, ksize)
            smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), self.pre_blur_sigma)

        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)

        L = lab[:, :, 0] * (100.0 / 255.0)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0

        L_dx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        L_dy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        a_dx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        a_dy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        b_dx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        b_dy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

        L_grad = np.sqrt(L_dx**2 + L_dy**2)
        a_grad = np.sqrt(a_dx**2 + a_dy**2)
        b_grad = np.sqrt(b_dx**2 + b_dy**2)

        return np.sqrt(L_grad**2 + a_grad**2 + b_grad**2)

    def _detect_edges_lab_delta_e(
        self, image_bgr: np.ndarray, bilateral: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stage 1: Multi-strategy edge detection with confidence scoring.

        Three complementary detection strategies:
        1. LAB ΔE Sobel (bilateral-based) — object boundaries
        2. Chrominance-only edges (LAB a*/b*, σ pre-smoothed) — grid-immune
        3. Persistence edges (σ blur survival) — blur-proof = content

        Combination:
        - Persistence edges are guaranteed (always included)
        - Among LAB ΔE and chrominance, 1+ agreement → included
        - Final confidence scoring filters remaining noise

        Args:
            image_bgr: Input BGR image (uint8).
            bilateral: Pre-computed bilateral filtered image.

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        # --- Strategy A: LAB ΔE Sobel (existing, on bilateral result) ---
        delta_e = self._compute_delta_e_map(image_bgr, bilateral)
        edges_lab = (delta_e > self.delta_e_low).astype(np.uint8) * 255

        # --- Strategy B: Chrominance-only edges (grid-immune) ---
        edges_chroma = self._detect_chrominance_edges(image_bgr)

        # --- Strategy C: Persistence edges (blur survival) ---
        edges_persist = self._detect_persistence_edges(image_bgr)

        # --- Combine via voting ---
        combined = self._combine_edge_strategies(
            edges_lab, edges_chroma, edges_persist, delta_e
        )

        # --- Strategy D: Periodicity rejection (post-filter) ---
        # Only persistence edges are protected (confirmed blur-surviving content)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        combined = self._reject_periodic_edges(combined, gray, protected=edges_persist)

        return combined

    def _detect_chrominance_edges(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect edges using LAB a*/b* chrominance channels only.

        Grid patterns are luminance (L) noise and do not affect chrominance
        (a*/b*). Pre-smoothing with sigma destroys grid-scale artifacts
        while preserving real content color boundaries.

        Args:
            image_bgr: BGR image (uint8).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

        a_ch = lab[:, :, 1]
        b_ch = lab[:, :, 2]

        # Pre-smooth chrominance to suppress grid-scale color artifacts
        a_ch = ndimage.gaussian_filter(a_ch, sigma=self.chrominance_sigma)
        b_ch = ndimage.gaussian_filter(b_ch, sigma=self.chrominance_sigma)

        # Compute gradients
        grad_a_x = cv2.Sobel(a_ch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_a_y = cv2.Sobel(a_ch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_b_x = cv2.Sobel(b_ch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_b_y = cv2.Sobel(b_ch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        delta_e_chroma = np.sqrt(
            grad_a_x**2 + grad_a_y**2 + grad_b_x**2 + grad_b_y**2
        )

        return (delta_e_chroma > self.chrominance_threshold).astype(np.uint8) * 255

    def _detect_persistence_edges(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect edges that survive Gaussian blur (content edges).

        Grid period is 7-9px, so sigma=3 blur destroys grid but preserves
        content edges that span 20+ pixels.

        Args:
            image_bgr: BGR image (uint8).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float64)

        # Apply large sigma blur to destroy grid patterns
        blurred = ndimage.gaussian_filter(gray_f, sigma=self.persistence_sigma)
        blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)

        # Use relaxed Canny thresholds on blurred image
        # (blurring already suppresses noise, so lower thresholds capture more content)
        edges = cv2.Canny(
            blurred_u8,
            self.delta_e_low * 4,   # e.g. 5.0 * 4 = 20
            self.delta_e_high * 4,  # e.g. 12.0 * 4 = 48
        )

        return edges

    def _combine_edge_strategies(
        self,
        edges_lab: np.ndarray,
        edges_chroma: np.ndarray,
        edges_persist: np.ndarray,
        delta_e_map: np.ndarray,
    ) -> np.ndarray:
        """Combine three edge strategies with confidence scoring.

        Persistence edges are guaranteed (always included). Among LAB ΔE
        and chrominance, 1+ vote includes the edge. Final confidence
        scoring filters noise using persistence weight (0.5),
        length weight (0.3), and chrominance bonus (0.2).

        Args:
            edges_lab: LAB ΔE edge map (uint8).
            edges_chroma: Chrominance edge map (uint8).
            edges_persist: Persistence edge map (uint8).
            delta_e_map: Raw ΔE gradient map (float32).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        # Dilate each detector by 1px to compensate for position differences
        kernel_3 = np.ones((3, 3), dtype=np.uint8)

        lab_d = cv2.dilate(edges_lab, kernel_3, iterations=1)
        chroma_d = cv2.dilate(edges_chroma, kernel_3, iterations=1)
        persist_d = cv2.dilate(edges_persist, kernel_3, iterations=1)

        # Vote: count agreements across all 3 detectors
        vote_count = (
            (lab_d > 0).astype(np.int32)
            + (chroma_d > 0).astype(np.int32)
            + (persist_d > 0).astype(np.int32)
        )

        # Combined: persistence (guaranteed) OR 2+ detectors agree
        # LAB ΔE alone is NOT sufficient (would pass grid edges)
        combined = (
            (edges_persist > 0) | (vote_count >= 2)
        ).astype(np.uint8) * 255

        # --- Confidence scoring to filter noise ---
        edge_binary = combined > 0

        # 1. Persistence score (weight 0.5): edge present in persistence map
        persist_score = (edges_persist > 0).astype(np.float32)

        # 2. Length score (weight 0.3): longer connected components score higher
        length_score = self._compute_length_score(combined)

        # 3. Chrominance bonus (weight 0.2): confirmed by chrominance detector
        chroma_score = (edges_chroma > 0).astype(np.float32)

        confidence = (
            0.5 * persist_score
            + 0.3 * length_score
            + 0.2 * chroma_score
        )

        # Strong LAB edges bypass confidence filter
        strong_lab = (delta_e_map > self.delta_e_high).astype(np.uint8) * 255

        # Final: edge pixels with sufficient confidence OR strong LAB ΔE
        final = (
            (edge_binary & (confidence >= self.confidence_threshold))
            | (strong_lab > 0)
        ).astype(np.uint8) * 255

        return final

    def _compute_length_score(self, edge_map: np.ndarray) -> np.ndarray:
        """Score edge pixels by connected component length.

        Longer edges are more likely to be content rather than noise.

        Args:
            edge_map: Binary edge map (uint8, 0 or 255).

        Returns:
            Float score map [0, 1] same shape as edge_map.
        """
        binary = (edge_map > 0).astype(np.uint8)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        score = np.zeros_like(edge_map, dtype=np.float32)
        if n_labels <= 1:
            return score

        # Get max component size for normalization
        sizes = stats[1:, cv2.CC_STAT_AREA]
        max_size = sizes.max() if len(sizes) > 0 else 1

        for label_idx in range(1, n_labels):
            mask = labels == label_idx
            size = stats[label_idx, cv2.CC_STAT_AREA]
            # sqrt normalization: diminishing returns for very long edges
            score[mask] = np.sqrt(size / max_size)

        return score

    # --- Periodicity rejection ---

    def _detect_grid_periods(self, gray: np.ndarray) -> Tuple[int, int]:
        """Detect grid periods via autocorrelation.

        Args:
            gray: Grayscale image (uint8).

        Returns:
            (period_x, period_y) in pixels.
        """
        gray_f = gray.astype(np.float64)
        gray_f -= gray_f.mean()
        h, w = gray_f.shape

        max_lag_x = min(w // 2, 50)
        acf_x = np.zeros(max_lag_x)
        for lag in range(max_lag_x):
            if w - lag > 0:
                acf_x[lag] = np.mean(gray_f[:, :w - lag] * gray_f[:, lag:])
        if acf_x[0] > 1e-10:
            acf_x /= acf_x[0]

        max_lag_y = min(h // 2, 50)
        acf_y = np.zeros(max_lag_y)
        for lag in range(max_lag_y):
            if h - lag > 0:
                acf_y[lag] = np.mean(gray_f[:h - lag, :] * gray_f[lag:, :])
        if acf_y[0] > 1e-10:
            acf_y /= acf_y[0]

        period_x = self._find_first_peak(acf_x)
        period_y = self._find_first_peak(acf_y)
        return period_x, period_y

    @staticmethod
    def _find_first_peak(
        acf: np.ndarray,
        search_min: int = 4,
        search_max: int = 30,
        fallback: int = 8,
    ) -> int:
        """Find first peak in autocorrelation."""
        search_end = min(len(acf) - 1, search_max)
        for i in range(search_min, search_end):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
                return i
        return fallback

    def _compute_periodicity_map(
        self, edge_map: np.ndarray, gray: np.ndarray,
        period_x: int, period_y: int,
    ) -> np.ndarray:
        """Compute per-pixel periodicity score using local normalized cross-correlation.

        High value = highly periodic (likely grid).
        Low value = not periodic (likely content).

        Args:
            edge_map: Binary edge map (uint8).
            gray: Grayscale image (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.

        Returns:
            Float periodicity map [0, 1].
        """
        h, w = edge_map.shape

        # Use gradient magnitude for correlation
        gray_f = gray.astype(np.float64)
        grad_x = ndimage.sobel(gray_f, axis=1)
        grad_y = ndimage.sobel(gray_f, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

        window = max(period_x, period_y) * 2 + 1
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, min(h, w) // 2))

        periodicity = np.zeros((h, w), dtype=np.float32)

        # Horizontal periodicity
        if 0 < period_x < w // 2:
            shifted = np.zeros_like(grad_mag)
            shifted[:, period_x:] = grad_mag[:, :-period_x]

            local_mean_g = ndimage.uniform_filter(grad_mag, size=window)
            local_mean_s = ndimage.uniform_filter(shifted, size=window)

            local_cross = ndimage.uniform_filter(grad_mag * shifted, size=window)
            local_var_g = ndimage.uniform_filter(grad_mag**2, size=window) - local_mean_g**2
            local_var_s = ndimage.uniform_filter(shifted**2, size=window) - local_mean_s**2

            denom = np.sqrt(np.maximum(local_var_g, 0) * np.maximum(local_var_s, 0)) + 1e-8
            ncc_x = np.clip(
                (local_cross - local_mean_g * local_mean_s) / denom, 0, 1
            )
            periodicity = np.maximum(periodicity, ncc_x)

        # Vertical periodicity
        if 0 < period_y < h // 2:
            shifted = np.zeros_like(grad_mag)
            shifted[period_y:, :] = grad_mag[:-period_y, :]

            local_mean_g = ndimage.uniform_filter(grad_mag, size=window)
            local_mean_s = ndimage.uniform_filter(shifted, size=window)

            local_cross = ndimage.uniform_filter(grad_mag * shifted, size=window)
            local_var_g = ndimage.uniform_filter(grad_mag**2, size=window) - local_mean_g**2
            local_var_s = ndimage.uniform_filter(shifted**2, size=window) - local_mean_s**2

            denom = np.sqrt(np.maximum(local_var_g, 0) * np.maximum(local_var_s, 0)) + 1e-8
            ncc_y = np.clip(
                (local_cross - local_mean_g * local_mean_s) / denom, 0, 1
            )
            periodicity = np.maximum(periodicity, ncc_y)

        return periodicity

    def _reject_periodic_edges(
        self, edge_map: np.ndarray, gray: np.ndarray,
        protected: np.ndarray | None = None,
    ) -> np.ndarray:
        """Remove edges that show grid-like periodicity.

        Persistence edges (protected) are never rejected — they survived
        heavy blur and are confirmed content edges.

        Args:
            edge_map: Binary edge map (uint8, 0 or 255).
            gray: Grayscale image (uint8).
            protected: Binary edge map of protected edges (persistence).

        Returns:
            Filtered edge map with periodic edges removed.
        """
        h, w = edge_map.shape
        if h < 20 or w < 20:
            return edge_map

        period_x, period_y = self._detect_grid_periods(gray)

        periodicity = self._compute_periodicity_map(edge_map, gray, period_x, period_y)

        # Reject edges where periodicity exceeds threshold
        edge_binary = edge_map > 0
        periodic_mask = periodicity > self.periodicity_threshold

        # Protect persistence edges from rejection
        if protected is not None:
            # Dilate protected edges to include their neighborhood
            kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            protected_zone = cv2.dilate(
                (protected > 0).astype(np.uint8), kernel_5, iterations=1
            ) > 0
            periodic_mask = periodic_mask & ~protected_zone

        filtered = edge_binary & ~periodic_mask

        return (filtered.astype(np.uint8) * 255)

    def _cleanup_edges(self, edge_map: np.ndarray) -> np.ndarray:
        """Clean up edge map using morphological operations.

        Operations:
        1. Remove small isolated fragments (opening)
        2. Close small gaps in edge lines (closing)
        3. Thin edges to consistent width
        4. Filter out very short edge fragments

        Args:
            edge_map: Binary edge map (uint8, 0 or 255).

        Returns:
            Cleaned binary edge map (uint8, 0 or 255).
        """
        # Convert to binary for morphological operations
        binary = (edge_map > 0).astype(np.uint8)

        # 1. Remove small isolated fragments (morphological opening)
        # Small kernel to remove noise while preserving thin lines
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

        # 2. Close small gaps to connect nearby edge segments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 3. Optional dilation to thicken edges (helps with connectivity)
        if self.edge_dilate > 0:
            kernel_dilate = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.edge_dilate + 1, 2 * self.edge_dilate + 1)
            )
            dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        else:
            dilated = closed

        # 4. Thin to consistent width using erosion (not full skeletonization)
        # This preserves connectivity better than full skeletonization
        kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thinned = cv2.erode(dilated, kernel_thin, iterations=1)

        # 5. Re-dilate to ensure edges are visible (1-2px wide)
        kernel_redilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final = cv2.dilate(thinned, kernel_redilate, iterations=1)

        # 6. Remove very short edge fragments (noise)
        # Label connected components and filter by size
        labeled, n_labels = label(final > 0)
        cleaned = np.zeros_like(final, dtype=np.uint8)

        for i in range(1, n_labels + 1):
            component = labeled == i
            if np.sum(component) >= self.min_edge_length:
                cleaned[component] = 255

        return cleaned

    def _segment_regions(
        self, edge_map: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Stage 2: Segment image into regions based on edge map.

        Inverts edge mask and finds connected components (regions).

        Args:
            edge_map: Binary edge map (uint8, 0 or 255).

        Returns:
            Tuple of (label_map, n_regions) where label_map has integer
            labels for each pixel (0 = edge, 1+ = region labels).
        """
        # Invert edge map: non-edge pixels are regions
        non_edge = edge_map == 0

        # Find connected components
        label_map, n_regions = label(non_edge)

        return label_map.astype(np.int32), n_regions

    def _merge_small_regions(
        self,
        label_map: np.ndarray,
        n_regions: int,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Merge small regions into their nearest neighbor by color.

        Args:
            label_map: Region labels (int32).
            n_regions: Number of regions.
            image_bgr: Original BGR image.

        Returns:
            Tuple of (new_label_map, new_n_regions).
        """
        if n_regions <= 1:
            return label_map, n_regions

        # Convert to LAB for color comparison
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute region statistics
        region_sizes = np.zeros(n_regions + 1, dtype=np.int32)
        region_colors = np.zeros((n_regions + 1, 3), dtype=np.float32)

        for i in range(1, n_regions + 1):
            mask = label_map == i
            region_sizes[i] = np.sum(mask)
            if region_sizes[i] > 0:
                region_colors[i] = np.median(lab[mask], axis=0)

        # Find small regions and their best merge target
        merge_map = np.arange(n_regions + 1, dtype=np.int32)  # Identity initially

        for i in range(1, n_regions + 1):
            if region_sizes[i] < self.min_region_area:
                # Find neighbors
                mask = label_map == i
                dilated = ndimage.binary_dilation(mask, iterations=1)
                border = dilated & ~mask
                neighbor_labels = np.unique(label_map[border])
                neighbor_labels = neighbor_labels[neighbor_labels > 0]
                neighbor_labels = neighbor_labels[neighbor_labels != i]

                if len(neighbor_labels) == 0:
                    # No neighbors, find nearest by color
                    best_target = self._find_nearest_by_color(
                        i, region_colors, region_sizes
                    )
                else:
                    # Find neighbor with most similar color
                    best_target = self._find_best_neighbor(
                        i, neighbor_labels, region_colors
                    )

                if best_target > 0:
                    merge_map[i] = best_target

        # Apply merges (handle chains)
        for _ in range(n_regions):  # Max iterations to resolve chains
            changed = False
            for i in range(1, n_regions + 1):
                if merge_map[i] != i and merge_map[merge_map[i]] != merge_map[i]:
                    merge_map[i] = merge_map[merge_map[i]]
                    changed = True
            if not changed:
                break

        # Apply merge map to labels
        new_label_map = merge_map[label_map]

        # Relabel to consecutive integers
        unique_labels = np.unique(new_label_map)
        unique_labels = unique_labels[unique_labels > 0]
        new_n_regions = len(unique_labels)

        relabel_map = np.zeros(n_regions + 1, dtype=np.int32)
        for new_idx, old_label in enumerate(unique_labels, start=1):
            relabel_map[old_label] = new_idx

        new_label_map = relabel_map[new_label_map]

        return new_label_map, new_n_regions

    def _find_nearest_by_color(
        self,
        region_idx: int,
        region_colors: np.ndarray,
        region_sizes: np.ndarray,
    ) -> int:
        """Find region with most similar color.

        Args:
            region_idx: Index of region to merge.
            region_colors: LAB colors for each region.
            region_sizes: Sizes of each region.

        Returns:
            Best target region index.
        """
        my_color = region_colors[region_idx]
        best_target = 0
        best_dist = float('inf')

        for j in range(1, len(region_colors)):
            if j == region_idx or region_sizes[j] == 0:
                continue
            dist = np.sqrt(np.sum((my_color - region_colors[j]) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_target = j

        return best_target

    def _find_best_neighbor(
        self,
        region_idx: int,
        neighbor_labels: np.ndarray,
        region_colors: np.ndarray,
    ) -> int:
        """Find neighbor with most similar color.

        Args:
            region_idx: Index of region to merge.
            neighbor_labels: Array of neighboring region indices.
            region_colors: LAB colors for each region.

        Returns:
            Best target region index.
        """
        my_color = region_colors[region_idx]
        best_target = neighbor_labels[0]
        best_dist = float('inf')

        for j in neighbor_labels:
            dist = np.sqrt(np.sum((my_color - region_colors[j]) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_target = j

        return int(best_target)

    def _flatten_regions(
        self,
        image_bgr: np.ndarray,
        label_map: np.ndarray,
        n_regions: int,
    ) -> np.ndarray:
        """Stage 3: Flatten each region to its median color.

        Computes median color in LAB space (more perceptually uniform),
        then fills region with that color.

        Args:
            image_bgr: Original BGR image.
            label_map: Region labels.
            n_regions: Number of regions.

        Returns:
            Flattened BGR image (uint8).
        """
        # Convert to LAB for median computation
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_lab = lab.copy()

        for i in range(1, n_regions + 1):
            mask = label_map == i
            if np.sum(mask) == 0:
                continue

            # Compute median color in LAB
            region_pixels = lab[mask]
            median_color = np.median(region_pixels, axis=0)

            # Fill region with median color
            result_lab[mask] = median_color

        # Handle edge pixels (label 0) - keep original
        edge_mask = label_map == 0
        result_lab[edge_mask] = lab[edge_mask]

        # Convert back to BGR
        result_bgr = cv2.cvtColor(
            np.clip(result_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        )

        return result_bgr

    def _flatten_single_region(self, image_bgr: np.ndarray) -> np.ndarray:
        """Flatten entire image as single region.

        Args:
            image_bgr: Original BGR image.

        Returns:
            Flattened BGR image (uint8).
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        median_color = np.median(lab.reshape(-1, 3), axis=0)
        result_lab = np.full_like(lab, median_color)
        result_bgr = cv2.cvtColor(
            np.clip(result_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        )
        return result_bgr

    def _blend_edge_zones(
        self,
        flattened: np.ndarray,
        original: np.ndarray,
        edge_map: np.ndarray,
    ) -> np.ndarray:
        """Blend flattened and original images at edge zones.

        Uses distance-weighted alpha blending near edges to preserve
        ink line quality while ensuring smooth transitions.

        Args:
            flattened: Flattened BGR image.
            original: Original BGR image.
            edge_map: Binary edge map (uint8, 0 or 255).

        Returns:
            Blended BGR image (uint8).
        """
        if self.blend_width <= 0:
            # No blending, just copy original at edges
            result = flattened.copy()
            edge_mask = edge_map > 0
            result[edge_mask] = original[edge_mask]
            return result

        # Compute distance transform from edges
        non_edge = edge_map == 0
        distance = ndimage.distance_transform_edt(non_edge)

        # Create alpha map: 0 at edges, 1 at blend_width distance
        alpha = np.clip(distance / self.blend_width, 0, 1)
        alpha = alpha[:, :, np.newaxis]  # Add channel dimension

        # Blend: near edges use original, far from edges use flattened
        result = (alpha * flattened + (1 - alpha) * original).astype(np.uint8)

        return result


def demo():
    """Quick demo/test of EdgeAwareFlatProcessor."""
    import os

    # Create a simple test image with flat regions and edges
    h, w = 100, 100
    test_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Background (light beige)
    test_image[:, :] = [200, 220, 230]  # BGR

    # Red region (top-left)
    test_image[10:40, 10:40] = [50, 50, 200]

    # Blue region (bottom-right)
    test_image[60:90, 60:90] = [200, 100, 50]

    # Black ink line (diagonal)
    for i in range(20, 80):
        test_image[i, i:i+2] = [20, 20, 20]

    # Add some noise to simulate degradation
    noise = np.random.normal(0, 10, test_image.shape).astype(np.int16)
    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Process
    processor = EdgeAwareFlatProcessor(
        delta_e_high=12.0,
        delta_e_low=5.0,
        min_region_area=20,
        edge_dilate=1,
        blend_width=2,
    )

    result, intermediates = processor.process(noisy_image)

    print(f"Input shape: {noisy_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Regions found: {intermediates['n_regions']}")
    print(f"Edge pixels: {np.sum(intermediates['edge_map'] > 0)}")
    print(f"Cleaned edge pixels: {np.sum(intermediates['cleaned_edge_map'] > 0)}")

    return result, intermediates


if __name__ == "__main__":
    demo()
