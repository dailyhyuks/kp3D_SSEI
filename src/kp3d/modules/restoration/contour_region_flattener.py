"""Contour-Based Region Flattening for grid pattern removal (v10).

Key insight: Object edges (ink lines) are 4-30x stronger than grid lines.
Thresholding extracts object edges → contour-based region segmentation →
per-region median flattening → grid removal.

Pipeline:
1. Grid Period Detection (autocorrelation)
2. Multi-Strategy Edge Detection (Sobel + Canny + chrominance + persistence)
3. Edge Confidence Scoring (periodicity + persistence + length)
4. Edge Mask Cleanup (morphology + filtering)
5. Region Segmentation (connected components + merge small regions)
6. Region Flattening (per-region median)
7. Edge Zone Blending (distance-weighted original/flattened blend)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter


class ContourRegionFlattener:
    """Contour-based region flattening for grid pattern removal.

    Pure algorithm class. No torch dependency.
    """

    def __init__(
        self,
        period_x: int = 0,
        period_y: int = 0,
        edge_low: float = 80.0,
        edge_high: float = 160.0,
        confidence_threshold: float = 0.5,
        min_region_area: int = 50,
        flatten_method: str = "median",
        blend_width: int = 1,
        min_edge_length: int = 15,
        chrominance_threshold: float = 15.0,
    ) -> None:
        """Initialize the ContourRegionFlattener.

        Args:
            period_x: Horizontal grid period (0 = auto-detect).
            period_y: Vertical grid period (0 = auto-detect).
            edge_low: Canny low threshold.
            edge_high: Canny high threshold.
            confidence_threshold: Threshold for edge confidence scoring.
            min_region_area: Minimum area for a region to be kept separately.
            flatten_method: "median" or "trimmed_mean".
            blend_width: Width of edge blending zone in pixels.
            min_edge_length: Minimum connected edge length to keep.
            chrominance_threshold: Threshold for chrominance-based edge detection.
        """
        self.period_x = period_x
        self.period_y = period_y
        self.edge_low = edge_low
        self.edge_high = edge_high
        self.confidence_threshold = confidence_threshold
        self.min_region_area = min_region_area
        self.flatten_method = flatten_method
        self.blend_width = blend_width
        self.min_edge_length = min_edge_length
        self.chrominance_threshold = chrominance_threshold

        # Scoring weights
        self._weight_periodicity = 0.25  # 0.35 → 0.25 (낮춤: 내부 엣지를 격자로 오판 방지)
        self._weight_persistence = 0.45  # 0.40 → 0.45 (올림: blur 생존이 가장 신뢰할 만한 지표)
        self._weight_length = 0.30       # 0.25 → 0.30 (올림: 긴 엣지는 거의 항상 content)

        # Multi-scale sigmas for edge detection
        self._sobel_sigmas = (0.5, 1.0, 2.0, 4.0)
        self._persistence_sigmas = (0.5, 1.0, 2.0, 3.0)  # 4.0 → 3.0: sigma=4는 내부 엣지까지 소멸시킴

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run full pipeline. Returns (result_bgr, intermediates_dict).

        Args:
            image_bgr: Input BGR image (uint8).

        Returns:
            Tuple of (processed BGR image, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "flattened": image_bgr.copy(),
                "n_regions": 0,
            }

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Phase 1: Grid Period Detection
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # Phase 2: Multi-Strategy Edge Detection
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        # Combine: 2+ detector agreement (with dilation for position tolerance)
        # + persistence edges always included (survived sigma=4 blur = certain content)
        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        # Handle case where no edges detected
        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "flattened": image_bgr.copy(),
                "n_regions": 0,
            }

        # Phase 3: Edge Confidence Scoring
        periodicity_score = self.compute_periodicity_score(raw_edge_map, period_x, period_y, gray=gray)
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)

        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # Phase 4: Edge Mask Cleanup
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # Phase 5: Region Segmentation
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # Phase 6: Region Flattening
        flattened = self.flatten_regions(image_bgr, label_map)

        # Phase 7: Edge Zone Blending
        blended = self.blend_edge_zones(flattened, image_bgr, final_edge_mask)

        # Phase 8: Edge-aware enhancement
        # Edges: bilateral-filtered original (removes grid, preserves ink lines)
        # Non-edges: extra-smoothed contour result (kills residual grid)
        result = self.edge_aware_enhance(blended, image_bgr, final_edge_mask)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "flattened": flattened,
            "n_regions": n_regions,
        }

        return result, intermediates

    def detect_grid_periods(self, gray: np.ndarray) -> Tuple[int, int]:
        """Phase 1: Detect grid periods via autocorrelation.

        Args:
            gray: Grayscale image (uint8 or float).

        Returns:
            (period_x, period_y) in pixels. Falls back to (9, 7) if detection fails.
        """
        gray_f = gray.astype(np.float64)
        gray_f -= gray_f.mean()
        h, w = gray_f.shape

        # Horizontal period: average autocorrelation along rows
        max_lag_x = min(w // 2, 50)
        acf_x = np.zeros(max_lag_x)
        for lag in range(max_lag_x):
            if w - lag > 0:
                acf_x[lag] = np.mean(gray_f[:, :w - lag] * gray_f[:, lag:])

        # Normalize
        if acf_x[0] > 1e-10:
            acf_x /= acf_x[0]

        # Vertical period: average autocorrelation along columns
        max_lag_y = min(h // 2, 50)
        acf_y = np.zeros(max_lag_y)
        for lag in range(max_lag_y):
            if h - lag > 0:
                acf_y[lag] = np.mean(gray_f[:h - lag, :] * gray_f[lag:, :])

        if acf_y[0] > 1e-10:
            acf_y /= acf_y[0]

        period_x = self._find_first_peak(acf_x, min_lag=3, search_min=4, search_max=30, fallback=9)
        period_y = self._find_first_peak(acf_y, min_lag=3, search_min=4, search_max=30, fallback=7)

        return period_x, period_y

    @staticmethod
    def _find_first_peak(
        acf: np.ndarray,
        min_lag: int = 3,
        search_min: int = 4,
        search_max: int = 30,
        fallback: int = 9,
    ) -> int:
        """Find first peak in autocorrelation function.

        Args:
            acf: Normalized autocorrelation array.
            min_lag: Minimum lag to start searching (skip DC region).
            search_min: Minimum search range.
            search_max: Maximum search range.
            fallback: Default value if no peak found.

        Returns:
            Lag of the first significant peak, or fallback.
        """
        search_start = max(min_lag, search_min)
        search_end = min(len(acf) - 1, search_max)

        for i in range(search_start, search_end):
            if i > 0 and i < len(acf) - 1:
                if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
                    return i

        return fallback

    def compute_multiscale_gradient(self, gray: np.ndarray) -> np.ndarray:
        """Multi-scale Sobel gradient (max across scales).

        Args:
            gray: Grayscale image (uint8).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        gray_f = gray.astype(np.float64)
        h, w = gray.shape
        max_gradient = np.zeros((h, w), dtype=np.float64)

        for sigma in self._sobel_sigmas:
            # Apply Gaussian smoothing
            if sigma > 0:
                smoothed = ndimage.gaussian_filter(gray_f, sigma=sigma)
            else:
                smoothed = gray_f

            # Compute Sobel gradients
            grad_x = ndimage.sobel(smoothed, axis=1)
            grad_y = ndimage.sobel(smoothed, axis=0)

            # Gradient magnitude
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Take max across scales
            max_gradient = np.maximum(max_gradient, magnitude)

        # Threshold using top percentile to control edge density
        # Percentile 95: keeps only ~5% strongest gradients (ink lines, not grid)
        nonzero = max_gradient[max_gradient > 1e-10]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, 95)
        else:
            threshold = 20
        edges = (max_gradient > threshold).astype(np.uint8) * 255

        return edges

    def detect_edges_canny(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detection with configured thresholds.

        Args:
            gray: Grayscale image (uint8).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        return edges

    def detect_chrominance_edges(self, bgr: np.ndarray) -> np.ndarray:
        """LAB chrominance ΔE-based edge detection.

        Grid is luminance-only, so chrominance edges are content edges.
        Pre-smoothing with sigma=2.0 destroys grid-scale (7-9px) color
        artifacts while preserving real content color boundaries.

        Args:
            bgr: BGR image (uint8).

        Returns:
            Binary edge map (uint8, 0 or 255).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

        # Extract a* and b* channels
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        # Pre-smooth chrominance to suppress grid-scale color artifacts
        # sigma=3.0 destroys variations at 7-9px period (grid) while
        # preserving content edges which span 20+ pixels
        a_channel = ndimage.gaussian_filter(a_channel, sigma=3.0)
        b_channel = ndimage.gaussian_filter(b_channel, sigma=3.0)

        # Compute gradients of smoothed chrominance channels
        grad_a_x = ndimage.sobel(a_channel, axis=1)
        grad_a_y = ndimage.sobel(a_channel, axis=0)
        grad_b_x = ndimage.sobel(b_channel, axis=1)
        grad_b_y = ndimage.sobel(b_channel, axis=0)

        # Combined chrominance gradient magnitude (ΔE-like)
        delta_e = np.sqrt(grad_a_x ** 2 + grad_a_y ** 2 + grad_b_x ** 2 + grad_b_y ** 2)

        # Threshold
        edges = (delta_e > self.chrominance_threshold).astype(np.uint8) * 255

        return edges

    def compute_multiscale_persistence(self, gray: np.ndarray) -> np.ndarray:
        """Edges that survive large-sigma blur are content edges.

        Grid period is 7-9px, so sigma=4 blur destroys grid but preserves content.

        Args:
            gray: Grayscale image (uint8).

        Returns:
            Binary edge map of persistent edges (uint8, 0 or 255).
        """
        gray_f = gray.astype(np.float64)

        # Apply large sigma blur
        blurred = ndimage.gaussian_filter(gray_f, sigma=3.0)  # 4.0 → 3.0

        # Compute edges on blurred image
        blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
        edges = cv2.Canny(blurred_u8, self.edge_low * 0.5, self.edge_high * 0.5)

        return edges

    def combine_raw_edges(
        self,
        *edge_maps: np.ndarray,
        guaranteed_edges: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine edge maps using 2+ detector agreement + guaranteed edges.

        Each detector's output is dilated by 2px before counting votes,
        to handle slight position differences between detectors.
        Guaranteed edges (e.g., persistence) are always included.

        Args:
            edge_maps: Variable number of binary edge maps for voting.
            guaranteed_edges: Optional edge map always included in result.

        Returns:
            Combined binary edge map (uint8, 0 or 255).
        """
        if not edge_maps:
            if guaranteed_edges is not None:
                return (guaranteed_edges > 0).astype(np.uint8) * 255
            return np.zeros((1, 1), dtype=np.uint8)

        # Dilate each map by 2px to compensate for position differences
        kernel = np.ones((3, 3), dtype=np.uint8)
        vote_count = np.zeros_like(edge_maps[0], dtype=np.int32)
        for edge_map in edge_maps:
            dilated = cv2.dilate(edge_map, kernel, iterations=1)
            vote_count += (dilated > 0).astype(np.int32)

        # Require at least 2 detectors to agree
        min_votes = 2 if len(edge_maps) >= 2 else 1
        combined = (vote_count >= min_votes).astype(np.uint8) * 255

        # Always include guaranteed edges (e.g., persistence = survived heavy blur)
        if guaranteed_edges is not None:
            combined = np.maximum(combined, (guaranteed_edges > 0).astype(np.uint8) * 255)

        return combined

    def compute_periodicity_score(
        self, edge_map: np.ndarray, period_x: int, period_y: int, gray: np.ndarray = None
    ) -> np.ndarray:
        """Score: edges repeating at grid period → low confidence (grid-like).

        Uses continuous gradient magnitude with local normalized cross-correlation
        instead of binary edge overlap. This works even when edge density is high.

        High score (1.0) = NOT periodic (likely content)
        Low score (0.0) = periodic (likely grid)

        Args:
            edge_map: Binary edge map.
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            gray: Optional grayscale image for gradient computation.

        Returns:
            Float array [0, 1] same shape as edge_map.
        """
        h, w = edge_map.shape
        edge_f = edge_map.astype(np.float32) / 255.0

        # Use gradient magnitude instead of binary edges
        if gray is not None:
            gray_f = gray.astype(np.float64)
            grad_x = ndimage.sobel(gray_f, axis=1)
            grad_y = ndimage.sobel(gray_f, axis=0)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
        else:
            grad_mag = edge_f  # fallback

        # Local window size for correlation (2x period)
        window = max(period_x, period_y) * 2 + 1
        if window % 2 == 0:
            window += 1
        window = min(window, min(h, w) // 2)
        if window < 3:
            window = 3

        periodicity = np.zeros((h, w), dtype=np.float32)

        # Horizontal periodicity via local normalized cross-correlation
        if period_x > 0 and period_x < w // 2:
            shifted = np.zeros_like(grad_mag)
            shifted[:, period_x:] = grad_mag[:, :-period_x]

            # Local means using uniform filter
            local_mean_g = ndimage.uniform_filter(grad_mag, size=window)
            local_mean_s = ndimage.uniform_filter(shifted, size=window)

            # Local cross-correlation
            product = grad_mag * shifted
            local_cross = ndimage.uniform_filter(product, size=window)

            # Local variances
            local_var_g = ndimage.uniform_filter(grad_mag**2, size=window) - local_mean_g**2
            local_var_s = ndimage.uniform_filter(shifted**2, size=window) - local_mean_s**2

            # Normalized cross-correlation
            denom = np.sqrt(np.maximum(local_var_g, 0) * np.maximum(local_var_s, 0)) + 1e-8
            ncc_x = (local_cross - local_mean_g * local_mean_s) / denom
            ncc_x = np.clip(ncc_x, 0, 1)  # only positive correlation = periodic
            periodicity = np.maximum(periodicity, ncc_x)

        # Vertical periodicity
        if period_y > 0 and period_y < h // 2:
            shifted = np.zeros_like(grad_mag)
            shifted[period_y:, :] = grad_mag[:-period_y, :]

            local_mean_g = ndimage.uniform_filter(grad_mag, size=window)
            local_mean_s = ndimage.uniform_filter(shifted, size=window)

            product = grad_mag * shifted
            local_cross = ndimage.uniform_filter(product, size=window)

            local_var_g = ndimage.uniform_filter(grad_mag**2, size=window) - local_mean_g**2
            local_var_s = ndimage.uniform_filter(shifted**2, size=window) - local_mean_s**2

            denom = np.sqrt(np.maximum(local_var_g, 0) * np.maximum(local_var_s, 0)) + 1e-8
            ncc_y = (local_cross - local_mean_g * local_mean_s) / denom
            ncc_y = np.clip(ncc_y, 0, 1)
            periodicity = np.maximum(periodicity, ncc_y)

        # Invert: high periodicity → low score (likely grid)
        score = 1.0 - periodicity

        # Only score edge pixels
        score = score * edge_f

        return score

    def compute_persistence_score(
        self, gray: np.ndarray, edge_map: np.ndarray
    ) -> np.ndarray:
        """Score: edges surviving multiple blur scales → high confidence.

        Args:
            gray: Grayscale image (uint8).
            edge_map: Binary edge map.

        Returns:
            Float array [0, 1] same shape as edge_map.
        """
        h, w = edge_map.shape
        gray_f = gray.astype(np.float64)
        edge_f = edge_map.astype(np.float32) / 255.0

        # Count how many scales each edge survives
        survival_count = np.zeros((h, w), dtype=np.float32)

        for sigma in self._persistence_sigmas:
            if sigma > 0:
                blurred = ndimage.gaussian_filter(gray_f, sigma=sigma)
            else:
                blurred = gray_f

            blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)

            # Detect edges at this scale
            scale_edges = cv2.Canny(blurred_u8, self.edge_low, self.edge_high)

            # Dilate slightly to account for position shift
            kernel = np.ones((3, 3), dtype=np.uint8)
            scale_edges_dilated = cv2.dilate(scale_edges, kernel, iterations=1)

            # Check if original edge pixels survive at this scale
            survives = (scale_edges_dilated > 0).astype(np.float32)
            survival_count += survives * edge_f

        # Normalize by number of scales
        n_scales = len(self._persistence_sigmas)
        score = survival_count / n_scales

        return score

    def compute_edge_length_score(self, edge_map: np.ndarray) -> np.ndarray:
        """Score: longer connected edges → higher confidence.

        Uses skeletonization to reduce dense edge blobs to thin lines
        before computing connected component sizes.

        Args:
            edge_map: Binary edge map.

        Returns:
            Float array [0, 1] same shape as edge_map.
        """
        from skimage.morphology import skeletonize

        edge_binary = (edge_map > 0).astype(np.uint8)

        # Skeletonize to get 1px-wide lines
        # This prevents dense edge blobs from forming one giant CC
        skeleton = skeletonize(edge_binary > 0).astype(np.uint8)

        # Find connected components on skeleton
        n_labels, labels_skel, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

        score = np.zeros_like(edge_map, dtype=np.float32)

        if n_labels <= 1:
            return score

        # Get component sizes
        sizes = []
        for label_idx in range(1, n_labels):
            sizes.append(stats[label_idx, cv2.CC_STAT_AREA])

        if not sizes:
            return score

        max_size = max(sizes)
        if max_size == 0:
            return score

        # Map skeleton CC labels back to original edge pixels
        # For each original edge pixel, find nearest skeleton label
        # Use distance transform to propagate labels from skeleton to edge

        # First, assign scores to skeleton pixels
        skel_score = np.zeros_like(score)
        for label_idx in range(1, n_labels):
            mask = labels_skel == label_idx
            size = stats[label_idx, cv2.CC_STAT_AREA]
            normalized = np.sqrt(size / max_size)
            skel_score[mask] = normalized

        # Propagate skeleton scores to nearby edge pixels using dilation
        # Iteratively dilate skeleton scores to fill edge regions
        propagated = skel_score.copy()
        edge_mask = edge_binary > 0
        for _ in range(5):  # 5 iterations of propagation
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(propagated, kernel, iterations=1)
            # Only fill unscored edge pixels
            unfilled = edge_mask & (propagated == 0)
            propagated[unfilled] = dilated[unfilled]

        # Final: only keep scores for edge pixels
        score = propagated * edge_mask.astype(np.float32)

        return score

    def compute_edge_confidence(
        self,
        edge_map: np.ndarray,
        periodicity: np.ndarray,
        persistence: np.ndarray,
        length: np.ndarray,
    ) -> np.ndarray:
        """Combine scores with weights → binary edge mask.

        Args:
            edge_map: Binary edge map.
            periodicity: Periodicity rejection score (high = not periodic).
            persistence: Multi-scale persistence score.
            length: Edge length score.

        Returns:
            Float confidence map [0, 1].
        """
        edge_f = (edge_map > 0).astype(np.float32)

        # Weighted combination
        confidence = (
            self._weight_periodicity * periodicity
            + self._weight_persistence * persistence
            + self._weight_length * length
        )

        # Only keep edge pixels above threshold
        confident_edges = np.where(
            (edge_f > 0) & (confidence >= self.confidence_threshold),
            confidence,
            0.0
        )

        return confident_edges.astype(np.float32)

    def cleanup_edge_mask(self, edge_mask: np.ndarray) -> np.ndarray:
        """Phase 4: morphological cleanup + length filtering.

        Args:
            edge_mask: Binary edge mask (bool or uint8).

        Returns:
            Cleaned binary edge mask (uint8, 0 or 255).
        """
        mask = (edge_mask > 0).astype(np.uint8) * 255

        # Morphological closing to close 1px gaps
        kernel_close = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Remove short connected components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        cleaned = np.zeros_like(mask)
        for label_idx in range(1, n_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= self.min_edge_length:
                cleaned[labels == label_idx] = 255

        return cleaned

    def segment_regions(self, edge_mask: np.ndarray) -> np.ndarray:
        """Phase 5: connected components + small region merging.

        Args:
            edge_mask: Binary edge mask (uint8).

        Returns:
            Label map where each region has a unique integer label.
        """
        # Invert edge mask to get regions (edges are boundaries)
        non_edge = (edge_mask == 0).astype(np.uint8)

        # Find connected components of non-edge regions
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            non_edge, connectivity=4
        )

        # Merge small regions into their largest adjacent neighbor
        if n_labels > 1:
            labels = self._merge_small_regions(labels, stats, n_labels)

        return labels

    def _merge_small_regions(
        self, labels: np.ndarray, stats: np.ndarray, n_labels: int
    ) -> np.ndarray:
        """Merge regions smaller than min_region_area into neighbors.

        Args:
            labels: Label map.
            stats: Component statistics from connectedComponentsWithStats.
            n_labels: Number of labels.

        Returns:
            Updated label map with small regions merged.
        """
        h, w = labels.shape
        result = labels.copy()

        # Find small regions
        small_regions = []
        for label_idx in range(1, n_labels):  # Skip background (0)
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < self.min_region_area:
                small_regions.append(label_idx)

        # For each small region, find largest adjacent neighbor
        for small_label in small_regions:
            mask = labels == small_label

            # Dilate to find neighbors
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

            # Find adjacent labels (in dilated area but not in original)
            adjacent_mask = (dilated > 0) & ~mask
            adjacent_labels = labels[adjacent_mask]

            if len(adjacent_labels) == 0:
                continue

            # Find most common adjacent label (excluding 0 and self)
            unique_labels, counts = np.unique(adjacent_labels, return_counts=True)

            # Filter out background and self
            valid_mask = (unique_labels != 0) & (unique_labels != small_label)
            if not np.any(valid_mask):
                continue

            valid_labels = unique_labels[valid_mask]
            valid_counts = counts[valid_mask]

            # Pick largest adjacent region by contact area
            best_neighbor = valid_labels[np.argmax(valid_counts)]

            # Merge: assign small region to neighbor
            result[mask] = best_neighbor

        return result

    def flatten_regions(self, bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Phase 6: gradient-preserving region flattening.

        Instead of replacing all pixels with a single median value (which
        destroys gradients), uses bilateral filtering to remove grid-frequency
        noise while preserving content gradients within each region.

        For small regions (< 200 pixels), falls back to median since bilateral
        filtering has limited effect.

        Args:
            bgr: BGR image (uint8).
            labels: Label map from segmentation.

        Returns:
            Flattened BGR image (uint8).
        """
        # Apply bilateral filter to entire image first (grid removal)
        # d=9 covers typical grid period (7-9px), sigmaColor=75 preserves
        # edges within region, sigmaSpace=75 for spatial smoothing
        bilateral = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

        result = bgr.copy()
        n_labels = int(labels.max()) + 1

        for label_idx in range(1, n_labels):  # Skip edges (label 0)
            mask = labels == label_idx
            if not np.any(mask):
                continue

            region_size = np.sum(mask)

            if region_size < 200:
                # Small regions: median is more robust
                for c in range(3):
                    channel_pixels = bgr[:, :, c][mask]
                    if len(channel_pixels) == 0:
                        continue

                    if self.flatten_method == "trimmed_mean":
                        low = np.percentile(channel_pixels, 10)
                        high = np.percentile(channel_pixels, 90)
                        trimmed = channel_pixels[
                            (channel_pixels >= low) & (channel_pixels <= high)
                        ]
                        flat_value = np.mean(trimmed) if len(trimmed) > 0 else np.median(channel_pixels)
                    else:
                        flat_value = np.median(channel_pixels)

                    result[:, :, c][mask] = np.clip(flat_value, 0, 255).astype(np.uint8)
            else:
                # Large regions: use bilateral-filtered result to preserve gradients
                result[mask] = bilateral[mask]

        return result

    def blend_edge_zones(
        self, flattened: np.ndarray, original: np.ndarray, edge_mask: np.ndarray
    ) -> np.ndarray:
        """Phase 7: distance-weighted blending near edges.

        Edge center preserves original, farther away uses flattened result.

        Args:
            flattened: Flattened BGR image.
            original: Original BGR image.
            edge_mask: Binary edge mask (uint8).

        Returns:
            Blended BGR image (uint8).
        """
        # Distance from edge
        non_edge = (edge_mask == 0).astype(np.uint8)
        distance = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3)

        # Compute alpha: 0 at edge, 1 far from edge
        alpha = np.clip(distance / max(self.blend_width, 1), 0, 1)

        # Expand alpha to 3 channels
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)

        # Blend: near edge = original, far from edge = flattened
        result = (original.astype(np.float32) * (1 - alpha_3ch)
                  + flattened.astype(np.float32) * alpha_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)

    def edge_aware_enhance(
        self,
        blended: np.ndarray,
        original: np.ndarray,
        edge_mask: np.ndarray,
    ) -> np.ndarray:
        """Phase 8: Edge-aware enhancement.

        Near edges: use bilateral-filtered original (clean edges without grid).
        Far from edges: apply extra bilateral smoothing to suppress residual grid.
        Soft Gaussian transition between the two.

        Args:
            blended: Blended BGR result from Phase 7.
            original: Original BGR image.
            edge_mask: Binary edge mask (uint8).

        Returns:
            Enhanced BGR image (uint8).
        """
        # Soft edge proximity: 1.0 at edges, fading to 0.0
        edge_float = edge_mask.astype(np.float32) / 255.0
        edge_proximity = cv2.GaussianBlur(edge_float, (0, 0), sigmaX=2.0)
        edge_proximity_3ch = np.stack([edge_proximity] * 3, axis=-1)

        # Edge source: bilateral-filtered original (removes grid, keeps ink lines)
        bilateral_original = cv2.bilateralFilter(original, d=9, sigmaColor=50, sigmaSpace=50)

        # Non-edge source: extra-smoothed contour result (kills residual grid)
        extra_smooth = cv2.bilateralFilter(blended, d=9, sigmaColor=40, sigmaSpace=40)

        # Blend: edge areas = bilateral original, non-edge = extra smooth
        result = (extra_smooth.astype(np.float32) * (1 - edge_proximity_3ch)
                  + bilateral_original.astype(np.float32) * edge_proximity_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _measure_channel_modulation(
        self,
        bgr: np.ndarray,
        px: int,
        py: int,
        n_harmonics: int = 3,
    ) -> Dict[int, float]:
        """Measure per-channel modulation depth at grid frequencies.

        For each BGR channel, computes the ratio of energy at grid
        harmonic frequencies to total energy (excluding DC).

        Args:
            bgr: Input image in BGR format (uint8).
            px: Horizontal grid period.
            py: Vertical grid period.
            n_harmonics: Number of harmonics to include.

        Returns:
            Dict mapping channel index (0=B, 1=G, 2=R) to modulation depth [0,1].
        """
        modulation: Dict[int, float] = {}
        h, w = bgr.shape[:2]

        for c in range(3):
            channel = bgr[:, :, c].astype(np.float64)

            # Row-wise 1D FFT for horizontal grid
            F_rows = np.fft.rfft(channel, axis=1)
            mag_rows = np.abs(F_rows)

            # Column-wise 1D FFT for vertical grid
            F_cols = np.fft.rfft(channel, axis=0)
            mag_cols = np.abs(F_cols)

            # Total energy (excluding DC)
            total_h = np.sum(mag_rows[:, 1:] ** 2)
            total_v = np.sum(mag_cols[1:, :] ** 2)
            total_energy = total_h + total_v

            # Grid harmonic energy
            grid_energy = 0.0
            n_freq_h = mag_rows.shape[1]
            n_freq_v = mag_cols.shape[0]

            for k in range(1, n_harmonics + 1):
                # Horizontal harmonics
                if px > 0:
                    fk_h = int(round(k * w / px))
                    for offset in range(-2, 3):
                        idx = fk_h + offset
                        if 1 <= idx < n_freq_h:
                            grid_energy += np.sum(mag_rows[:, idx] ** 2)

                # Vertical harmonics
                if py > 0:
                    fk_v = int(round(k * h / py))
                    for offset in range(-2, 3):
                        idx = fk_v + offset
                        if 1 <= idx < n_freq_v:
                            grid_energy += np.sum(mag_cols[idx, :] ** 2)

            modulation[c] = grid_energy / max(total_energy, 1e-10)

        return modulation

    @staticmethod
    def _create_1d_notch_mask(
        n_freq: int,
        period: float,
        img_len: int,
        n_harmonics: int = 3,
        sigma: float = 0.8,
        attenuation: float = 0.05,
    ) -> np.ndarray:
        """Create 1D Gaussian notch mask for rfft output.

        Suppresses grid harmonic frequencies with narrow Gaussian notches.
        DC component is always preserved.

        Args:
            n_freq: Number of frequency bins (rfft output length).
            period: Grid period in pixels.
            img_len: Image dimension (width for rows, height for cols).
            n_harmonics: Number of harmonics to suppress.
            sigma: Gaussian notch width in frequency bins (narrower = less content damage).
            attenuation: Minimum value at notch center (0 = total removal).

        Returns:
            1D array of shape (n_freq,) with values in [attenuation, 1.0].
        """
        mask = np.ones(n_freq, dtype=np.float64)

        if period <= 0:
            return mask

        freq_bins = np.arange(n_freq, dtype=np.float64)

        for k in range(1, n_harmonics + 1):
            # Harmonic frequency in bin index
            fk = k * img_len / period

            if fk >= n_freq:
                break

            # Gaussian notch: 1 - (1-attenuation) * exp(-0.5 * ((f - fk) / sigma)^2)
            gaussian = np.exp(-0.5 * ((freq_bins - fk) / sigma) ** 2)
            mask *= 1.0 - (1.0 - attenuation) * gaussian

        # Ensure DC is preserved
        mask[0] = 1.0

        return mask

    def _apply_separable_notch(
        self,
        bgr_f64: np.ndarray,
        px: int,
        py: int,
        modulation: Dict[int, float],
        n_harmonics: int = 3,
        sigma: float = 0.8,
        attenuation: float = 0.05,
        channel_adaptive: bool = True,
    ) -> np.ndarray:
        """Apply separable 1D FFT notch filter to remove grid patterns.

        For each channel:
        1. Row-wise rfft -> apply horizontal notch mask -> irfft
        2. Column-wise rfft -> apply vertical notch mask -> irfft

        Channel-adaptive mode scales notch strength by per-channel modulation depth.

        Args:
            bgr_f64: Input image as float64 BGR.
            px: Horizontal grid period.
            py: Vertical grid period.
            modulation: Per-channel modulation depths from _measure_channel_modulation.
            n_harmonics: Number of harmonics to filter.
            sigma: Notch width in frequency bins.
            attenuation: Minimum value at notch center.
            channel_adaptive: Whether to scale notch strength per channel.

        Returns:
            Filtered image as float64 BGR.
        """
        result = bgr_f64.copy()
        h, w = result.shape[:2]

        # Pre-compute notch masks
        n_freq_h = w // 2 + 1  # rfft output length for rows
        n_freq_v = h // 2 + 1  # rfft output length for cols

        mask_h = self._create_1d_notch_mask(n_freq_h, px, w, n_harmonics, sigma, attenuation)
        mask_v = self._create_1d_notch_mask(n_freq_v, py, h, n_harmonics, sigma, attenuation)

        # Max modulation for normalization
        max_mod = max(modulation.values()) if modulation else 1.0
        if max_mod < 1e-10:
            max_mod = 1.0

        for c in range(3):
            channel = result[:, :, c]

            # Channel-adaptive scaling
            if channel_adaptive and c in modulation:
                ch_scale = modulation[c] / max_mod
            else:
                ch_scale = 1.0

            # Scaled notch masks: blend between identity (1.0) and full notch
            # effective_mask = 1 - ch_scale * (1 - mask)
            eff_mask_h = 1.0 - ch_scale * (1.0 - mask_h)
            eff_mask_v = 1.0 - ch_scale * (1.0 - mask_v)

            # Horizontal: row-wise FFT
            F_rows = np.fft.rfft(channel, axis=1)
            F_rows *= eff_mask_h[np.newaxis, :]  # broadcast across rows
            channel = np.fft.irfft(F_rows, n=w, axis=1)

            # Vertical: column-wise FFT
            F_cols = np.fft.rfft(channel, axis=0)
            F_cols *= eff_mask_v[:, np.newaxis]  # broadcast across columns
            channel = np.fft.irfft(F_cols, n=h, axis=0)

            result[:, :, c] = channel

        return result

    def _apply_directional_median(
        self,
        bgr: np.ndarray,
        px: int,
        py: int,
        strength: float = 0.3,
    ) -> np.ndarray:
        """Apply directional median filter to clean residual grid patterns.

        Uses 1D median filters aligned with grid directions to remove
        remaining grid artifacts without blurring content edges.

        Args:
            bgr: Input image (float64 BGR).
            px: Horizontal grid period.
            py: Vertical grid period.
            strength: Blending strength (0 = no effect, 1 = full median).

        Returns:
            Cleaned image as float64 BGR.
        """
        if strength <= 0:
            return bgr.copy()

        result = bgr.copy()

        # Kernel sizes: use grid period (odd, at least 3)
        kx = max(3, px if px % 2 == 1 else px + 1)
        ky = max(3, py if py % 2 == 1 else py + 1)

        for c in range(3):
            channel = result[:, :, c]

            # Horizontal median (removes vertical grid lines)
            h_med = median_filter(channel, size=(1, kx))

            # Vertical median (removes horizontal grid lines)
            v_med = median_filter(channel, size=(ky, 1))

            # Average of directional medians
            avg_med = (h_med + v_med) * 0.5

            # Blend
            result[:, :, c] = (1.0 - strength) * channel + strength * avg_med

        return result

    def process_v2(
        self,
        image_bgr: np.ndarray,
        notch_sigma: float = 0.8,
        notch_attenuation: float = 0.05,
        n_harmonics: int = 3,
        median_strength: float = 0.3,
        channel_adaptive: bool = True,
        edge_restore_sigma: float = 1.5,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Contour v2 pipeline: Separable 1D Notch Filter for grid removal.

        Replaces bilateral filter (Phase 6/8) with frequency-domain notch
        filtering that specifically targets grid harmonics.

        Pipeline:
        1-5. Same as process() (period detection, edge detection, scoring, cleanup, segmentation)
        6A. Separable 1D FFT Notch Filter (NEW - replaces bilateral)
        6B. Directional Median Residual Cleaning (NEW)
        7. Edge Zone Blending (same as process())
        8. Simplified Edge Restoration (NEW - replaces dual bilateral)

        Args:
            image_bgr: Input BGR image (uint8).
            notch_sigma: Frequency-domain notch width (narrower = less content damage).
            notch_attenuation: Notch center minimum (0 = total removal).
            n_harmonics: Number of grid harmonics to filter.
            median_strength: Directional median cleaning strength.
            channel_adaptive: Enable per-channel modulation-based adaptation.
            edge_restore_sigma: Gaussian sigma for edge restoration blending.

        Returns:
            Tuple of (processed BGR image uint8, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v2",
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # === Phase 1: Grid Period Detection (reuse) ===
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # === Phase 2: Multi-Strategy Edge Detection (reuse) ===
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v2",
            }

        # === Phase 3: Edge Confidence Scoring (reuse) ===
        periodicity_score = self.compute_periodicity_score(
            raw_edge_map, period_x, period_y, gray=gray
        )
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)
        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # === Phase 4: Edge Mask Cleanup (reuse) ===
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # === Phase 5: Region Segmentation (reuse) ===
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # === Phase 6A: Separable 1D Notch Filter (NEW) ===
        # Measure per-channel modulation depth
        modulation = self._measure_channel_modulation(
            image_bgr, period_x, period_y, n_harmonics=n_harmonics
        )

        # Apply separable notch filter
        bgr_f64 = image_bgr.astype(np.float64)
        notch_filtered = self._apply_separable_notch(
            bgr_f64, period_x, period_y, modulation,
            n_harmonics=n_harmonics,
            sigma=notch_sigma,
            attenuation=notch_attenuation,
            channel_adaptive=channel_adaptive,
        )

        # === Phase 6B: Directional Median Residual Cleaning (NEW) ===
        cleaned = self._apply_directional_median(
            notch_filtered, period_x, period_y, strength=median_strength
        )

        # Clip and convert to uint8
        processed = np.clip(cleaned, 0, 255).astype(np.uint8)

        # === Phase 7: Edge Zone Blending (reuse) ===
        blended = self.blend_edge_zones(processed, image_bgr, final_edge_mask)

        # === Phase 8: Simplified Edge Restoration (NEW) ===
        # Simple Gaussian-weighted blend: edges → original, non-edges → processed
        edge_float = final_edge_mask.astype(np.float32) / 255.0
        edge_proximity = cv2.GaussianBlur(edge_float, (0, 0), sigmaX=edge_restore_sigma)
        edge_proximity_3ch = np.stack([edge_proximity] * 3, axis=-1)

        result = (blended.astype(np.float32) * (1.0 - edge_proximity_3ch)
                  + image_bgr.astype(np.float32) * edge_proximity_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "n_regions": n_regions,
            "channel_modulation": modulation,
            "version": "v2",
        }

        return result, intermediates

    def process_v3(
        self,
        image_bgr: np.ndarray,
        notch_sigma: float = 2.0,
        notch_attenuation: float = 0.10,
        n_harmonics: int = 3,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Contour v3 pipeline: Notch + Bilateral hybrid for grid removal.

        Combines v2's FFT notch filter (grid frequency removal) with v1's
        bilateral filtering (artifact cleanup and visual quality). The notch
        filter removes grid harmonics, then bilateral post-smoothing cleans
        ringing/banding artifacts. Phase 7-8 reuse v1's edge blending and
        dual bilateral for visual quality parity.

        Pipeline:
        1-5. Same as process() (period detection -> edge -> scoring -> cleanup -> segmentation)
        6. Notch + Bilateral Hybrid (NEW):
           6A. Separable 1D FFT notch filter (conservative params)
           6B. Bilateral post-smoothing (cleans notch artifacts)
           6C. Region-aware application (large=bilateral, small=median)
        7. Edge Zone Blending (same as v1)
        8. Dual Bilateral Edge Restoration (same as v1)

        Args:
            image_bgr: Input BGR image (uint8).
            notch_sigma: Frequency-domain notch width (narrower = less content damage).
            notch_attenuation: Notch center minimum (higher = more conservative).
            n_harmonics: Number of grid harmonics to filter.
            bilateral_d: Bilateral filter kernel diameter.
            bilateral_sigma_color: Bilateral color sigma.
            bilateral_sigma_space: Bilateral spatial sigma.

        Returns:
            Tuple of (processed BGR image uint8, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v3",
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # === Phase 1: Grid Period Detection (reuse) ===
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # === Phase 2: Multi-Strategy Edge Detection (reuse) ===
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v3",
            }

        # === Phase 3: Edge Confidence Scoring (reuse) ===
        periodicity_score = self.compute_periodicity_score(
            raw_edge_map, period_x, period_y, gray=gray
        )
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)
        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # === Phase 4: Edge Mask Cleanup (reuse) ===
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # === Phase 5: Region Segmentation (reuse) ===
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # === Phase 6: Notch + Bilateral Hybrid (NEW) ===

        # Step 6A: Separable 1D Notch Filter (v2 helper, conservative params)
        modulation = self._measure_channel_modulation(
            image_bgr, period_x, period_y, n_harmonics=n_harmonics
        )
        bgr_f64 = image_bgr.astype(np.float64)
        notch_result = self._apply_separable_notch(
            bgr_f64, period_x, period_y, modulation,
            n_harmonics=n_harmonics,
            sigma=notch_sigma,
            attenuation=notch_attenuation,
            channel_adaptive=False,  # v2 results showed noAdapt is better
        )
        notch_u8 = np.clip(notch_result, 0, 255).astype(np.uint8)

        # Step 6B: Bilateral Post-Smoothing (cleans ringing/banding from notch)
        bilateral_smoothed = cv2.bilateralFilter(
            notch_u8, d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )

        # Step 6C: Region-aware application
        result = image_bgr.copy()
        n_labels = int(label_map.max()) + 1

        for label_idx in range(1, n_labels):  # Skip edges (label 0)
            mask = label_map == label_idx
            if not np.any(mask):
                continue

            region_size = np.sum(mask)

            if region_size < 200:
                # Small regions: median fallback (same as v1)
                for c in range(3):
                    channel_pixels = image_bgr[:, :, c][mask]
                    if len(channel_pixels) == 0:
                        continue
                    flat_value = np.median(channel_pixels)
                    result[:, :, c][mask] = np.clip(flat_value, 0, 255).astype(np.uint8)
            else:
                # Large regions: use bilateral-smoothed notch result
                result[mask] = bilateral_smoothed[mask]

        # === Phase 7: Edge Zone Blending (v1 reuse) ===
        blended = self.blend_edge_zones(result, image_bgr, final_edge_mask)

        # === Phase 8: Dual Bilateral Edge Restoration (v1 reuse) ===
        result = self.edge_aware_enhance(blended, image_bgr, final_edge_mask)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "n_regions": n_regions,
            "channel_modulation": modulation,
            "version": "v3",
        }

        return result, intermediates

    def _build_dark_object_mask(
        self,
        gray: np.ndarray,
        edge_mask: np.ndarray,
        dark_block_size: int = 51,
        dark_c_offset: float = 10.0,
        min_dark_area: int = 30,
        dark_edge_dilate: int = 2,
    ) -> np.ndarray:
        """Build a mask of dark objects (ink lines, dark regions) to protect.

        Combines adaptive thresholding (detects locally dark regions) with
        dilated edge mask to cover the full width of ink strokes.

        Args:
            gray: Grayscale image (uint8).
            edge_mask: Binary edge mask from Phase 4 (uint8, 0 or 255).
            dark_block_size: Block size for adaptive threshold (must be odd).
            dark_c_offset: Offset for adaptive threshold (higher = more aggressive dark detection).
            min_dark_area: Minimum area for dark blobs (removes noise).
            dark_edge_dilate: Pixels to dilate edge mask to cover ink stroke width.

        Returns:
            Binary mask (uint8, 0 or 255) where 255 = dark object region to protect.
        """
        # Ensure block_size is odd
        if dark_block_size % 2 == 0:
            dark_block_size += 1

        # Step 1: Adaptive threshold to detect locally dark regions
        # THRESH_BINARY_INV + ADAPTIVE_GAUSSIAN: pixels darker than local mean - c_offset → 255
        dark_adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            dark_block_size,
            dark_c_offset,
        )

        # Step 2: Dilate edge mask to cover full ink stroke width
        if dark_edge_dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dark_edge_dilate + 1, 2 * dark_edge_dilate + 1),
            )
            dilated_edges = cv2.dilate(edge_mask, kernel, iterations=1)
        else:
            dilated_edges = edge_mask.copy()

        # Step 3: Combine — protect any pixel that is dark OR near an edge
        dark_mask = np.maximum(dark_adaptive, dilated_edges)

        # Step 4: Remove small noise blobs
        if min_dark_area > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                dark_mask, connectivity=8
            )
            cleaned = np.zeros_like(dark_mask)
            for label_idx in range(1, n_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area >= min_dark_area:
                    cleaned[labels == label_idx] = 255
            dark_mask = cleaned

        return dark_mask

    def process_v4(
        self,
        image_bgr: np.ndarray,
        # Dark object mask params
        dark_block_size: int = 51,
        dark_c_offset: float = 10.0,
        dark_edge_dilate: int = 2,
        # Notch params (v3 best vicinity)
        notch_sigma: float = 3.0,
        notch_attenuation: float = 0.05,
        n_harmonics: int = 4,
        # Light region post-processing
        light_bilateral_d: int = 7,
        light_bilateral_sigma: float = 50.0,
        # Dark region handling
        dark_preserve_mode: str = "original",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Contour v4 pipeline: Dark Object Protection + Aggressive Interior Flattening.

        Key insight: Instead of protecting only thin edge boundaries (1-2px),
        protect the ENTIRE dark object region (ink lines = 3-20px wide).
        Apply aggressive grid removal ONLY to bright/light regions.

        Pipeline:
        1-5. Same as v1/v3 (period detection, edge detection, scoring, cleanup, segmentation)
        5.5. Dark Object Mask generation (NEW)
        6. Differential processing:
           - Light regions: notch filter + optional bilateral (aggressive)
           - Dark regions: original preservation or light bilateral
        7. Edge Zone Blending (v1, enhanced with dark mask)
        8. Edge-aware Enhancement (v1, enhanced with dark mask)

        Args:
            image_bgr: Input BGR image (uint8).
            dark_block_size: Adaptive threshold block size for dark detection.
            dark_c_offset: Adaptive threshold offset (higher = more dark detected).
            dark_edge_dilate: Edge mask dilation for ink stroke width coverage.
            notch_sigma: Notch filter width in frequency bins.
            notch_attenuation: Notch center minimum value.
            n_harmonics: Number of grid harmonics to filter.
            light_bilateral_d: Bilateral filter diameter for light regions (0 = notch only).
            light_bilateral_sigma: Bilateral sigma for light regions.
            dark_preserve_mode: "original" or "light_bilateral".

        Returns:
            Tuple of (processed BGR image uint8, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "dark_object_mask": np.zeros((h, w), dtype=np.uint8),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v4",
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # === Phase 1: Grid Period Detection (reuse) ===
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # === Phase 2: Multi-Strategy Edge Detection (reuse) ===
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "dark_object_mask": np.zeros((h, w), dtype=np.uint8),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v4",
            }

        # === Phase 3: Edge Confidence Scoring (reuse) ===
        periodicity_score = self.compute_periodicity_score(
            raw_edge_map, period_x, period_y, gray=gray
        )
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)
        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # === Phase 4: Edge Mask Cleanup (reuse) ===
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # === Phase 5: Region Segmentation (reuse) ===
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # === Phase 5.5: Dark Object Mask (NEW) ===
        dark_object_mask = self._build_dark_object_mask(
            gray, final_edge_mask,
            dark_block_size=dark_block_size,
            dark_c_offset=dark_c_offset,
            dark_edge_dilate=dark_edge_dilate,
        )

        # === Phase 6: Differential Processing (NEW) ===
        # Step 6A: Apply notch filter to entire image (grid harmonic removal)
        modulation = self._measure_channel_modulation(
            image_bgr, period_x, period_y, n_harmonics=n_harmonics
        )
        bgr_f64 = image_bgr.astype(np.float64)
        notch_result = self._apply_separable_notch(
            bgr_f64, period_x, period_y, modulation,
            n_harmonics=n_harmonics,
            sigma=notch_sigma,
            attenuation=notch_attenuation,
            channel_adaptive=False,
        )
        notch_u8 = np.clip(notch_result, 0, 255).astype(np.uint8)

        # Step 6B: Prepare light region result (notch + optional bilateral)
        if light_bilateral_d > 0:
            processed_light = cv2.bilateralFilter(
                notch_u8, d=light_bilateral_d,
                sigmaColor=light_bilateral_sigma,
                sigmaSpace=light_bilateral_sigma,
            )
        else:
            processed_light = notch_u8

        # Step 6C: Prepare dark region result
        if dark_preserve_mode == "light_bilateral":
            # Light bilateral: gentle smoothing that preserves edges
            dark_result = cv2.bilateralFilter(
                image_bgr, d=5, sigmaColor=30, sigmaSpace=30,
            )
        else:
            # "original": keep original pixels for dark regions
            dark_result = image_bgr

        # Step 6D: Composite — apply differentially based on dark mask
        result = image_bgr.copy()
        light_mask = dark_object_mask == 0  # regions NOT in dark mask

        result[light_mask] = processed_light[light_mask]
        result[~light_mask] = dark_result[~light_mask]

        # === Phase 7: Edge Zone Blending (v1, enhanced with dark mask) ===
        # Use standard edge blending but incorporate dark mask
        non_edge = (final_edge_mask == 0).astype(np.uint8)
        distance = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3)
        alpha = np.clip(distance / max(self.blend_width, 1), 0, 1)

        # Reduce alpha (more original) where dark mask is active
        dark_float = dark_object_mask.astype(np.float32) / 255.0
        alpha = alpha * (1.0 - 0.5 * dark_float)  # dark regions: halve the flattening weight

        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        blended = (image_bgr.astype(np.float32) * (1 - alpha_3ch)
                   + result.astype(np.float32) * alpha_3ch)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # === Phase 8: Edge-aware Enhancement (v1, enhanced with dark mask) ===
        # Soft edge proximity combining edge mask AND dark object mask
        edge_float = final_edge_mask.astype(np.float32) / 255.0
        dark_proximity = cv2.GaussianBlur(dark_float, (0, 0), sigmaX=2.0)
        edge_proximity = cv2.GaussianBlur(edge_float, (0, 0), sigmaX=2.0)

        # Combined protection: max of edge proximity and dark proximity
        combined_proximity = np.maximum(edge_proximity, dark_proximity)
        combined_proximity_3ch = np.stack([combined_proximity] * 3, axis=-1)

        # Edge source: bilateral-filtered original
        bilateral_original = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=50, sigmaSpace=50)

        # Non-edge source: extra-smoothed result
        extra_smooth = cv2.bilateralFilter(blended, d=9, sigmaColor=40, sigmaSpace=40)

        # Blend with enhanced proximity (dark regions → more original preservation)
        final = (extra_smooth.astype(np.float32) * (1 - combined_proximity_3ch)
                 + bilateral_original.astype(np.float32) * combined_proximity_3ch)
        final = np.clip(final, 0, 255).astype(np.uint8)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "dark_object_mask": dark_object_mask,
            "n_regions": n_regions,
            "channel_modulation": modulation,
            "version": "v4",
        }

        return final, intermediates

    # ------------------------------------------------------------------
    # V5: Brightness-Adaptive Median Template Subtraction
    # ------------------------------------------------------------------

    def _estimate_grid_template_brightness_adaptive(
        self,
        image: np.ndarray,
        period_x: int,
        period_y: int,
        n_brightness_bands: int = 4,
        use_median: bool = True,
    ) -> np.ndarray:
        """Estimate grid template adaptively per brightness band.

        Unlike V3.5 estimate_grid_template_2d() which uses only flat regions
        and mean statistics, this method:
        - Uses ALL pixels (partitioned by brightness band)
        - Computes MEDIAN of residuals (robust to content outliers)
        - Builds separate templates per brightness band
        - Interpolates across bands for smooth grid estimate

        Args:
            image: Single-channel grayscale image (uint8 or float).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            n_brightness_bands: Number of brightness bands.
            use_median: If True use median; if False use mean.

        Returns:
            Grid estimate map (float32, ~0 centered) same size as input.
        """
        gray = image.astype(np.float32)
        h, w = gray.shape

        # Local mean (kernel = 2x max period)
        ksize = max(period_x, period_y) * 2 + 1
        if ksize % 2 == 0:
            ksize += 1
        local_mean = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Brightness band thresholds (percentile-based)
        percentiles = np.linspace(0, 100, n_brightness_bands + 1)
        thresholds = np.percentile(local_mean, percentiles)
        # Ensure last threshold captures all pixels
        thresholds[-1] = thresholds[-1] + 1.0

        # Pre-compute period coordinate maps
        yy_mod = np.arange(h)[:, None] % period_y  # (h, 1)
        xx_mod = np.arange(w)[None, :] % period_x  # (1, w)

        # Residuals: actual value - local mean = grid component
        residuals = gray - local_mean

        # Estimate template per brightness band
        band_means = []
        band_templates = []
        min_samples = 5

        for b in range(n_brightness_bands):
            band_mask = (local_mean >= thresholds[b]) & (local_mean < thresholds[b + 1])
            if not band_mask.any():
                band_means.append((thresholds[b] + thresholds[b + 1]) / 2.0)
                band_templates.append(np.zeros((period_y, period_x), dtype=np.float32))
                continue

            band_mean_val = float(local_mean[band_mask].mean())
            band_means.append(band_mean_val)

            template = np.zeros((period_y, period_x), dtype=np.float32)
            for py in range(period_y):
                for px in range(period_x):
                    cell_mask = band_mask & (yy_mod == py) & (xx_mod == px)
                    count = cell_mask.sum()
                    if count > min_samples:
                        vals = residuals[cell_mask]
                        template[py, px] = float(
                            np.median(vals) if use_median else np.mean(vals)
                        )
            band_templates.append(template)

        # Build full-size grid estimate via brightness interpolation
        n_bands = len(band_means)
        band_means_arr = np.array(band_means, dtype=np.float32)

        # Stack templates: (n_bands, Py, Px)
        templates_stack = np.stack(band_templates, axis=0)

        # For each pixel: find two nearest bands and interpolate
        grid_estimate = np.zeros((h, w), dtype=np.float32)

        # Vectorized: map each pixel to its folded template coordinates
        fy = (np.arange(h)[:, None] % period_y).astype(np.intp)  # (h, 1)
        fx = (np.arange(w)[None, :] % period_x).astype(np.intp)  # (1, w)

        # Tile-index arrays (full 2D via broadcasting)
        fy_full = np.broadcast_to(fy, (h, w))
        fx_full = np.broadcast_to(fx, (h, w))

        if n_bands == 1:
            # Single band: vectorized template lookup
            grid_estimate = templates_stack[0, fy_full, fx_full]
        else:
            # Multi-band interpolation (fully vectorized)
            bmin = band_means_arr[0]
            bmax = band_means_arr[-1]
            brightness_clamped = np.clip(local_mean, bmin, bmax)

            # Find band indices for interpolation
            band_idx = np.searchsorted(
                band_means_arr, brightness_clamped.ravel(), side='right',
            ).reshape(h, w) - 1
            band_idx = np.clip(band_idx, 0, n_bands - 2)

            # Interpolation weights
            lower_means = band_means_arr[band_idx]
            upper_means = band_means_arr[band_idx + 1]
            denom = np.where(
                (upper_means - lower_means) < 1e-6, 1.0,
                upper_means - lower_means,
            )
            weight_upper = np.clip(
                (brightness_clamped - lower_means) / denom, 0.0, 1.0,
            )
            weight_lower = 1.0 - weight_upper

            # Vectorized gather: templates_stack[band_idx, fy, fx]
            lower_vals = templates_stack[band_idx, fy_full, fx_full]
            upper_vals = templates_stack[band_idx + 1, fy_full, fx_full]

            grid_estimate = weight_lower * lower_vals + weight_upper * upper_vals

        return grid_estimate

    def _subtract_grid_per_channel(
        self,
        image_bgr: np.ndarray,
        period_x: int,
        period_y: int,
        n_brightness_bands: int = 4,
        subtraction_strength: float = 1.0,
        use_median: bool = True,
    ) -> np.ndarray:
        """Estimate and subtract grid template per BGR channel.

        Grid modulation depth varies across channels (typically Blue > Green > Red).
        Processing each channel independently captures this variation.

        Args:
            image_bgr: Input BGR image (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            n_brightness_bands: Number of brightness bands for template estimation.
            subtraction_strength: Multiplier for grid subtraction (1.0 = full).
            use_median: Use median (True) or mean (False) for template estimation.

        Returns:
            Grid-subtracted BGR image (uint8).
        """
        result = np.zeros_like(image_bgr, dtype=np.float32)

        for c in range(3):
            channel = image_bgr[:, :, c]
            grid_est = self._estimate_grid_template_brightness_adaptive(
                channel, period_x, period_y,
                n_brightness_bands=n_brightness_bands,
                use_median=use_median,
            )
            result[:, :, c] = channel.astype(np.float32) - grid_est * subtraction_strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def process_v5(
        self,
        image_bgr: np.ndarray,
        # Template estimation params
        n_brightness_bands: int = 4,
        subtraction_strength: float = 1.0,
        use_median: bool = True,
        # Residual notch params
        residual_notch: bool = True,
        residual_sigma: float = 1.5,
        residual_attenuation: float = 0.10,
        # Edge restoration params
        edge_restore_sigma: float = 1.5,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Contour v5: Brightness-Adaptive Median Template Subtraction.

        Key idea: Estimate the grid pattern directly by period-folding with
        brightness-adaptive median statistics, then subtract it. This avoids
        frequency-domain filtering that damages edges.

        Pipeline:
        1-5. Contour infrastructure (period detection, edge detection,
             scoring, cleanup, segmentation)
        6.   Brightness-band median grid template estimation (per channel)
        7.   Channel-wise grid subtraction
        8.   Optional gentle residual notch filtering
        9.   TRUE original edge restoration (Gaussian-weighted blend)

        Args:
            image_bgr: Input BGR image (uint8).
            n_brightness_bands: Number of brightness bands for template estimation.
            subtraction_strength: Grid subtraction multiplier.
            use_median: Use median (True) or mean (False) for template estimation.
            residual_notch: Apply residual notch filtering after subtraction.
            residual_sigma: Notch width for residual filtering (0 = skip).
            residual_attenuation: Attenuation for residual notch.
            edge_restore_sigma: Gaussian sigma for edge restoration blending.

        Returns:
            Tuple of (processed BGR image uint8, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "version": "v5",
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # === Phase 1: Grid Period Detection (reuse) ===
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # Guard: if no valid periods detected, return original
        if period_x <= 0 or period_y <= 0:
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "version": "v5",
            }

        # === Phase 2: Multi-Strategy Edge Detection (reuse) ===
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "version": "v5",
            }

        # === Phase 3: Edge Confidence Scoring (reuse) ===
        periodicity_score = self.compute_periodicity_score(
            raw_edge_map, period_x, period_y, gray=gray
        )
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)
        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # === Phase 4: Edge Mask Cleanup (reuse) ===
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # === Phase 5: Region Segmentation (reuse) ===
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # === Phase 6-7: Channel-wise grid template subtraction ===
        subtracted = self._subtract_grid_per_channel(
            image_bgr, period_x, period_y,
            n_brightness_bands=n_brightness_bands,
            subtraction_strength=subtraction_strength,
            use_median=use_median,
        )

        # === Phase 8: Optional residual notch filtering ===
        processed = subtracted
        if residual_notch and residual_sigma > 0:
            modulation = self._measure_channel_modulation(
                subtracted, period_x, period_y, n_harmonics=3,
            )
            bgr_f64 = subtracted.astype(np.float64)
            notch_result = self._apply_separable_notch(
                bgr_f64, period_x, period_y, modulation,
                n_harmonics=3,
                sigma=residual_sigma,
                attenuation=residual_attenuation,
                channel_adaptive=True,
            )
            processed = np.clip(notch_result, 0, 255).astype(np.uint8)

        # === Phase 9: TRUE original edge restoration ===
        edge_float = final_edge_mask.astype(np.float32) / 255.0
        edge_proximity = cv2.GaussianBlur(
            edge_float, (0, 0), sigmaX=edge_restore_sigma,
        )
        edge_proximity_3ch = np.stack([edge_proximity] * 3, axis=-1)

        result = (
            processed.astype(np.float32) * (1.0 - edge_proximity_3ch)
            + image_bgr.astype(np.float32) * edge_proximity_3ch
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "n_regions": n_regions,
            "version": "v5",
        }

        return result, intermediates

    def process_v6(
        self,
        image_bgr: np.ndarray,
        # Strong pass (non-edge regions)
        sigma_strong: float = 3.0,
        attenuation_strong: float = 0.05,
        n_harmonics_strong: int = 5,
        # Weak pass (edge regions)
        sigma_weak: float = 1.5,
        attenuation_weak: float = 0.25,
        n_harmonics_weak: int = 3,
        # Blend
        blend_sigma: float = 3.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Contour v6 pipeline: Dual-Strength Notch with Edge-Adaptive Blending.

        Key insight: v2's Phase 7-8 re-introduce grid patterns by blending with
        the original image near edges. v6 eliminates this by using two notch
        filter strengths: strong (aggressive grid removal for flat regions) and
        weak (gentle grid removal for edge regions), then blending between them
        based on edge proximity. The original image is NEVER used in the output.

        Pipeline:
        1-5. Same as process_v2 (period detection, edge detection, scoring,
             cleanup, segmentation)
        6A. Strong Notch Filter (aggressive grid removal for non-edge areas)
        6B. Weak Notch Filter (gentle grid removal for edge areas)
        7.  Edge-Proximity Adaptive Blending (strong <-> weak, NO original)

        Args:
            image_bgr: Input BGR image (uint8).
            sigma_strong: Notch width for strong pass.
            attenuation_strong: Notch center minimum for strong pass.
            n_harmonics_strong: Number of harmonics for strong pass.
            sigma_weak: Notch width for weak pass.
            attenuation_weak: Notch center minimum for weak pass.
            n_harmonics_weak: Number of harmonics for weak pass.
            blend_sigma: Gaussian sigma for edge proximity blending.

        Returns:
            Tuple of (processed BGR image uint8, dict of intermediate results).
        """
        h, w = image_bgr.shape[:2]

        # Handle edge cases
        if h < 10 or w < 10:
            return image_bgr.copy(), {
                "detected_periods": (0, 0),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v6",
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # === Phase 1: Grid Period Detection (reuse) ===
        if self.period_x <= 0 or self.period_y <= 0:
            period_x, period_y = self.detect_grid_periods(gray)
        else:
            period_x, period_y = self.period_x, self.period_y

        # Guard: if no valid periods detected, return original
        if period_x <= 0 or period_y <= 0:
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": np.zeros((h, w), dtype=np.uint8),
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v6",
            }

        # === Phase 2: Multi-Strategy Edge Detection (reuse) ===
        gradient_edges = self.compute_multiscale_gradient(gray)
        canny_edges = self.detect_edges_canny(gray)
        chrominance_edges = self.detect_chrominance_edges(image_bgr)
        persistence_edges = self.compute_multiscale_persistence(gray)

        raw_edge_map = self.combine_raw_edges(
            gradient_edges, canny_edges, chrominance_edges,
            guaranteed_edges=persistence_edges,
        )

        if not np.any(raw_edge_map):
            return image_bgr.copy(), {
                "detected_periods": (period_x, period_y),
                "raw_edge_map": raw_edge_map,
                "confidence_map": np.zeros((h, w), dtype=np.float32),
                "final_edge_mask": np.zeros((h, w), dtype=np.uint8),
                "label_map": np.zeros((h, w), dtype=np.int32),
                "n_regions": 0,
                "channel_modulation": {},
                "version": "v6",
            }

        # === Phase 3: Edge Confidence Scoring (reuse) ===
        periodicity_score = self.compute_periodicity_score(
            raw_edge_map, period_x, period_y, gray=gray
        )
        persistence_score = self.compute_persistence_score(gray, raw_edge_map)
        length_score = self.compute_edge_length_score(raw_edge_map)
        confidence_map = self.compute_edge_confidence(
            raw_edge_map, periodicity_score, persistence_score, length_score
        )

        # === Phase 4: Edge Mask Cleanup (reuse) ===
        final_edge_mask = self.cleanup_edge_mask(confidence_map > 0)

        # === Phase 5: Region Segmentation (reuse) ===
        label_map = self.segment_regions(final_edge_mask)
        n_regions = int(label_map.max())

        # === Phase 6A: Strong Notch Filter (aggressive grid removal) ===
        modulation = self._measure_channel_modulation(
            image_bgr, period_x, period_y,
            n_harmonics=max(n_harmonics_strong, n_harmonics_weak),
        )
        bgr_f64 = image_bgr.astype(np.float64)

        strong_result = self._apply_separable_notch(
            bgr_f64, period_x, period_y, modulation,
            n_harmonics=n_harmonics_strong,
            sigma=sigma_strong,
            attenuation=attenuation_strong,
            channel_adaptive=False,
        )

        # === Phase 6B: Weak Notch Filter (gentle grid removal) ===
        weak_result = self._apply_separable_notch(
            bgr_f64, period_x, period_y, modulation,
            n_harmonics=n_harmonics_weak,
            sigma=sigma_weak,
            attenuation=attenuation_weak,
            channel_adaptive=False,
        )

        # === Phase 7: Edge-Proximity Adaptive Blending ===
        # edge proximity: 1.0 at edges, 0.0 far from edges
        edge_float = final_edge_mask.astype(np.float32) / 255.0
        edge_proximity = cv2.GaussianBlur(
            edge_float, (0, 0), sigmaX=blend_sigma,
        )
        edge_proximity_3ch = np.stack([edge_proximity] * 3, axis=-1)

        # Blend: far from edge → strong (aggressive), near edge → weak (gentle)
        result = (strong_result * (1.0 - edge_proximity_3ch)
                  + weak_result * edge_proximity_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)

        intermediates = {
            "detected_periods": (period_x, period_y),
            "raw_edge_map": raw_edge_map,
            "confidence_map": confidence_map,
            "final_edge_mask": final_edge_mask,
            "label_map": label_map,
            "n_regions": n_regions,
            "channel_modulation": modulation,
            "version": "v6",
        }

        return result, intermediates
