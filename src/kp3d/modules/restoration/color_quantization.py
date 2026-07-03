"""Color Quantization based restoration for Korean traditional paintings.

v12: Paradigm shift from edge detection to color quantization.
Instead of detecting edges (which are confused by grid patterns),
we exploit the limited color palette (5-20 colors) of traditional paintings.

Pipeline:
    Stage 1: Rolling Guidance Filter (pre-processing)
    Stage 2: LAB K-means palette extraction (adaptive k)
    Stage 3: Guided pixel assignment
    Stage 4: Ink line detection and preservation
    Stage 5: Region refinement (small region merging, boundary smoothing)
    Stage 6: Final rendering (palette fill, ink restoration, boundary blending)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import label


class ColorQuantizationProcessor:
    """Color quantization based painting restoration.

    Core algorithm with no torch dependency - works with BGR uint8 numpy arrays.
    """

    def __init__(
        self,
        k_min: int = 5,
        k_max: int = 30,
        k_selection: str = "elbow",
        pre_filter: str = "rolling_guidance",
        rolling_iterations: int = 4,
        rolling_sigma_s: float = 3.0,
        rolling_sigma_r: float = 0.05,
        ink_l_threshold: float = 40.0,
        ink_chroma_threshold: float = 20.0,
        quantization_method: str = "guided",
        min_region_area: int = 50,
        blend_width: int = 2,
        flatten_strength: float = 0.6,
        adaptive_flatten: bool = True,
        variance_threshold: float = 80.0,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.k_selection = k_selection
        self.pre_filter = pre_filter
        self.rolling_iterations = rolling_iterations
        self.rolling_sigma_s = rolling_sigma_s
        self.rolling_sigma_r = rolling_sigma_r
        self.ink_l_threshold = ink_l_threshold
        self.ink_chroma_threshold = ink_chroma_threshold
        self.quantization_method = quantization_method
        self.min_region_area = min_region_area
        self.blend_width = blend_width
        self.flatten_strength = flatten_strength
        self.adaptive_flatten = adaptive_flatten
        self.variance_threshold = variance_threshold

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full color quantization restoration pipeline.

        Args:
            image_bgr: Input image in BGR uint8 format.

        Returns:
            Tuple of (result_bgr, intermediates_dict).
        """
        h, w = image_bgr.shape[:2]

        # Stage 1: Rolling Guidance Filter
        filtered = self._rolling_guidance_filter(image_bgr)

        # Stage 2: LAB K-means palette extraction
        lab_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(np.float32)
        palette_lab, labels, k = self._extract_palette(lab_filtered)

        # Stage 3: Guided pixel assignment
        labels_refined = self._assign_pixels(lab_filtered, palette_lab, labels)

        # Stage 4: Ink line detection
        lab_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ink_mask = self._detect_ink_lines(lab_original, image_bgr)

        # Stage 5: Region refinement
        labels_refined, n_regions = self._refine_regions(
            labels_refined, lab_filtered, palette_lab
        )

        # Stage 6: Final rendering
        result = self._render_final(
            image_bgr, filtered, lab_filtered, palette_lab,
            labels_refined, ink_mask
        )

        # Build intermediates for debugging
        intermediates = {
            'filtered': filtered,
            'palette_lab': palette_lab,
            'labels': labels_refined,
            'ink_mask': (ink_mask * 255).astype(np.uint8),
            'k': k,
            'n_regions': n_regions,
            'quantized_preview': self._render_quantized_preview(
                palette_lab, labels_refined, h, w
            ),
        }

        return result, intermediates

    # ------------------------------------------------------------------ #
    # Stage 1: Rolling Guidance Filter
    # ------------------------------------------------------------------ #
    def _rolling_guidance_filter(self, image_bgr: np.ndarray) -> np.ndarray:
        """Rolling Guidance Filter (ECCV 2014).

        Iteratively applies joint bilateral filter to remove small-scale
        structures (grid pattern, ~7-9px) while preserving large-scale content.

        Algorithm:
            J_0 = GaussianBlur(I, sigma_s)
            J_t = JointBilateral(I, J_{t-1}, sigma_s, sigma_r)
        """
        if self.pre_filter == "none":
            return image_bgr.copy()

        if self.pre_filter == "bilateral":
            result = image_bgr.copy()
            for _ in range(self.rolling_iterations):
                result = cv2.bilateralFilter(
                    result, d=-1,
                    sigmaColor=self.rolling_sigma_r * 255,
                    sigmaSpace=self.rolling_sigma_s,
                )
            return result

        # Rolling Guidance Filter
        img_f = image_bgr.astype(np.float64) / 255.0

        # J_0: Gaussian blur
        ksize = int(np.ceil(self.rolling_sigma_s * 3) * 2 + 1)
        j = cv2.GaussianBlur(img_f, (ksize, ksize), self.rolling_sigma_s)

        # Iterative joint bilateral filtering
        for _ in range(self.rolling_iterations):
            j = self._joint_bilateral_filter(
                img_f, j,
                sigma_s=self.rolling_sigma_s,
                sigma_r=self.rolling_sigma_r,
            )

        return (np.clip(j, 0, 1) * 255).astype(np.uint8)

    def _joint_bilateral_filter(
        self,
        source: np.ndarray,
        guide: np.ndarray,
        sigma_s: float,
        sigma_r: float,
    ) -> np.ndarray:
        """Joint bilateral filter: uses guide for range weights, source for values.

        Approximated using OpenCV's bilateral filter on the guide,
        then blending with the source based on range similarity.
        """
        h, w, c = source.shape
        radius = int(np.ceil(sigma_s * 2))
        ksize = 2 * radius + 1

        # Use cv2.bilateralFilter as approximation
        # Apply bilateral to guide to get structure-aware smoothing
        guide_u8 = (np.clip(guide, 0, 1) * 255).astype(np.uint8)
        source_u8 = (np.clip(source, 0, 1) * 255).astype(np.uint8)

        # Joint bilateral: smooth source using guide's edges
        # OpenCV doesn't have a direct joint bilateral, so we approximate:
        # 1. Compute guide's bilateral result for edge structure
        # 2. Use guide similarity to weight source pixels

        result = np.zeros_like(source)
        for ch in range(c):
            # For each channel, do a guided-style filtering
            guide_ch = guide[:, :, ch]
            source_ch = source[:, :, ch]

            # Pad arrays
            pad = radius
            guide_pad = np.pad(guide_ch, pad, mode='reflect')
            source_pad = np.pad(source_ch, pad, mode='reflect')

            # Vectorized sliding window approach for speed
            # Use separable approximation for large images
            if h * w > 500 * 500:
                # Fast approximation: use cv2 bilateral on source
                # with guide-derived sigma
                filtered = cv2.bilateralFilter(
                    source_u8[:, :, ch], d=ksize,
                    sigmaColor=sigma_r * 255,
                    sigmaSpace=sigma_s,
                ).astype(np.float64) / 255.0
                result[:, :, ch] = filtered
            else:
                # Accurate joint bilateral for smaller images
                filtered = self._joint_bilateral_channel(
                    source_pad, guide_pad, radius, sigma_s, sigma_r
                )
                result[:, :, ch] = filtered[pad:pad+h, pad:pad+w]

        return result

    def _joint_bilateral_channel(
        self,
        source_pad: np.ndarray,
        guide_pad: np.ndarray,
        radius: int,
        sigma_s: float,
        sigma_r: float,
    ) -> np.ndarray:
        """Joint bilateral filter for a single channel (padded arrays)."""
        h_pad, w_pad = source_pad.shape
        h = h_pad - 2 * radius
        w = w_pad - 2 * radius

        result = np.zeros((h_pad, w_pad), dtype=np.float64)
        weight_sum = np.zeros((h_pad, w_pad), dtype=np.float64)

        center_guide = guide_pad[radius:radius+h, radius:radius+w]

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Spatial weight
                spatial_w = np.exp(-(dx*dx + dy*dy) / (2 * sigma_s * sigma_s))

                # Range weight (based on guide similarity)
                neighbor_guide = guide_pad[
                    radius+dy:radius+dy+h,
                    radius+dx:radius+dx+w
                ]
                range_diff = center_guide - neighbor_guide
                range_w = np.exp(-(range_diff * range_diff) / (2 * sigma_r * sigma_r))

                # Combined weight
                w_total = spatial_w * range_w

                # Weighted source value
                neighbor_source = source_pad[
                    radius+dy:radius+dy+h,
                    radius+dx:radius+dx+w
                ]
                result[radius:radius+h, radius:radius+w] += w_total * neighbor_source
                weight_sum[radius:radius+h, radius:radius+w] += w_total

        # Normalize
        mask = weight_sum > 0
        result[mask] /= weight_sum[mask]
        return result

    # ------------------------------------------------------------------ #
    # Stage 2: LAB K-means Palette Extraction
    # ------------------------------------------------------------------ #
    def _extract_palette(
        self, lab_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Extract color palette using K-means in LAB space.

        Args:
            lab_image: LAB float32 image (H, W, 3).

        Returns:
            (palette_lab, labels, k) where:
                palette_lab: (k, 3) array of LAB palette colors
                labels: (H, W) integer label map
                k: number of colors selected
        """
        h, w = lab_image.shape[:2]

        # Subsample for speed (every 4th pixel)
        step = 4
        samples = lab_image[::step, ::step].reshape(-1, 3)

        # Remove NaN/Inf
        valid = np.isfinite(samples).all(axis=1)
        samples = samples[valid]

        if len(samples) < self.k_max:
            # Too few samples, use all pixels
            samples = lab_image.reshape(-1, 3)

        # Select k
        if self.k_selection == "fixed":
            k = self.k_min
        elif self.k_selection == "silhouette":
            k = self._select_k_silhouette(samples)
        else:  # "elbow"
            k = self._select_k_elbow(samples)

        # Final K-means with selected k on full resolution
        pixels = lab_image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels_flat, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        labels = labels_flat.reshape(h, w).astype(np.int32)
        palette_lab = centers.astype(np.float32)

        return palette_lab, labels, k

    def _select_k_elbow(self, samples: np.ndarray) -> int:
        """Select optimal k using elbow method on inertia curve."""
        samples_f32 = samples.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2.0)

        inertias = []
        k_range = range(self.k_min, self.k_max + 1)

        for k in k_range:
            compactness, _, _ = cv2.kmeans(
                samples_f32, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS
            )
            inertias.append(compactness)

        if len(inertias) < 3:
            return self.k_min

        # Find elbow: point of maximum curvature
        inertias = np.array(inertias)

        # Normalize to [0, 1] range
        if inertias[0] > inertias[-1] and inertias[0] > 0:
            inertias_norm = (inertias - inertias[-1]) / (inertias[0] - inertias[-1])
        else:
            return self.k_min

        # Distance from line connecting first and last point
        n = len(inertias_norm)
        p1 = np.array([0, inertias_norm[0]])
        p2 = np.array([n - 1, inertias_norm[-1]])

        distances = []
        for i, val in enumerate(inertias_norm):
            p = np.array([i, val])
            # Distance from point to line
            d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(d)

        best_idx = np.argmax(distances)
        return list(k_range)[best_idx]

    def _select_k_silhouette(self, samples: np.ndarray) -> int:
        """Select optimal k using silhouette score."""
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans as SKKMeans
        except ImportError:
            return self._select_k_elbow(samples)

        # Limit sample size for silhouette computation
        if len(samples) > 5000:
            idx = np.random.choice(len(samples), 5000, replace=False)
            samples_sub = samples[idx]
        else:
            samples_sub = samples

        best_k = self.k_min
        best_score = -1

        for k in range(self.k_min, min(self.k_max + 1, 16)):
            km = SKKMeans(n_clusters=k, n_init=3, max_iter=20, random_state=42)
            labels = km.fit_predict(samples_sub)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(samples_sub, labels, sample_size=2000)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    # ------------------------------------------------------------------ #
    # Stage 3: Guided Pixel Assignment
    # ------------------------------------------------------------------ #
    def _assign_pixels(
        self,
        lab_image: np.ndarray,
        palette_lab: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Assign pixels to palette colors with spatial coherence.

        Args:
            lab_image: LAB float32 image (H, W, 3).
            palette_lab: (k, 3) palette colors in LAB.
            labels: (H, W) initial label assignments.

        Returns:
            Refined labels (H, W) int32.
        """
        if self.quantization_method == "hard":
            return labels

        # Guided method: soft probability + guided filter
        h, w = lab_image.shape[:2]
        k = len(palette_lab)

        # Compute soft probabilities for each palette color
        # P(c|pixel) = exp(-d^2 / (2 * sigma^2)) where d = LAB distance
        sigma = 15.0  # Softness parameter
        pixels = lab_image.reshape(-1, 3)  # (N, 3)

        # Distance to each palette color
        probs = np.zeros((h * w, k), dtype=np.float32)
        for i in range(k):
            diff = pixels - palette_lab[i]
            dist_sq = np.sum(diff * diff, axis=1)
            probs[:, i] = np.exp(-dist_sq / (2 * sigma * sigma))

        # Normalize
        prob_sum = probs.sum(axis=1, keepdims=True)
        prob_sum[prob_sum == 0] = 1
        probs /= prob_sum

        # Reshape to (H, W, k)
        prob_maps = probs.reshape(h, w, k)

        # Apply guided filter to each probability channel for spatial coherence
        try:
            guided_available = hasattr(cv2, 'ximgproc')
            if not guided_available:
                # Try importing
                import cv2 as cv2_test
                guided_available = hasattr(cv2_test, 'ximgproc')
        except Exception:
            guided_available = False

        if guided_available:
            guide = cv2.cvtColor(
                (np.clip(lab_image[:, :, 0], 0, 100) * 2.55).astype(np.uint8),
                cv2.COLOR_GRAY2BGR
            )
            for i in range(k):
                prob_maps[:, :, i] = cv2.ximgproc.guidedFilter(
                    guide=guide,
                    src=prob_maps[:, :, i],
                    radius=8,
                    eps=0.02,
                )
        else:
            # Fallback: simple Gaussian smoothing for spatial coherence
            for i in range(k):
                prob_maps[:, :, i] = cv2.GaussianBlur(
                    prob_maps[:, :, i], (9, 9), 2.0
                )

        # Argmax for final assignment
        refined_labels = np.argmax(prob_maps, axis=2).astype(np.int32)
        return refined_labels

    # ------------------------------------------------------------------ #
    # Stage 4: Ink Line Detection
    # ------------------------------------------------------------------ #
    def _detect_ink_lines(
        self,
        lab_image: np.ndarray,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        """Detect ink lines (먹선) in the original image.

        Conditions for ink pixels:
            1. Dark: L* < threshold (default 40)
            2. Neutral: chroma < threshold (default 20)
            3. High gradient: gradient > 80th percentile

        Args:
            lab_image: LAB float32 image from original.
            image_bgr: Original BGR uint8 image.

        Returns:
            Binary mask (H, W) bool.
        """
        h, w = lab_image.shape[:2]

        # Condition 1: Dark pixels
        L = lab_image[:, :, 0]  # L channel (0-255 in OpenCV LAB)
        L_normalized = L * (100.0 / 255.0)  # Normalize to 0-100
        dark_mask = L_normalized < self.ink_l_threshold

        # Condition 2: Neutral (low chroma)
        a = lab_image[:, :, 1] - 128.0
        b = lab_image[:, :, 2] - 128.0
        chroma = np.sqrt(a * a + b * b)
        neutral_mask = chroma < self.ink_chroma_threshold

        # Condition 3: High gradient (edge-like)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        grad_threshold = np.percentile(gradient, 80)
        high_gradient_mask = gradient > grad_threshold

        # Combine all three conditions
        ink_mask = dark_mask & neutral_mask & high_gradient_mask

        # Morphological closing to connect nearby ink pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink_mask_u8 = ink_mask.astype(np.uint8)
        ink_mask_u8 = cv2.morphologyEx(ink_mask_u8, cv2.MORPH_CLOSE, kernel)

        return ink_mask_u8.astype(bool)

    # ------------------------------------------------------------------ #
    # Stage 5: Region Refinement
    # ------------------------------------------------------------------ #
    def _refine_regions(
        self,
        labels: np.ndarray,
        lab_image: np.ndarray,
        palette_lab: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Refine regions by merging small ones and smoothing boundaries.

        Args:
            labels: (H, W) label map.
            lab_image: LAB float32 image.
            palette_lab: (k, 3) palette colors.

        Returns:
            (refined_labels, n_regions).
        """
        h, w = labels.shape
        k = len(palette_lab)

        # For each palette color, find connected components
        new_labels = np.zeros_like(labels)
        region_id = 0

        for color_idx in range(k):
            color_mask = labels == color_idx
            if not np.any(color_mask):
                continue

            # Connected component analysis within this color
            labeled, n_comp = label(color_mask)
            for comp_id in range(1, n_comp + 1):
                comp_mask = labeled == comp_id
                area = np.sum(comp_mask)

                if area < self.min_region_area:
                    # Small region: find nearest large neighbor by LAB distance
                    # Dilate to find neighbors
                    dilated = cv2.dilate(
                        comp_mask.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                        iterations=2,
                    )
                    neighbor_mask = (dilated > 0) & ~comp_mask
                    if np.any(neighbor_mask):
                        # Find most common label among neighbors
                        neighbor_labels = new_labels[neighbor_mask]
                        neighbor_labels = neighbor_labels[neighbor_labels > 0]
                        if len(neighbor_labels) > 0:
                            # Assign to most frequent neighbor
                            values, counts = np.unique(
                                neighbor_labels, return_counts=True
                            )
                            best_label = values[np.argmax(counts)]
                            new_labels[comp_mask] = best_label
                            continue

                # Assign new region ID
                region_id += 1
                new_labels[comp_mask] = region_id

        # Handle any unassigned pixels (label 0)
        unassigned = new_labels == 0
        if np.any(unassigned):
            # Assign to nearest labeled pixel
            dist, idx = ndimage.distance_transform_edt(
                unassigned, return_indices=True
            )
            new_labels[unassigned] = new_labels[idx[0][unassigned], idx[1][unassigned]]

        # Boundary smoothing: median filter on labels
        # Convert labels to color, median blur, then re-label
        new_labels_smooth = cv2.medianBlur(
            new_labels.astype(np.uint16), 3
        ).astype(np.int32)

        n_regions = len(np.unique(new_labels_smooth))
        return new_labels_smooth, n_regions

    # ------------------------------------------------------------------ #
    # Stage 6: Final Rendering — Region-Boundary-Aware Smoothing
    # ------------------------------------------------------------------ #
    def _render_final(
        self,
        original_bgr: np.ndarray,
        filtered_bgr: np.ndarray,
        lab_filtered: np.ndarray,
        palette_lab: np.ndarray,
        labels: np.ndarray,
        ink_mask: np.ndarray,
    ) -> np.ndarray:
        """Render via region-boundary-aware iterative bilateral smoothing.

        Key insight: instead of replacing pixels with median colors (too flat)
        or blending with filtered (just blur), we use the quantized region map
        to define WHERE smoothing should NOT cross.

        Algorithm:
            1. Extract region boundaries from label map
            2. Iterative bilateral on filtered image, snapping boundary pixels
               back after each iteration → smooth interiors, sharp boundaries
            3. Gentle per-region flattening toward median (adaptive)
            4. Ink line restoration

        This produces: smooth regions (not flat) + sharp boundaries (not blurry).

        Args:
            original_bgr: Original BGR image.
            filtered_bgr: Rolling-guidance filtered BGR image.
            lab_filtered: LAB float32 of filtered image.
            palette_lab: (k, 3) palette in LAB.
            labels: (H, W) refined label map.
            ink_mask: (H, W) boolean ink mask.

        Returns:
            Result BGR uint8 image.
        """
        h, w = original_bgr.shape[:2]

        # Step 1: Extract region boundaries from quantized label map
        labels_f = labels.astype(np.float32)
        gx = cv2.Sobel(labels_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(labels_f, cv2.CV_32F, 0, 1, ksize=3)
        boundary_mask = (np.abs(gx) + np.abs(gy)) > 0

        # Dilate boundaries by 1px for stronger region separation
        boundary_dilated = cv2.dilate(
            boundary_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)

        # Step 2: Iterative bilateral with boundary snapping
        # Each iteration smooths region interiors more while boundaries stay sharp
        result = filtered_bgr.copy()
        boundary_ref = filtered_bgr.copy()

        for _ in range(3):
            smoothed = cv2.bilateralFilter(
                result, d=9, sigmaColor=40.0, sigmaSpace=40.0
            )
            # Snap boundary pixels back — prevents cross-region bleeding
            smoothed[boundary_dilated] = boundary_ref[boundary_dilated]
            result = smoothed

        # Step 3: Per-region gentle flattening toward median
        if self.flatten_strength > 0:
            result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            unique_labels = np.unique(labels)

            for lbl in unique_labels:
                mask = labels == lbl
                if not np.any(mask):
                    continue

                region_pixels = result_lab[mask]
                median_color = np.median(region_pixels, axis=0)

                if self.adaptive_flatten:
                    variance = np.mean(np.var(region_pixels, axis=0))
                    local_s = self.flatten_strength * np.clip(
                        1.0 - (variance / self.variance_threshold), 0.0, 1.0
                    )
                    local_s = max(local_s, 0.02)
                else:
                    local_s = self.flatten_strength

                blended = local_s * median_color + (1.0 - local_s) * region_pixels
                result_lab[mask] = blended

            result = cv2.cvtColor(
                np.clip(result_lab, 0, 255).astype(np.uint8),
                cv2.COLOR_LAB2BGR,
            )

        # Step 4: Ink line restoration from median-filtered source
        if np.any(ink_mask):
            ink_source = cv2.medianBlur(filtered_bgr, 3)
            result[ink_mask] = ink_source[ink_mask]

        return result

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def _render_quantized_preview(
        self,
        palette_lab: np.ndarray,
        labels: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Render a preview showing only palette colors (no blending)."""
        preview_lab = np.zeros((h, w, 3), dtype=np.float32)
        for i, color in enumerate(palette_lab):
            mask = labels == i
            if np.any(mask):
                preview_lab[mask] = color

        # Handle any labels >= len(palette) (from region refinement)
        # These keep the closest palette color
        for lbl in np.unique(labels):
            if lbl >= len(palette_lab):
                mask = labels == lbl
                if np.any(mask):
                    # Use the color from the quantized result
                    preview_lab[mask] = palette_lab[0]  # fallback

        return cv2.cvtColor(
            np.clip(preview_lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR,
        )
