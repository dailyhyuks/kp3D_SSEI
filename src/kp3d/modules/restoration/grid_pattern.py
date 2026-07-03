"""Grid pattern removal for traditional Korean paintings.

Removes fabric/canvas grid patterns while preserving painting contours.
Based on V13d algorithm: Cartoon-Texture Decomposition + FFT + Guided Filter.

Reference: IEEE 2016 - "Removal of Canvas Patterns in Digital Acquisitions of Paintings"
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig


class GridPatternRestorer(BaseRestoration):
    """격자 패턴 제거 복원기 (V13d)

    직물(캔버스/비단)의 격자 무늬를 제거하면서 회화의 윤곽선을 보존합니다.

    알고리즘:
    1. Cartoon-Texture 분해 (Iterative Bilateral Filter)
    2. FFT 방향성 필터 (0°/90° 격자 주파수 제거)
    3. Guided Filter (Edge-aware smoothing)
    4. Structure Tensor Edge Enhancement
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True

    @property
    def name(self) -> str:
        return "grid_pattern"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_rgb(self, tensor: Tensor) -> np.ndarray:
        """Tensor to RGB numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray) -> Tensor:
        """RGB numpy array to tensor."""
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    # =========================================================================
    # Core Decomposition Functions
    # =========================================================================

    def iterative_bilateral_decomposition(
        self,
        image: np.ndarray,
        iterations: int = 5,
        d: int = 9,
        sigma_color: float = 100,
        sigma_space: float = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iterative bilateral filter for cartoon-texture separation.

        Args:
            image: Input image (float32)
            iterations: Number of bilateral iterations
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space

        Returns:
            Tuple of (cartoon, texture) components
        """
        img = image.astype(np.float32)
        cartoon = img.copy()

        for _ in range(iterations):
            cartoon = cv2.bilateralFilter(cartoon, d, sigma_color, sigma_space)

        texture = img - cartoon
        return cartoon, texture

    def detect_grid_frequencies(
        self,
        image: np.ndarray,
        num_peaks: int = 5
    ) -> Tuple[list, list]:
        """Auto-detect grid frequencies from FFT spectrum.

        Returns:
            Tuple of (horizontal_freqs, vertical_freqs) - distance from DC
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Compute FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Analyze horizontal axis (for vertical grid lines)
        h_profile = magnitude[cy, :].copy()
        h_profile[max(0, cx-5):min(w, cx+6)] = 0  # Exclude DC

        # Analyze vertical axis (for horizontal grid lines)
        v_profile = magnitude[:, cx].copy()
        v_profile[max(0, cy-5):min(h, cy+6)] = 0  # Exclude DC

        def find_peaks(profile, center, threshold=0.3):
            max_val = profile.max()
            threshold_val = max_val * threshold
            peaks = []
            for i in range(3, len(profile) - 3):
                if profile[i] > threshold_val:
                    if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                        if profile[i] > profile[i-2] and profile[i] > profile[i+2]:
                            peaks.append(abs(i - center))
            return sorted(set(peaks), key=lambda x: -profile[center + x] if center + x < len(profile) else 0)[:num_peaks]

        h_freqs = find_peaks(h_profile, cx)
        v_freqs = find_peaks(v_profile, cy)

        return h_freqs, v_freqs

    def fft_directional_filter(
        self,
        image: np.ndarray,
        line_width: int = 5,
        dc_exclusion: int = 15,
        angles: list = None,
        target_freqs: list = None,
        freq_band_width: int = 3
    ) -> np.ndarray:
        """Apply directional FFT filter to remove grid frequencies.

        Args:
            image: Input image (single channel or 3-channel)
            line_width: Width of notch filter lines
            dc_exclusion: Radius around DC to preserve
            angles: List of angles to filter (default: [0, 90] for grid)
            target_freqs: List of specific frequencies to target (distance from DC)
            freq_band_width: Width of frequency bands to remove

        Returns:
            Filtered image
        """
        if angles is None:
            angles = [0, 90]

        if image.ndim == 3:
            channels = []
            for i in range(3):
                filtered = self.fft_directional_filter(
                    image[:, :, i], line_width, dc_exclusion, angles,
                    target_freqs, freq_band_width
                )
                channels.append(filtered)
            return np.stack(channels, axis=2)

        h, w = image.shape
        cy, cx = h // 2, w // 2

        # Create filter mask
        filter_mask = np.ones((h, w), dtype=np.float32)

        y, x = np.ogrid[:h, :w]
        y = y - cy
        x = x - cx

        # Create notch filters for each angle
        for angle in angles:
            rad = np.deg2rad(angle)
            dist_from_line = np.abs(x * np.sin(rad) - y * np.cos(rad))
            notch = (dist_from_line < line_width / 2).astype(np.float32)
            filter_mask = filter_mask * (1 - notch)

        # Add targeted frequency bands if specified
        if target_freqs:
            dist_from_center = np.sqrt(x**2 + y**2)
            for freq in target_freqs:
                if freq > dc_exclusion:
                    # Ring filter at this frequency
                    ring = np.abs(dist_from_center - freq) < freq_band_width
                    filter_mask = filter_mask * (1 - ring.astype(np.float32) * 0.7)

        # Preserve DC component
        y_grid, x_grid = np.ogrid[:h, :w]
        dc_dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        filter_mask[dc_dist < dc_exclusion] = 1.0

        # Smooth filter edges
        notch_mask = 1 - filter_mask
        notch_mask = cv2.GaussianBlur(notch_mask, (5, 5), 1.0)
        filter_mask = 1 - notch_mask

        # Apply FFT filter
        f = np.fft.fft2(image.astype(np.float32))
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * filter_mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered = np.fft.ifft2(f_ishift)

        return np.real(filtered).astype(np.float32)

    def fft_aggressive_filter(
        self,
        image: np.ndarray,
        passes: int = 2
    ) -> np.ndarray:
        """Aggressive FFT filtering with auto-detected frequencies.

        Multi-pass processing with progressively stronger filtering.
        """
        result = image.astype(np.float32)

        # Auto-detect grid frequencies
        h_freqs, v_freqs = self.detect_grid_frequencies(image)
        all_freqs = list(set(h_freqs + v_freqs))

        for p in range(passes):
            # Increase filter strength with each pass
            line_width = 9 + p * 4  # 9, 13, 17...
            dc_exclusion = max(5, 10 - p * 2)  # 10, 8, 6...

            # Apply directional filter
            result = self.fft_directional_filter(
                result,
                line_width=line_width,
                dc_exclusion=dc_exclusion,
                target_freqs=all_freqs[:5],
                freq_band_width=3 + p
            )

        return result

    # =========================================================================
    # Grid Detection
    # =========================================================================

    def compute_local_variance_map(
        self,
        image: np.ndarray,
        window_size: int = 7
    ) -> np.ndarray:
        """Compute local variance map for grid detection.

        High variance in small windows often indicates grid pattern.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_mean_sq = cv2.filter2D(gray ** 2, -1, kernel)
        local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)

        local_var_norm = local_var / (local_var.max() + 1e-6)
        return local_var_norm.astype(np.float32)

    def compute_grid_likelihood_map(
        self,
        image: np.ndarray,
        small_window: int = 5,
        large_window: int = 15
    ) -> np.ndarray:
        """Detect grid regions by comparing variance at different scales.

        Grid patterns have high variance at small scale but lower at large scale.
        True edges have high variance at both scales.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        var_small = self.compute_local_variance_map(gray, small_window)
        var_large = self.compute_local_variance_map(gray, large_window)

        grid_likelihood = var_small * (1 - var_large * 0.5)
        grid_likelihood = np.clip(grid_likelihood, 0, 1)
        grid_likelihood = cv2.GaussianBlur(grid_likelihood, (5, 5), 1.0)

        return grid_likelihood.astype(np.float32)

    # =========================================================================
    # Guided Filter
    # =========================================================================

    def guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int = 8,
        eps: float = 0.01
    ) -> np.ndarray:
        """Edge-preserving smoothing using guided filter.

        Args:
            guide: Guide image for edge preservation
            src: Source image to filter
            radius: Filter radius
            eps: Regularization parameter

        Returns:
            Filtered image
        """
        if guide.ndim == 3:
            channels = []
            for i in range(3):
                filtered = self.guided_filter(guide[:, :, i], src[:, :, i], radius, eps)
                channels.append(filtered)
            return np.stack(channels, axis=2)

        guide = guide.astype(np.float32) / 255.0
        src = src.astype(np.float32) / 255.0

        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (2*r+1, 2*r+1))

        mean_guide = box_filter(guide, radius)
        mean_src = box_filter(src, radius)
        corr_guide = box_filter(guide * guide, radius)
        corr_guide_src = box_filter(guide * src, radius)

        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src

        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        output = mean_a * guide + mean_b
        return (output * 255).astype(np.float32)

    def adaptive_guided_filter(
        self,
        image: np.ndarray,
        grid_map: np.ndarray,
        radius_grid: int = 10,
        radius_edge: int = 3,
        eps: float = 0.01
    ) -> np.ndarray:
        """Apply stronger guided filter in grid regions, lighter in edge regions."""
        filtered_strong = self.guided_filter(image, image, radius=radius_grid, eps=eps)
        filtered_light = self.guided_filter(image, image, radius=radius_edge, eps=eps*10)

        if image.ndim == 3:
            grid_map_3d = grid_map[:, :, np.newaxis]
        else:
            grid_map_3d = grid_map

        result = grid_map_3d * filtered_strong + (1 - grid_map_3d) * filtered_light
        return result.astype(np.float32)

    # =========================================================================
    # Structure Tensor Edge Enhancement
    # =========================================================================

    def compute_structure_tensor(
        self,
        image: np.ndarray,
        rho: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute structure tensor for edge direction analysis.

        Returns:
            Tuple of (coherence, orientation)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        Jxx = gx * gx
        Jxy = gx * gy
        Jyy = gy * gy

        ksize = int(rho * 6) | 1
        Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), rho)
        Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), rho)
        Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), rho)

        tmp = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
        lambda1 = 0.5 * (Jxx + Jyy + tmp)
        lambda2 = 0.5 * (Jxx + Jyy - tmp)

        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
        coherence = np.clip(coherence, 0, 1)

        orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)

        return coherence.astype(np.float32), orientation.astype(np.float32)

    def structure_tensor_edge_enhance(
        self,
        image: np.ndarray,
        coherence: np.ndarray,
        strength: float = 0.4,
        threshold: float = 0.3
    ) -> np.ndarray:
        """Enhance edges based on structure tensor coherence."""
        edge_mask = np.clip((coherence - threshold) / (1 - threshold + 1e-6), 0, 1)
        edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0.5)

        blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
        detail = image.astype(np.float32) - blurred.astype(np.float32)

        if image.ndim == 3:
            edge_mask_3d = edge_mask[:, :, np.newaxis]
        else:
            edge_mask_3d = edge_mask

        enhanced = image.astype(np.float32) + strength * edge_mask_3d * detail
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    # =========================================================================
    # Contour Preservation
    # =========================================================================

    def compute_multiscale_contour_map(self, image: np.ndarray) -> np.ndarray:
        """Compute multi-scale contour map for edge detection."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        gx3 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy3 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag3 = np.sqrt(gx3**2 + gy3**2)

        gray_blur5 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        gx5 = cv2.Sobel(gray_blur5, cv2.CV_64F, 1, 0, ksize=5)
        gy5 = cv2.Sobel(gray_blur5, cv2.CV_64F, 0, 1, ksize=5)
        mag5 = np.sqrt(gx5**2 + gy5**2)

        gray_blur7 = cv2.GaussianBlur(gray, (5, 5), 1.5)
        gx7 = cv2.Sobel(gray_blur7, cv2.CV_64F, 1, 0, ksize=7)
        gy7 = cv2.Sobel(gray_blur7, cv2.CV_64F, 0, 1, ksize=7)
        mag7 = np.sqrt(gx7**2 + gy7**2)

        combined = np.maximum(mag3, np.maximum(mag5 * 1.2, mag7 * 1.5))
        combined_norm = combined / (combined.max() + 1e-6)

        return combined_norm.astype(np.float32)

    def contour_aware_blend(
        self,
        original: np.ndarray,
        filtered: np.ndarray,
        contour_map: np.ndarray,
        preservation_strength: float = 0.5,
        threshold: float = 0.12
    ) -> np.ndarray:
        """Blend with contour-aware preservation."""
        preservation = np.clip((contour_map - threshold * 0.5) / (threshold + 1e-6), 0, 1)
        preservation = preservation * preservation_strength
        preservation = cv2.GaussianBlur(preservation, (5, 5), 1.0)

        if original.ndim == 3:
            preservation_3d = preservation[:, :, np.newaxis]
        else:
            preservation_3d = preservation

        result = preservation_3d * original.astype(np.float32) + \
                 (1 - preservation_3d) * filtered.astype(np.float32)

        return result.astype(np.uint8)

    # =========================================================================
    # Main Restoration Pipeline
    # =========================================================================

    def restore_grid_pattern(
        self,
        image_bgr: np.ndarray,
        method: str = "guided_only"
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Main grid pattern removal pipeline.

        Args:
            image_bgr: Input BGR image
            method: Restoration method
                - "guided_only": Guided filter only (best for most cases, ~22%)
                - "triple_medium": All features, medium strength (~20%)
                - "triple_strong": All features, strong strength (~18%)
                - "aggressive": Strong FFT + guided, less contour preservation (~33%)
                - "ultra": Maximum grid removal, may affect contours (~37%)
                - "extreme": Multi-pass with auto frequency detection (~45%)

        Returns:
            Tuple of (restored_image, intermediates)
        """
        intermediates = {}
        config = self.config

        # Get method-specific parameters
        use_auto_freq = False
        multi_pass = 1

        if method == "guided_only":
            grid_detection = False
            use_guided_filter = True
            structure_enhance = False
            edge_preservation = 0.6
            structure_strength = 0.0
            guided_radius = config.grid_guided_radius
            fft_line_width = config.grid_fft_line_width
            fft_iterations = 1
            final_fft = False
            texture_weight = 0.6
        elif method == "triple_strong":
            grid_detection = True
            use_guided_filter = True
            structure_enhance = True
            edge_preservation = 0.7
            structure_strength = 0.5
            guided_radius = 10
            fft_line_width = config.grid_fft_line_width
            fft_iterations = 1
            final_fft = False
            texture_weight = 0.6
        elif method == "aggressive":
            # Stronger FFT, less contour preservation
            grid_detection = False
            use_guided_filter = True
            structure_enhance = False
            edge_preservation = 0.3  # Less contour preservation
            structure_strength = 0.0
            guided_radius = 12  # Larger radius
            fft_line_width = 11  # Wider FFT notch
            fft_iterations = 2  # Double FFT
            final_fft = True  # Apply FFT to final result
            texture_weight = 0.4  # Less texture in recombination
            use_auto_freq = True
        elif method == "ultra":
            # Maximum grid removal
            grid_detection = False
            use_guided_filter = True
            structure_enhance = False
            edge_preservation = 0.2  # Minimal contour preservation
            structure_strength = 0.0
            guided_radius = 15  # Large radius
            fft_line_width = 15  # Very wide FFT notch
            fft_iterations = 3  # Triple FFT
            final_fft = True
            texture_weight = 0.3
            use_auto_freq = True
        elif method == "extreme":
            # Extreme grid removal with multi-pass processing
            grid_detection = False
            use_guided_filter = True
            structure_enhance = False
            edge_preservation = 0.1  # Almost no contour preservation
            structure_strength = 0.0
            guided_radius = 18  # Very large radius
            fft_line_width = 19  # Maximum FFT notch
            fft_iterations = 4  # Quad FFT
            final_fft = True
            texture_weight = 0.2
            use_auto_freq = True
            multi_pass = 2  # Run entire pipeline twice
        else:  # triple_medium (default)
            grid_detection = True
            use_guided_filter = True
            structure_enhance = True
            edge_preservation = 0.6
            structure_strength = 0.4
            guided_radius = config.grid_guided_radius
            fft_line_width = config.grid_fft_line_width
            fft_iterations = 1
            final_fft = False
            texture_weight = 0.6

        # Auto-detect grid frequencies if enabled
        detected_freqs = None
        if use_auto_freq:
            h_freqs, v_freqs = self.detect_grid_frequencies(image_bgr)
            detected_freqs = list(set(h_freqs + v_freqs))[:7]

        image = image_bgr.astype(np.float32)

        # Step 1: Compute contour map
        contour_map = self.compute_multiscale_contour_map(image)
        if config.store_intermediates:
            intermediates['contour_map'] = (contour_map * 255).astype(np.uint8)

        # Step 2: Grid likelihood detection
        if grid_detection:
            grid_map = self.compute_grid_likelihood_map(image)
            if config.store_intermediates:
                intermediates['grid_map'] = (grid_map * 255).astype(np.uint8)
        else:
            grid_map = np.ones_like(contour_map) * 0.5

        # Multi-pass processing for extreme mode
        current_image = image
        for pass_idx in range(multi_pass):
            # Step 3: Cartoon-Texture decomposition
            bilateral_iters = config.grid_bilateral_iterations
            if method in ("aggressive", "ultra", "extreme"):
                bilateral_iters = max(bilateral_iters, 7 + pass_idx * 2)

            cartoon, texture = self.iterative_bilateral_decomposition(
                current_image,
                iterations=bilateral_iters,
                d=9,
                sigma_color=100 + pass_idx * 20,
                sigma_space=100 + pass_idx * 20
            )
            if config.store_intermediates and pass_idx == 0:
                intermediates['cartoon'] = np.clip(cartoon, 0, 255).astype(np.uint8)

            # Step 4: FFT on texture (iterative for aggressive modes)
            texture_filtered = texture
            dc_excl = 8 if method in ("aggressive", "ultra", "extreme") else 10
            dc_excl = max(5, dc_excl - pass_idx * 2)

            for _ in range(fft_iterations):
                texture_filtered = self.fft_directional_filter(
                    texture_filtered,
                    line_width=fft_line_width + pass_idx * 2,
                    dc_exclusion=dc_excl,
                    target_freqs=detected_freqs,
                    freq_band_width=3 + pass_idx
                )

            # Step 5: FFT on cartoon (stronger for aggressive modes)
            cartoon_fft_width = 5 if method in ("aggressive", "ultra") else 3
            if method == "extreme":
                cartoon_fft_width = 7 + pass_idx * 2
            cartoon_filtered = self.fft_directional_filter(
                cartoon,
                line_width=cartoon_fft_width,
                dc_exclusion=15 if method in ("aggressive", "ultra", "extreme") else 20,
                target_freqs=detected_freqs if method == "extreme" else None
            )

            # Step 6: Recombine
            tw = texture_weight - pass_idx * 0.1
            combined = cartoon_filtered + texture_filtered * tw
            combined = np.clip(combined, 0, 255).astype(np.uint8)

            # Step 6.5: Additional FFT on combined result for aggressive modes
            if final_fft:
                combined_float = combined.astype(np.float32)
                combined_float = self.fft_directional_filter(
                    combined_float,
                    line_width=fft_line_width - 2 + pass_idx * 2,
                    dc_exclusion=max(6, 12 - pass_idx * 2),
                    target_freqs=detected_freqs
                )
                combined = np.clip(combined_float, 0, 255).astype(np.uint8)

            # Step 7: Adaptive Guided Filter
            if use_guided_filter:
                gr = guided_radius + pass_idx * 3
                guided_result = self.adaptive_guided_filter(
                    combined, grid_map,
                    radius_grid=gr,
                    radius_edge=3,
                    eps=config.grid_guided_eps
                )
                guided_result = np.clip(guided_result, 0, 255).astype(np.uint8)
            else:
                guided_result = combined

            # Step 8: Contour-aware blending (less preservation in later passes)
            ep = edge_preservation - pass_idx * 0.05
            blended = self.contour_aware_blend(
                current_image.astype(np.uint8), guided_result, contour_map,
                preservation_strength=max(0.05, ep),
                threshold=config.grid_contour_threshold
            )

            # Step 9: Bilateral smoothing (stronger for aggressive modes)
            if method in ("aggressive", "ultra", "extreme"):
                d = 7 + pass_idx * 2
                sigma = 35 + pass_idx * 10
                smoothed = cv2.bilateralFilter(blended, d, sigma, sigma)
            else:
                smoothed = cv2.bilateralFilter(blended, 5, 25, 25)

            current_image = smoothed.astype(np.float32)

        # Step 10: Structure tensor edge enhancement
        if structure_enhance:
            coherence, _ = self.compute_structure_tensor(current_image.astype(np.uint8))
            if config.store_intermediates:
                intermediates['coherence'] = (coherence * 255).astype(np.uint8)

            result = self.structure_tensor_edge_enhance(
                current_image.astype(np.uint8), coherence,
                strength=structure_strength,
                threshold=config.grid_structure_threshold
            )
        else:
            result = current_image.astype(np.uint8)

        # Extra smoothing for extreme mode
        if method == "extreme":
            result = cv2.bilateralFilter(result, 5, 30, 30)

        if config.store_intermediates:
            intermediates['result'] = result

        return result, intermediates

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Grid pattern removal forward pass.

        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            **kwargs: Additional arguments
                - method: "guided_only", "triple_medium", "triple_strong"

        Returns:
            ModuleOutput with restored image
        """
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy BGR
        img_np = self._tensor_to_numpy_rgb(image[0])
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Get method from kwargs or config
        method = kwargs.get('method', self.config.grid_method)

        # Run restoration
        result_bgr, intermediates = self.restore_grid_pattern(img_bgr, method=method)

        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - start

        # Compute texture reduction metric
        original_std = np.std(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32))
        restored_std = np.std(cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
        texture_reduction = (1 - restored_std / original_std) * 100

        # Build intermediates for output
        intermediate_tensors = {}
        if self.config.store_intermediates:
            for key, arr in intermediates.items():
                if arr is not None:
                    if arr.ndim == 2:
                        arr = np.stack([arr]*3, axis=-1)
                    intermediate_tensors[key] = self._numpy_to_tensor(arr)

        result_tensor = self._numpy_to_tensor(result_rgb).unsqueeze(0)

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediate_tensors,
            metadata={
                'method': f'grid_pattern_{method}',
                'processing_time': elapsed,
                'texture_reduction_percent': texture_reduction,
                'grid_bilateral_iterations': self.config.grid_bilateral_iterations,
                'grid_fft_line_width': self.config.grid_fft_line_width,
                'grid_guided_radius': self.config.grid_guided_radius,
            }
        )


__all__ = ["GridPatternRestorer"]
