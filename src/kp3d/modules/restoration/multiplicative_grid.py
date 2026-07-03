"""Multiplicative grid deconvolution for Korean painting restoration (v14.1).

This module implements a multiplicative deconvolution approach for removing
periodic grid patterns from digitized Korean paintings. The grid pattern is
modeled as a multiplicative degradation: I_observed = I_clean * G_pattern.

Recovery is performed via division: I_clean = I_observed / G_normalized.

Key features:
- FFT-based primary period detection with autocorrelation cross-validation
- Robust template estimation via median tiling
- Clamped division for numerical stability
- Ink-line edge protection to preserve brush strokes
- Object-Edge-Only Enhancement (OEE) via scale separation + chrominance immunity
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


class MultiplicativeGridRemover:
    """Grid removal via multiplicative deconvolution.

    Models grid degradation as I_observed = I_clean * G where G is a periodic
    grid pattern. Recovers clean image via I_clean = I_observed / G_normalized.
    """

    def detect_period(self, image_gray: np.ndarray) -> Tuple[int, int]:
        """Detect grid periods via FFT with autocorrelation cross-validation.

        Uses FFT-based detection as primary method (more robust when content
        autocorrelation dominates), with autocorrelation for validation.

        Args:
            image_gray: Grayscale image as 2D numpy array.

        Returns:
            Tuple of (period_x, period_y) representing horizontal and vertical
            grid periods in pixels.
        """
        # Primary: FFT-based detection
        fft_px, fft_py = self.detect_period_fft(image_gray)

        # Secondary: Autocorrelation (for validation)
        acf_px, acf_py = self._detect_period_acf(image_gray)

        # Cross-validate: prefer FFT, but use ACF if FFT fails
        period_x = fft_px if 4 <= fft_px <= 30 else acf_px
        period_y = fft_py if 4 <= fft_py <= 30 else acf_py

        return period_x, period_y

    def detect_period_fft(self, image_gray: np.ndarray) -> Tuple[int, int]:
        """FFT-based grid period detection with harmonics validation.

        Grid patterns produce harmonics at f, 2f, 3f, etc.
        This distinguishes grid from content frequencies which lack harmonics.
        """
        gray = image_gray.astype(np.float64)
        h, w = gray.shape

        # Row-wise FFT -> X period
        row_fft = np.fft.fft(gray, axis=1)
        row_power = np.mean(np.abs(row_fft)**2, axis=0)
        period_x = self._find_period_with_harmonics(row_power, w, min_period=5, max_period=30)

        # Col-wise FFT -> Y period
        col_fft = np.fft.fft(gray, axis=0)
        col_power = np.mean(np.abs(col_fft)**2, axis=1)
        period_y = self._find_period_with_harmonics(col_power, h, min_period=5, max_period=30)

        return period_x, period_y

    @staticmethod
    def _find_period_with_harmonics(
        power: np.ndarray,
        size: int,
        min_period: int = 4,
        max_period: int = 30,
        n_harmonics: int = 3,
    ) -> int:
        """Find grid period using harmonic series validation.

        For each candidate period p, checks whether harmonics at
        size/p, 2*size/p, 3*size/p all show elevated power.
        Scores each candidate by sum of harmonic prominences.

        Args:
            power: FFT power spectrum.
            size: Signal length (image width or height).
            min_period: Minimum candidate period.
            max_period: Maximum candidate period.
            n_harmonics: Number of harmonics to check (including fundamental).

        Returns:
            Best period in pixels.
        """
        from scipy.ndimage import median_filter

        half = size // 2
        # Compute baseline using median filter
        spectrum = power[1:half].copy()  # skip DC
        kernel = max(5, len(spectrum) // 5)
        if kernel % 2 == 0:
            kernel += 1
        baseline = median_filter(spectrum, size=kernel)
        baseline = np.maximum(baseline, 1e-10)

        # Prominence of each frequency bin
        prominence = spectrum / baseline

        best_score = 0
        best_period = 8  # fallback

        for period in range(min_period, max_period + 1):
            fundamental_k = size / period  # fractional frequency index

            score = 0.0
            valid_harmonics = 0

            for h in range(1, n_harmonics + 1):
                k = fundamental_k * h
                k_int = int(round(k)) - 1  # -1 because spectrum starts at index 1 (we skipped DC)

                if k_int < 0 or k_int >= len(prominence):
                    continue

                # Check neighbors for slight frequency mismatch
                k_lo = max(0, k_int - 1)
                k_hi = min(len(prominence) - 1, k_int + 1)
                local_prom = max(prominence[k_lo], prominence[k_int], prominence[k_hi])

                if local_prom > 1.5:  # above baseline
                    score += local_prom
                    valid_harmonics += 1

            # Require at least 2 harmonics to confirm it's a grid (not content)
            if valid_harmonics >= 2 and score > best_score:
                best_score = score
                best_period = period

        # If no period found with harmonics, fall back to simple peak finding
        if best_score == 0:
            # Simple: find strongest prominence in grid period range
            for period in range(min_period, max_period + 1):
                k = int(round(size / period)) - 1
                if 0 <= k < len(prominence):
                    if prominence[k] > best_score:
                        best_score = prominence[k]
                        best_period = period

        return best_period

    def _detect_period_acf(self, image_gray: np.ndarray) -> Tuple[int, int]:
        """Autocorrelation-based grid period detection.

        Computes normalized autocorrelation along rows and columns to find
        the dominant periodic structure in the grid pattern.

        Args:
            image_gray: Grayscale image as 2D numpy array.

        Returns:
            Tuple of (period_x, period_y) representing horizontal and vertical
            grid periods in pixels.
        """
        gray = image_gray.astype(np.float64)
        gray -= gray.mean()
        h, w = gray.shape

        # Horizontal period: average autocorrelation along rows
        max_lag = min(w // 2, 50)
        acf_x = np.zeros(max_lag)
        for lag in range(max_lag):
            acf_x[lag] = np.mean(gray[:, :w - lag] * gray[:, lag:])
        if acf_x[0] > 0:
            acf_x /= acf_x[0]

        # Vertical period: average autocorrelation along columns
        max_lag_y = min(h // 2, 50)
        acf_y = np.zeros(max_lag_y)
        for lag in range(max_lag_y):
            acf_y[lag] = np.mean(gray[:h - lag, :] * gray[lag:, :])
        if acf_y[0] > 0:
            acf_y /= acf_y[0]

        period_x = self._find_first_peak(acf_x, min_lag=3, fallback=9)
        period_y = self._find_first_peak(acf_y, min_lag=3, fallback=7)

        return period_x, period_y

    @staticmethod
    def _find_first_peak(acf: np.ndarray, min_lag: int = 3, fallback: int = 9) -> int:
        """Find the first significant peak in an autocorrelation function.

        Args:
            acf: Normalized autocorrelation array.
            min_lag: Minimum lag to consider (avoids trivial peak at 0).
            fallback: Default period if no peak found.

        Returns:
            Index of first peak, or fallback value.
        """
        for i in range(min_lag, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
                return i
        return fallback

    def estimate_template(
        self,
        image: np.ndarray,
        period_x: int,
        period_y: int,
        method: str = "median"
    ) -> np.ndarray:
        """Tile averaging to extract grid pattern template.

        Collects all period_x x period_y tiles from the image and computes
        the median (or mean) to extract the underlying grid pattern. The
        median is robust to outliers such as ink lines and painted details.

        The template is normalized so that mean=1 (multiplicative unit),
        making it suitable for deconvolution via division.

        Args:
            image: Input image (grayscale or BGR color).
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.
            method: Aggregation method, either "median" (default) or "mean".

        Returns:
            Normalized template array with same channel count as input.
        """
        H, W = image.shape[:2]
        is_color = len(image.shape) == 3

        # Guard: image must be at least one period in both dimensions
        if H < period_y or W < period_x:
            if is_color:
                return np.ones((period_y, period_x, image.shape[2]), dtype=np.float64)
            return np.ones((period_y, period_x), dtype=np.float64)

        tiles = []
        for y in range(0, H - period_y + 1, period_y):
            for x in range(0, W - period_x + 1, period_x):
                tile = image[y:y + period_y, x:x + period_x]
                tiles.append(tile.astype(np.float64))

        tiles = np.array(tiles)

        if method == "median":
            template = np.median(tiles, axis=0)
        else:
            template = np.mean(tiles, axis=0)

        # Normalize to mean=1 per channel
        if is_color:
            for c in range(template.shape[2]):
                ch_mean = template[:, :, c].mean()
                if ch_mean > 1e-8:
                    template[:, :, c] /= ch_mean
        else:
            t_mean = template.mean()
            if t_mean > 1e-8:
                template /= t_mean

        return template

    def estimate_grid_modulation(
        self,
        image: np.ndarray,
        period_x: int,
        period_y: int,
    ) -> np.ndarray:
        """Estimate per-pixel grid modulation via Gaussian low-pass division.

        Instead of tiling a small template, this method:
        1. Blurs the image with sigma ~ max(period)/2 to remove grid
        2. Divides original by blurred to isolate the multiplicative grid
        3. Normalizes the ratio to mean=1

        This produces a full-resolution modulation map that adapts to
        local content, avoiding the content-leakage problem of tile averaging.

        Args:
            image: Input image (BGR or grayscale).
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.

        Returns:
            Full-size modulation map (same shape as image), mean=1.
        """
        sigma = max(period_x, period_y) / 2.0
        sigma = max(sigma, 2.0)  # minimum sigma

        img_f = image.astype(np.float64)
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
        blurred = np.maximum(blurred, 1.0)  # avoid div by zero

        ratio = img_f / blurred

        # Normalize to mean=1 per channel
        if len(ratio.shape) == 3:
            for c in range(ratio.shape[2]):
                ch_mean = ratio[:, :, c].mean()
                if ch_mean > 1e-8:
                    ratio[:, :, c] /= ch_mean
        else:
            r_mean = ratio.mean()
            if r_mean > 1e-8:
                ratio /= r_mean

        return ratio

    def _guided_filter(
        self,
        guide: np.ndarray,
        src: np.ndarray,
        radius: int,
        eps: float,
    ) -> np.ndarray:
        """Edge-preserving guided filter (O(1) box-filter implementation).

        Smooths src while preserving edges detected in guide.
        Equivalent to cv2.ximgproc.guidedFilter but without opencv-contrib.

        Args:
            guide: Guide image (grayscale float64).
            src: Source image to filter (float64, same shape as guide or per-channel).
            radius: Filter radius (window = 2*radius+1).
            eps: Regularization (larger = smoother, ~0.01-0.04 typical).

        Returns:
            Filtered image (same shape as src).
        """
        def box_filter(img, r):
            """O(1) box filter via integral image."""
            return cv2.blur(img, (2*r+1, 2*r+1))

        guide_f = guide.astype(np.float64)
        src_f = src.astype(np.float64)

        mean_g = box_filter(guide_f, radius)
        mean_s = box_filter(src_f, radius)
        corr_gs = box_filter(guide_f * src_f, radius)
        corr_gg = box_filter(guide_f * guide_f, radius)

        var_g = corr_gg - mean_g * mean_g
        cov_gs = corr_gs - mean_g * mean_s

        a = cov_gs / (var_g + eps)
        b = mean_s - a * mean_g

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        return mean_a * guide_f + mean_b

    def estimate_grid_modulation_guided(
        self,
        image: np.ndarray,
        period_x: int,
        period_y: int,
        guided_radius: int = 0,
        guided_eps: float = 0.02,
    ) -> np.ndarray:
        """Edge-preserving grid modulation estimation via guided filter division.

        Like estimate_grid_modulation but uses a guided filter instead of
        Gaussian blur for the low-pass. The guided filter preserves edges
        in the denominator, so division doesn't damage content edges.

        Args:
            image: Input image (BGR or grayscale).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            guided_radius: Filter radius. 0 = auto (based on period).
            guided_eps: Regularization parameter. Higher = smoother.

        Returns:
            Full-size modulation map (same shape as image), mean=1.
        """
        if guided_radius <= 0:
            guided_radius = max(period_x, period_y)

        img_f = image.astype(np.float64)

        if len(img_f.shape) == 3:
            # Use grayscale as guide for color image
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

            blurred = np.zeros_like(img_f)
            for c in range(img_f.shape[2]):
                blurred[:, :, c] = self._guided_filter(
                    guide, img_f[:, :, c], guided_radius,
                    eps=guided_eps * (255.0**2)
                )
        else:
            guide = img_f / 255.0
            blurred = self._guided_filter(guide, img_f, guided_radius, eps=guided_eps * (255.0**2))

        blurred = np.maximum(blurred, 1.0)
        ratio = img_f / blurred

        # Normalize to mean=1 per channel
        if len(ratio.shape) == 3:
            for c in range(ratio.shape[2]):
                ch_mean = ratio[:, :, c].mean()
                if ch_mean > 1e-8:
                    ratio[:, :, c] /= ch_mean
        else:
            r_mean = ratio.mean()
            if r_mean > 1e-8:
                ratio /= r_mean

        return ratio

    def estimate_grid_notch(
        self,
        image: np.ndarray,
        period_x: int,
        period_y: int,
        notch_width: float = 1.5,
        n_harmonics: int = 4,
        base_attenuation: float = 0.0,
    ) -> np.ndarray:
        """Remove grid via 2D FFT notch filter at grid harmonic frequencies.

        Unlike Gaussian division which removes ALL high frequencies,
        this method only suppresses the specific periodic frequencies
        corresponding to the grid pattern and its harmonics.

        This preserves texture and fine detail that happen to fall
        at non-grid frequencies.

        Args:
            image: Input image (BGR or grayscale).
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.
            notch_width: Gaussian notch width in frequency bins (sigma).
            n_harmonics: Number of harmonics to suppress.
            base_attenuation: Minimum attenuation at notch center
                (0=full removal, 0.3=keep 30% of grid).

        Returns:
            Grid-removed image as uint8, same shape as input.
        """
        is_color = len(image.shape) == 3

        if is_color:
            # Process each channel separately
            channels = cv2.split(image)
            result_channels = []
            for ch in channels:
                result_channels.append(
                    self._notch_filter_channel(
                        ch, period_x, period_y, notch_width, n_harmonics, base_attenuation
                    )
                )
            return cv2.merge(result_channels)
        else:
            return self._notch_filter_channel(
                image, period_x, period_y, notch_width, n_harmonics, base_attenuation
            )

    def _notch_filter_channel(
        self,
        channel: np.ndarray,
        period_x: int,
        period_y: int,
        notch_width: float,
        n_harmonics: int,
        base_attenuation: float,
    ) -> np.ndarray:
        """Apply notch filter to a single channel.

        Args:
            channel: Single-channel uint8 image.
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            notch_width: Notch width (sigma in freq bins).
            n_harmonics: Number of harmonics.
            base_attenuation: Min attenuation at notch center.

        Returns:
            Filtered channel as uint8.
        """
        h, w = channel.shape
        img_f = channel.astype(np.float64)

        # Limit harmonics for small periods to avoid over-suppression
        # Small period = high base frequency, harmonics reach Nyquist quickly
        max_safe_harmonic_x = max(1, (w // 2) // max(1, w // period_x)) if period_x > 0 else 1
        max_safe_harmonic_y = max(1, (h // 2) // max(1, h // period_y)) if period_y > 0 else 1
        effective_harmonics = min(n_harmonics, max_safe_harmonic_x, max_safe_harmonic_y)

        # 2D FFT
        F = np.fft.fft2(img_f)

        # Build notch mask (1 = keep, base_attenuation = suppress)
        mask = np.ones((h, w), dtype=np.float64)

        for hx in range(1, effective_harmonics + 1):
            for hy in range(0, effective_harmonics + 1):
                if hx == 0 and hy == 0:
                    continue

                # Grid frequencies in both positive and negative spectrum
                kx = hx * w / period_x if period_x > 0 else 0
                ky = hy * h / period_y if period_y > 0 else 0

                # Apply Gaussian notch at all 4 symmetric positions
                for fx_sign in [1, -1]:
                    for fy_sign in [1, -1]:
                        fx = fx_sign * kx
                        fy = fy_sign * ky

                        # Wrap to FFT indices
                        fx_idx = fx % w
                        fy_idx = fy % h

                        # Create distance map from this notch center
                        cx = np.arange(w)
                        cy = np.arange(h)

                        # Efficient: compute distance only near the notch
                        # (Gaussian falls off quickly)
                        dx = np.minimum(np.abs(cx - fx_idx), np.abs(cx - fx_idx + w))
                        dx = np.minimum(dx, np.abs(cx - fx_idx - w))

                        dy = np.minimum(np.abs(cy - fy_idx), np.abs(cy - fy_idx + h))
                        dy = np.minimum(dy, np.abs(cy - fy_idx - h))

                        dist2 = dy[:, None]**2 + dx[None, :]**2

                        # Gaussian notch: suppress this frequency
                        notch = 1.0 - (1.0 - base_attenuation) * np.exp(-dist2 / (2 * notch_width**2))
                        mask = mask * notch

        # Also suppress pure horizontal/vertical grid lines (hy=0 or hx=0 cases)
        for h_idx in range(1, effective_harmonics + 1):
            # Pure vertical lines (kx only, ky=0)
            kx = h_idx * w / period_x if period_x > 0 else 0
            for fx_sign in [1, -1]:
                fx_idx = (fx_sign * kx) % w
                dx = np.arange(w, dtype=np.float64)
                dx = np.minimum(np.abs(dx - fx_idx), np.abs(dx - fx_idx + w))
                dx = np.minimum(dx, np.abs(dx - fx_idx - w))
                notch_1d = 1.0 - (1.0 - base_attenuation) * np.exp(-dx**2 / (2 * notch_width**2))
                mask *= notch_1d[None, :]

            # Pure horizontal lines (ky only, kx=0)
            ky = h_idx * h / period_y if period_y > 0 else 0
            for fy_sign in [1, -1]:
                fy_idx = (fy_sign * ky) % h
                dy = np.arange(h, dtype=np.float64)
                dy = np.minimum(np.abs(dy - fy_idx), np.abs(dy - fy_idx + h))
                dy = np.minimum(dy, np.abs(dy - fy_idx - h))
                notch_1d = 1.0 - (1.0 - base_attenuation) * np.exp(-dy**2 / (2 * notch_width**2))
                mask *= notch_1d[:, None]

        # Apply mask and inverse FFT
        F_filtered = F * mask
        result = np.real(np.fft.ifft2(F_filtered))

        return np.clip(result, 0, 255).astype(np.uint8)

    def deconvolve(
        self,
        image: np.ndarray,
        template_normalized: np.ndarray,
        period_x: int,
        period_y: int,
        strength: float = 1.0,
        clamp_min: float = 0.5,
        clamp_max: float = 2.0
    ) -> np.ndarray:
        """Multiplicative deconvolution: I_clean = I_observed / G_normalized.

        Tiles the normalized template to cover the full image and performs
        element-wise division. Template values are clamped to prevent
        extreme corrections from numerical instability.

        Args:
            image: Input degraded image (grayscale or BGR).
            template_normalized: Normalized grid template (mean=1).
            period_x: Horizontal period (must match template width).
            period_y: Vertical period (must match template height).
            strength: Blending factor 0-1. At 1.0, full deconvolution.
                At 0.0, returns original. Values between blend linearly.
            clamp_min: Minimum template value (prevents over-brightening).
            clamp_max: Maximum template value (prevents over-darkening).

        Returns:
            Deconvolved image as uint8.
        """
        H, W = image.shape[:2]
        is_color = len(image.shape) == 3

        # Tile template to full image size
        if is_color:
            ty, tx = template_normalized.shape[:2]
            reps_y = H // ty + 1
            reps_x = W // tx + 1
            G_full = np.tile(template_normalized, (reps_y, reps_x, 1))[:H, :W]
        else:
            ty, tx = template_normalized.shape
            reps_y = H // ty + 1
            reps_x = W // tx + 1
            G_full = np.tile(template_normalized, (reps_y, reps_x))[:H, :W]

        # Clamp for safe division
        G_safe = np.clip(G_full, clamp_min, clamp_max)

        # Deconvolution
        img_f = image.astype(np.float64)
        result = img_f / G_safe

        # Brightness correction (preserve global mean)
        scale = img_f.mean() / (result.mean() + 1e-8)
        result *= scale

        # Blend with original based on strength
        if strength < 1.0:
            result = img_f * (1 - strength) + result * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def edge_protect_blend(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        ink_l_threshold: float = 40.0
    ) -> np.ndarray:
        """Protect ink lines and strong edges by blending with original.

        Creates a mask identifying dark regions (likely ink strokes) with
        high gradient (edges). In these areas, the original image is
        preserved to avoid artifacts on fine brush details.

        Args:
            original: Original BGR image before restoration.
            restored: Restored BGR image after deconvolution.
            ink_l_threshold: L* threshold in LAB space. Pixels darker than
                this (lower L*) are candidates for ink protection.

        Returns:
            Blended image preserving ink lines as uint8 BGR.
        """
        # Extract L channel from LAB color space
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float64) * (100.0 / 255.0)

        # Compute gradient magnitude
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        # Ink mask: dark pixels with strong edges
        ink_mask = (L < ink_l_threshold) & (grad > np.percentile(grad, 70))

        # Smooth the mask for gradual blending
        weight = cv2.GaussianBlur(ink_mask.astype(np.float32), (7, 7), 2.0)

        # Blend: restored where weight is low, original where weight is high
        result = (
            restored.astype(np.float64) * (1 - weight[..., None])
            + original.astype(np.float64) * weight[..., None]
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def edge_detail_transfer(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        detail_strength: float = 0.5,
        detail_sigma: float = 1.0,
    ) -> np.ndarray:
        """Transfer edge details from original to restored image.

        Extracts high-frequency edge detail from the original image
        and injects it into the restored result. This recovers internal
        object edges that were damaged during grid removal.

        detail = original - GaussianBlur(original, detail_sigma)
        result = restored + detail_strength * detail * edge_weight

        The edge_weight ensures detail is only injected where actual
        edges exist, avoiding amplification of grid residuals in flat areas.

        Args:
            original: Original BGR image.
            restored: Restored BGR image (after grid removal).
            detail_strength: How much detail to inject (0-1).
            detail_sigma: Sigma for detail extraction blur.

        Returns:
            Edge-enhanced result as uint8 BGR.
        """
        if detail_strength <= 0:
            return restored

        orig_f = original.astype(np.float64)
        rest_f = restored.astype(np.float64)

        # Extract high-frequency detail from original
        blurred_orig = cv2.GaussianBlur(orig_f, (0, 0), detail_sigma)
        detail = orig_f - blurred_orig  # high-pass

        # Compute edge confidence map from original
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        # Normalize gradient to [0, 1], suppress weak gradients (grid noise)
        grad_norm = grad / (np.percentile(grad, 95) + 1e-8)
        grad_norm = np.clip(grad_norm, 0, 1)

        # Threshold: only keep strong edges (above median)
        # This avoids re-injecting grid pattern residuals
        edge_weight = np.where(grad_norm > 0.3, grad_norm, 0.0)
        edge_weight = cv2.GaussianBlur(edge_weight.astype(np.float64), (5, 5), 1.0)

        # Inject detail weighted by edge confidence
        result = rest_f + detail_strength * detail * edge_weight[..., None]

        return np.clip(result, 0, 255).astype(np.uint8)

    def adaptive_usm(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        sigma: float = 1.5,
        edge_threshold: float = 0.2,
    ) -> np.ndarray:
        """Adaptive Unsharp Masking — edge-aware sharpening on restored image.

        Unlike edge_detail_transfer which references the original (and may
        re-inject grid residuals), this operates solely on the restored image.
        Sharpening is applied proportionally to local edge strength, so flat
        areas (where grid residuals live) stay untouched.

        sharpened = image + strength * highpass * edge_weight

        Args:
            image: Restored BGR image (uint8).
            strength: Sharpening strength (0=none, 1=moderate, 2=strong).
            sigma: Gaussian sigma for high-pass extraction.
            edge_threshold: Minimum normalized gradient to apply sharpening.
                Lower = sharpen more areas. Range 0-1.

        Returns:
            Sharpened image as uint8 BGR.
        """
        if strength <= 0:
            return image

        img_f = image.astype(np.float64)

        # High-pass: detail to amplify
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
        highpass = img_f - blurred

        # Edge weight from the restored image itself (not original)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)

        # Normalize
        grad_norm = grad / (np.percentile(grad, 95) + 1e-8)
        grad_norm = np.clip(grad_norm, 0, 1)

        # Suppress below threshold (flat areas / grid residuals)
        edge_weight = np.where(grad_norm > edge_threshold, grad_norm, 0.0)
        edge_weight = cv2.GaussianBlur(edge_weight, (3, 3), 0.8)

        # Apply adaptive sharpening
        result = img_f + strength * highpass * edge_weight[..., None]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _compute_periodicity_map(
        self,
        image_gray: np.ndarray,
        period_x: int,
        period_y: int,
        window_radius: int = 0,
    ) -> np.ndarray:
        """Compute local periodicity map via normalized cross-correlation at grid lags.

        For each pixel, computes NCC between the local patch and the patch shifted
        by the grid period. High NCC = periodic = likely grid edge, not object edge.

        Args:
            image_gray: Grayscale image (float64, 0-255 range).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            window_radius: Radius for local NCC window. 0 = auto (2 * max_period).

        Returns:
            Periodicity map in [0, 1], same shape as input.
        """
        if window_radius <= 0:
            window_radius = 2 * max(period_x, period_y)

        gray = image_gray.astype(np.float64)
        h, w = gray.shape
        win_size = 2 * window_radius + 1

        # Local mean and variance via uniform filter
        local_mean = uniform_filter(gray, size=win_size)
        local_sq_mean = uniform_filter(gray ** 2, size=win_size)
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-8)
        local_std = np.sqrt(local_var)

        # NCC at horizontal lag (period_x)
        ncc_x = np.zeros_like(gray)
        if period_x < w:
            shifted_x = np.zeros_like(gray)
            shifted_x[:, :w - period_x] = gray[:, period_x:]
            shifted_x[:, w - period_x:] = gray[:, :period_x]

            shifted_mean = uniform_filter(shifted_x, size=win_size)
            cross = uniform_filter(gray * shifted_x, size=win_size)
            cov = cross - local_mean * shifted_mean
            shifted_var = np.maximum(
                uniform_filter(shifted_x ** 2, size=win_size) - shifted_mean ** 2, 1e-8
            )
            shifted_std = np.sqrt(shifted_var)
            ncc_x = cov / (local_std * shifted_std + 1e-8)

        # NCC at vertical lag (period_y)
        ncc_y = np.zeros_like(gray)
        if period_y < h:
            shifted_y = np.zeros_like(gray)
            shifted_y[:h - period_y, :] = gray[period_y:, :]
            shifted_y[h - period_y:, :] = gray[:period_y, :]

            shifted_mean = uniform_filter(shifted_y, size=win_size)
            cross = uniform_filter(gray * shifted_y, size=win_size)
            cov = cross - local_mean * shifted_mean
            shifted_var = np.maximum(
                uniform_filter(shifted_y ** 2, size=win_size) - shifted_mean ** 2, 1e-8
            )
            shifted_std = np.sqrt(shifted_var)
            ncc_y = cov / (local_std * shifted_std + 1e-8)

        # Combine: max of x and y periodicity
        periodicity = np.maximum(np.clip(ncc_x, 0, 1), np.clip(ncc_y, 0, 1))

        return periodicity

    def object_edge_enhance(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        period_x: int,
        period_y: int,
        edge_sigma_scale: float = 2.0,
        detail_source: str = "original",
        detail_sigma: float = 1.5,
        enhance_strength: float = 0.3,
        edge_threshold_low: float = 0.05,
        edge_threshold_high: float = 0.2,
        periodicity_rejection: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Object-Edge-Only Enhancement: enhances only true object edges, not grid edges.

        Combines three physical separation strategies:
        1. Scale Separation: large-sigma gradient from restored (grid already removed)
        2. Chrominance Immunity: LAB a*/b* gradient from original (immune to multiplicative grid)
        3. Periodicity Rejection: local NCC suppresses periodic (grid) structures

        Args:
            original: Original BGR image (uint8).
            restored: Restored BGR image after grid removal (uint8).
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.
            edge_sigma_scale: sigma = scale * max(period) for large-scale gradient.
            detail_source: "restored" (safe) or "original" (stronger detail).
            detail_sigma: Sigma for detail extraction high-pass.
            enhance_strength: Enhancement intensity (0-1).
            edge_threshold_low: Soft threshold lower bound.
            edge_threshold_high: Soft threshold upper bound.
            periodicity_rejection: Periodicity suppression strength (0=disabled).

        Returns:
            Tuple of (enhanced_image, oee_mask) both as uint8/float64.
        """
        if enhance_strength <= 0:
            mask = np.zeros(restored.shape[:2], dtype=np.float64)
            return restored, mask

        max_period = max(period_x, period_y)

        # --- Step A: Large-Scale Luminance Gradient (from RESTORED) ---
        gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sigma_large = edge_sigma_scale * max_period

        # Blur to eliminate grid-scale features, keep only object-scale edges
        ksize = int(sigma_large * 6) | 1  # ensure odd
        gray_blurred = cv2.GaussianBlur(gray_restored, (ksize, ksize), sigma_large)

        # Sobel gradient on blurred image
        lum_gx = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
        lum_gy = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
        lum_grad = np.sqrt(lum_gx ** 2 + lum_gy ** 2)

        # Normalize to [0, 1] using 95th percentile
        p95_lum = np.percentile(lum_grad, 95)
        lum_norm = np.clip(lum_grad / (p95_lum + 1e-8), 0, 1)

        # --- Step B: Chrominance Gradient (from ORIGINAL) ---
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float64)
        a_ch = lab[:, :, 1] - 128.0  # center around 0
        b_ch = lab[:, :, 2] - 128.0

        # Smooth chrominance to suppress cross-channel noise
        chroma_sigma = max_period / 2.0
        chroma_ksize = int(chroma_sigma * 6) | 1
        a_smooth = cv2.GaussianBlur(a_ch, (chroma_ksize, chroma_ksize), chroma_sigma)
        b_smooth = cv2.GaussianBlur(b_ch, (chroma_ksize, chroma_ksize), chroma_sigma)

        # Chrominance gradient
        a_gx = cv2.Sobel(a_smooth, cv2.CV_64F, 1, 0, ksize=3)
        a_gy = cv2.Sobel(a_smooth, cv2.CV_64F, 0, 1, ksize=3)
        b_gx = cv2.Sobel(b_smooth, cv2.CV_64F, 1, 0, ksize=3)
        b_gy = cv2.Sobel(b_smooth, cv2.CV_64F, 0, 1, ksize=3)
        chroma_grad = np.sqrt(a_gx ** 2 + a_gy ** 2 + b_gx ** 2 + b_gy ** 2)

        # Normalize to [0, 1]
        p95_chroma = np.percentile(chroma_grad, 95)
        chroma_norm = np.clip(chroma_grad / (p95_chroma + 1e-8), 0, 1)

        # --- Step C: Combine via max() ---
        # max() ensures monochrome paintings (low chroma) still get lum detection
        combined = np.maximum(lum_norm, chroma_norm)

        # --- Step D: Periodicity Rejection ---
        if periodicity_rejection > 0:
            periodicity = self._compute_periodicity_map(
                gray_restored, period_x, period_y
            )
            combined = combined * (1.0 - periodicity_rejection * periodicity)

        # --- Step E: Soft Thresholding + Smoothing ---
        denom = edge_threshold_high - edge_threshold_low
        if denom < 1e-8:
            denom = 1e-8
        mask = np.clip((combined - edge_threshold_low) / denom, 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)

        # --- Step F: Detail Extraction + Application ---
        if detail_source == "original":
            source = original.astype(np.float64)
            # When using original, use larger sigma to avoid grid re-injection
            eff_sigma = max(detail_sigma, max_period / 3.0)
        else:
            source = restored.astype(np.float64)
            eff_sigma = detail_sigma

        detail_ksize = int(eff_sigma * 6) | 1
        source_blurred = cv2.GaussianBlur(source, (detail_ksize, detail_ksize), eff_sigma)
        detail = source - source_blurred  # high-pass

        rest_f = restored.astype(np.float64)
        result = rest_f + enhance_strength * detail * mask[..., None]
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Return mask as float for visualization (0-255 uint8)
        mask_vis = (mask * 255).astype(np.uint8)

        return result, mask_vis

    def process(
        self,
        image_bgr: np.ndarray,
        period_detection: str = "auto",
        manual_period_x: int = 8,
        manual_period_y: int = 8,
        template_method: str = "median",
        deconv_strength: float = 1.0,
        clamp_min: float = 0.5,
        clamp_max: float = 2.0,
        edge_protection: bool = True,
        ink_l_threshold: float = 40.0,
        notch_width: float = 1.5,
        notch_harmonics: int = 4,
        notch_attenuation: float = 0.0,
        edge_enhance: bool = False,
        edge_detail_strength: float = 0.5,
        edge_detail_sigma: float = 1.0,
        final_sharpen: bool = False,
        final_sharpen_strength: float = 1.0,
        final_sharpen_sigma: float = 1.5,
        final_sharpen_edge_threshold: float = 0.2,
        object_edge_enhance: bool = False,
        oee_edge_sigma_scale: float = 2.0,
        oee_detail_source: str = "original",
        oee_detail_sigma: float = 1.5,
        oee_enhance_strength: float = 0.3,
        oee_edge_low: float = 0.05,
        oee_edge_high: float = 0.2,
        oee_periodicity_rejection: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Full multiplicative grid removal pipeline.

        Executes the complete restoration workflow:
        1. Period detection (automatic or manual)
        2. Template estimation via tile aggregation
        3. Multiplicative deconvolution
        4. Optional edge/ink-line protection

        Args:
            image_bgr: Input BGR image with grid degradation.
            period_detection: Either "auto" for autocorrelation detection
                or "manual" to use manual_period_x/y values.
            manual_period_x: Horizontal period when using manual detection.
            manual_period_y: Vertical period when using manual detection.
            template_method: Tile aggregation method ("median" or "mean").
            deconv_strength: Deconvolution blending strength (0-1).
            clamp_min: Minimum template clamp value for safe division.
            clamp_max: Maximum template clamp value for safe division.
            edge_protection: Whether to apply ink-line edge protection.
            ink_l_threshold: L* threshold for ink detection (0-100).

        Returns:
            Tuple of (result_bgr, intermediates) where intermediates is a
            dict containing:
            - period_x: Detected/used horizontal period
            - period_y: Detected/used vertical period
            - template: Normalized grid template
            - grid_full: Full-size tiled grid pattern
            - deconvolved: Result before edge protection
        """
        intermediates: Dict = {}

        # Step 1: Period detection
        if period_detection == "auto":
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            period_x, period_y = self.detect_period(gray)
        else:
            period_x = manual_period_x
            period_y = manual_period_y

        intermediates["period_x"] = period_x
        intermediates["period_y"] = period_y

        # Step 2: Grid modulation estimation
        H, W = image_bgr.shape[:2]

        if template_method == "gaussian_guided":
            # Guided filter division: edge-preserving low-pass
            grid_full = self.estimate_grid_modulation_guided(
                image_bgr, period_x, period_y,
                guided_radius=0,  # auto
                guided_eps=0.02,
            )
            intermediates["grid_full"] = grid_full
            intermediates["template"] = grid_full[:period_y, :period_x]

            G_safe = np.clip(grid_full, clamp_min, clamp_max)
            img_f = image_bgr.astype(np.float64)
            result = img_f / G_safe
            scale = img_f.mean() / (result.mean() + 1e-8)
            result *= scale
            if deconv_strength < 1.0:
                result = img_f * (1 - deconv_strength) + result * deconv_strength
            deconvolved = np.clip(result, 0, 255).astype(np.uint8)
        elif template_method == "gaussian":
            # Gaussian low-pass division: full-resolution modulation map
            grid_full = self.estimate_grid_modulation(
                image_bgr, period_x, period_y
            )
            intermediates["grid_full"] = grid_full
            # No small template in this mode
            intermediates["template"] = grid_full[:period_y, :period_x]

            # Deconvolve directly with full-resolution modulation
            G_safe = np.clip(grid_full, clamp_min, clamp_max)
            img_f = image_bgr.astype(np.float64)
            result = img_f / G_safe
            scale = img_f.mean() / (result.mean() + 1e-8)
            result *= scale
            if deconv_strength < 1.0:
                result = img_f * (1 - deconv_strength) + result * deconv_strength
            deconvolved = np.clip(result, 0, 255).astype(np.uint8)
        elif template_method == "notch":
            # FFT notch filter: selective grid frequency suppression
            deconvolved = self.estimate_grid_notch(
                image_bgr, period_x, period_y,
                notch_width=notch_width,
                n_harmonics=notch_harmonics,
                base_attenuation=notch_attenuation,
            )
            intermediates["grid_full"] = np.ones_like(image_bgr, dtype=np.float64)
            intermediates["template"] = np.ones((period_y, period_x, 3), dtype=np.float64)
        else:
            # Tile-based template estimation (median or mean)
            template = self.estimate_template(
                image_bgr, period_x, period_y, method=template_method
            )
            intermediates["template"] = template

            # Store full-size grid for visualization
            ty, tx = template.shape[:2]
            reps_y = H // ty + 1
            reps_x = W // tx + 1
            grid_full = np.tile(template, (reps_y, reps_x, 1))[:H, :W]
            intermediates["grid_full"] = grid_full

            # Deconvolve with tiled template
            deconvolved = self.deconvolve(
                image_bgr,
                template,
                period_x,
                period_y,
                strength=deconv_strength,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
        intermediates["deconvolved"] = deconvolved

        # Step 4: Edge protection (optional)
        if edge_protection:
            result = self.edge_protect_blend(
                image_bgr, deconvolved, ink_l_threshold=ink_l_threshold
            )
        else:
            result = deconvolved

        # Step 5: Object-Edge-Only Enhancement (v14.1) OR legacy edge enhancement
        if object_edge_enhance:
            result, oee_mask = self.object_edge_enhance(
                image_bgr, result,
                period_x, period_y,
                edge_sigma_scale=oee_edge_sigma_scale,
                detail_source=oee_detail_source,
                detail_sigma=oee_detail_sigma,
                enhance_strength=oee_enhance_strength,
                edge_threshold_low=oee_edge_low,
                edge_threshold_high=oee_edge_high,
                periodicity_rejection=oee_periodicity_rejection,
            )
            intermediates["oee_mask"] = oee_mask
        elif edge_enhance:
            # Legacy: Edge detail transfer
            result = self.edge_detail_transfer(
                image_bgr, result,
                detail_strength=edge_detail_strength,
                detail_sigma=edge_detail_sigma,
            )
            intermediates["before_edge_enhance"] = deconvolved

        # Step 6: Final adaptive sharpening (skipped when OEE active)
        if not object_edge_enhance and final_sharpen:
            intermediates["before_sharpen"] = result.copy()
            result = self.adaptive_usm(
                result,
                strength=final_sharpen_strength,
                sigma=final_sharpen_sigma,
                edge_threshold=final_sharpen_edge_threshold,
            )

        return result, intermediates
