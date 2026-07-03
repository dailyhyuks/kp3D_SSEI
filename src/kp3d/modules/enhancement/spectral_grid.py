"""Spectral (frequency-domain) grid removal using Butterworth notch filters.

Removes periodic grid artifacts from digitized images by:
1. Converting multiplicative grid to additive via log transform
2. Detecting grid harmonics in the frequency domain with sub-pixel precision
3. Applying Butterworth notch filters with smooth roll-off (no Gibbs ringing)
4. Iterating until convergence

Designed to run BEFORE upscaling to prevent grid amplification.
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class SpectralGridRemover:
    """Frequency-domain grid removal using Butterworth notch filters."""

    def __init__(self, config: Any) -> None:
        """Initialize with configuration parameters.

        Args:
            config: Configuration object with spectral_* attributes
        """
        self.period_min = config.spectral_period_min
        self.period_max = config.spectral_period_max
        self.butterworth_order = config.spectral_butterworth_order
        self.max_iterations = config.spectral_max_iterations
        self.convergence_threshold = config.spectral_convergence_threshold
        self.min_notch_width = config.spectral_min_notch_width
        self.max_notch_width = config.spectral_max_notch_width
        self.padding_factor = config.spectral_padding_factor
        self.harmonic_threshold = config.spectral_harmonic_threshold
        self.min_peak_prominence = config.spectral_min_peak_prominence

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Main entry point for spectral grid removal.

        Args:
            image_bgr: Input image in BGR format (uint8)

        Returns:
            Tuple of (cleaned_bgr, metadata_dict)
        """
        metadata: Dict[str, Any] = {
            "iterations": 0,
            "converged": False,
            "periods_detected": {"horizontal": [], "vertical": []},
            "notches_applied": 0,
            "residual_energy_ratio": 1.0,
        }

        try:
            h, w = image_bgr.shape[:2]

            # Skip processing for images too small for reliable spectral analysis
            min_dim = min(h, w)
            if min_dim < self.period_max * 4:
                logger.debug(f"Image too small ({w}x{h}) for spectral grid removal, "
                           f"need at least {int(self.period_max * 4)}px")
                return image_bgr.copy(), {"skipped": True, "reason": "image_too_small"}

            # Convert BGR -> LAB, extract L channel
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0].astype(np.float64)
            original_shape = L.shape

            # Log transform to convert multiplicative grid to additive
            epsilon = 1.0
            L_log = np.log(L + epsilon)
            L_log_original = L_log.copy()

            # Iterative filtering
            for iteration in range(self.max_iterations):
                logger.debug(f"Spectral grid removal iteration {iteration + 1}")

                # Detect grid periods with sub-pixel precision
                period_info = self._detect_periods_subpixel(L_log)

                if not period_info["found"]:
                    logger.debug("No grid periods detected, stopping iteration")
                    break

                # Store detected periods in metadata
                metadata["periods_detected"]["horizontal"].extend(
                    [p["period"] for p in period_info["horizontal"]]
                )
                metadata["periods_detected"]["vertical"].extend(
                    [p["period"] for p in period_info["vertical"]]
                )

                # Compute padded FFT for notch width estimation
                padded_h = original_shape[0] * self.padding_factor
                padded_w = original_shape[1] * self.padding_factor
                padded_shape = (padded_h, padded_w)

                # Apply Hann window to reduce spectral leakage
                window_h = np.hanning(original_shape[0])
                window_w = np.hanning(original_shape[1])
                window_2d = np.outer(window_h, window_w)
                L_windowed = L_log * window_2d

                # Zero-pad and compute FFT
                L_padded = np.zeros(padded_shape, dtype=np.float64)
                L_padded[:original_shape[0], :original_shape[1]] = L_windowed
                F = np.fft.fftshift(np.fft.fft2(L_padded))
                F_mag = np.abs(F)

                # Combine all peaks for notch placement
                all_peaks = period_info["horizontal"] + period_info["vertical"]

                # Estimate notch widths adaptively
                notch_params = self._estimate_notch_widths(F_mag, all_peaks, padded_shape)

                if not notch_params:
                    logger.debug("No valid notch parameters, stopping iteration")
                    break

                metadata["notches_applied"] += len(notch_params)

                # Build 2D Butterworth notch filter
                notch_filter = self._build_butterworth_notch(
                    padded_shape, notch_params, self.butterworth_order
                )

                # Apply filter
                L_log = self._apply_filter(L_log, notch_filter, padded_shape)

                metadata["iterations"] = iteration + 1

                # Check convergence
                residual_ratio = self._compute_residual_ratio(
                    L_log_original, L_log, notch_params, padded_shape
                )
                metadata["residual_energy_ratio"] = residual_ratio

                if residual_ratio < self.convergence_threshold:
                    logger.debug(f"Converged at iteration {iteration + 1} "
                               f"(residual ratio: {residual_ratio:.4f})")
                    metadata["converged"] = True
                    break

            # Inverse log transform
            L_restored = np.exp(L_log) - epsilon
            L_restored = np.clip(L_restored, 0, 255).astype(np.uint8)

            # Replace L channel in LAB and convert back to BGR
            lab[:, :, 0] = L_restored
            result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            logger.info(f"Spectral grid removal complete: {metadata['iterations']} iterations, "
                       f"{metadata['notches_applied']} notches applied")

            return result_bgr, metadata

        except Exception as e:
            logger.error(f"Spectral grid removal failed: {e}")
            # Return original image on failure
            return image_bgr.copy(), {"error": str(e), "iterations": 0}

    def _detect_periods_subpixel(self, L_log: np.ndarray) -> Dict:
        """Detect grid periods with sub-pixel precision using FFT + parabolic interpolation.

        Args:
            L_log: Log-transformed L channel

        Returns:
            Dict with 'horizontal', 'vertical' lists and 'found' bool
        """
        h, w = L_log.shape
        padded_h = h * self.padding_factor
        padded_w = w * self.padding_factor

        # Apply Hann window before FFT to suppress spectral leakage
        # (critical for avoiding false peaks from non-periodic signals like gradients)
        window_h = np.hanning(h)
        window_w = np.hanning(w)
        window_2d = np.outer(window_h, window_w)
        L_windowed = L_log * window_2d

        # Zero-pad and compute 2D FFT
        L_padded = np.zeros((padded_h, padded_w), dtype=np.float64)
        L_padded[:h, :w] = L_windowed
        F = np.fft.fftshift(np.fft.fft2(L_padded))
        F_mag = np.abs(F)

        result: Dict[str, Any] = {
            "horizontal": [],
            "vertical": [],
            "found": False
        }

        center_y = padded_h // 2
        center_x = padded_w // 2

        # For horizontal grid lines: project onto vertical frequency axis
        # Average magnitude along horizontal frequency axis (axis=1)
        vertical_profile = np.mean(F_mag, axis=1)

        # For vertical grid lines: project onto horizontal frequency axis
        # Average magnitude along vertical frequency axis (axis=0)
        horizontal_profile = np.mean(F_mag, axis=0)

        # Valid frequency range based on period constraints
        min_freq_h = padded_h / self.period_max
        max_freq_h = padded_h / self.period_min
        min_freq_w = padded_w / self.period_max
        max_freq_w = padded_w / self.period_min

        # Detect horizontal grid (vertical frequency peaks)
        h_peaks = self._find_peaks_in_profile(
            vertical_profile, center_y, min_freq_h, max_freq_h, padded_h, "horizontal"
        )
        result["horizontal"] = h_peaks

        # Detect vertical grid (horizontal frequency peaks)
        v_peaks = self._find_peaks_in_profile(
            horizontal_profile, center_x, min_freq_w, max_freq_w, padded_w, "vertical"
        )
        result["vertical"] = v_peaks

        result["found"] = len(h_peaks) > 0 or len(v_peaks) > 0

        return result

    def _find_peaks_in_profile(
        self,
        profile: np.ndarray,
        center: int,
        min_freq: float,
        max_freq: float,
        padded_size: int,
        direction: str
    ) -> List[Dict]:
        """Find peaks in a 1D frequency profile with sub-pixel interpolation.

        Uses both global threshold (median-based) and local prominence
        (peak must stand out from its local neighborhood) to avoid
        false positives on non-grid images like smooth gradients.

        Args:
            profile: 1D magnitude profile
            center: Center index (DC component)
            min_freq: Minimum frequency bin to search
            max_freq: Maximum frequency bin to search
            padded_size: Size of padded FFT
            direction: 'horizontal' or 'vertical' (for logging)

        Returns:
            List of peak dicts with 'period', 'freq_bin', 'amplitude', 'direction'
        """
        peaks = []
        n = len(profile)

        # Search in positive frequencies only (symmetric)
        start_idx = int(center + min_freq)
        end_idx = int(min(center + max_freq, n - 1))

        if start_idx >= end_idx - 1:
            return peaks

        # Calculate threshold from the SEARCH REGION only (not global median)
        # This avoids being fooled by high DC energy pulling down the threshold
        search_region = profile[start_idx:end_idx]
        region_median = np.median(search_region)
        threshold = region_median * self.min_peak_prominence

        # Local neighborhood size for prominence check (±window bins)
        local_window = max(5, int((end_idx - start_idx) * 0.1))

        # Find local maxima above threshold with local prominence check
        for k in range(start_idx + 1, end_idx - 1):
            # Check if local maximum
            if (profile[k] > profile[k - 1] and
                profile[k] > profile[k + 1] and
                profile[k] > threshold):

                # Local prominence check: peak must be significantly above
                # the local neighborhood average (not just a slight ripple)
                local_start = max(start_idx, k - local_window)
                local_end = min(end_idx, k + local_window + 1)
                local_region = profile[local_start:local_end]
                local_mean = np.mean(local_region)

                # Peak must be at least 2x the local mean to be a real grid peak
                if profile[k] < local_mean * 2.0:
                    continue

                # Sharpness check: a real grid peak is narrowband (spike-like).
                # Compute the ratio of peak to the average of its immediate
                # neighbors (±2 bins). For a true grid harmonic this should be
                # high (>1.5). Smooth spectra have ratio ~1.
                neighbor_start = max(0, k - 3)
                neighbor_end = min(n, k + 4)
                neighbors = np.concatenate([
                    profile[neighbor_start:k-1],
                    profile[k+2:neighbor_end]
                ])
                if len(neighbors) > 0:
                    neighbor_mean = np.mean(neighbors)
                    if neighbor_mean > 1e-10:
                        sharpness = profile[k] / neighbor_mean
                        if sharpness < 1.5:
                            continue

                # Parabolic interpolation for sub-pixel precision
                y_minus = profile[k - 1]
                y_center = profile[k]
                y_plus = profile[k + 1]

                denom = y_minus - 2 * y_center + y_plus
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y_minus - y_plus) / denom
                else:
                    delta = 0.0

                # Clamp delta to reasonable range
                delta = np.clip(delta, -0.5, 0.5)

                subpixel_freq_bin = (k - center) + delta

                # Convert frequency bin to period
                if abs(subpixel_freq_bin) > 0.1:
                    period = padded_size / abs(subpixel_freq_bin)

                    # Validate period is in expected range
                    if self.period_min <= period <= self.period_max:
                        peaks.append({
                            "period": period,
                            "freq_bin": subpixel_freq_bin,
                            "freq_idx": k,
                            "amplitude": y_center,
                            "direction": direction
                        })

        # Group harmonics and keep fundamental + harmonics
        if peaks:
            peaks = self._group_harmonics(peaks)

        return peaks

    def _group_harmonics(self, peaks: List[Dict]) -> List[Dict]:
        """Group peaks by harmonic relationship, keeping fundamental and harmonics.

        Args:
            peaks: List of detected peaks

        Returns:
            Filtered list including fundamental and valid harmonics
        """
        if len(peaks) <= 1:
            return peaks

        # Sort by period (descending) - fundamental has longest period
        sorted_peaks = sorted(peaks, key=lambda p: p["period"], reverse=True)

        result = []
        fundamental = sorted_peaks[0]
        result.append(fundamental)
        fundamental_freq = abs(fundamental["freq_bin"])

        # Check remaining peaks for harmonic relationship
        for peak in sorted_peaks[1:]:
            peak_freq = abs(peak["freq_bin"])

            # Check if this is a harmonic (integer multiple of fundamental)
            if fundamental_freq > 0.1:
                ratio = peak_freq / fundamental_freq
                nearest_int = round(ratio)

                if nearest_int >= 2 and abs(ratio - nearest_int) < 0.1:
                    # This is a harmonic
                    if peak["amplitude"] > np.median([p["amplitude"] for p in peaks]) * self.harmonic_threshold:
                        result.append(peak)
                else:
                    # Not a harmonic - might be a different fundamental
                    # Only add if prominent enough
                    if peak["amplitude"] > fundamental["amplitude"] * 0.5:
                        result.append(peak)

        return result

    def _estimate_notch_widths(
        self,
        F_mag: np.ndarray,
        peaks: List[Dict],
        padded_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Measure frequency spread at each peak for adaptive notch width.

        Args:
            F_mag: 2D FFT magnitude (shifted)
            peaks: List of detected peaks
            padded_shape: Shape of padded FFT

        Returns:
            List of notch parameter dicts with 'freq_x', 'freq_y', 'width', 'amplitude'
        """
        notch_params = []
        center_y = padded_shape[0] // 2
        center_x = padded_shape[1] // 2

        for peak in peaks:
            direction = peak["direction"]
            freq_bin = peak["freq_bin"]

            # Determine notch center coordinates
            if direction == "horizontal":
                # Horizontal grid -> vertical frequency component
                freq_y = freq_bin
                freq_x = 0.0
                # Extract vertical slice through center
                profile = F_mag[:, center_x]
                idx = int(center_y + freq_bin)
            else:
                # Vertical grid -> horizontal frequency component
                freq_y = 0.0
                freq_x = freq_bin
                # Extract horizontal slice through center
                profile = F_mag[center_y, :]
                idx = int(center_x + freq_bin)

            # Measure half-maximum width
            if 0 <= idx < len(profile):
                peak_val = profile[idx]
                half_max = peak_val / 2

                # Find half-max points on both sides
                left_idx = idx
                while left_idx > 0 and profile[left_idx] > half_max:
                    left_idx -= 1

                right_idx = idx
                while right_idx < len(profile) - 1 and profile[right_idx] > half_max:
                    right_idx += 1

                width = (right_idx - left_idx) / 2.0
                width = np.clip(width, self.min_notch_width, self.max_notch_width)

                notch_params.append({
                    "freq_x": freq_x,
                    "freq_y": freq_y,
                    "width": width,
                    "amplitude": peak["amplitude"]
                })

        return notch_params

    def _build_butterworth_notch(
        self,
        shape: Tuple[int, int],
        notch_params: List[Dict],
        order: int
    ) -> np.ndarray:
        """Construct 2D Butterworth notch filter.

        Creates a filter that rejects specific frequencies while passing others.
        The Butterworth design provides smooth roll-off without Gibbs ringing.

        Args:
            shape: (height, width) of the filter
            notch_params: List of dicts with 'freq_x', 'freq_y', 'width', 'amplitude'
            order: Butterworth filter order

        Returns:
            2D filter array with values in [0, 1]
        """
        h, w = shape
        center_y = h // 2
        center_x = w // 2

        # Create frequency coordinate meshgrid (centered at DC)
        u = np.arange(h) - center_y
        v = np.arange(w) - center_x
        V, U = np.meshgrid(v, u)  # Note: V is horizontal, U is vertical

        # Start with all-pass filter
        H = np.ones((h, w), dtype=np.float64)

        for notch in notch_params:
            fx = notch["freq_x"]
            fy = notch["freq_y"]
            width = notch["width"]

            # Place notches at all 4 symmetric positions
            # (fx, fy), (-fx, fy), (fx, -fy), (-fx, -fy)
            notch_positions = [
                (fx, fy),
                (-fx, fy),
                (fx, -fy),
                (-fx, -fy)
            ]

            # Remove duplicates (when fx=0 or fy=0)
            unique_positions = list(set(notch_positions))

            for px, py in unique_positions:
                # Distance from each frequency point to the notch center
                # U corresponds to vertical freq (fy direction)
                # V corresponds to horizontal freq (fx direction)
                D = np.sqrt((V - px) ** 2 + (U - py) ** 2)

                # Avoid division by zero at exact notch center
                D_safe = np.maximum(D, 1e-10)

                # Butterworth notch formula
                # H_notch = 1 - 1 / (1 + (D/W)^(2n))
                ratio = D_safe / width
                H_single = 1.0 - 1.0 / (1.0 + ratio ** (2 * order))

                # Set to 0 at exact center (where D < small threshold)
                H_single[D < 0.1] = 0.0

                # Multiply into combined filter
                H = H * H_single

        return H

    def _apply_filter(
        self,
        L_log: np.ndarray,
        notch_filter: np.ndarray,
        padded_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Apply notch filter in frequency domain.

        No windowing is applied during filtering — windowing is only needed
        for spectral estimation (peak detection), not for notch filter
        application. The notch filter removes narrow frequency bands directly.

        Args:
            L_log: Log-transformed L channel
            notch_filter: 2D Butterworth notch filter
            padded_shape: Shape of padded arrays

        Returns:
            Filtered L_log (cropped to original size)
        """
        original_shape = L_log.shape

        # Zero-pad to match filter size (no windowing for filtering)
        L_padded = np.zeros(padded_shape, dtype=np.float64)
        L_padded[:original_shape[0], :original_shape[1]] = L_log

        # Forward FFT
        F = np.fft.fft2(L_padded)
        F_shifted = np.fft.fftshift(F)

        # Apply notch filter
        F_filtered = F_shifted * notch_filter

        # Inverse FFT
        F_unshifted = np.fft.ifftshift(F_filtered)
        L_filtered_padded = np.fft.ifft2(F_unshifted)

        # Take real part and crop to original size
        result = np.real(L_filtered_padded[:original_shape[0], :original_shape[1]])

        return result

    def _compute_residual_ratio(
        self,
        L_log_original: np.ndarray,
        L_log_filtered: np.ndarray,
        notch_params: List[Dict],
        padded_shape: Tuple[int, int]
    ) -> float:
        """Compute residual grid energy ratio for convergence check.

        Args:
            L_log_original: Original log-transformed L
            L_log_filtered: Filtered log-transformed L
            notch_params: Notch filter parameters
            padded_shape: Shape of padded FFT

        Returns:
            Ratio of residual energy at notch locations to total energy
        """
        original_shape = L_log_filtered.shape

        # Compute FFT of residual (what was removed)
        residual = L_log_original - L_log_filtered

        # Zero-pad
        residual_padded = np.zeros(padded_shape, dtype=np.float64)
        residual_padded[:original_shape[0], :original_shape[1]] = residual

        F_residual = np.fft.fftshift(np.fft.fft2(residual_padded))
        F_residual_mag = np.abs(F_residual)

        # Also compute FFT of filtered signal
        filtered_padded = np.zeros(padded_shape, dtype=np.float64)
        filtered_padded[:original_shape[0], :original_shape[1]] = L_log_filtered

        F_filtered = np.fft.fftshift(np.fft.fft2(filtered_padded))
        F_filtered_mag = np.abs(F_filtered)

        # Measure energy at notch locations in filtered signal
        center_y = padded_shape[0] // 2
        center_x = padded_shape[1] // 2

        notch_energy = 0.0
        total_energy = np.sum(F_filtered_mag ** 2)

        if total_energy < 1e-10:
            return 0.0

        for notch in notch_params:
            fx = int(notch["freq_x"])
            fy = int(notch["freq_y"])
            width = int(np.ceil(notch["width"]))

            # Sample around notch location
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    y_idx = center_y + fy + dy
                    x_idx = center_x + fx + dx

                    if (0 <= y_idx < padded_shape[0] and
                        0 <= x_idx < padded_shape[1]):
                        notch_energy += F_filtered_mag[y_idx, x_idx] ** 2

        return notch_energy / total_energy
