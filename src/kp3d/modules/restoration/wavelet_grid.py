"""
Wavelet-based grid pattern removal using Stationary Wavelet Transform (SWT).

This module implements grid pattern detection and suppression for Korean painting
restoration using the a trous (with holes) algorithm for undecimated wavelet
decomposition. The stationary wavelet transform preserves translation invariance,
making it ideal for detecting and removing regular grid patterns without
introducing artifacts.

Key Features:
    - Stationary Wavelet Transform (SWT) using a trous algorithm
    - Automatic grid pattern detection via autocorrelation analysis
    - Soft thresholding for artifact-free grid suppression
    - Detail preservation to maintain painting texture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import pywt for filter coefficients, fall back to hardcoded values
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


class WaveletGridDecomposer:
    """
    Grid pattern removal using Stationary Wavelet Transform (SWT).

    The SWT (a trous algorithm) provides translation-invariant decomposition,
    making it ideal for detecting regular grid patterns in scanned paintings.
    Grid patterns appear as high-energy periodic structures in specific
    wavelet subbands, which can be identified and suppressed while
    preserving artistic detail.

    Attributes:
        wavelet_type: Type of wavelet to use (default: "db4")
        levels: Number of decomposition levels (default: 3)
        suppression_strength: How strongly to suppress grid (0-1, default: 0.3)
        detail_preservation: How much non-grid detail to preserve (0-1, default: 0.7)
    """

    # Hardcoded Daubechies-4 filter coefficients (normalized)
    # These are the standard db4 coefficients from the literature
    _DB4_LOW_DEC = np.array([
        -0.010597401785069032,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.027983769416983849,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523
    ])

    _DB4_HIGH_DEC = np.array([
        -0.23037781330885523,
        0.7148465705525415,
        -0.6308807679295904,
        -0.027983769416983849,
        0.18703481171888114,
        0.030841381835986965,
        -0.032883011666982945,
        -0.010597401785069032
    ])

    def __init__(
        self,
        wavelet_type: str = "db4",
        levels: int = 3,
        suppression_strength: float = 0.3,
        detail_preservation: float = 0.7
    ) -> None:
        """
        Initialize the wavelet grid decomposer.

        Args:
            wavelet_type: Wavelet type to use. Currently supports "db4".
            levels: Number of SWT decomposition levels. More levels capture
                    coarser grid patterns but increase computation.
            suppression_strength: Strength of grid suppression (0-1).
                                  Higher values remove more grid but risk
                                  removing detail.
            detail_preservation: Factor for preserving non-grid detail (0-1).
                                 Higher values keep more texture.
        """
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.suppression_strength = np.clip(suppression_strength, 0.0, 1.0)
        self.detail_preservation = np.clip(detail_preservation, 0.0, 1.0)

        # Get filter bank on initialization
        self._filters = self._get_filter_bank()

    def _get_filter_bank(self) -> dict[str, NDArray[np.floating]]:
        """
        Get the wavelet filter bank for decomposition and reconstruction.

        Returns:
            Dictionary containing:
                - low_dec: Low-pass decomposition filter
                - high_dec: High-pass decomposition filter
                - low_rec: Low-pass reconstruction filter
                - high_rec: High-pass reconstruction filter
        """
        if _HAS_PYWT and self.wavelet_type in pywt.wavelist():
            # Use pywt for filter coefficients
            wavelet = pywt.Wavelet(self.wavelet_type)
            return {
                "low_dec": np.array(wavelet.dec_lo, dtype=np.float64),
                "high_dec": np.array(wavelet.dec_hi, dtype=np.float64),
                "low_rec": np.array(wavelet.rec_lo, dtype=np.float64),
                "high_rec": np.array(wavelet.rec_hi, dtype=np.float64)
            }
        else:
            # Fall back to hardcoded db4 coefficients
            # Reconstruction filters are time-reversed decomposition filters
            return {
                "low_dec": self._DB4_LOW_DEC.copy(),
                "high_dec": self._DB4_HIGH_DEC.copy(),
                "low_rec": self._DB4_LOW_DEC[::-1].copy(),
                "high_rec": self._DB4_HIGH_DEC[::-1].copy()
            }

    def _upsample_filter(
        self,
        filt: NDArray[np.floating],
        level: int
    ) -> NDArray[np.floating]:
        """
        Upsample filter for a trous algorithm by inserting zeros.

        At level j, insert 2^(j-1) - 1 zeros between each coefficient.

        Args:
            filt: Original filter coefficients
            level: Current decomposition level (1-indexed)

        Returns:
            Upsampled filter with zeros inserted
        """
        if level <= 1:
            return filt

        # Number of zeros to insert between coefficients
        zeros_between = (1 << (level - 1)) - 1  # 2^(level-1) - 1

        # Create upsampled filter
        new_length = len(filt) + (len(filt) - 1) * zeros_between
        upsampled = np.zeros(new_length, dtype=filt.dtype)

        # Place original coefficients at correct positions
        step = zeros_between + 1
        upsampled[::step] = filt

        return upsampled

    def _convolve_2d_separable(
        self,
        image: NDArray[np.floating],
        filt_row: NDArray[np.floating],
        filt_col: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Apply separable 2D convolution using row and column filters.

        Args:
            image: 2D input image
            filt_row: Filter for row convolution
            filt_col: Filter for column convolution

        Returns:
            Filtered image
        """
        # Apply row filter (convolve along columns)
        temp = fftconvolve(image, filt_row.reshape(1, -1), mode='same')
        # Apply column filter (convolve along rows)
        result = fftconvolve(temp, filt_col.reshape(-1, 1), mode='same')
        return result

    def stationary_wavelet_decompose(
        self,
        image_2d: NDArray[np.floating]
    ) -> tuple[list[tuple[NDArray, NDArray, NDArray]], NDArray[np.floating]]:
        """
        Perform Stationary Wavelet Transform using a trous algorithm.

        The a trous algorithm performs undecimated wavelet decomposition
        by upsampling the filters at each level instead of downsampling
        the signal. This preserves translation invariance.

        Args:
            image_2d: 2D grayscale image as float array

        Returns:
            Tuple of:
                - List of (LH, HL, HH) detail coefficient tuples per level
                - Final LL approximation coefficients
        """
        detail_coeffs: list[tuple[NDArray, NDArray, NDArray]] = []
        approx = image_2d.astype(np.float64)

        low_dec = self._filters["low_dec"]
        high_dec = self._filters["high_dec"]

        for level in range(1, self.levels + 1):
            # Upsample filters for this level
            low_up = self._upsample_filter(low_dec, level)
            high_up = self._upsample_filter(high_dec, level)

            # Compute 2D subbands using separable filtering
            # LL: low-pass in both directions (becomes next level's approximation)
            ll = self._convolve_2d_separable(approx, low_up, low_up)

            # LH: low-pass rows, high-pass columns (horizontal edges)
            lh = self._convolve_2d_separable(approx, low_up, high_up)

            # HL: high-pass rows, low-pass columns (vertical edges)
            hl = self._convolve_2d_separable(approx, high_up, low_up)

            # HH: high-pass in both directions (diagonal edges)
            hh = self._convolve_2d_separable(approx, high_up, high_up)

            detail_coeffs.append((lh, hl, hh))
            approx = ll

        return detail_coeffs, approx

    def identify_grid_subbands(
        self,
        detail_coeffs: list[tuple[NDArray, NDArray, NDArray]]
    ) -> dict[tuple[int, str], float]:
        """
        Identify which subbands contain grid patterns using autocorrelation.

        Grid patterns exhibit high periodicity, which manifests as strong
        periodic peaks in the autocorrelation function. We compute a
        "regularity score" based on the ratio of periodic peaks to the
        center peak.

        Args:
            detail_coeffs: List of (LH, HL, HH) tuples from SWT decomposition

        Returns:
            Dictionary mapping (level, subband_type) to confidence score (0-1).
            Higher scores indicate more likely grid patterns.
        """
        grid_confidence: dict[tuple[int, str], float] = {}
        subband_names = ["LH", "HL", "HH"]

        for level_idx, (lh, hl, hh) in enumerate(detail_coeffs):
            level = level_idx + 1

            for subband, name in zip([lh, hl, hh], subband_names):
                confidence = self._compute_regularity_score(subband)
                grid_confidence[(level, name)] = confidence

        return grid_confidence

    def _compute_regularity_score(
        self,
        subband: NDArray[np.floating],
        top_k: int = 5
    ) -> float:
        """
        Compute regularity score from autocorrelation analysis.

        Grid patterns produce periodic peaks in the autocorrelation.
        The regularity score is the ratio of average top-k peaks
        (excluding center) to the center peak value.

        Args:
            subband: 2D subband coefficients
            top_k: Number of top peaks to average

        Returns:
            Regularity score in range [0, 1]
        """
        # Compute 2D autocorrelation using FFT
        # Autocorr(f) = IFFT(|FFT(f)|^2)
        f_transform = np.fft.fft2(subband)
        power_spectrum = np.abs(f_transform) ** 2
        autocorr = np.fft.ifft2(power_spectrum).real

        # Normalize by center value (variance)
        center_val = autocorr[0, 0]
        if center_val < 1e-10:
            return 0.0

        autocorr_norm = autocorr / center_val

        # Shift so center is in the middle
        autocorr_shifted = np.fft.fftshift(autocorr_norm)

        # Exclude center region (self-correlation)
        h, w = autocorr_shifted.shape
        cy, cx = h // 2, w // 2
        exclude_radius = max(3, min(h, w) // 20)

        # Mask out center
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = ((y_coords - cy) ** 2 + (x_coords - cx) ** 2) <= exclude_radius ** 2
        autocorr_masked = autocorr_shifted.copy()
        autocorr_masked[center_mask] = 0

        # Find top-k peaks
        flat_autocorr = autocorr_masked.flatten()
        if len(flat_autocorr) < top_k:
            top_k = len(flat_autocorr)

        # Get top k absolute values (peaks can be positive or negative)
        top_indices = np.argpartition(np.abs(flat_autocorr), -top_k)[-top_k:]
        top_peaks = np.abs(flat_autocorr[top_indices])

        # Regularity score: mean of top peaks
        regularity = np.mean(top_peaks)

        # Clamp to [0, 1]
        return float(np.clip(regularity, 0.0, 1.0))

    def suppress_grid_coefficients(
        self,
        detail_coeffs: list[tuple[NDArray, NDArray, NDArray]],
        grid_confidence: dict[tuple[int, str], float]
    ) -> list[tuple[NDArray, NDArray, NDArray]]:
        """
        Suppress grid patterns using soft thresholding based on confidence.

        Coefficients are multiplied by a suppression factor that depends
        on the grid confidence for that subband. Non-grid detail is
        preserved using the detail_preservation parameter.

        Suppression formula:
            suppressed = coeff * (1 - conf * suppression) * preservation_factor

        Where preservation_factor ensures non-grid content is maintained.

        Args:
            detail_coeffs: List of (LH, HL, HH) tuples from decomposition
            grid_confidence: Confidence scores from identify_grid_subbands

        Returns:
            Suppressed detail coefficients in same format as input
        """
        suppressed_coeffs: list[tuple[NDArray, NDArray, NDArray]] = []
        subband_names = ["LH", "HL", "HH"]

        for level_idx, (lh, hl, hh) in enumerate(detail_coeffs):
            level = level_idx + 1
            suppressed_subbands: list[NDArray] = []

            for subband, name in zip([lh, hl, hh], subband_names):
                conf = grid_confidence.get((level, name), 0.0)

                # Compute suppression factor
                # Higher confidence and suppression_strength = more suppression
                suppression_factor = 1.0 - conf * self.suppression_strength

                # Preservation factor ensures we keep non-grid detail
                # When suppressing, we still want to preserve some detail
                preservation_factor = max(
                    1.0 - self.suppression_strength + self.detail_preservation,
                    self.detail_preservation
                )

                # Apply soft thresholding
                suppressed = subband * suppression_factor * preservation_factor
                suppressed_subbands.append(suppressed)

            suppressed_coeffs.append((
                suppressed_subbands[0],
                suppressed_subbands[1],
                suppressed_subbands[2]
            ))

        return suppressed_coeffs

    def reconstruct(
        self,
        suppressed_coeffs: list[tuple[NDArray, NDArray, NDArray]],
        approx: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Reconstruct image from SWT coefficients.

        For SWT, reconstruction is simplified: we sum all subbands
        weighted appropriately. The a trous reconstruction uses
        the inverse relationship of the decomposition.

        Args:
            suppressed_coeffs: Suppressed detail coefficients
            approx: Final approximation (LL) coefficients

        Returns:
            Reconstructed 2D image
        """
        # Get reconstruction filters
        low_rec = self._filters["low_rec"]
        high_rec = self._filters["high_rec"]

        # Start with approximation
        result = approx.copy()

        # Reconstruct from coarsest to finest level
        for level_idx in range(len(suppressed_coeffs) - 1, -1, -1):
            level = level_idx + 1
            lh, hl, hh = suppressed_coeffs[level_idx]

            # Upsample reconstruction filters
            low_up = self._upsample_filter(low_rec, level)
            high_up = self._upsample_filter(high_rec, level)

            # Reconstruct each subband and accumulate
            # For simplified SWT inverse, we convolve with reconstruction filters
            # and sum the contributions

            # Approximation contribution (low-low)
            ll_contrib = self._convolve_2d_separable(result, low_up, low_up)

            # Detail contributions
            lh_contrib = self._convolve_2d_separable(lh, low_up, high_up)
            hl_contrib = self._convolve_2d_separable(hl, high_up, low_up)
            hh_contrib = self._convolve_2d_separable(hh, high_up, high_up)

            # Combine all contributions
            # Normalization factor for a trous (approximately 4 subbands summed)
            result = (ll_contrib + lh_contrib + hl_contrib + hh_contrib) / 4.0

        return result

    def process(
        self,
        image_bgr: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.floating]]:
        """
        Full grid removal pipeline for BGR color image.

        Process each color channel independently through the wavelet
        pipeline and merge results. Returns both the processed image
        and a confidence map showing detected grid locations.

        Args:
            image_bgr: Input BGR image as uint8 array (H, W, 3)

        Returns:
            Tuple of:
                - Processed BGR image with grid suppressed (uint8)
                - Confidence map showing grid detection strength (float32)
        """
        # Convert to float32 for processing
        image_float = image_bgr.astype(np.float32) / 255.0

        # Process each channel
        result_channels: list[NDArray] = []
        all_confidences: list[dict[tuple[int, str], float]] = []

        for channel_idx in range(3):
            channel = image_float[:, :, channel_idx]

            # Decompose
            detail_coeffs, approx = self.stationary_wavelet_decompose(channel)

            # Identify grid subbands
            grid_confidence = self.identify_grid_subbands(detail_coeffs)
            all_confidences.append(grid_confidence)

            # Suppress grid coefficients
            suppressed = self.suppress_grid_coefficients(detail_coeffs, grid_confidence)

            # Reconstruct
            reconstructed = self.reconstruct(suppressed, approx)

            # Clip to valid range
            reconstructed = np.clip(reconstructed, 0.0, 1.0)
            result_channels.append(reconstructed)

        # Merge channels
        result_float = np.stack(result_channels, axis=-1)

        # Convert back to uint8
        result_bgr = (result_float * 255.0).astype(np.uint8)

        # Create confidence map by averaging across channels and subbands
        confidence_map = self._create_confidence_map(
            all_confidences,
            image_bgr.shape[:2]
        )

        return result_bgr, confidence_map

    def _create_confidence_map(
        self,
        all_confidences: list[dict[tuple[int, str], float]],
        shape: tuple[int, int]
    ) -> NDArray[np.floating]:
        """
        Create a spatial confidence map from subband confidences.

        Averages confidence across channels and creates a uniform
        map (since SWT is translation-invariant, confidence is
        the same across spatial locations).

        Args:
            all_confidences: List of confidence dicts, one per channel
            shape: Output shape (H, W)

        Returns:
            Confidence map as float32 array
        """
        # Average confidence across all channels and subbands
        total_conf = 0.0
        count = 0

        for channel_conf in all_confidences:
            for conf_value in channel_conf.values():
                total_conf += conf_value
                count += 1

        avg_conf = total_conf / count if count > 0 else 0.0

        # Create uniform confidence map
        # In practice, grid patterns are often uniform across the image
        confidence_map = np.full(shape, avg_conf, dtype=np.float32)

        return confidence_map


__all__ = ["WaveletGridDecomposer"]
