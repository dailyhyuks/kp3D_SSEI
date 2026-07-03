"""Skip logic utilities for the Enhancement Pipeline.

Provides resolution checking and grid presence detection to avoid
unnecessary processing stages.
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from torch import Tensor


class ResolutionChecker:
    """Check if image resolution exceeds threshold for skipping upscale."""

    @staticmethod
    def should_skip(image: Tensor, max_pixels: int) -> bool:
        """Determine if upscaling should be skipped based on resolution.

        Args:
            image: Input tensor of shape (B, C, H, W).
            max_pixels: Maximum number of pixels before skipping.

        Returns:
            True if H*W >= max_pixels (image is already large enough).
        """
        if image.dim() == 4:
            _, _, h, w = image.shape
        elif image.dim() == 3:
            _, h, w = image.shape
        else:
            return False
        return h * w >= max_pixels


class GridPresenceChecker:
    """Detect whether a grid pattern is present in the image via FFT analysis.

    Uses MultiplicativeGridRemover's FFT detection to check for periodic
    grid artifacts and computes a harmonic confidence score.
    """

    def __init__(self, confidence_threshold: float = 3.0):
        """Initialize grid presence checker.

        Args:
            confidence_threshold: Minimum harmonic score to consider
                grid as present. Higher = more strict.
        """
        self.confidence_threshold = confidence_threshold

    def check(self, image_bgr: np.ndarray) -> Tuple[bool, Dict]:
        """Check if grid pattern is present in the image.

        Performs FFT analysis and checks harmonic peaks to determine
        if periodic grid artifacts exist.

        Args:
            image_bgr: Input image in BGR uint8 format.

        Returns:
            Tuple of (grid_detected, info_dict) where info_dict contains:
                - period_x: Detected horizontal period
                - period_y: Detected vertical period
                - harmonic_score: Confidence score of grid presence
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Compute FFT magnitude spectrum
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Search for periodic peaks in horizontal direction
        period_x, score_x = self._find_period(magnitude[cy, cx:], min_period=4, max_period=30)
        # Search for periodic peaks in vertical direction
        period_y, score_y = self._find_period(magnitude[cy:, cx], min_period=4, max_period=30)

        harmonic_score = (score_x + score_y) / 2.0

        info = {
            "period_x": period_x,
            "period_y": period_y,
            "harmonic_score": harmonic_score,
        }

        grid_detected = harmonic_score >= self.confidence_threshold
        return grid_detected, info

    def _find_period(
        self, spectrum_1d: np.ndarray, min_period: int = 4, max_period: int = 30
    ) -> Tuple[int, float]:
        """Find dominant period and its harmonic score in 1D spectrum.

        Looks for peaks at fundamental frequency and validates harmonics.

        Args:
            spectrum_1d: 1D magnitude spectrum from center outward.
            min_period: Minimum period to search for.
            max_period: Maximum period to search for.

        Returns:
            Tuple of (period, harmonic_score).
        """
        n = len(spectrum_1d)
        if n < max_period * 2:
            return 8, 0.0

        # Convert period range to frequency range
        freq_min = n // max_period
        freq_max = n // min_period

        if freq_min >= freq_max or freq_max >= n:
            return 8, 0.0

        # Find peak in valid frequency range
        search_region = spectrum_1d[freq_min:freq_max]
        if len(search_region) == 0:
            return 8, 0.0

        peak_idx = np.argmax(search_region) + freq_min
        if peak_idx == 0:
            return 8, 0.0

        period = n // peak_idx

        # Compute harmonic score: check if harmonics 2f, 3f, 4f also have peaks
        baseline = np.median(spectrum_1d[1:])
        if baseline == 0:
            baseline = 1.0

        peak_val = spectrum_1d[peak_idx]
        score = (peak_val - baseline) / baseline

        # Check harmonics
        for harmonic in range(2, 5):
            h_idx = peak_idx * harmonic
            if h_idx < n:
                h_val = spectrum_1d[h_idx]
                score += 0.5 * (h_val - baseline) / baseline

        return int(period), float(max(score, 0.0))
