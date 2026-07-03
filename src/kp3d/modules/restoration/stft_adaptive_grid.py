"""STFT + Channel-Adaptive + Edge Protection grid removal (v9).

Detect-then-remove approach for grid pattern removal in scanned Korean paintings.
Instead of blanket filtering, this module first detects where the grid is and
how strong it is, then selectively removes it with per-channel adaptation.

Pipeline:
1. Grid Period Detection: Autocorrelation-based period estimation (or use known values)
2. Channel Modulation Measurement: Per-channel FFT energy at grid frequencies
3. STFT Local Energy Map: Windowed FFT to compute spatially-varying grid strength
4. Channel-Adaptive Notch Filtering: Per-channel attenuation proportional to modulation
5. Edge Protection: Blend original in edge regions to preserve brushstrokes

Key innovations over previous approaches:
- Spatial adaptivity: local energy-guided attenuation (strong grid -> aggressive, weak -> gentle)
- Per-channel adaptation: B(14.8%) > G(7.4%) > R(4.5%) modulation depth matching
- Edge-aware blending: DoG/LoG/ST mask protects painting content
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class STFTAdaptiveGridRemover:
    """STFT + Channel-Adaptive + Edge Protection grid removal.

    Attributes:
        period_x: Horizontal grid period in pixels (0 = auto-detect).
        period_y: Vertical grid period in pixels (0 = auto-detect).
        window_size: STFT window size. Use LCM of periods for full coverage.
        hop_size: STFT hop size (smaller = smoother, larger = faster).
        notch_sigma: Width of Gaussian notch in frequency domain.
        base_attenuation: Baseline attenuation at grid frequencies (0 = full removal).
        channel_adaptive: Whether to adapt notch strength per channel.
        n_harmonics: Number of harmonic frequencies to filter.
    """

    def __init__(
        self,
        period_x: int = 0,
        period_y: int = 0,
        window_size: int = 63,
        hop_size: int = 16,
        notch_sigma: float = 1.5,
        base_attenuation: float = 0.15,
        channel_adaptive: bool = True,
        n_harmonics: int = 5,
    ) -> None:
        self.period_x = period_x
        self.period_y = period_y
        self.window_size = window_size
        self.hop_size = hop_size
        self.notch_sigma = notch_sigma
        self.base_attenuation = base_attenuation
        self.channel_adaptive = channel_adaptive
        self.n_harmonics = n_harmonics

    # ------------------------------------------------------------------
    # Grid Period Detection
    # ------------------------------------------------------------------

    def detect_grid_periods(
        self, image_gray: np.ndarray
    ) -> Tuple[int, int]:
        """Detect grid periods via autocorrelation peak finding.

        Computes the mean-subtracted autocorrelation along rows (horizontal
        period) and columns (vertical period), then finds the first peak
        beyond lag 3 to avoid the DC component.

        Args:
            image_gray: Grayscale image (uint8 or float).

        Returns:
            (period_x, period_y) in pixels. Falls back to (9, 7) if
            detection fails.
        """
        gray = image_gray.astype(np.float64)
        gray -= gray.mean()
        h, w = gray.shape

        # Horizontal period: average autocorrelation along rows
        max_lag = min(w // 2, 50)
        acf_x = np.zeros(max_lag)
        for lag in range(max_lag):
            acf_x[lag] = np.mean(gray[:, :w - lag] * gray[:, lag:])
        # Normalize
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

    def detect_grid_periods_subpixel(
        self, image_gray: np.ndarray
    ) -> Tuple[float, float]:
        """Detect grid periods with subpixel accuracy via parabolic interpolation.

        Same as detect_grid_periods but returns float periods refined by
        fitting a parabola around the autocorrelation peak.

        Args:
            image_gray: Grayscale image (uint8 or float).

        Returns:
            (period_x, period_y) as floats with subpixel accuracy.
        """
        gray = image_gray.astype(np.float64)
        gray -= gray.mean()
        h, w = gray.shape

        # Horizontal period
        max_lag = min(w // 2, 50)
        acf_x = np.zeros(max_lag)
        for lag in range(max_lag):
            acf_x[lag] = np.mean(gray[:, :w - lag] * gray[:, lag:])
        if acf_x[0] > 0:
            acf_x /= acf_x[0]

        # Vertical period
        max_lag_y = min(h // 2, 50)
        acf_y = np.zeros(max_lag_y)
        for lag in range(max_lag_y):
            acf_y[lag] = np.mean(gray[:h - lag, :] * gray[lag:, :])
        if acf_y[0] > 0:
            acf_y /= acf_y[0]

        peak_x = self._find_first_peak(acf_x, min_lag=3, fallback=9)
        peak_y = self._find_first_peak(acf_y, min_lag=3, fallback=7)

        period_x = self._refine_peak_subpixel(acf_x, peak_x)
        period_y = self._refine_peak_subpixel(acf_y, peak_y)

        return period_x, period_y

    @staticmethod
    def _find_first_peak(
        acf: np.ndarray, min_lag: int = 3, fallback: int = 9
    ) -> int:
        """Find first peak in autocorrelation function after min_lag.

        A peak is defined as a local maximum where acf[i] > acf[i-1]
        and acf[i] > acf[i+1], with a minimum height of 0.05.

        Args:
            acf: Normalized autocorrelation array.
            min_lag: Minimum lag to start searching (skip DC region).
            fallback: Default value if no peak found.

        Returns:
            Lag of the first significant peak, or fallback.
        """
        for i in range(min_lag, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
                return i
        return fallback

    @staticmethod
    def _refine_peak_subpixel(acf: np.ndarray, peak_idx: int) -> float:
        """Refine integer peak to subpixel accuracy via parabolic interpolation.

        Fits a parabola to the peak and its neighbors to find the true maximum.

        Args:
            acf: Autocorrelation array.
            peak_idx: Integer peak index.

        Returns:
            Subpixel-accurate peak position.
        """
        if peak_idx <= 0 or peak_idx >= len(acf) - 1:
            return float(peak_idx)
        y0, y1, y2 = acf[peak_idx - 1], acf[peak_idx], acf[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) < 1e-10:
            return float(peak_idx)
        delta = 0.5 * (y0 - y2) / denom
        return peak_idx + np.clip(delta, -0.5, 0.5)

    # ------------------------------------------------------------------
    # Channel Modulation Depth
    # ------------------------------------------------------------------

    def measure_channel_modulation(
        self,
        image_bgr: np.ndarray,
        period_x: int,
        period_y: int,
    ) -> Dict[int, float]:
        """Measure per-channel modulation depth at grid frequencies.

        For each BGR channel, computes the ratio of energy at grid
        harmonic frequencies to total energy (excluding DC).

        Args:
            image_bgr: Input image in BGR format (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.

        Returns:
            Dict mapping channel index (0=B, 1=G, 2=R) to modulation depth.
        """
        modulation = {}
        h, w = image_bgr.shape[:2]

        for c in range(3):
            channel = image_bgr[:, :, c].astype(np.float64)
            f = np.fft.fft2(channel)
            magnitude = np.abs(np.fft.fftshift(f))
            cy, cx = h // 2, w // 2

            grid_energy = 0.0
            total_energy = np.sum(magnitude ** 2) - magnitude[cy, cx] ** 2

            for harmonic in range(1, self.n_harmonics + 1):
                # Horizontal grid frequencies
                fx = int(round(harmonic * w / period_x))
                if cx + fx + 3 < w:
                    grid_energy += np.sum(magnitude[:, cx + fx - 2:cx + fx + 3] ** 2)
                if cx - fx - 2 >= 0:
                    grid_energy += np.sum(magnitude[:, cx - fx - 2:cx - fx + 3] ** 2)

                # Vertical grid frequencies
                fy = int(round(harmonic * h / period_y))
                if cy + fy + 3 < h:
                    grid_energy += np.sum(magnitude[cy + fy - 2:cy + fy + 3, :] ** 2)
                if cy - fy - 2 >= 0:
                    grid_energy += np.sum(magnitude[cy - fy - 2:cy - fy + 3, :] ** 2)

            modulation[c] = grid_energy / max(total_energy, 1e-10)

        return modulation

    # ------------------------------------------------------------------
    # Modulation-Weighted Grayscale
    # ------------------------------------------------------------------

    def compute_modulation_weighted_gray(
        self,
        image_bgr: np.ndarray,
        modulation: Dict[int, float],
    ) -> np.ndarray:
        """Compute modulation-weighted grayscale for energy map.

        Weights channels proportional to their grid modulation depth,
        so channels where the grid is most visible contribute more
        to the energy map. This improves spatial accuracy of grid
        detection compared to standard luminance conversion.

        Args:
            image_bgr: Input image in BGR format (uint8).
            modulation: Per-channel modulation depths {0: B, 1: G, 2: R}.

        Returns:
            Weighted grayscale image (float64, same H×W as input).
        """
        max_mod = max(modulation.values()) if modulation else 0
        if max_mod < 1e-10:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        weights = np.array([modulation.get(c, 0) / max_mod for c in range(3)])
        # Normalize to sum to 1
        weights /= weights.sum() + 1e-10

        img_f = image_bgr.astype(np.float64)
        weighted_gray = (
            img_f[:, :, 0] * weights[0]
            + img_f[:, :, 1] * weights[1]
            + img_f[:, :, 2] * weights[2]
        )
        return weighted_gray

    # ------------------------------------------------------------------
    # STFT Local Energy Map
    # ------------------------------------------------------------------

    def compute_local_energy_map(
        self,
        image_gray: np.ndarray,
        period_x: int,
        period_y: int,
    ) -> np.ndarray:
        """Compute spatially-varying grid energy via STFT.

        Divides the image into overlapping windows and measures grid
        energy in each window. The result is a smooth energy map
        indicating grid strength at each pixel.

        Args:
            image_gray: Grayscale image (uint8 or float).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.

        Returns:
            Grid energy map (float32, same size as input), values in [0, 1].
        """
        h, w = image_gray.shape
        ws = self.window_size
        hop = self.hop_size

        # Pad for edge windows
        pad = ws // 2
        padded = np.pad(
            image_gray.astype(np.float64),
            ((pad, pad), (pad, pad)),
            mode='reflect',
        )
        hp, wp = padded.shape

        energy_map = np.zeros((hp, wp), dtype=np.float64)
        weight_map = np.zeros((hp, wp), dtype=np.float64)

        hann_2d = np.outer(np.hanning(ws), np.hanning(ws))

        for y0 in range(0, hp - ws + 1, hop):
            for x0 in range(0, wp - ws + 1, hop):
                patch = padded[y0:y0 + ws, x0:x0 + ws]
                local_energy = self._measure_patch_grid_energy(
                    patch, period_x, period_y
                )
                energy_map[y0:y0 + ws, x0:x0 + ws] += local_energy * hann_2d
                weight_map[y0:y0 + ws, x0:x0 + ws] += hann_2d

        # Normalize
        weight_map = np.maximum(weight_map, 1e-10)
        energy_map /= weight_map

        # Crop padding
        energy_map = energy_map[pad:pad + h, pad:pad + w]

        # Normalize to [0, 1]
        emax = energy_map.max()
        if emax > 1e-10:
            energy_map /= emax

        return energy_map.astype(np.float32)

    def _measure_patch_grid_energy(
        self,
        patch: np.ndarray,
        period_x: int,
        period_y: int,
    ) -> float:
        """Measure grid energy in a local patch.

        Args:
            patch: 2D image patch (float64).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.

        Returns:
            Ratio of grid frequency energy to total energy.
        """
        f = np.fft.fft2(patch)
        magnitude = np.abs(np.fft.fftshift(f))
        ph, pw = patch.shape
        cy, cx = ph // 2, pw // 2

        grid_energy = 0.0
        total_energy = np.sum(magnitude ** 2) - magnitude[cy, cx] ** 2 + 1e-10

        n_harmonics = min(3, self.n_harmonics)  # Fewer harmonics for small patches
        for harmonic in range(1, n_harmonics + 1):
            fx = int(round(harmonic * pw / period_x))
            if 1 < fx < pw // 2 - 1:
                for sign in [1, -1]:
                    col = cx + sign * fx
                    if 0 <= col - 1 and col + 2 <= pw:
                        grid_energy += np.sum(magnitude[:, col - 1:col + 2] ** 2)

            fy = int(round(harmonic * ph / period_y))
            if 1 < fy < ph // 2 - 1:
                for sign in [1, -1]:
                    row = cy + sign * fy
                    if 0 <= row - 1 and row + 2 <= ph:
                        grid_energy += np.sum(magnitude[row - 1:row + 2, :] ** 2)

        return grid_energy / total_energy

    # ------------------------------------------------------------------
    # Gaussian Notch Mask
    # ------------------------------------------------------------------

    def create_gaussian_notch_mask(
        self,
        h: int,
        w: int,
        period_x: int,
        period_y: int,
        attenuation: float = 0.0,
    ) -> np.ndarray:
        """Create Gaussian-shaped notch mask for grid frequencies.

        The mask is 1.0 everywhere except near grid harmonic frequencies,
        where it drops toward `attenuation`.

        Args:
            h: Image height.
            w: Image width.
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            attenuation: Minimum value at grid frequencies (0 = full removal).

        Returns:
            Mask array of shape (h, w), float64 in [attenuation, 1.0].
        """
        cy, cx = h // 2, w // 2
        sigma = self.notch_sigma
        mask = np.ones((h, w), dtype=np.float64)

        for harmonic in range(1, self.n_harmonics + 1):
            # Vertical grid -> horizontal frequency lines
            fy = int(round(harmonic * h / period_y))
            if fy < h // 2 - 2:
                for y in range(h):
                    dist_pos = abs(y - (cy + fy))
                    dist_neg = abs(y - (cy - fy))
                    mask[y, :] *= 1 - (1 - attenuation) * np.exp(
                        -dist_pos ** 2 / (2 * sigma ** 2)
                    )
                    mask[y, :] *= 1 - (1 - attenuation) * np.exp(
                        -dist_neg ** 2 / (2 * sigma ** 2)
                    )

            # Horizontal grid -> vertical frequency lines
            fx = int(round(harmonic * w / period_x))
            if fx < w // 2 - 2:
                for x in range(w):
                    dist_pos = abs(x - (cx + fx))
                    dist_neg = abs(x - (cx - fx))
                    mask[:, x] *= 1 - (1 - attenuation) * np.exp(
                        -dist_pos ** 2 / (2 * sigma ** 2)
                    )
                    mask[:, x] *= 1 - (1 - attenuation) * np.exp(
                        -dist_neg ** 2 / (2 * sigma ** 2)
                    )

        return mask

    def create_gaussian_notch_mask_v2(
        self,
        h: int,
        w: int,
        period_x: float,
        period_y: float,
        attenuation: float = 0.0,
        sigma: Optional[float] = None,
        n_harmonics: Optional[int] = None,
    ) -> np.ndarray:
        """Create Gaussian notch mask with float period support and custom params.

        Enhanced version supporting subpixel-accurate period placement and
        configurable notch width/harmonics count.

        Args:
            h: Image height.
            w: Image width.
            period_x: Horizontal grid period (float for subpixel accuracy).
            period_y: Vertical grid period (float for subpixel accuracy).
            attenuation: Minimum value at grid frequencies.
            sigma: Notch width override. If None, uses self.notch_sigma.
            n_harmonics: Number of harmonics override. If None, uses self.n_harmonics.

        Returns:
            Mask array of shape (h, w), float64 in [attenuation, 1.0].
        """
        cy, cx = h / 2.0, w / 2.0
        sig = sigma if sigma is not None else self.notch_sigma
        n_harm = n_harmonics if n_harmonics is not None else self.n_harmonics
        mask = np.ones((h, w), dtype=np.float64)

        yy = np.arange(h, dtype=np.float64)
        xx = np.arange(w, dtype=np.float64)

        for harmonic in range(1, n_harm + 1):
            # Vertical grid -> horizontal frequency lines (float position)
            fy = harmonic * h / period_y
            if fy < h / 2.0 - 2:
                for sign in [1, -1]:
                    fy_pos = cy + sign * fy
                    dist = np.abs(yy - fy_pos)
                    line_notch = 1 - (1 - attenuation) * np.exp(
                        -dist ** 2 / (2 * sig ** 2)
                    )
                    mask *= line_notch[:, np.newaxis]

            # Horizontal grid -> vertical frequency lines (float position)
            fx = harmonic * w / period_x
            if fx < w / 2.0 - 2:
                for sign in [1, -1]:
                    fx_pos = cx + sign * fx
                    dist = np.abs(xx - fx_pos)
                    line_notch = 1 - (1 - attenuation) * np.exp(
                        -dist ** 2 / (2 * sig ** 2)
                    )
                    mask *= line_notch[np.newaxis, :]

        return mask

    # ------------------------------------------------------------------
    # Core: Channel-Adaptive STFT Grid Removal
    # ------------------------------------------------------------------

    def channel_adaptive_stft_removal(
        self,
        image_bgr: np.ndarray,
        period_x: int,
        period_y: int,
        energy_map: np.ndarray,
        modulation_depths: Dict[int, float],
    ) -> np.ndarray:
        """Apply STFT notch filtering with channel-adaptive attenuation.

        For each overlapping window and each channel:
        1. Measure local grid energy (from pre-computed energy map)
        2. Scale attenuation inversely with local energy and channel modulation
        3. Apply Gaussian notch filter in frequency domain
        4. Reconstruct via overlap-add

        Args:
            image_bgr: Input image in BGR format (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            energy_map: Pre-computed local grid energy map (float32, [0, 1]).
            modulation_depths: Per-channel modulation depths.

        Returns:
            Filtered image as uint8 BGR.
        """
        h, w = image_bgr.shape[:2]
        ws = self.window_size
        hop = self.hop_size

        # Compute per-channel attenuation scaling
        max_mod = max(modulation_depths.values()) if modulation_depths else 1.0
        channel_attn_scale = {}
        for c, mod in modulation_depths.items():
            # Stronger modulation -> lower attenuation (more removal)
            # B(14.8%) -> ~0.15, G(7.4%) -> ~0.27, R(4.5%) -> ~0.34
            if self.channel_adaptive:
                channel_attn_scale[c] = self.base_attenuation + 0.25 * (
                    1 - mod / max(max_mod, 1e-10)
                )
            else:
                channel_attn_scale[c] = self.base_attenuation

        # Pad image
        pad_y = max((ws - h % ws) % ws, ws // 2)
        pad_x = max((ws - w % ws) % ws, ws // 2)
        img_padded = np.pad(
            image_bgr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='reflect'
        )
        hp, wp = img_padded.shape[:2]

        # Pad energy map similarly
        energy_padded = np.pad(energy_map, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

        result = np.zeros((hp, wp, 3), dtype=np.float64)
        weight = np.zeros((hp, wp), dtype=np.float64)

        hann_2d = np.outer(np.hanning(ws), np.hanning(ws))

        for y0 in range(0, hp - ws + 1, hop):
            for x0 in range(0, wp - ws + 1, hop):
                # Local energy (average over window)
                local_energy = np.mean(
                    energy_padded[y0:y0 + ws, x0:x0 + ws]
                )

                for c in range(3):
                    patch = img_padded[y0:y0 + ws, x0:x0 + ws, c].astype(
                        np.float64
                    )

                    # Compute adaptive attenuation
                    base_attn = channel_attn_scale.get(c, self.base_attenuation)

                    # Scale by local energy: high energy -> low attenuation
                    # Low energy -> high attenuation (preserve detail)
                    energy_factor = np.clip(local_energy * 2, 0.3, 2.0)
                    adaptive_attn = base_attn / energy_factor
                    adaptive_attn = np.clip(adaptive_attn, 0.02, 0.5)

                    # Apply windowed FFT + notch
                    windowed = patch * hann_2d
                    f = np.fft.fft2(windowed)
                    f_shifted = np.fft.fftshift(f)

                    mask = self.create_gaussian_notch_mask(
                        ws, ws, period_x, period_y, attenuation=adaptive_attn
                    )
                    f_filtered = f_shifted * mask
                    patch_filtered = np.real(
                        np.fft.ifft2(np.fft.ifftshift(f_filtered))
                    )

                    result[y0:y0 + ws, x0:x0 + ws, c] += patch_filtered * hann_2d

                weight[y0:y0 + ws, x0:x0 + ws] += hann_2d ** 2

        # Normalize by overlap weights
        weight = np.maximum(weight, 1e-10)
        for c in range(3):
            result[:, :, c] /= weight

        # Remove padding
        result = result[pad_y:pad_y + h, pad_x:pad_x + w, :]

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Global (non-STFT) Channel-Adaptive Notch
    # ------------------------------------------------------------------

    def channel_adaptive_notch_global(
        self,
        image_bgr: np.ndarray,
        period_x: int,
        period_y: int,
        modulation_depths: Dict[int, float],
    ) -> np.ndarray:
        """Apply global channel-adaptive Gaussian notch filter.

        Faster than STFT version (no windowing), but does not adapt
        to local grid strength. Useful for images with uniform grid.

        Args:
            image_bgr: Input image in BGR format (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            modulation_depths: Per-channel modulation depths.

        Returns:
            Filtered image as uint8 BGR.
        """
        h, w = image_bgr.shape[:2]
        max_mod = max(modulation_depths.values()) if modulation_depths else 1.0

        result = np.zeros_like(image_bgr, dtype=np.float64)

        for c in range(3):
            mod = modulation_depths.get(c, max_mod)
            if self.channel_adaptive:
                attn = self.base_attenuation + 0.25 * (1 - mod / max(max_mod, 1e-10))
            else:
                attn = self.base_attenuation

            channel = image_bgr[:, :, c].astype(np.float64)
            f = np.fft.fft2(channel)
            f_shifted = np.fft.fftshift(f)

            mask = self.create_gaussian_notch_mask(
                h, w, period_x, period_y, attenuation=attn
            )
            f_filtered = f_shifted * mask
            result[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # V2: Point Notch Mask (intersection points only)
    # ------------------------------------------------------------------

    def create_point_notch_mask(
        self,
        h: int,
        w: int,
        period_x: int,
        period_y: int,
        attenuation: float = 0.0,
        point_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Create 2D Gaussian notch mask at intersection points only.

        Unlike create_gaussian_notch_mask which attenuates entire rows/columns,
        this places 2D Gaussian notches ONLY at the intersection points where
        horizontal and vertical grid harmonics meet, preserving non-grid frequencies.

        Args:
            h: Image height.
            w: Image width.
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            attenuation: Minimum value at grid frequencies (0 = full removal).
            point_sigma: Width of 2D Gaussian notch. Defaults to notch_sigma * 1.5.

        Returns:
            Mask array of shape (h, w), float64 in [attenuation, 1.0].
        """
        cy, cx = h // 2, w // 2
        sigma = point_sigma if point_sigma is not None else self.notch_sigma * 1.5

        # Create coordinate grids
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        mask = np.ones((h, w), dtype=np.float64)

        for hx in range(1, self.n_harmonics + 1):
            fx = int(round(hx * w / period_x))
            if fx >= w // 2 - 2:
                continue

            for hy in range(1, self.n_harmonics + 1):
                fy = int(round(hy * h / period_y))
                if fy >= h // 2 - 2:
                    continue

                # Place 2D Gaussian notches at all 4 quadrant intersection points
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        fx_pos = cx + sx * fx
                        fy_pos = cy + sy * fy
                        dist_sq = (yy - fy_pos) ** 2 + (xx - fx_pos) ** 2
                        notch = 1 - (1 - attenuation) * np.exp(-dist_sq / (2 * sigma ** 2))
                        mask *= notch

        # Also place notches at axis-only harmonics (fx, 0) and (0, fy)
        for hx in range(1, self.n_harmonics + 1):
            fx = int(round(hx * w / period_x))
            if fx >= w // 2 - 2:
                continue
            # Points at (cx +/- fx, cy)
            for sx in [1, -1]:
                fx_pos = cx + sx * fx
                fy_pos = cy
                dist_sq = (yy - fy_pos) ** 2 + (xx - fx_pos) ** 2
                notch = 1 - (1 - attenuation) * np.exp(-dist_sq / (2 * sigma ** 2))
                mask *= notch

        for hy in range(1, self.n_harmonics + 1):
            fy = int(round(hy * h / period_y))
            if fy >= h // 2 - 2:
                continue
            # Points at (cx, cy +/- fy)
            for sy in [1, -1]:
                fx_pos = cx
                fy_pos = cy + sy * fy
                dist_sq = (yy - fy_pos) ** 2 + (xx - fx_pos) ** 2
                notch = 1 - (1 - attenuation) * np.exp(-dist_sq / (2 * sigma ** 2))
                mask *= notch

        return mask

    # ------------------------------------------------------------------
    # V2: Adaptive Edge Mask (grid-energy weighted)
    # ------------------------------------------------------------------

    def compute_adaptive_edge_mask(
        self,
        edge_mask: np.ndarray,
        energy_map: np.ndarray,
        grid_weight: float = 2.0,
    ) -> np.ndarray:
        """Compute adaptive edge mask that reduces protection where grid is strong.

        Instead of blanket edge protection, this reduces protection in edge
        regions where grid energy is high (allowing filtering there), while
        preserving edges in low-grid-energy regions.

        Args:
            edge_mask: Original edge protection mask (float32, [0, 1]).
            energy_map: Grid energy map from compute_local_energy_map (float32, [0, 1]).
            grid_weight: Scaling factor for grid energy influence. Higher = less
                edge protection where grid is present.

        Returns:
            Adaptive edge mask (float32, [0, 1]).
        """
        # grid_factor: high grid energy -> lower edge protection
        grid_factor = 1.0 - np.clip(energy_map * grid_weight, 0, 0.9)
        adaptive_mask = edge_mask * grid_factor
        return adaptive_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # V2: Two-Pass Removal (Global point notch + STFT residual)
    # ------------------------------------------------------------------

    def two_pass_removal(
        self,
        image_bgr: np.ndarray,
        period_x: int,
        period_y: int,
        energy_map: np.ndarray,
        modulation_depths: Dict[int, float],
        edge_mask: Optional[np.ndarray] = None,
        edge_preservation: float = 0.5,
        residual_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Two-pass grid removal: global point notch + STFT on residual.

        Pass 1 applies global channel-adaptive POINT notch filtering which is
        safe and preserves PSNR (~34dB) while achieving ~60% grid removal.

        Pass 2 measures residual grid energy on the pass 1 result and applies
        STFT windowed filtering only where residual energy is above threshold.

        This approach combines the best of both methods: global notch's
        high PSNR with STFT's high grid removal rate.

        Args:
            image_bgr: Input image in BGR format (uint8).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            energy_map: Pre-computed local grid energy map (float32, [0, 1]).
            modulation_depths: Per-channel modulation depths.
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
            edge_preservation: How much to preserve edges (0 = ignore, 1 = full).
            residual_threshold: Minimum residual energy to trigger pass 2.

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}
        h, w = image_bgr.shape[:2]
        max_mod = max(modulation_depths.values()) if modulation_depths else 1.0

        # ------------------------------------------------------------------
        # Pass 1: Global channel-adaptive POINT notch
        # ------------------------------------------------------------------
        pass1_result = np.zeros_like(image_bgr, dtype=np.float64)

        for c in range(3):
            mod = modulation_depths.get(c, max_mod)
            if self.channel_adaptive:
                attn = self.base_attenuation + 0.25 * (1 - mod / max(max_mod, 1e-10))
            else:
                attn = self.base_attenuation

            channel = image_bgr[:, :, c].astype(np.float64)
            f = np.fft.fft2(channel)
            f_shifted = np.fft.fftshift(f)
            mask = self.create_point_notch_mask(h, w, period_x, period_y, attenuation=attn)
            f_filtered = f_shifted * mask
            pass1_result[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

        pass1_result = np.clip(pass1_result, 0, 255).astype(np.uint8)
        intermediates['pass1_result'] = pass1_result

        # ------------------------------------------------------------------
        # Measure residual grid energy on pass 1 result
        # ------------------------------------------------------------------
        gray_pass1 = cv2.cvtColor(pass1_result, cv2.COLOR_BGR2GRAY)
        residual_energy = self.compute_local_energy_map(gray_pass1, period_x, period_y)
        intermediates['residual_energy'] = (residual_energy * 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # Pass 2: STFT only on regions with residual energy > threshold
        # ------------------------------------------------------------------
        max_residual = residual_energy.max()

        if max_residual > residual_threshold:
            # Run STFT on pass1 result (not original)
            pass2_result = self.channel_adaptive_stft_removal(
                pass1_result, period_x, period_y, residual_energy, modulation_depths
            )
            intermediates['pass2_result'] = pass2_result

            # Blend pass2 with pass1 based on residual energy
            residual_mask = np.clip(residual_energy / max(max_residual, 1e-10), 0, 1)
            residual_mask_3ch = residual_mask[:, :, np.newaxis]
            result = (
                residual_mask_3ch * pass2_result.astype(np.float32) +
                (1 - residual_mask_3ch) * pass1_result.astype(np.float32)
            )
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            # Skip pass 2 if residual is low enough
            result = pass1_result
            intermediates['pass2_result'] = pass1_result

        # ------------------------------------------------------------------
        # Adaptive edge protection
        # ------------------------------------------------------------------
        if edge_mask is not None and edge_preservation > 0:
            adaptive_edge = self.compute_adaptive_edge_mask(edge_mask, energy_map)
            intermediates['adaptive_edge_mask'] = (adaptive_edge * 255).astype(np.uint8)

            mask_3ch = (adaptive_edge * edge_preservation)[:, :, np.newaxis]
            original_f = image_bgr.astype(np.float32)
            result_f = result.astype(np.float32)
            result = mask_3ch * original_f + (1 - mask_3ch) * result_f
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result, intermediates

    # ------------------------------------------------------------------
    # V2: Full Pipeline with All Improvements
    # ------------------------------------------------------------------

    def process_v2(
        self,
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        edge_preservation: float = 0.5,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Full V2 detect-then-remove pipeline with all improvements.

        V2 improvements over process():
        1. Point notch mask (intersection points only, not full lines)
        2. Two-pass removal (global first, STFT on residual)
        3. Adaptive edge protection (grid-energy weighted)

        Args:
            image_bgr: Input image in BGR format (uint8).
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
                If None, no edge protection is applied.
            edge_preservation: How much to preserve edges (0 = ignore, 1 = full).

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}

        # Step 1: Detect or use configured grid periods
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.period_x > 0 and self.period_y > 0:
            period_x, period_y = self.period_x, self.period_y
        else:
            period_x, period_y = self.detect_grid_periods(gray)

        intermediates['detected_periods'] = np.array(
            [period_x, period_y], dtype=np.int32
        )

        # Step 2: Measure per-channel modulation depth
        modulation = self.measure_channel_modulation(
            image_bgr, period_x, period_y
        )
        intermediates['modulation_depths'] = np.array(
            [modulation.get(c, 0) for c in range(3)], dtype=np.float64
        )

        # Step 3: Compute STFT local energy map (modulation-weighted)
        weighted_gray = self.compute_modulation_weighted_gray(
            image_bgr, modulation
        )
        energy_map = self.compute_local_energy_map(weighted_gray, period_x, period_y)
        intermediates['energy_map'] = (energy_map * 255).astype(np.uint8)

        # Step 4: Two-pass removal with adaptive edge protection
        result, two_pass_intermediates = self.two_pass_removal(
            image_bgr,
            period_x,
            period_y,
            energy_map,
            modulation,
            edge_mask=edge_mask,
            edge_preservation=edge_preservation,
        )

        # Merge intermediates
        intermediates.update(two_pass_intermediates)
        intermediates['result'] = result

        return result, intermediates

    # ------------------------------------------------------------------
    # V3: Edge High-Frequency Extraction
    # ------------------------------------------------------------------

    def extract_edge_highfreq(
        self,
        image_bgr: np.ndarray,
        edge_mask: np.ndarray,
        highpass_sigma: float = 1.5,
        channel_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4),
    ) -> np.ndarray:
        """Extract high-frequency edge content from original image.

        Computes per-channel high-frequency content (original - GaussianBlur),
        masks to edge regions only, and applies channel weights to emphasize
        channels with less grid contamination (R/G over B).

        Args:
            image_bgr: Original input image in BGR format (uint8).
            edge_mask: Edge mask (float32, [0, 1]). 1 = edge region.
            highpass_sigma: Sigma for Gaussian blur used in high-pass filter.
                Tuned to capture brushstroke-scale features (2-3px).
            channel_weights: Per-channel weights (B, G, R). Lower B weight
                because B channel has strongest grid contamination (14.8%).

        Returns:
            High-frequency edge content as float32 BGR, same shape as input.
        """
        result = np.zeros_like(image_bgr, dtype=np.float32)
        for c in range(3):
            channel = image_bgr[:, :, c].astype(np.float32)
            low_freq = cv2.GaussianBlur(channel, (0, 0), highpass_sigma)
            high_freq = channel - low_freq
            result[:, :, c] = high_freq * edge_mask * channel_weights[c]
        return result

    # ------------------------------------------------------------------
    # V3.1: Multi-Scale DoG Edge Extraction with Grid Avoidance
    # ------------------------------------------------------------------

    def extract_edge_highfreq_multiscale(
        self,
        image_bgr: np.ndarray,
        edge_mask: np.ndarray,
        period_x: int,
        period_y: int,
        scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        channel_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4),
    ) -> np.ndarray:
        """Extract high-frequency edge content using multi-scale DoG pyramid
        with grid frequency avoidance.

        Instead of a single-scale Gaussian high-pass (which overlaps with grid
        frequencies at sigma~1.5), this decomposes the signal into octave bands
        via Difference-of-Gaussians and weights each band inversely to its
        overlap with grid harmonic frequencies.

        Grid periods 7-9px correspond to spatial frequencies ~0.11-0.14 cyc/px.
        A DoG band centered near this range gets low weight, while bands at
        finer (calligraphy) and coarser (broad strokes) scales are preserved.

        Args:
            image_bgr: Original input image in BGR format (uint8).
            edge_mask: Edge mask (float32, [0, 1]). 1 = edge region.
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.
            scales: Tuple of sigma values for DoG pyramid bands.
                Each band is DoG(sigma, sigma*2).
            channel_weights: Per-channel weights (B, G, R).

        Returns:
            High-frequency edge content as float32 BGR, same shape as input.
        """
        # Average grid spatial frequency (cycles/pixel)
        avg_grid_freq = 0.5 * (1.0 / period_x + 1.0 / period_y)

        # Compute per-band weight based on distance from grid frequency
        band_weights = []
        for sigma in scales:
            sigma2 = sigma * 2.0
            # DoG peak frequency: 1 / (2*pi*sqrt(sigma1*sigma2))
            center_freq = 1.0 / (2.0 * np.pi * np.sqrt(sigma * sigma2))
            # Distance in log-frequency space (octave distance)
            log_dist = abs(np.log2(center_freq / avg_grid_freq))
            # Gaussian weighting: high when far from grid, low when near
            weight = 1.0 - np.exp(-log_dist ** 2 / (2.0 * 0.5 ** 2))
            band_weights.append(max(weight, 0.05))  # Floor at 5%

        # Normalize
        total_w = sum(band_weights)
        band_weights = [w / total_w for w in band_weights]

        result = np.zeros_like(image_bgr, dtype=np.float32)

        for c in range(3):
            channel = image_bgr[:, :, c].astype(np.float32)
            band_sum = np.zeros_like(channel)

            for sigma, bw in zip(scales, band_weights):
                sigma2 = sigma * 2.0
                blur1 = cv2.GaussianBlur(channel, (0, 0), sigma)
                blur2 = cv2.GaussianBlur(channel, (0, 0), sigma2)
                dog_band = blur1 - blur2
                band_sum += dog_band * bw

            result[:, :, c] = band_sum * edge_mask * channel_weights[c]

        return result

    # ------------------------------------------------------------------
    # V3.3: Intensity-Based Ink Protection
    # ------------------------------------------------------------------

    @staticmethod
    def compute_intensity_protection_mask(
        image_gray: np.ndarray,
        threshold: float = 60.0,
        steepness: float = 0.08,
        blur_sigma: float = 3.0,
    ) -> np.ndarray:
        """Compute intensity-based protection mask for ink line preservation.

        Uses a sigmoid function to map pixel intensity to protection weight.
        Dark pixels (ink lines) get high protection (~1.0), bright pixels
        (background where grid is visible) get low protection (~0.0).

        Args:
            image_gray: Grayscale image (uint8, range 0-255).
            threshold: Sigmoid center point (pixel value). Pixels darker
                than this are protected. Default 60 covers ink lines (0-40)
                with margin.
            steepness: Sigmoid steepness parameter. Controls transition width.
                0.08 gives ~25 pixel-value transition zone.
            blur_sigma: Gaussian blur sigma for mask smoothing. Prevents
                hard mask edges from creating artifacts.

        Returns:
            Protection mask as float32 array, same shape as input.
            Values in [0, 1] where 1 = fully protected (dark/ink),
            0 = no protection (bright/background).
        """
        gray_f = image_gray.astype(np.float32)
        mask = 1.0 / (1.0 + np.exp(steepness * (gray_f - threshold)))
        if blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # V3: Grid Removal from Arbitrary Content
    # ------------------------------------------------------------------

    def filter_grid_from_content(
        self,
        content_bgr: np.ndarray,
        period_x: int,
        period_y: int,
        notch_attenuation: float = 0.02,
    ) -> np.ndarray:
        """Remove grid harmonics from arbitrary content using point notch filter.

        Applies the same point notch mask used in V2 to filter grid frequencies
        from any content (e.g., extracted edge high-frequency signals).

        Args:
            content_bgr: Content to filter (float32 or float64 BGR, 3-channel).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            notch_attenuation: Attenuation at grid frequencies. Very low (0.02)
                to nearly fully remove grid residuals from edge content.

        Returns:
            Filtered content as float32 BGR, same shape as input.
        """
        h, w = content_bgr.shape[:2]
        result = np.zeros_like(content_bgr, dtype=np.float32)
        mask = self.create_point_notch_mask(
            h, w, period_x, period_y, attenuation=notch_attenuation
        )
        for c in range(3):
            f = np.fft.fft2(content_bgr[:, :, c].astype(np.float64))
            f_shifted = np.fft.fftshift(f)
            f_filtered = f_shifted * mask
            result[:, :, c] = np.real(
                np.fft.ifft2(np.fft.ifftshift(f_filtered))
            ).astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # V3: Full Pipeline (Grid Removal + Edge Restoration)
    # ------------------------------------------------------------------

    def process_v3(
        self,
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        edge_strength: float = 0.7,
        aggressive_attenuation: float = 0.05,
        highpass_sigma: float = 1.5,
        channel_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4),
        apply_diffusion: bool = False,
        multiscale_edge: bool = False,
        adaptive_weights: bool = False,
        energy_normalize: bool = False,
        # V3.2 parameters
        notch_sigma_override: Optional[float] = None,
        n_harmonics_override: Optional[int] = None,
        n_passes: int = 1,
        subpixel_period: bool = False,
        edge_hf_notch_attenuation: float = 0.02,
        # V3.3: intensity-based ink protection
        intensity_protection: bool = False,
        intensity_threshold: float = 60.0,
        intensity_steepness: float = 0.08,
        intensity_blur_sigma: float = 3.0,
        intensity_protection_strength: float = 0.8,
        # V3.4: energy-based grid protection (uses pre-computed energy_map)
        energy_protection: bool = False,
        energy_protection_strength: float = 0.85,
        energy_protection_blur: float = 2.0,
        energy_protection_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """V3 Grid Removal + Edge Restoration pipeline.

        Dual-domain approach that decouples grid removal from edge preservation:
        1. Aggressively remove grid (high GR, accepting edge damage)
        2. Extract high-frequency edge content from original
        3. Filter grid harmonics from the extracted edge content
        4. Add cleaned edge content back to the filtered result

        This breaks the GR-vs-PSNR tradeoff by restoring edge detail
        AFTER grid removal rather than protecting edges DURING removal.

        V3.1 improvements (opt-in via flags):
        - multiscale_edge: Multi-scale DoG pyramid with grid frequency avoidance
          instead of single-scale Gaussian HP. Avoids the scale overlap between
          grid (7-9px period) and edge features.
        - adaptive_weights: Compute channel weights from measured modulation
          depths instead of using fixed (0.2, 0.4, 0.4). Channels with less
          grid contamination contribute more edge information.
        - energy_normalize: Complement-based blending that measures residual
          HF energy in the filtered image and adjusts edge restoration strength
          to prevent double-counting (overshoot/halo artifacts).

        Args:
            image_bgr: Input image in BGR format (uint8).
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
                If None, a simple gradient-based mask is computed.
            edge_strength: Strength of edge restoration (0 = no restoration,
                1 = full restoration). Controls how much cleaned edge HF
                is added back.
            aggressive_attenuation: Attenuation for aggressive grid removal
                in step 1. Lower = more aggressive removal.
            highpass_sigma: Sigma for high-pass filter in edge extraction.
            channel_weights: Per-channel weights for edge extraction (B, G, R).
                Ignored when adaptive_weights=True.
            apply_diffusion: If True, apply Perona-Malik anisotropic diffusion
                as final post-processing step.
            multiscale_edge: If True, use multi-scale DoG extraction with
                grid frequency avoidance instead of single-scale high-pass.
            adaptive_weights: If True, compute channel weights from measured
                modulation depths. Overrides channel_weights parameter.
            energy_normalize: If True, use complement-based energy normalization
                to prevent double-counting of edge content.

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}

        # Step 1: Detect or use configured grid periods
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.period_x > 0 and self.period_y > 0:
            if subpixel_period:
                period_x_f, period_y_f = self.detect_grid_periods_subpixel(gray)
                # Use subpixel if close to configured value, else use configured
                if abs(period_x_f - self.period_x) < 1.0:
                    period_x_f = period_x_f
                else:
                    period_x_f = float(self.period_x)
                if abs(period_y_f - self.period_y) < 1.0:
                    period_y_f = period_y_f
                else:
                    period_y_f = float(self.period_y)
            else:
                period_x_f = float(self.period_x)
                period_y_f = float(self.period_y)
            period_x = self.period_x
            period_y = self.period_y
        else:
            period_x, period_y = self.detect_grid_periods(gray)
            period_x_f = float(period_x)
            period_y_f = float(period_y)
            if subpixel_period:
                period_x_f, period_y_f = self.detect_grid_periods_subpixel(gray)

        intermediates['detected_periods'] = np.array(
            [period_x_f, period_y_f], dtype=np.float64
        )

        # Step 2: Measure per-channel modulation depth
        modulation = self.measure_channel_modulation(
            image_bgr, period_x, period_y
        )
        intermediates['modulation_depths'] = np.array(
            [modulation.get(c, 0) for c in range(3)], dtype=np.float64
        )

        # Step 3a: Compute adaptive channel weights if enabled
        if adaptive_weights and modulation:
            max_mod = max(modulation.values())
            if max_mod > 1e-10:
                raw_weights = []
                for c in range(3):
                    mod = modulation.get(c, 0)
                    # Less contaminated channels get higher weight
                    # Cap reduction at 80% to keep all channels contributing
                    raw_w = 1.0 - (mod / max_mod) * 0.8
                    raw_weights.append(raw_w)
                total_w = sum(raw_weights) + 1e-10
                channel_weights = tuple(w / total_w for w in raw_weights)
            intermediates['adaptive_channel_weights'] = np.array(
                channel_weights, dtype=np.float64
            )

        # Step 3b: Compute STFT local energy map (modulation-weighted)
        weighted_gray = self.compute_modulation_weighted_gray(
            image_bgr, modulation
        )
        energy_map = self.compute_local_energy_map(weighted_gray, period_x, period_y)
        intermediates['energy_map'] = (energy_map * 255).astype(np.uint8)

        # Step 4: Aggressive grid removal (multi-pass support)
        saved_attn = self.base_attenuation
        saved_sigma = self.notch_sigma
        saved_harmonics = self.n_harmonics

        # Apply overrides
        if notch_sigma_override is not None:
            self.notch_sigma = notch_sigma_override
        if n_harmonics_override is not None:
            self.n_harmonics = n_harmonics_override
        self.base_attenuation = aggressive_attenuation

        filtered = image_bgr
        for pass_idx in range(n_passes):
            if pass_idx > 0:
                # Widen notch slightly for subsequent passes to catch residuals
                self.notch_sigma = (notch_sigma_override or saved_sigma) + pass_idx * 0.5
            filtered, v2_intermediates = self.process_v2(
                filtered,
                edge_mask=None,  # No edge protection - remove everything
                edge_preservation=0.0,
            )

        # Restore original settings
        self.base_attenuation = saved_attn
        self.notch_sigma = saved_sigma
        self.n_harmonics = saved_harmonics
        intermediates['aggressive_filtered'] = filtered
        intermediates.update({
            f'v2_{k}': v for k, v in v2_intermediates.items()
        })

        # Step 4b: Intensity-based protection (ink line preservation)
        if intensity_protection:
            i_mask = self.compute_intensity_protection_mask(
                gray,
                threshold=intensity_threshold,
                steepness=intensity_steepness,
                blur_sigma=intensity_blur_sigma,
            )
            prot_mask_3ch = (i_mask * intensity_protection_strength)[:, :, np.newaxis]
            original_f = image_bgr.astype(np.float32)
            filtered_f = filtered.astype(np.float32)
            filtered = np.clip(
                prot_mask_3ch * original_f + (1 - prot_mask_3ch) * filtered_f,
                0, 255,
            ).astype(np.uint8)
            intermediates['intensity_protection_mask'] = (i_mask * 255).astype(np.uint8)

        # Step 4c: Energy-based protection (grid-confidence mask)
        if energy_protection:
            # energy_map: [0,1], 1 = strong grid.
            if energy_protection_threshold > 0:
                # Hard threshold: only allow filtering where grid energy > threshold
                e_prot = (energy_map < energy_protection_threshold).astype(np.float32)
            else:
                # Linear inversion (original behavior)
                e_prot = 1.0 - energy_map
            if energy_protection_blur > 0:
                e_prot = cv2.GaussianBlur(e_prot, (0, 0), energy_protection_blur)
            e_prot_3ch = (e_prot * energy_protection_strength)[:, :, np.newaxis]
            original_f = image_bgr.astype(np.float32)
            filtered_f = filtered.astype(np.float32)
            filtered = np.clip(
                e_prot_3ch * original_f + (1 - e_prot_3ch) * filtered_f,
                0, 255,
            ).astype(np.uint8)
            intermediates['energy_protection_mask'] = (e_prot * 255).astype(np.uint8)

        # Step 5: Compute edge mask if not provided
        if edge_mask is None:
            # Simple gradient-based edge mask
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_mask = np.clip(grad_mag / (grad_mag.max() + 1e-10), 0, 1).astype(
                np.float32
            )
        intermediates['edge_mask'] = (edge_mask * 255).astype(np.uint8)

        # Step 6: Extract high-frequency edge content from original
        if multiscale_edge:
            edge_hf = self.extract_edge_highfreq_multiscale(
                image_bgr, edge_mask,
                period_x=period_x, period_y=period_y,
                channel_weights=channel_weights,
            )
        else:
            edge_hf = self.extract_edge_highfreq(
                image_bgr, edge_mask,
                highpass_sigma=highpass_sigma,
                channel_weights=channel_weights,
            )
        intermediates['edge_hf'] = np.clip(
            edge_hf + 128, 0, 255
        ).astype(np.uint8)

        # Step 7: Filter grid harmonics from edge HF content
        edge_hf_clean = self.filter_grid_from_content(
            edge_hf, period_x, period_y,
            notch_attenuation=edge_hf_notch_attenuation,
        )
        intermediates['edge_hf_clean'] = np.clip(
            edge_hf_clean + 128, 0, 255
        ).astype(np.uint8)

        # Step 8: Composite: filtered + cleaned edge HF
        filtered_f = filtered.astype(np.float32)

        if energy_normalize:
            # Energy-normalized composition: measure residual HF in filtered
            # at edge locations to avoid double-counting with restored edges.
            residual_hf = np.zeros_like(filtered_f)
            for c in range(3):
                low = cv2.GaussianBlur(filtered_f[:, :, c], (0, 0), 1.5)
                residual_hf[:, :, c] = filtered_f[:, :, c] - low

            # Per-pixel complement weight: reduce restoration where residual
            # edge content already exists in the filtered image.
            edge_mask_3ch = edge_mask[:, :, np.newaxis]
            residual_energy = np.sum(
                residual_hf ** 2 * edge_mask_3ch, axis=2, keepdims=True
            )
            restore_energy = np.sum(
                edge_hf_clean ** 2, axis=2, keepdims=True
            )
            complement_weight = 1.0 - residual_energy / (
                residual_energy + restore_energy + 1e-10
            )
            complement_weight = np.clip(complement_weight, 0.2, 1.0)
            result = filtered_f + edge_hf_clean * edge_strength * complement_weight

            intermediates['complement_weight'] = np.clip(
                complement_weight[:, :, 0] * 255, 0, 255
            ).astype(np.uint8)
        else:
            result = filtered_f + edge_hf_clean * edge_strength

        # Step 9: Optional Perona-Malik anisotropic diffusion
        if apply_diffusion:
            result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            result = self._perona_malik_diffusion(result_uint8).astype(np.float32)

        result = np.clip(result, 0, 255).astype(np.uint8)
        intermediates['result'] = result

        return result, intermediates

    @staticmethod
    def _perona_malik_diffusion(
        image_bgr: np.ndarray,
        iterations: int = 10,
        kappa: float = 30.0,
        gamma: float = 0.1,
    ) -> np.ndarray:
        """Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.

        Smooths flat regions while preserving edges, useful as final
        post-processing to reduce ringing artifacts from frequency filtering.

        Args:
            image_bgr: Input image (uint8 BGR).
            iterations: Number of diffusion iterations.
            kappa: Edge stopping parameter. Higher = more smoothing across edges.
            gamma: Diffusion rate (stability requires gamma <= 0.25).

        Returns:
            Diffused image as uint8 BGR.
        """
        img = image_bgr.astype(np.float64)

        for _ in range(iterations):
            # Compute gradients in 4 directions
            dn = np.roll(img, -1, axis=0) - img  # North
            ds = np.roll(img, 1, axis=0) - img   # South
            de = np.roll(img, -1, axis=1) - img   # East
            dw = np.roll(img, 1, axis=1) - img    # West

            # Perona-Malik edge stopping function (option 1)
            cn = np.exp(-(dn / kappa) ** 2)
            cs = np.exp(-(ds / kappa) ** 2)
            ce = np.exp(-(de / kappa) ** 2)
            cw = np.exp(-(dw / kappa) ** 2)

            img += gamma * (cn * dn + cs * ds + ce * de + cw * dw)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # V3.5 Experiment 3: Content-Gradient Edge Mask
    # ------------------------------------------------------------------

    @staticmethod
    def compute_content_gradient_mask(
        image_gray: np.ndarray,
        period_x: int,
        period_y: int,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Compute edge mask using content gradient with grid-period smoothing.

        Smooths the gradient magnitude with a kernel larger than grid period
        so that periodic grid gradients cancel out while non-periodic content
        gradients survive.

        Args:
            image_gray: Grayscale image (uint8 or float).
            period_x: Horizontal grid period in pixels.
            period_y: Vertical grid period in pixels.
            strength: Mask strength scaling (0-2). Default 1.0.

        Returns:
            Content gradient mask (float32, [0, 1]). 1 = strong content edge.
        """
        gray = image_gray.astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Kernel size = 2 * max(period) + 1 to span at least one full grid cycle
        k = 2 * max(period_x, period_y) + 1
        if k % 2 == 0:
            k += 1

        # Gaussian blur averages out periodic grid gradients
        content_grad = cv2.GaussianBlur(grad_mag, (k, k), 0)

        # Normalize to [0, 1]
        gmax = content_grad.max()
        if gmax > 1e-10:
            content_grad /= gmax

        content_grad = np.clip(content_grad * strength, 0.0, 1.0)
        return content_grad.astype(np.float32)

    # ------------------------------------------------------------------
    # V3.5 Experiment 2: Period-Folded Grid Template Estimation & Division
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_grid_template_2d(
        image_gray: np.ndarray,
        period_x: int,
        period_y: int,
        flat_threshold: float = 10.0,
    ) -> np.ndarray:
        """Estimate 2D grid template by period-folding (averaging over cycles).

        In flat (low-variance) regions, the only pattern is the grid itself.
        By averaging many such regions at (y % Py, x % Px), random content
        cancels and only the periodic grid template remains.

        Args:
            image_gray: Grayscale image (uint8 or float).
            period_x: Horizontal grid period.
            period_y: Vertical grid period.
            flat_threshold: Maximum local std-dev to consider a pixel "flat".

        Returns:
            Grid template of shape (period_y, period_x) as float64.
            Values represent multiplicative grid factor (centered around 1.0).
        """
        gray = image_gray.astype(np.float64)
        h, w = gray.shape

        # Compute local standard deviation to find flat regions
        kernel_size = max(period_x, period_y) * 2 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        local_sq = cv2.GaussianBlur(gray ** 2, (kernel_size, kernel_size), 0)
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))

        flat_mask = local_std < flat_threshold

        # Accumulate period-folded ratio: pixel / local_mean
        template_sum = np.zeros((period_y, period_x), dtype=np.float64)
        template_count = np.zeros((period_y, period_x), dtype=np.float64)

        # Vectorized period folding
        yy, xx = np.mgrid[0:h, 0:w]
        fy = yy % period_y
        fx = xx % period_x

        safe_mean = np.maximum(local_mean, 1.0)
        ratio = gray / safe_mean  # ~1.0 + grid modulation

        for py in range(period_y):
            for px in range(period_x):
                cell_mask = flat_mask & (fy == py) & (fx == px)
                if cell_mask.any():
                    template_sum[py, px] = ratio[cell_mask].mean()
                    template_count[py, px] = cell_mask.sum()

        # Fill cells with no flat pixels using overall mean
        valid = template_count > 0
        if valid.any():
            overall_mean = template_sum[valid].mean()
        else:
            overall_mean = 1.0
        template_sum[~valid] = overall_mean

        return template_sum

    def process_multiplicative(
        self,
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        flat_threshold: float = 10.0,
        residual_notch: bool = True,
        residual_attenuation: float = 0.10,
        residual_sigma: float = 1.5,
        edge_strength: float = 0.5,
        # V3.2-style params for residual pass
        notch_sigma_override: Optional[float] = None,
        n_harmonics_override: Optional[int] = None,
        subpixel_period: bool = False,
        content_gradient_mask: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """V3.5 Multiplicative grid removal via period-folded template division.

        Pipeline:
        1. Detect grid periods
        2. Estimate multiplicative grid template via period folding
        3. Tile template to image size and divide
        4. Optionally apply gentle residual notch filtering
        5. Edge restoration

        Args:
            image_bgr: Input image in BGR format (uint8).
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
            flat_threshold: Local std threshold for flat region detection.
            residual_notch: Whether to apply residual notch after division.
            residual_attenuation: Attenuation for residual notch pass.
            residual_sigma: Sigma for residual notch pass.
            edge_strength: Strength of edge restoration.
            notch_sigma_override: Override notch sigma for residual pass.
            n_harmonics_override: Override n_harmonics for residual pass.
            subpixel_period: Use subpixel period detection.
            content_gradient_mask: Use content gradient mask instead of Sobel.

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect periods
        if self.period_x > 0 and self.period_y > 0:
            period_x, period_y = self.period_x, self.period_y
        else:
            period_x, period_y = self.detect_grid_periods(gray)

        intermediates['detected_periods'] = np.array(
            [period_x, period_y], dtype=np.int32
        )

        # Step 2: Estimate grid template
        template = self.estimate_grid_template_2d(
            gray, period_x, period_y, flat_threshold=flat_threshold,
        )
        intermediates['grid_template'] = (
            np.clip((template - 0.8) * 255 / 0.4, 0, 255).astype(np.uint8)
        )

        # Step 3: Tile template and divide
        h, w = image_bgr.shape[:2]
        tile_y = (h + period_y - 1) // period_y
        tile_x = (w + period_x - 1) // period_x
        tiled = np.tile(template, (tile_y, tile_x))[:h, :w]

        # Prevent division by zero
        tiled = np.maximum(tiled, 0.5)

        # Apply division per channel
        divided = np.zeros_like(image_bgr, dtype=np.float64)
        for c in range(3):
            divided[:, :, c] = image_bgr[:, :, c].astype(np.float64) / tiled

        divided = np.clip(divided, 0, 255).astype(np.uint8)
        intermediates['divided_result'] = divided

        # Step 4: Optional residual notch filtering
        if residual_notch:
            saved_sigma = self.notch_sigma
            saved_harmonics = self.n_harmonics
            saved_attn = self.base_attenuation

            self.notch_sigma = notch_sigma_override or residual_sigma
            if n_harmonics_override is not None:
                self.n_harmonics = n_harmonics_override
            self.base_attenuation = residual_attenuation

            filtered, _ = self.process_v2(
                divided, edge_mask=None, edge_preservation=0.0,
            )

            self.notch_sigma = saved_sigma
            self.n_harmonics = saved_harmonics
            self.base_attenuation = saved_attn
        else:
            filtered = divided

        intermediates['filtered'] = filtered

        # Step 5: Edge restoration
        if edge_mask is None:
            if content_gradient_mask:
                edge_mask = self.compute_content_gradient_mask(
                    gray, period_x, period_y,
                )
            else:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                edge_mask = np.clip(
                    grad_mag / (grad_mag.max() + 1e-10), 0, 1,
                ).astype(np.float32)
        intermediates['edge_mask'] = (edge_mask * 255).astype(np.uint8)

        # Extract and clean edge HF
        edge_hf = self.extract_edge_highfreq(
            image_bgr, edge_mask, highpass_sigma=1.5,
        )
        edge_hf_clean = self.filter_grid_from_content(
            edge_hf, period_x, period_y, notch_attenuation=0.02,
        )

        result = filtered.astype(np.float32) + edge_hf_clean * edge_strength
        result = np.clip(result, 0, 255).astype(np.uint8)
        intermediates['result'] = result

        return result, intermediates

    # ------------------------------------------------------------------
    # V3.5 Experiment 1: Log-Domain Notch Filtering
    # ------------------------------------------------------------------

    def process_v3_log(
        self,
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        epsilon: float = 3.0,
        edge_strength: float = 0.6,
        aggressive_attenuation: float = 0.02,
        # Pass-through V3.2 params
        notch_sigma_override: Optional[float] = None,
        n_harmonics_override: Optional[int] = None,
        n_passes: int = 1,
        subpixel_period: bool = True,
        edge_hf_notch_attenuation: float = 0.012,
        multiscale_edge: bool = True,
        adaptive_weights: bool = True,
        energy_normalize: bool = True,
        content_gradient_mask: bool = False,
        channel_weights: Tuple[float, float, float] = (0.2, 0.4, 0.4),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """V3.5 Log-domain notch filtering for multiplicative grid removal.

        Converts the image to log domain where multiplicative grid becomes
        additive, applies FFT notch filtering ENTIRELY IN FLOAT64 (no uint8
        quantization), then converts back via exp.

        This eliminates quantization artifacts that were caused by:
        1. Converting log-domain [0,255] to uint8 before filtering
        2. exp() amplifying those 1/256 quantization errors

        Math:
            I_obs = I_clean * (1 + m * grid)
            log(I_obs + eps) = log(I_clean + eps) + log(1 + m*grid)
                             ≈ log(I_clean + eps) + m*grid  (for small m)

        Args:
            image_bgr: Input image in BGR format (uint8).
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
            epsilon: Offset to prevent log(0). Controls noise amplification
                in dark regions. Higher = safer but less effective in shadows.
            edge_strength: Strength of edge restoration.
            aggressive_attenuation: Attenuation for aggressive grid removal.
            notch_sigma_override: Override notch sigma.
            n_harmonics_override: Override n_harmonics.
            n_passes: Number of notch filtering passes.
            subpixel_period: Use subpixel period detection.
            edge_hf_notch_attenuation: Attenuation for edge HF notch.
            multiscale_edge: Use multi-scale DoG edge extraction.
            adaptive_weights: Use adaptive channel weights.
            energy_normalize: Use energy-normalized composition.
            content_gradient_mask: Use content gradient mask.
            channel_weights: Per-channel weights for edge extraction.

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}

        # ================================================================
        # Step 1: Log domain conversion (KEEP AS FLOAT64 - no uint8!)
        # ================================================================
        img_f = image_bgr.astype(np.float64)
        log_img = np.log(img_f + epsilon)

        # Scale to [0, 255] range for consistency but STAY IN FLOAT64
        log_min = log_img.min()
        log_max = log_img.max()
        log_range = log_max - log_min
        if log_range < 1e-10:
            log_range = 1.0
        log_scaled = (log_img - log_min) / log_range * 255.0
        # NO np.uint8 conversion here! This is the key fix.

        intermediates['log_image'] = np.clip(log_scaled, 0, 255).astype(np.uint8)

        # ================================================================
        # Step 2: Period detection (from original domain for stability)
        # ================================================================
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if self.period_x > 0 and self.period_y > 0:
            if subpixel_period:
                period_x_f, period_y_f = self.detect_grid_periods_subpixel(gray)
                # Use subpixel if close to configured value
                if abs(period_x_f - self.period_x) >= 1.0:
                    period_x_f = float(self.period_x)
                if abs(period_y_f - self.period_y) >= 1.0:
                    period_y_f = float(self.period_y)
            else:
                period_x_f = float(self.period_x)
                period_y_f = float(self.period_y)
            period_x = self.period_x
            period_y = self.period_y
        else:
            period_x, period_y = self.detect_grid_periods(gray)
            period_x_f = float(period_x)
            period_y_f = float(period_y)
            if subpixel_period:
                period_x_f, period_y_f = self.detect_grid_periods_subpixel(gray)

        intermediates['detected_periods'] = np.array(
            [period_x_f, period_y_f], dtype=np.float64
        )

        # ================================================================
        # Step 3: Measure modulation and compute adaptive channel weights
        # ================================================================
        # Use uint8 version for modulation measurement (expects uint8)
        log_uint8_for_modulation = np.clip(log_scaled, 0, 255).astype(np.uint8)
        modulation = self.measure_channel_modulation(
            log_uint8_for_modulation, period_x, period_y
        )
        intermediates['modulation_depths'] = np.array(
            [modulation.get(c, 0) for c in range(3)], dtype=np.float64
        )

        if adaptive_weights and modulation:
            max_mod = max(modulation.values())
            if max_mod > 1e-10:
                raw_weights = []
                for c in range(3):
                    mod = modulation.get(c, 0)
                    raw_w = 1.0 - (mod / max_mod) * 0.8
                    raw_weights.append(raw_w)
                total_w = sum(raw_weights) + 1e-10
                channel_weights = tuple(w / total_w for w in raw_weights)
            intermediates['adaptive_channel_weights'] = np.array(
                channel_weights, dtype=np.float64
            )

        # ================================================================
        # Step 4: Multi-pass notch filtering in LOG domain (FLOAT64!)
        # ================================================================
        saved_attn = self.base_attenuation
        saved_sigma = self.notch_sigma
        saved_harmonics = self.n_harmonics

        if notch_sigma_override is not None:
            self.notch_sigma = notch_sigma_override
        if n_harmonics_override is not None:
            self.n_harmonics = n_harmonics_override
        self.base_attenuation = aggressive_attenuation

        h, w = log_scaled.shape[:2]
        max_mod = max(modulation.values()) if modulation else 1.0
        filtered_log = np.zeros_like(log_scaled, dtype=np.float64)

        for pass_idx in range(n_passes):
            if pass_idx > 0:
                # Widen notch slightly for subsequent passes
                self.notch_sigma = (notch_sigma_override or saved_sigma) + pass_idx * 0.5

            src = log_scaled if pass_idx == 0 else filtered_log

            for c in range(3):
                # Channel-adaptive attenuation
                mod = modulation.get(c, max_mod)
                if self.channel_adaptive:
                    attn = self.base_attenuation + 0.25 * (1 - mod / max(max_mod, 1e-10))
                else:
                    attn = self.base_attenuation

                # FFT filtering on float64 channel directly
                channel = src[:, :, c]  # Already float64
                f = np.fft.fft2(channel)
                f_shifted = np.fft.fftshift(f)
                mask = self.create_point_notch_mask(h, w, period_x, period_y, attenuation=attn)
                f_filtered = f_shifted * mask
                filtered_log[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

        # Restore settings
        self.base_attenuation = saved_attn
        self.notch_sigma = saved_sigma
        self.n_harmonics = saved_harmonics

        intermediates['aggressive_filtered_log'] = np.clip(filtered_log, 0, 255).astype(np.uint8)

        # ================================================================
        # Step 4b: STFT residual pass on remaining grid artifacts
        # ================================================================
        # Compute residual energy on pass1 result
        gray_filtered_log = np.zeros((h, w), dtype=np.float64)
        for c in range(3):
            gray_filtered_log += filtered_log[:, :, c] * [0.114, 0.587, 0.299][c]
        gray_filtered_uint8 = np.clip(gray_filtered_log, 0, 255).astype(np.uint8)
        residual_energy = self.compute_local_energy_map(gray_filtered_uint8, period_x, period_y)
        intermediates['residual_energy'] = (residual_energy * 255).astype(np.uint8)

        max_residual = residual_energy.max()
        if max_residual > 0.3:  # Same threshold as two_pass_removal
            # STFT pass on log-domain float64 data (key: no uint8 conversion!)
            ws = self.window_size
            hop_size = self.hop_size

            # Re-apply saved settings with aggressive attenuation
            saved_attn2 = self.base_attenuation
            self.base_attenuation = aggressive_attenuation

            # Pad
            pad_y = max((ws - h % ws) % ws, ws // 2)
            pad_x = max((ws - w % ws) % ws, ws // 2)
            img_padded = np.pad(filtered_log, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='reflect')
            hp, wp = img_padded.shape[:2]
            energy_padded = np.pad(residual_energy, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')

            stft_result = np.zeros((hp, wp, 3), dtype=np.float64)
            stft_weight = np.zeros((hp, wp), dtype=np.float64)
            hann_2d = np.outer(np.hanning(ws), np.hanning(ws))

            max_mod = max(modulation.values()) if modulation else 1.0

            for y0 in range(0, hp - ws + 1, hop_size):
                for x0 in range(0, wp - ws + 1, hop_size):
                    local_energy = np.mean(energy_padded[y0:y0 + ws, x0:x0 + ws])

                    for c in range(3):
                        patch = img_padded[y0:y0 + ws, x0:x0 + ws, c]  # Already float64

                        # Channel-adaptive attenuation
                        mod = modulation.get(c, max_mod)
                        if self.channel_adaptive:
                            base_attn = aggressive_attenuation + 0.25 * (1 - mod / max(max_mod, 1e-10))
                        else:
                            base_attn = aggressive_attenuation

                        energy_factor = np.clip(local_energy * 2, 0.3, 2.0)
                        adaptive_attn = base_attn / energy_factor
                        adaptive_attn = np.clip(adaptive_attn, 0.02, 0.5)

                        windowed = patch * hann_2d
                        f = np.fft.fft2(windowed)
                        f_shifted = np.fft.fftshift(f)
                        mask = self.create_gaussian_notch_mask(ws, ws, period_x, period_y, attenuation=adaptive_attn)
                        f_filtered = f_shifted * mask
                        patch_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))

                        stft_result[y0:y0 + ws, x0:x0 + ws, c] += patch_filtered * hann_2d

                    stft_weight[y0:y0 + ws, x0:x0 + ws] += hann_2d ** 2

            stft_weight = np.maximum(stft_weight, 1e-10)
            for c in range(3):
                stft_result[:, :, c] /= stft_weight

            stft_result = stft_result[pad_y:pad_y + h, pad_x:pad_x + w, :]

            self.base_attenuation = saved_attn2

            # Blend STFT result with pass1 result based on residual energy
            residual_mask = np.clip(residual_energy / max(max_residual, 1e-10), 0, 1)
            residual_mask_3ch = residual_mask[:, :, np.newaxis]
            filtered_log = residual_mask_3ch * stft_result + (1 - residual_mask_3ch) * filtered_log

            intermediates['stft_pass_applied'] = np.array([1.0])

        # ================================================================
        # Step 5: Convert back from log domain (FLOAT64 throughout!)
        # ================================================================
        log_filtered = filtered_log / 255.0 * log_range + log_min
        result_f = np.exp(log_filtered) - epsilon

        # ================================================================
        # Step 6: Edge restoration from ORIGINAL domain
        # ================================================================
        if edge_mask is None:
            if content_gradient_mask:
                edge_mask = self.compute_content_gradient_mask(
                    gray, period_x, period_y,
                )
            else:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                edge_mask = np.clip(
                    grad_mag / (grad_mag.max() + 1e-10), 0, 1,
                ).astype(np.float32)

        intermediates['edge_mask'] = (edge_mask * 255).astype(np.uint8)

        # Extract edge HF from original image
        if multiscale_edge:
            edge_hf = self.extract_edge_highfreq_multiscale(
                image_bgr, edge_mask,
                period_x=period_x, period_y=period_y,
                channel_weights=channel_weights,
            )
        else:
            edge_hf = self.extract_edge_highfreq(
                image_bgr, edge_mask,
                highpass_sigma=1.5,
                channel_weights=channel_weights,
            )

        intermediates['edge_hf'] = np.clip(edge_hf + 128, 0, 255).astype(np.uint8)

        # Filter grid from edge HF
        edge_hf_clean = self.filter_grid_from_content(
            edge_hf, period_x, period_y,
            notch_attenuation=edge_hf_notch_attenuation,
        )
        intermediates['edge_hf_clean'] = np.clip(edge_hf_clean + 128, 0, 255).astype(np.uint8)

        # ================================================================
        # Step 7: Composite filtered + edge HF
        # ================================================================
        if energy_normalize:
            # Energy-normalized composition
            residual_hf = np.zeros_like(result_f)
            for c in range(3):
                low = cv2.GaussianBlur(result_f[:, :, c].astype(np.float32), (0, 0), 1.5)
                residual_hf[:, :, c] = result_f[:, :, c] - low.astype(np.float64)

            edge_mask_3ch = edge_mask[:, :, np.newaxis]
            residual_energy = np.sum(residual_hf ** 2 * edge_mask_3ch, axis=2, keepdims=True)
            restore_energy = np.sum(edge_hf_clean ** 2, axis=2, keepdims=True)
            complement_weight = 1.0 - residual_energy / (residual_energy + restore_energy + 1e-10)
            complement_weight = np.clip(complement_weight, 0.2, 1.0)
            result_f = result_f + edge_hf_clean * edge_strength * complement_weight

            intermediates['complement_weight'] = np.clip(
                complement_weight[:, :, 0] * 255, 0, 255
            ).astype(np.uint8)
        else:
            result_f = result_f + edge_hf_clean * edge_strength

        result = np.clip(result_f, 0, 255).astype(np.uint8)
        intermediates['result'] = result

        return result, intermediates

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        edge_preservation: float = 0.5,
        use_stft: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Full detect-then-remove pipeline.

        Steps:
        1. Detect grid periods (or use configured values)
        2. Measure per-channel modulation depth
        3. Compute STFT local energy map
        4. Apply channel-adaptive notch filtering (STFT or global)
        5. Blend with original using edge protection mask

        Args:
            image_bgr: Input image in BGR format (uint8).
            edge_mask: Pre-computed edge protection mask (float32, [0, 1]).
                If None, no edge protection is applied.
            edge_preservation: How much to preserve edges (0 = ignore, 1 = full).
            use_stft: If True, use STFT (locally-adaptive). If False, global notch.

        Returns:
            Tuple of (result_bgr_uint8, intermediates_dict).
        """
        intermediates: Dict[str, np.ndarray] = {}

        # Step 1: Detect or use configured grid periods
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.period_x > 0 and self.period_y > 0:
            period_x, period_y = self.period_x, self.period_y
        else:
            period_x, period_y = self.detect_grid_periods(gray)

        intermediates['detected_periods'] = np.array(
            [period_x, period_y], dtype=np.int32
        )

        # Step 2: Measure per-channel modulation depth
        modulation = self.measure_channel_modulation(
            image_bgr, period_x, period_y
        )
        intermediates['modulation_depths'] = np.array(
            [modulation.get(c, 0) for c in range(3)], dtype=np.float64
        )

        # Step 3: Compute STFT local energy map (modulation-weighted)
        weighted_gray = self.compute_modulation_weighted_gray(
            image_bgr, modulation
        )
        energy_map = self.compute_local_energy_map(weighted_gray, period_x, period_y)
        intermediates['energy_map'] = (energy_map * 255).astype(np.uint8)

        # Step 4: Apply filtering
        if use_stft:
            filtered = self.channel_adaptive_stft_removal(
                image_bgr, period_x, period_y, energy_map, modulation
            )
        else:
            filtered = self.channel_adaptive_notch_global(
                image_bgr, period_x, period_y, modulation
            )
        intermediates['filtered'] = filtered

        # Step 5: Edge protection blending
        if edge_mask is not None and edge_preservation > 0:
            mask_3ch = (edge_mask * edge_preservation)[:, :, np.newaxis]
            original_f = image_bgr.astype(np.float32)
            filtered_f = filtered.astype(np.float32)
            result = mask_3ch * original_f + (1 - mask_3ch) * filtered_f
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = filtered

        intermediates['result'] = result

        return result, intermediates


__all__ = ["STFTAdaptiveGridRemover"]
