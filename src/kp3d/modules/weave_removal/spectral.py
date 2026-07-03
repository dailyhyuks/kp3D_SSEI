"""FFT-based spectral interpolation for grid artifact removal.

This module provides spectral interpolation algorithms to remove periodic
grid patterns (weave artifacts) from scanned images of traditional paintings.

Phase 1: Single patch (64x64) grid removal.
Phase 2: Full image patch-wise processing with overlap-add.
"""

import numpy as np
import cv2
from scipy import signal
from typing import Dict, List, Tuple, Optional

from loguru import logger


# =============================================================================
# Constants
# =============================================================================

# Default parameters
DEFAULT_PATCH_SIZE = 64
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_ALPHA = 1.0
DEFAULT_MIN_PROMINENCE = 0.1
DEFAULT_EXCLUDE_DC_RADIUS = 5
DEFAULT_BAND_RADIUS = 2
DEFAULT_MIN_DISTANCE = 3
DEFAULT_PROMINENCE_RATIO = 0.05
DEFAULT_CROSS_HARMONIC_THRESHOLD = 1.5
DEFAULT_ANNULUS_INNER = 2
DEFAULT_ANNULUS_OUTER = 5
DEFAULT_PEAK_RADIUS = 3
DEFAULT_CROSS_PEAK_RADIUS = 1
DEFAULT_EDGE_ALPHA_MIN = 0.3
DEFAULT_EDGE_THRESHOLD_PERCENTILE = 75.0

# Supported methods
SUPPORTED_METHODS = ("annular_mean", "radial_median", "directional_interp")
SUPPORTED_CHANNEL_MODES = ("lab_l", "bgr_independent")


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _get_conjugate_position(y: int, x: int, H: int, W: int) -> Tuple[int, int]:
    """Return the conjugate symmetric position in fftshift coordinates.

    After fftshift, DC is located at (H//2, W//2).
    The symmetric position of frequency (fy, fx) is its reflection around center.

    Args:
        y: y coordinate (fftshift coordinate system)
        x: x coordinate (fftshift coordinate system)
        H: Image height
        W: Image width

    Returns:
        (sym_y, sym_x) conjugate symmetric position
    """
    cy, cx = H // 2, W // 2
    sym_y = (2 * cy - y) % H
    sym_x = (2 * cx - x) % W
    return sym_y, sym_x


def _detect_grid_peaks_2d(
    fft_magnitude: np.ndarray,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    prominence_ratio: float = DEFAULT_MIN_PROMINENCE,
    exclude_dc_radius: int = DEFAULT_EXCLUDE_DC_RADIUS,
    band_radius: int = DEFAULT_BAND_RADIUS
) -> List[Tuple[int, int]]:
    """Detect grid peaks from fftshifted 2D magnitude spectrum (ad-hoc method).

    Excludes DC neighborhood and performs band search along horizontal/vertical axes.
    Used as fallback when harmonic structure is unknown.

    Args:
        fft_magnitude: fftshifted 2D magnitude spectrum
        min_distance: Minimum distance between peaks
        prominence_ratio: Minimum prominence ratio relative to max magnitude
        exclude_dc_radius: Radius around DC to exclude
        band_radius: Search band radius around axes

    Returns:
        List of (y, x) coordinates (fftshift coordinate system)
    """
    H, W = fft_magnitude.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.ogrid[:H, :W]
    dc_mask = (yy - cy)**2 + (xx - cx)**2 <= exclude_dc_radius**2

    mag_masked = fft_magnitude.copy()
    mag_masked[dc_mask] = 0

    max_magnitude = np.max(mag_masked)
    if max_magnitude == 0:
        return []

    min_prominence = prominence_ratio * max_magnitude
    peaks = []

    # Horizontal axis band search
    h_band_start = max(0, cy - band_radius)
    h_band_end = min(H, cy + band_radius + 1)
    horizontal_band = mag_masked[h_band_start:h_band_end, :]
    horizontal_profile = np.max(horizontal_band, axis=0)
    h_peaks, _ = signal.find_peaks(
        horizontal_profile, distance=min_distance, prominence=min_prominence
    )
    for x_idx in h_peaks:
        band_column = mag_masked[h_band_start:h_band_end, x_idx]
        y_offset = np.argmax(band_column)
        peaks.append((int(h_band_start + y_offset), int(x_idx)))

    # Vertical axis band search
    v_band_start = max(0, cx - band_radius)
    v_band_end = min(W, cx + band_radius + 1)
    vertical_band = mag_masked[:, v_band_start:v_band_end]
    vertical_profile = np.max(vertical_band, axis=1)
    v_peaks, _ = signal.find_peaks(
        vertical_profile, distance=min_distance, prominence=min_prominence
    )
    for y_idx in v_peaks:
        band_row = mag_masked[y_idx, v_band_start:v_band_end]
        x_offset = np.argmax(band_row)
        peaks.append((int(y_idx), int(v_band_start + x_offset)))

    # Remove duplicates
    unique_peaks = []
    seen = set()
    for (y, x) in peaks:
        if (y, x) not in seen and not (y == cy and x == cx):
            seen.add((y, x))
            unique_peaks.append((y, x))

    return unique_peaks


def _estimate_fundamental_frequency_1d(
    profile: np.ndarray,
    center: int,
    exclude_dc_radius: int = DEFAULT_EXCLUDE_DC_RADIUS,
    min_distance: int = DEFAULT_MIN_DISTANCE
) -> Optional[float]:
    """Estimate fundamental frequency (bin spacing) from 1D FFT magnitude profile.

    Uses the mode of peak spacings as fundamental frequency.
    Applies parabolic interpolation for sub-pixel precision.

    Args:
        profile: 1D magnitude profile
        center: Center index (DC position)
        exclude_dc_radius: Radius around DC to exclude
        min_distance: Minimum distance between peaks

    Returns:
        Estimated fundamental frequency in bins, or None if detection fails
    """
    # Exclude DC neighborhood
    masked = profile.copy()
    dc_lo = max(0, center - exclude_dc_radius)
    dc_hi = min(len(masked), center + exclude_dc_radius + 1)
    masked[dc_lo:dc_hi] = 0

    max_mag = np.max(masked)
    if max_mag == 0:
        return None

    # Detect peaks on right half only (symmetric)
    right_half = masked[center + exclude_dc_radius:]
    if len(right_half) < 3:
        return None

    peaks, props = signal.find_peaks(
        right_half, distance=min_distance,
        prominence=DEFAULT_PROMINENCE_RATIO * max_mag
    )

    if len(peaks) < 2:
        # If only 1 peak, use its position as fundamental
        if len(peaks) == 1:
            pk = peaks[0] + exclude_dc_radius
            # Parabolic interpolation for sub-pixel
            idx = peaks[0]
            if 0 < idx < len(right_half) - 1:
                a, b, c = right_half[idx-1], right_half[idx], right_half[idx+1]
                if 2*b - a - c != 0:
                    delta = 0.5 * (a - c) / (2*b - a - c)
                    return float(pk + delta)
            return float(pk)
        return None

    # Calculate peak spacings
    spacings = np.diff(peaks).astype(float)

    # Sub-pixel: parabolic interpolation for each peak
    refined_peaks = []
    for pk in peaks:
        idx = pk
        if 0 < idx < len(right_half) - 1:
            a, b, c = right_half[idx-1], right_half[idx], right_half[idx+1]
            denom = 2*b - a - c
            if denom != 0:
                delta = 0.5 * (a - c) / denom
                refined_peaks.append(float(pk + delta))
            else:
                refined_peaks.append(float(pk))
        else:
            refined_peaks.append(float(pk))

    refined_spacings = np.diff(refined_peaks)

    if len(refined_spacings) == 0:
        return None

    # Mode spacing = fundamental frequency (median is robust to outliers)
    fundamental = float(np.median(refined_spacings))

    return fundamental


def _find_peak_in_region(
    mag: np.ndarray,
    center_y: int, center_x: int,
    search_radius: int = DEFAULT_BAND_RADIUS
) -> Tuple[int, int, float]:
    """Find the actual maximum position around a given center coordinate.

    Args:
        mag: Magnitude array
        center_y: Center y coordinate
        center_x: Center x coordinate
        search_radius: Search radius around center

    Returns:
        (best_y, best_x, best_mag) position and magnitude of maximum
    """
    H, W = mag.shape
    y_lo = max(0, center_y - search_radius)
    y_hi = min(H, center_y + search_radius + 1)
    x_lo = max(0, center_x - search_radius)
    x_hi = min(W, center_x + search_radius + 1)

    best_y, best_x, best_mag = center_y, center_x, 0.0
    for sy in range(y_lo, y_hi):
        for sx in range(x_lo, x_hi):
            if mag[sy, sx] > best_mag:
                best_mag = mag[sy, sx]
                best_y, best_x = sy, sx

    return best_y, best_x, best_mag


def _detect_grid_peaks_harmonic(
    fft_magnitude: np.ndarray,
    exclude_dc_radius: int = DEFAULT_EXCLUDE_DC_RADIUS,
    band_radius: int = DEFAULT_BAND_RADIUS,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    prominence_ratio: float = DEFAULT_PROMINENCE_RATIO,
    include_cross_harmonics: bool = True,
    cross_harmonic_threshold: float = DEFAULT_CROSS_HARMONIC_THRESHOLD
) -> Tuple[List[Tuple[int, int]], Optional[float], Optional[float]]:
    """Detect grid peaks based on harmonic structure.

    Step 1: Estimate fundamental frequencies (f_h, f_v) from FFT peak spacings
    Step 2: Generate axis harmonic positions (n*fh, 0), (0, m*fv)
    Step 3: Generate cross-harmonic positions (n*fh, m*fv) if enabled
    Step 4: Refine each position to actual peak location

    Args:
        fft_magnitude: fftshifted 2D magnitude spectrum
        exclude_dc_radius: Radius around DC to exclude
        band_radius: Search band radius around axes
        min_distance: Minimum distance for initial peak detection
        prominence_ratio: Prominence ratio for initial peak detection
        include_cross_harmonics: If True, include cross-harmonic (n*fh, m*fv)
        cross_harmonic_threshold: Cross-harmonic inclusion threshold.
            Include only if magnitude exceeds this multiple of background.

    Returns:
        (peaks, fundamental_h, fundamental_v) tuple
    """
    H, W = fft_magnitude.shape
    cy, cx = H // 2, W // 2

    # DC masking
    yy, xx = np.ogrid[:H, :W]
    dc_mask = (yy - cy)**2 + (xx - cx)**2 <= exclude_dc_radius**2
    mag_masked = fft_magnitude.copy()
    mag_masked[dc_mask] = 0

    peaks = []

    # --- Horizontal fundamental frequency estimation ---
    h_band_start = max(0, cy - band_radius)
    h_band_end = min(H, cy + band_radius + 1)
    h_profile = np.max(mag_masked[h_band_start:h_band_end, :], axis=0)

    fund_h = _estimate_fundamental_frequency_1d(
        h_profile, cx, exclude_dc_radius, min_distance
    )

    # --- Vertical fundamental frequency estimation ---
    v_band_start = max(0, cx - band_radius)
    v_band_end = min(W, cx + band_radius + 1)
    v_profile = np.max(mag_masked[:, v_band_start:v_band_end], axis=1)

    fund_v = _estimate_fundamental_frequency_1d(
        v_profile, cy, exclude_dc_radius, min_distance
    )

    # --- Horizontal axis harmonics (n*fh, 0) ---
    if fund_h is not None and fund_h > 1.0:
        n = 1
        while True:
            target_x_right = cx + round(n * fund_h)
            target_x_left = cx - round(n * fund_h)

            if target_x_right >= W and target_x_left < 0:
                break

            for target_x in [target_x_right, target_x_left]:
                if target_x < 0 or target_x >= W:
                    continue
                if abs(target_x - cx) <= exclude_dc_radius:
                    continue

                by, bx, bm = _find_peak_in_region(
                    mag_masked, cy, target_x, search_radius=band_radius
                )
                if bm > 0:
                    peaks.append((by, bx))

            n += 1

    # --- Vertical axis harmonics (0, m*fv) ---
    if fund_v is not None and fund_v > 1.0:
        n = 1
        while True:
            target_y_down = cy + round(n * fund_v)
            target_y_up = cy - round(n * fund_v)

            if target_y_down >= H and target_y_up < 0:
                break

            for target_y in [target_y_down, target_y_up]:
                if target_y < 0 or target_y >= H:
                    continue
                if abs(target_y - cy) <= exclude_dc_radius:
                    continue

                by, bx, bm = _find_peak_in_region(
                    mag_masked, target_y, cx, search_radius=band_radius
                )
                if bm > 0:
                    peaks.append((by, bx))

            n += 1

    # --- Cross-harmonics (n*fh, m*fv) ---
    if include_cross_harmonics and fund_h is not None and fund_v is not None:
        if fund_h > 1.0 and fund_v > 1.0:
            # Estimate background level: median of non-DC magnitude
            non_dc = mag_masked[~dc_mask]
            bg_median = np.median(non_dc[non_dc > 0]) if np.any(non_dc > 0) else 0
            cross_min_mag = bg_median * cross_harmonic_threshold

            n = 1
            while True:
                target_x = cx + round(n * fund_h)
                if target_x >= W:
                    break

                m = 1
                while True:
                    target_y = cy + round(m * fund_v)
                    if target_y >= H:
                        break

                    # 4-quadrant symmetry: (+n, +m), (+n, -m), (-n, +m), (-n, -m)
                    for sy_sign in [1, -1]:
                        for sx_sign in [1, -1]:
                            tx = cx + sx_sign * round(n * fund_h)
                            ty = cy + sy_sign * round(m * fund_v)

                            if tx < 0 or tx >= W or ty < 0 or ty >= H:
                                continue

                            # DC region check
                            dist_dc = np.sqrt((ty - cy)**2 + (tx - cx)**2)
                            if dist_dc <= exclude_dc_radius:
                                continue

                            by, bx, bm = _find_peak_in_region(
                                mag_masked, ty, tx, search_radius=2
                            )

                            # Include only if above background threshold
                            if bm >= cross_min_mag:
                                peaks.append((by, bx))

                    m += 1
                n += 1

    # Fallback if harmonic detection fails
    if len(peaks) == 0:
        logger.debug("Harmonic detection failed, falling back to ad-hoc method")
        fallback = _detect_grid_peaks_2d(
            fft_magnitude, min_distance, prominence_ratio,
            exclude_dc_radius, band_radius
        )
        return fallback, None, None

    # Remove duplicates
    unique_peaks = []
    seen = set()
    for (y, x) in peaks:
        if (y, x) not in seen and not (y == cy and x == cx):
            seen.add((y, x))
            unique_peaks.append((y, x))

    return unique_peaks, fund_h, fund_v


def _classify_peaks_axis_cross(
    peaks: List[Tuple[int, int]],
    fft_shape: Tuple[int, int],
    band_radius: int = DEFAULT_BAND_RADIUS
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Classify peaks into axis harmonics and cross harmonics.

    Peaks within band_radius of axes are axis harmonics, others are cross.

    Args:
        peaks: List of peak coordinates
        fft_shape: (H, W) shape of FFT
        band_radius: Band radius for axis detection

    Returns:
        (axis_peaks, cross_peaks) tuple
    """
    H, W = fft_shape
    cy, cx = H // 2, W // 2
    axis_peaks = []
    cross_peaks = []

    for (y, x) in peaks:
        on_h_axis = abs(y - cy) <= band_radius
        on_v_axis = abs(x - cx) <= band_radius
        if on_h_axis or on_v_axis:
            axis_peaks.append((y, x))
        else:
            cross_peaks.append((y, x))

    return axis_peaks, cross_peaks


def _estimate_annular_mean(
    fft_complex: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    annulus_inner: int = DEFAULT_ANNULUS_INNER,
    annulus_outer: int = DEFAULT_ANNULUS_OUTER,
    exclude_harmonics: Optional[List[Tuple[int, int]]] = None
) -> Dict[Tuple[int, int], complex]:
    """Estimate background spectrum using annular mean method.

    Computes mean of values in annular (donut) region around each peak.

    Args:
        fft_complex: fftshifted complex FFT
        peak_locations: List of peak coordinates
        annulus_inner: Inner radius of annulus
        annulus_outer: Outer radius of annulus
        exclude_harmonics: Additional positions to exclude from averaging

    Returns:
        Dict mapping (y, x) -> complex background estimate
    """
    H, W = fft_complex.shape
    background = {}

    # Build exclusion set
    exclude_set = set(peak_locations)
    if exclude_harmonics:
        exclude_set.update(exclude_harmonics)

    yy, xx = np.ogrid[:H, :W]

    for (py, px) in peak_locations:
        # Distance calculation for annulus
        dist_sq = (yy - py)**2 + (xx - px)**2
        annulus_mask = (dist_sq >= annulus_inner**2) & (dist_sq <= annulus_outer**2)

        # Exclude other peak/harmonic positions
        for (ey, ex) in exclude_set:
            annulus_mask[ey, ex] = False

        # Mean of values in annulus
        values = fft_complex[annulus_mask]
        if len(values) > 0:
            background[(py, px)] = np.mean(values)
        else:
            # Fallback: keep original (no correction)
            background[(py, px)] = fft_complex[py, px]

    return background


def _estimate_radial_median(
    fft_complex: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    radial_band_width: int = DEFAULT_BAND_RADIUS
) -> Dict[Tuple[int, int], complex]:
    """Estimate background spectrum using radial median method.

    Uses magnitude median and phase circular mean of frequencies at same
    radial distance from DC.

    Args:
        fft_complex: fftshifted complex FFT
        peak_locations: List of peak coordinates
        radial_band_width: Radial band width for same-distance grouping

    Returns:
        Dict mapping (y, x) -> complex background estimate
    """
    H, W = fft_complex.shape
    cy, cx = H // 2, W // 2
    background = {}

    # Pre-compute DC distance for all pixels
    yy, xx = np.ogrid[:H, :W]
    dist_from_dc = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    for (py, px) in peak_locations:
        # Peak's DC distance
        peak_dist = dist_from_dc[py, px]

        # Find pixels at same radial distance
        band_mask = np.abs(dist_from_dc - peak_dist) <= radial_band_width

        # Exclude peak itself
        band_mask[py, px] = False

        values = fft_complex[band_mask]
        if len(values) > 0:
            # Magnitude: median
            magnitudes = np.abs(values)
            median_mag = np.median(magnitudes)

            # Phase: circular mean
            phases = np.angle(values)
            mean_phase = np.arctan2(
                np.mean(np.sin(phases)),
                np.mean(np.cos(phases))
            )

            background[(py, px)] = median_mag * np.exp(1j * mean_phase)
        else:
            background[(py, px)] = fft_complex[py, px]

    return background


def _estimate_directional_interp(
    fft_complex: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    interp_offset: int = 3
) -> Dict[Tuple[int, int], complex]:
    """Estimate background spectrum using directional interpolation.

    Assumes grid peaks are along axes, interpolates from orthogonal direction.

    Args:
        fft_complex: fftshifted complex FFT
        peak_locations: List of peak coordinates
        interp_offset: Offset distance for interpolation

    Returns:
        Dict mapping (y, x) -> complex background estimate
    """
    H, W = fft_complex.shape
    cy, cx = H // 2, W // 2
    background = {}

    for (py, px) in peak_locations:
        # Determine axis direction: horizontal axis peak (cy row) vs vertical axis peak (cx col)
        if py == cy:
            # Horizontal axis peak -> interpolate vertically
            y1, y2 = py - interp_offset, py + interp_offset
            if 0 <= y1 < H and 0 <= y2 < H:
                v1, v2 = fft_complex[y1, px], fft_complex[y2, px]
                background[(py, px)] = (v1 + v2) / 2
            else:
                background[(py, px)] = fft_complex[py, px]
        elif px == cx:
            # Vertical axis peak -> interpolate horizontally
            x1, x2 = px - interp_offset, px + interp_offset
            if 0 <= x1 < W and 0 <= x2 < W:
                v1, v2 = fft_complex[py, x1], fft_complex[py, x2]
                background[(py, px)] = (v1 + v2) / 2
            else:
                background[(py, px)] = fft_complex[py, px]
        else:
            # Cross-harmonic: average of both directions (fallback)
            vals = []
            if py - interp_offset >= 0 and py + interp_offset < H:
                vals.append((fft_complex[py - interp_offset, px] +
                            fft_complex[py + interp_offset, px]) / 2)
            if px - interp_offset >= 0 and px + interp_offset < W:
                vals.append((fft_complex[py, px - interp_offset] +
                            fft_complex[py, px + interp_offset]) / 2)
            if vals:
                background[(py, px)] = np.mean(vals)
            else:
                background[(py, px)] = fft_complex[py, px]

    return background


def _estimate_spectral_background(
    fft_complex: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    method: str = "annular_mean",
    **kwargs
) -> Dict[Tuple[int, int], complex]:
    """Estimate background spectrum at peak locations.

    Estimates the natural spectrum level excluding grid spikes.

    Args:
        fft_complex: fftshifted complex FFT
        peak_locations: List of peak coordinates [(y, x), ...]
        method: Estimation method
            - "annular_mean": Annular region average
            - "radial_median": Radial median
            - "directional_interp": Orthogonal direction interpolation
        **kwargs: Method-specific parameters
            annular_mean: annulus_inner=2, annulus_outer=5, exclude_harmonics=None
            radial_median: radial_band_width=2
            directional_interp: interp_offset=3

    Returns:
        Dict mapping (y, x) -> complex background estimate

    Raises:
        ValueError: If method is not supported
    """
    if not peak_locations:
        return {}

    if method == "annular_mean":
        annulus_inner = kwargs.get("annulus_inner", DEFAULT_ANNULUS_INNER)
        annulus_outer = kwargs.get("annulus_outer", DEFAULT_ANNULUS_OUTER)
        exclude_harmonics = kwargs.get("exclude_harmonics", None)
        return _estimate_annular_mean(
            fft_complex, peak_locations,
            annulus_inner=annulus_inner,
            annulus_outer=annulus_outer,
            exclude_harmonics=exclude_harmonics
        )
    elif method == "radial_median":
        radial_band_width = kwargs.get("radial_band_width", DEFAULT_BAND_RADIUS)
        return _estimate_radial_median(
            fft_complex, peak_locations,
            radial_band_width=radial_band_width
        )
    elif method == "directional_interp":
        interp_offset = kwargs.get("interp_offset", 3)
        return _estimate_directional_interp(
            fft_complex, peak_locations,
            interp_offset=interp_offset
        )
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Supported: {', '.join(SUPPORTED_METHODS)}")


def _interpolate_peaks(
    fft_complex: np.ndarray,
    peak_locations: List[Tuple[int, int]],
    background_values: Dict[Tuple[int, int], complex],
    alpha: float = DEFAULT_ALPHA,
    peak_radius: int = DEFAULT_PEAK_RADIUS,
    adaptive_radius: bool = True
) -> np.ndarray:
    """Interpolate peaks with background values to remove grid.

    F_corrected = F_original - alpha * weight * (F_original - F_background)
    Conjugate symmetric positions are always modified together.

    When adaptive_radius=True, radius is dynamically adjusted based on each
    peak's prominence (center magnitude minus background):
      - Strong peaks -> wider radius (energy spreads wider)
      - Weak peaks -> narrower radius (prevent over-correction)

    Args:
        fft_complex: fftshifted complex FFT (original)
        peak_locations: List of peak coordinates
        background_values: Background estimate dict
        alpha: Correction strength (1.0 = full replacement)
        peak_radius: Maximum correction radius (fixed value when adaptive=False)
        adaptive_radius: If True, use prominence-based dynamic radius

    Returns:
        Corrected fft_complex copy
    """
    H, W = fft_complex.shape
    magnitude = np.abs(fft_complex)

    fft_corrected = fft_complex.copy()
    processed = set()

    # Adaptive radius calculation: determine radius per peak based on prominence
    if adaptive_radius and len(peak_locations) > 0:
        prominences = []
        for (y, x) in peak_locations:
            bg = background_values.get((y, x), fft_complex[y, x])
            prom = magnitude[y, x] - np.abs(bg)
            prominences.append(max(prom, 0))

        max_prom = max(prominences) if prominences else 1.0
        if max_prom == 0:
            max_prom = 1.0

        # Prominence ratio -> radius mapping
        # sqrt scale gives weaker peaks sufficient radius
        # ratio=0.1 -> sqrt(0.1)=0.32 -> r=max(2, round(3*0.32))=2
        # ratio=0.5 -> sqrt(0.5)=0.71 -> r=max(2, round(3*0.71))=2
        # ratio=1.0 -> sqrt(1.0)=1.00 -> r=max(2, round(3*1.00))=3
        min_radius = max(1, peak_radius // 2)  # Minimum radius = half of max
        per_peak_radius = []
        for prom in prominences:
            ratio = np.sqrt(prom / max_prom)  # sqrt gives lower peaks wider radius
            r = max(min_radius, round(peak_radius * ratio))
            per_peak_radius.append(r)
    else:
        per_peak_radius = [peak_radius] * len(peak_locations)

    for idx, (y, x) in enumerate(peak_locations):
        if (y, x) in processed:
            continue

        bg_value = background_values.get((y, x), fft_complex[y, x])
        r = per_peak_radius[idx]
        sigma = float(r) if r > 0 else 1.0

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = y + dy, x + dx

                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if (ny, nx) in processed:
                    continue

                dist_sq = dy * dy + dx * dx
                weight = np.exp(-dist_sq / (2.0 * sigma * sigma))

                correction = fft_corrected[ny, nx] - alpha * weight * (fft_corrected[ny, nx] - bg_value)
                fft_corrected[ny, nx] = correction
                processed.add((ny, nx))

                sym_y, sym_x = _get_conjugate_position(ny, nx, H, W)
                if (sym_y, sym_x) != (ny, nx) and (sym_y, sym_x) not in processed:
                    fft_corrected[sym_y, sym_x] = np.conj(correction)
                    processed.add((sym_y, sym_x))

    return fft_corrected


def _process_single_channel(
    channel: np.ndarray,
    alpha: float,
    method: str,
    min_prominence: float,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Apply spectral interpolation to a single channel.

    Grid confidence is computed as the peak prominence (peak magnitude - background
    estimate) relative to non-DC median magnitude, measuring actual grid energy excess.

    Args:
        channel: Single channel 2D array
        alpha: Correction strength
        method: Background estimation method
        min_prominence: Minimum grid confidence to process
        **kwargs: Additional parameters

    Returns:
        (processed_channel, grid_confidence) tuple
    """
    H, W = channel.shape

    # FFT
    fft = np.fft.fft2(channel.astype(np.float64))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    # Peak detection (harmonic-based, with ad-hoc fallback)
    use_harmonic = kwargs.get("use_harmonic", True)
    exclude_dc = kwargs.get("exclude_dc_radius", DEFAULT_EXCLUDE_DC_RADIUS)
    band_r = kwargs.get("band_radius", DEFAULT_BAND_RADIUS)

    if use_harmonic:
        peaks, fund_h, fund_v = _detect_grid_peaks_harmonic(
            magnitude,
            exclude_dc_radius=exclude_dc,
            band_radius=band_r,
            min_distance=kwargs.get("min_distance", DEFAULT_MIN_DISTANCE),
            prominence_ratio=min_prominence,
            include_cross_harmonics=kwargs.get("include_cross_harmonics", True),
            cross_harmonic_threshold=kwargs.get("cross_harmonic_threshold", DEFAULT_CROSS_HARMONIC_THRESHOLD)
        )
    else:
        peaks = _detect_grid_peaks_2d(
            magnitude,
            min_distance=kwargs.get("min_distance", DEFAULT_MIN_DISTANCE),
            prominence_ratio=min_prominence,
            exclude_dc_radius=exclude_dc,
            band_radius=band_r
        )

    # Early exit if no peaks
    if len(peaks) == 0:
        return channel.copy(), 0.0

    # Compute non-DC magnitude (for median)
    cy, cx = H // 2, W // 2
    exclude_dc_radius = kwargs.get("exclude_dc_radius", DEFAULT_EXCLUDE_DC_RADIUS)
    yy, xx = np.ogrid[:H, :W]
    dc_mask = (yy - cy)**2 + (xx - cx)**2 <= exclude_dc_radius**2
    mag_no_dc = magnitude.copy()
    mag_no_dc[dc_mask] = 0

    # Non-DC median magnitude
    non_dc_values = magnitude[~dc_mask]
    if len(non_dc_values) == 0:
        return channel.copy(), 0.0
    median_non_dc = np.median(non_dc_values)

    if median_non_dc == 0:
        return channel.copy(), 0.0

    # Estimate background BEFORE confidence calculation (used for confidence)
    background = _estimate_spectral_background(
        fft_shifted, peaks, method=method, **kwargs
    )

    # Grid confidence: peak prominence (peak mag - background) / median
    peak_prominences = []
    for (py, px) in peaks:
        peak_mag = magnitude[py, px]
        bg_estimate = np.abs(background.get((py, px), 0))
        prominence = peak_mag - bg_estimate
        peak_prominences.append(prominence)

    max_prominence = max(peak_prominences) if peak_prominences else 0.0
    grid_confidence = max_prominence / median_non_dc

    # Return original if confidence is low
    if grid_confidence < min_prominence:
        return channel.copy(), 0.0

    # Peak interpolation (neighborhood correction with adaptive/fixed peak_radius)
    split_radius = kwargs.get("split_radius", True)
    include_cross = kwargs.get("include_cross_harmonics", True)

    if split_radius and include_cross:
        # Separate axis and cross peaks for different radius interpolation
        axis_peaks, cross_peaks = _classify_peaks_axis_cross(peaks, (H, W), band_radius=band_r)

        # First interpolate axis peaks
        fft_corrected = _interpolate_peaks(
            fft_shifted, axis_peaks, background, alpha=alpha,
            peak_radius=kwargs.get("peak_radius", DEFAULT_PEAK_RADIUS),
            adaptive_radius=kwargs.get("adaptive_radius", True)
        )

        # Then interpolate cross peaks (smaller radius)
        fft_corrected = _interpolate_peaks(
            fft_corrected, cross_peaks, background, alpha=alpha,
            peak_radius=kwargs.get("cross_peak_radius", DEFAULT_CROSS_PEAK_RADIUS),
            adaptive_radius=kwargs.get("adaptive_radius", True)
        )
    else:
        # All peaks with same radius
        fft_corrected = _interpolate_peaks(
            fft_shifted, peaks, background, alpha=alpha,
            peak_radius=kwargs.get("peak_radius", DEFAULT_PEAK_RADIUS),
            adaptive_radius=kwargs.get("adaptive_radius", True)
        )

    # IFFT
    fft_unshifted = np.fft.ifftshift(fft_corrected)
    result = np.fft.ifft2(fft_unshifted)
    result_real = np.real(result)

    return result_real, grid_confidence


def _make_hann_2d(size: int) -> np.ndarray:
    """Create 2D Hann window.

    Args:
        size: Window size

    Returns:
        (size, size) float64 array
    """
    hann_1d = np.hanning(size)
    return np.outer(hann_1d, hann_1d)


def _compute_edge_strength_map(img_bgr: np.ndarray) -> np.ndarray:
    """Compute edge strength map.

    Args:
        img_bgr: BGR uint8 image

    Returns:
        Edge strength map in 0~1 range, float64
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    max_grad = grad.max()
    if max_grad > 0:
        grad /= max_grad
    return grad


# =============================================================================
# Public Functions
# =============================================================================


def spectral_interpolation_single(
    patch_bgr: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    method: str = "annular_mean",
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    channel_mode: str = "lab_l",
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Apply spectral interpolation to a single patch.

    Args:
        patch_bgr: BGR uint8 patch (H, W, 3)
        alpha: Correction strength (0~1, 1=full replacement)
        method: Background estimation method (annular_mean, radial_median, directional_interp)
        min_prominence: Minimum grid confidence threshold (skip if below)
        channel_mode:
            - "lab_l": Convert BGR->LAB, process L channel only, preserve a/b
            - "bgr_independent": Process each channel independently
        **kwargs: Method-specific additional parameters

    Returns:
        (result_bgr_uint8, grid_confidence) tuple
        - grid_confidence: Detected grid strength (0~1)
        - If confidence < min_prominence, returns original unchanged

    Raises:
        ValueError: If method or channel_mode is not supported
        TypeError: If patch_bgr is not a valid numpy array
    """
    # Input validation
    if not isinstance(patch_bgr, np.ndarray):
        raise TypeError(f"patch_bgr must be numpy array, got {type(patch_bgr)}")

    if patch_bgr.ndim != 3 or patch_bgr.shape[2] != 3:
        raise ValueError(f"patch_bgr must have shape (H, W, 3), got {patch_bgr.shape}")

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Supported: {', '.join(SUPPORTED_METHODS)}")

    if channel_mode not in SUPPORTED_CHANNEL_MODES:
        raise ValueError(f"Unknown channel_mode: {channel_mode}. Supported: {', '.join(SUPPORTED_CHANNEL_MODES)}")

    if not 0.0 <= alpha <= 1.0:
        logger.warning(f"alpha={alpha} outside [0, 1] range, clamping")
        alpha = np.clip(alpha, 0.0, 1.0)

    logger.debug(f"Processing patch {patch_bgr.shape}, method={method}, channel_mode={channel_mode}")

    if channel_mode == "lab_l":
        # BGR -> LAB
        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)

        # Process L channel only
        L_float = L.astype(np.float64)
        L_processed, confidence = _process_single_channel(
            L_float, alpha, method, min_prominence, **kwargs
        )

        if confidence < min_prominence:
            return patch_bgr.copy(), 0.0

        # Clip and convert to uint8
        L_result = np.clip(L_processed, 0, 255).astype(np.uint8)

        # Recombine LAB -> BGR
        lab_result = cv2.merge([L_result, a, b])
        result_bgr = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

        logger.debug(f"Patch processed with confidence={confidence:.3f}")
        return result_bgr, confidence

    elif channel_mode == "bgr_independent":
        # Process each channel independently
        channels = cv2.split(patch_bgr)
        processed_channels = []
        confidences = []

        for ch in channels:
            ch_float = ch.astype(np.float64)
            ch_processed, conf = _process_single_channel(
                ch_float, alpha, method, min_prominence, **kwargs
            )
            processed_channels.append(ch_processed)
            confidences.append(conf)

        # Average confidence
        avg_confidence = np.mean(confidences)

        if avg_confidence < min_prominence:
            return patch_bgr.copy(), 0.0

        # Recombine channels
        result_channels = [
            np.clip(ch, 0, 255).astype(np.uint8)
            for ch in processed_channels
        ]
        result_bgr = cv2.merge(result_channels)

        logger.debug(f"Patch processed with avg_confidence={avg_confidence:.3f}")
        return result_bgr, float(avg_confidence)

    else:
        # Should not reach here due to validation above
        raise ValueError(f"Unknown channel_mode: {channel_mode}. "
                        f"Supported: {', '.join(SUPPORTED_CHANNEL_MODES)}")


def process_image_patchwise(
    img_bgr: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    alpha: float = DEFAULT_ALPHA,
    method: str = "annular_mean",
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    channel_mode: str = "lab_l",
    edge_aware: bool = True,
    edge_alpha_min: float = DEFAULT_EDGE_ALPHA_MIN,
    edge_threshold_percentile: float = DEFAULT_EDGE_THRESHOLD_PERCENTILE,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply patch-wise spectral interpolation to full image.

    Uses Hann windowing + overlap-add to prevent patch boundary artifacts.

    Args:
        img_bgr: Input BGR uint8 image (H, W, 3)
        patch_size: Patch size (default 64)
        overlap_ratio: Overlap ratio between adjacent patches (0.5 = 50%)
        alpha: Correction strength (max value when edge_aware=True)
        method: Background estimation method
        min_prominence: Grid-absent patch skip threshold
        channel_mode: Channel processing mode
        edge_aware: If True, automatically reduce alpha for edge-heavy patches
        edge_alpha_min: Minimum alpha for edge_aware (default 0.3)
        edge_threshold_percentile: Edge detection percentile threshold (default 75)
        **kwargs: Additional parameters passed to spectral_interpolation_single

    Returns:
        (result_bgr_uint8, confidence_map) tuple

    Raises:
        ValueError: If method or channel_mode is not supported
        TypeError: If img_bgr is not a valid numpy array
    """
    # Input validation
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError(f"img_bgr must be numpy array, got {type(img_bgr)}")

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"img_bgr must have shape (H, W, 3), got {img_bgr.shape}")

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}. Supported: {', '.join(SUPPORTED_METHODS)}")

    if channel_mode not in SUPPORTED_CHANNEL_MODES:
        raise ValueError(f"Unknown channel_mode: {channel_mode}. Supported: {', '.join(SUPPORTED_CHANNEL_MODES)}")

    if patch_size < 8:
        raise ValueError(f"patch_size must be >= 8, got {patch_size}")

    if not 0.0 < overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio must be in (0, 1), got {overlap_ratio}")

    H, W = img_bgr.shape[:2]
    overlap = int(patch_size * overlap_ratio)
    stride = patch_size - overlap

    logger.info(f"Processing image {W}x{H} with patch_size={patch_size}, "
                f"overlap_ratio={overlap_ratio}, method={method}")

    hann = _make_hann_2d(patch_size)
    hann_3ch = hann[:, :, np.newaxis]

    output = np.zeros((H, W, 3), dtype=np.float64)
    weight_map = np.zeros((H, W), dtype=np.float64)

    n_patches_y = max(1, (H - overlap) // stride)
    n_patches_x = max(1, (W - overlap) // stride)
    confidence_map = np.zeros((n_patches_y, n_patches_x), dtype=np.float64)

    # Edge-aware: pre-compute edge strength map for full image
    edge_map = None
    edge_threshold = 0.0
    if edge_aware:
        edge_map = _compute_edge_strength_map(img_bgr)
        edge_threshold = np.percentile(edge_map, edge_threshold_percentile)
        logger.debug(f"Edge-aware mode enabled, threshold={edge_threshold:.3f}")

    total_patches = n_patches_y * n_patches_x
    processed_patches = 0

    patch_idx_y = 0
    for y in range(0, H - overlap, stride):
        patch_idx_x = 0
        for x in range(0, W - overlap, stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            y_start = y
            x_start = x

            actual_h = y_end - y_start
            actual_w = x_end - x_start

            if actual_h < patch_size or actual_w < patch_size:
                patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                crop = img_bgr[y_start:y_end, x_start:x_end]
                patch[:actual_h, :actual_w] = crop
                if actual_h < patch_size:
                    for py in range(actual_h, patch_size):
                        src_y = actual_h - 1 - (py - actual_h) % actual_h
                        patch[py, :actual_w] = crop[src_y]
                if actual_w < patch_size:
                    for px in range(actual_w, patch_size):
                        src_x = actual_w - 1 - (px - actual_w) % actual_w
                        patch[:actual_h, px] = crop[:, src_x]
                if actual_h < patch_size and actual_w < patch_size:
                    for py in range(actual_h, patch_size):
                        for px in range(actual_w, patch_size):
                            src_y = actual_h - 1 - (py - actual_h) % actual_h
                            src_x = actual_w - 1 - (px - actual_w) % actual_w
                            patch[py, px] = crop[src_y, src_x]
            else:
                patch = img_bgr[y_start:y_end, x_start:x_end].copy()

            # Edge-aware alpha calculation
            patch_alpha = alpha
            if edge_aware and edge_map is not None:
                patch_edge = edge_map[y_start:y_end, x_start:x_end]
                edge_ratio = (patch_edge > edge_threshold).mean()
                # Higher edge ratio -> lower alpha
                patch_alpha = max(edge_alpha_min, alpha * (1.0 - 0.7 * edge_ratio))

            result_patch, confidence = spectral_interpolation_single(
                patch, alpha=patch_alpha, method=method,
                min_prominence=min_prominence,
                channel_mode=channel_mode, **kwargs
            )

            if patch_idx_y < n_patches_y and patch_idx_x < n_patches_x:
                confidence_map[patch_idx_y, patch_idx_x] = confidence

            result_float = result_patch.astype(np.float64)

            window = hann_3ch[:actual_h, :actual_w]
            window_2d = hann[:actual_h, :actual_w]

            output[y_start:y_end, x_start:x_end] += result_float[:actual_h, :actual_w] * window
            weight_map[y_start:y_end, x_start:x_end] += window_2d

            patch_idx_x += 1
            processed_patches += 1

        patch_idx_y += 1

    valid = weight_map > 1e-8
    weight_map_3ch = np.maximum(weight_map[:, :, np.newaxis], 1e-8)

    result = np.where(
        valid[:, :, np.newaxis],
        output / weight_map_3ch,
        img_bgr.astype(np.float64)
    )

    result_bgr = np.clip(result, 0, 255).astype(np.uint8)

    avg_confidence = np.mean(confidence_map[confidence_map > 0]) if np.any(confidence_map > 0) else 0.0
    logger.info(f"Processed {processed_patches} patches, avg_confidence={avg_confidence:.3f}")

    return result_bgr, confidence_map
