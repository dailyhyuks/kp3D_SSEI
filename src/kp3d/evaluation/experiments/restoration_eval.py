"""Enhancement (Stage 1) evaluation: Weave/Grid Removal.

Evaluates our Spectral Interpolation approach (WeaveRemovalModule) against
traditional filtering baselines on REAL scanned images with actual grid patterns.

Pipeline V2.3 context:
- Stage 1 = Upscale 2x -> Weave Removal (Spectral Interpolation + Contour) -> Upscale 2x
- This experiment evaluates the Weave Removal step in isolation
- "Ours" = WeaveRemovalModule with QUALITY or CLEAN preset

Metrics (no-reference, applied to real grid images):
- Edge Preservation: Sobel-based edge retention ratio (vs original input)
- Grid Energy Ratio: FFT harmonic energy (lower = better grid removal)
- Band SNR: Content-band vs grid-band energy in dB (higher = better)
- SSIM: Structural similarity to original (measures fidelity)
- PSNR: Peak SNR vs original (measures fidelity)
- Naturalness: MSCN-based no-reference naturalness score
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from kp3d.evaluation.config import EvalConfig
from kp3d.evaluation.datasets import find_images


@dataclass
class EnhancementResult:
    """Result for a single image + method combination."""

    image_name: str
    method: str
    edge_preservation: float = 0.0
    grid_energy: float = 0.0
    band_snr: float = 0.0
    ssim: float = 0.0
    naturalness: float = 0.0
    psnr: float = 0.0


class RestorationExperiment:
    """Enhancement (Weave Removal) evaluation experiment.

    Workflow (Real Grid mode):
    1. Load real scanned images (which already have grid/weave patterns)
    2. Apply each method directly to the original image
    3. Compute no-reference metrics (Grid Energy, Band SNR, Naturalness)
    4. Compute fidelity metrics vs original (Edge Preservation, SSIM, PSNR)
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: List[EnhancementResult] = []

    def run(self) -> List[EnhancementResult]:
        """Run the full enhancement evaluation on real grid images."""
        images = find_images(self.config.data_dir)
        if self.config.max_images > 0:
            images = images[: self.config.max_images]

        if not images:
            raise FileNotFoundError(
                f"No images found in {self.config.data_dir}"
            )

        # Load baselines
        from kp3d.evaluation.baselines import get_enhancement_baseline

        baselines = {}
        for name in self.config.enhancement_baselines:
            try:
                baselines[name] = get_enhancement_baseline(name)
            except KeyError:
                print(f"  [WARN] Baseline '{name}' not available, skipping")

        # Our method: WeaveRemovalModule
        ours_method = self._create_ours_method()

        for img_path in images:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            img_name = img_path.stem

            if self.config.dry_run:
                print(f"  [DRY RUN] {img_name}: loaded ({img_bgr.shape})")
                continue

            # Evaluate each baseline on the real image directly
            for name, baseline in baselines.items():
                result_img = baseline.process(img_bgr)
                metrics = self._compute_metrics(img_bgr, result_img)
                self.results.append(
                    EnhancementResult(
                        image_name=img_name, method=name, **metrics
                    )
                )

            # Evaluate ours
            if ours_method is not None:
                result_img = ours_method(img_bgr)
                metrics = self._compute_metrics(img_bgr, result_img)
                self.results.append(
                    EnhancementResult(
                        image_name=img_name, method="ours_spectral", **metrics
                    )
                )

            # Also record "no processing" baseline
            metrics = self._compute_metrics(img_bgr, img_bgr)
            self.results.append(
                EnhancementResult(
                    image_name=img_name, method="no_processing", **metrics
                )
            )

        return self.results

    def _create_ours_method(self):
        """Create our WeaveRemoval method callable."""
        try:
            from kp3d.modules.weave_removal import (
                WeaveRemovalConfig,
                WeaveRemovalModule,
                WeaveRemovalPreset,
            )

            preset = (
                WeaveRemovalPreset.QUALITY
                if self.config.weave_removal_preset == "quality"
                else WeaveRemovalPreset.CLEAN
            )
            config = WeaveRemovalConfig(preset=preset)
            module = WeaveRemovalModule(config=config)

            def apply(img_bgr: np.ndarray) -> np.ndarray:
                result, _ = module.process_bgr(img_bgr)
                return result

            return apply

        except ImportError as e:
            print(f"  [WARN] WeaveRemovalModule not available: {e}")
            return None

    def _compute_metrics(
        self,
        original: np.ndarray,
        result: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all enhancement metrics.

        Args:
            original: Original input image (real grid, used as fidelity reference).
            result: Processed image (after grid removal).

        Returns:
            Dict of metric name -> value.
        """
        return {
            "edge_preservation": compute_edge_preservation(original, result),
            "grid_energy": measure_grid_energy(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
                self.config.grid_period_x,
                self.config.grid_period_y,
            ),
            "band_snr": compute_band_snr(
                cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
                self.config.grid_period_x,
                self.config.grid_period_y,
            ),
            "ssim": compute_ssim_numpy(original, result),
            "psnr": compute_psnr_numpy(original, result),
            "naturalness": compute_naturalness_score(result),
        }

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Aggregate results by method (mean across images)."""
        from collections import defaultdict

        method_results = defaultdict(list)
        for r in self.results:
            method_results[r.method].append(r)

        aggregated = {}
        for method, results in method_results.items():
            aggregated[method] = {
                "edge_preservation": np.mean([r.edge_preservation for r in results]),
                "grid_energy": np.mean([r.grid_energy for r in results]),
                "band_snr": np.mean([r.band_snr for r in results]),
                "ssim": np.mean([r.ssim for r in results]),
                "psnr": np.mean([r.psnr for r in results]),
                "naturalness": np.mean([r.naturalness for r in results]),
            }

        return aggregated


# =============================================================================
# Metric Functions
# =============================================================================


def compute_edge_preservation(original: np.ndarray, result: np.ndarray) -> float:
    """Compute edge preservation ratio using Sobel gradients.

    Measures what fraction of strong edges in the original are
    preserved in the result. Uses 80th percentile as edge threshold.

    Args:
        original: Reference BGR image (pseudo-clean GT).
        result: Processed image.

    Returns:
        Edge preservation ratio (0-1, higher is better).
    """
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    grad_orig = np.sqrt(
        cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3) ** 2
    )
    grad_result = np.sqrt(
        cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=3) ** 2
        + cv2.Sobel(gray_result, cv2.CV_64F, 0, 1, ksize=3) ** 2
    )

    threshold = np.percentile(grad_orig, 80)
    edge_mask = grad_orig > threshold

    if edge_mask.sum() == 0:
        return 1.0

    orig_strength = grad_orig[edge_mask].mean()
    result_strength = grad_result[edge_mask].mean()

    return min(result_strength / (orig_strength + 1e-10), 1.0)


def measure_grid_energy(
    img_gray: np.ndarray,
    period_x: int = 9,
    period_y: int = 7,
) -> float:
    """Measure residual grid energy at harmonic frequencies via FFT.

    Lower value = better grid removal.

    Args:
        img_gray: Grayscale image.
        period_x: Horizontal grid period.
        period_y: Vertical grid period.

    Returns:
        Grid energy ratio (fraction of total spectral energy at grid harmonics).
    """
    f = np.fft.fft2(img_gray.astype(np.float64))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2

    grid_energy = 0.0
    total_energy = np.sum(magnitude ** 2) - magnitude[cy, cx] ** 2

    for harmonic in range(1, 6):
        fx = int(round(harmonic * w / period_x))
        if cx + fx + 3 < w:
            grid_energy += np.sum(magnitude[:, cx + fx - 2 : cx + fx + 3] ** 2)
        if cx - fx - 2 >= 0:
            grid_energy += np.sum(magnitude[:, cx - fx - 2 : cx - fx + 3] ** 2)

        fy = int(round(harmonic * h / period_y))
        if cy + fy + 3 < h:
            grid_energy += np.sum(magnitude[cy + fy - 2 : cy + fy + 3, :] ** 2)
        if cy - fy - 2 >= 0:
            grid_energy += np.sum(magnitude[cy - fy - 2 : cy - fy + 3, :] ** 2)

    return grid_energy / max(total_energy, 1e-10)


def compute_band_snr(
    img_gray: np.ndarray,
    period_x: int = 9,
    period_y: int = 7,
    n_harmonics: int = 5,
    band_radius: int = 3,
) -> float:
    """Compute content-band vs grid-band energy ratio in dB.

    Higher value = better (content dominates over residual grid).

    Args:
        img_gray: Grayscale image.
        period_x: Horizontal grid period.
        period_y: Vertical grid period.
        n_harmonics: Number of harmonics to check.
        band_radius: Frequency band radius around each harmonic.

    Returns:
        Band SNR in dB.
    """
    f = np.fft.fft2(img_gray.astype(np.float64))
    magnitude = np.abs(np.fft.fftshift(f))
    power = magnitude ** 2

    h, w = img_gray.shape
    cy, cx = h // 2, w // 2

    grid_mask = np.zeros((h, w), dtype=bool)
    r = band_radius

    for harmonic in range(1, n_harmonics + 1):
        fx = int(round(harmonic * w / period_x))
        if fx < w // 2 - r:
            grid_mask[:, cx + fx - r : cx + fx + r + 1] = True
            grid_mask[:, cx - fx - r : cx - fx + r + 1] = True

        fy = int(round(harmonic * h / period_y))
        if fy < h // 2 - r:
            grid_mask[cy + fy - r : cy + fy + r + 1, :] = True
            grid_mask[cy - fy - r : cy - fy + r + 1, :] = True

    # Exclude DC
    dc_r = 3
    dc_mask = np.zeros((h, w), dtype=bool)
    dc_mask[cy - dc_r : cy + dc_r + 1, cx - dc_r : cx + dc_r + 1] = True

    content_mask = ~grid_mask & ~dc_mask

    grid_energy = np.sum(power[grid_mask])
    content_energy = np.sum(power[content_mask])

    if grid_energy < 1e-10:
        return 60.0

    return 10.0 * np.log10(content_energy / grid_energy)


def compute_naturalness_score(img_bgr: np.ndarray, block_size: int = 7) -> float:
    """MSCN-based no-reference naturalness score.

    Based on NSS (Natural Scene Statistics). Lower = more natural.

    Args:
        img_bgr: BGR image.
        block_size: Gaussian window size.

    Returns:
        Naturalness score (lower is better/more natural).
    """
    from scipy.special import gamma as gamma_fn

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    kernel = cv2.getGaussianKernel(block_size, block_size / 6.0)
    window = np.outer(kernel, kernel)

    mu = cv2.filter2D(gray, -1, window)
    mu_sq = cv2.filter2D(gray ** 2, -1, window)
    sigma = np.sqrt(np.maximum(mu_sq - mu ** 2, 0)) + 1e-7

    mscn = (gray - mu) / sigma
    b = block_size
    mscn = mscn[b:-b, b:-b].ravel()

    if len(mscn) < 100:
        return 5.0

    mean_abs = np.mean(np.abs(mscn))
    mean_sq = np.mean(mscn ** 2)
    if mean_sq < 1e-10:
        return 5.0
    r = (mean_abs ** 2) / mean_sq

    # Binary search for generalized Gaussian shape parameter
    lo, hi = 0.1, 4.0
    alpha_est = 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        r_mid = gamma_fn(2.0 / mid) ** 2 / (gamma_fn(1.0 / mid) * gamma_fn(3.0 / mid))
        if r_mid > r:
            lo = mid
        else:
            hi = mid
        alpha_est = mid

    natural_alpha = 0.4
    score = abs(alpha_est - natural_alpha)
    var_penalty = abs(np.log(mean_sq + 1e-10) - np.log(1.0)) * 0.3

    return score + var_penalty


def compute_psnr_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two BGR images (numpy, no torch)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two BGR images (numpy, no torch)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    ssim_vals = []
    n_channels = img1.shape[2] if img1.ndim == 3 else 1

    for c in range(min(n_channels, 3)):
        c1 = img1[:, :, c] if img1.ndim == 3 else img1
        c2 = img2[:, :, c] if img2.ndim == 3 else img2

        mu1 = cv2.filter2D(c1, -1, window)
        mu2 = cv2.filter2D(c2, -1, window)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(c1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(c2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(c1 * c2, -1, window) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim_vals.append(np.mean(ssim_map))

    return float(np.mean(ssim_vals))
