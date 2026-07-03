"""Smoke test: Verify V3 pipeline produces metrics in expected range.

Expected ranges (from experiment variant_r_base + contour):
- GridE: 0.03 ~ 0.05
- EdgePres: 0.60 ~ 0.75
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from loguru import logger

from kp3d.modules.weave_removal import (
    WeaveRemovalModule,
    WeaveRemovalPreset,
    WeaveRemovalConfig,
)


def measure_grid_energy_simple(gray: np.ndarray, period_x: int = 9, period_y: int = 7) -> float:
    """Simple grid energy metric via FFT."""
    H, W = gray.shape
    f = np.fft.fft2(gray.astype(np.float64))
    f_shifted = np.fft.fftshift(f)
    mag = np.abs(f_shifted)

    cy, cx = H // 2, W // 2
    energy = 0.0
    count = 0

    # Sum energy at grid harmonic positions
    for k in range(1, 4):  # First 3 harmonics
        for sy, sx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            py = int(cy + sy * k * H / period_y)
            px = int(cx + sx * k * W / period_x)
            if 0 <= py < H and 0 <= px < W:
                r = 3
                y1, y2 = max(0, py - r), min(H, py + r + 1)
                x1, x2 = max(0, px - r), min(W, px + r + 1)
                energy += np.sum(mag[y1:y2, x1:x2])
                count += (y2 - y1) * (x2 - x1)

    # Normalize by total energy
    total = np.sum(mag) + 1e-8
    return energy / total if count > 0 else 0.0


def compute_edge_preservation_simple(original: np.ndarray, result: np.ndarray) -> float:
    """Simple edge preservation metric."""
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Canny edges
    edges_orig = cv2.Canny(gray_orig, 50, 150)
    edges_result = cv2.Canny(gray_result, 50, 150)

    # Overlap ratio
    intersection = np.sum((edges_orig > 0) & (edges_result > 0))
    orig_total = np.sum(edges_orig > 0) + 1e-8

    return intersection / orig_total


def main():
    """Run smoke test on first available image."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data_original_painting" / "data_anno"

    # Find first image
    image_paths = sorted(data_dir.glob("*.png"))
    if not image_paths:
        logger.error(f"No images found in {data_dir}")
        return

    img_path = image_paths[0]
    logger.info(f"Testing V3 pipeline on: {img_path.name}")

    # Load image
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.error(f"Failed to load {img_path}")
        return

    # Resize if too large
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized to {img_bgr.shape[:2]}")

    # Test V3 pipeline
    logger.info("Running V3 pipeline...")
    cfg = WeaveRemovalPreset.V3.to_config()
    logger.info(f"  Config: split_radius={cfg.split_radius}, use_nlm_adaptive={cfg.use_nlm_adaptive}, contour_boost={cfg.contour_boost}")

    module = WeaveRemovalModule(cfg)
    result, _ = module.process_bgr(img_bgr)

    # Compute metrics
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    grid_e = measure_grid_energy_simple(gray_result)
    edge_pres = compute_edge_preservation_simple(img_bgr, result)

    logger.info("=" * 50)
    logger.info("V3 PIPELINE METRICS:")
    logger.info(f"  GridE:    {grid_e:.4f}  (expected: 0.03 ~ 0.05)")
    logger.info(f"  EdgePres: {edge_pres:.3f}  (expected: 0.60 ~ 0.75)")
    logger.info("=" * 50)

    # Validate ranges
    grid_ok = 0.02 <= grid_e <= 0.08
    edge_ok = 0.50 <= edge_pres <= 0.85

    if grid_ok and edge_ok:
        logger.success("PASS: Metrics within expected ranges")
    else:
        if not grid_ok:
            logger.warning(f"GridE {grid_e:.4f} outside expected range [0.02, 0.08]")
        if not edge_ok:
            logger.warning(f"EdgePres {edge_pres:.3f} outside expected range [0.50, 0.85]")

    # Also test legacy presets for regression
    logger.info("\nTesting legacy presets (regression check)...")

    for preset in [WeaveRemovalPreset.QUALITY, WeaveRemovalPreset.CLEAN]:
        cfg = preset.to_config()
        module = WeaveRemovalModule(cfg)
        result, _ = module.process_bgr(img_bgr)

        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        grid_e = measure_grid_energy_simple(gray_result)
        edge_pres = compute_edge_preservation_simple(img_bgr, result)

        logger.info(f"  {preset.value}: GridE={grid_e:.4f}, EdgePres={edge_pres:.3f}")

    logger.info("\nSmoke test complete.")


if __name__ == "__main__":
    main()
