"""Spatial-Adaptive Non-Local Means for periodic artifact removal.

Distance-transform + medial-axis based region-adaptive NLM.
Applies stronger NLM to narrow regions and near object edges where
standard methods leave grid artifacts.

V3 Pipeline:
    Stage 1: Split Radius spectral interpolation (grid removal base)
    Stage 2: Spatial-adaptive NLM blending (narrow-region targeted)
    Stage 3: Contour enhancement

The NLM stage operates ON TOP OF Split Radius output, blending
strong NLM into narrow regions only.

Reference:
    - WORK_REPORT_260522_restoration_v3_nlm_contour.md
    - experiments/spatial_adaptive_experiment.py (variant R)
    - experiments/oee_vs_contour_combined.py (variant_r_base)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class SpatialAdaptiveNLMConfig:
    """Configuration for Spatial-Adaptive NLM processing.

    Attributes:
        h_base: NLM filter strength for flat regions (lower = less smoothing).
            Not used in V3 pipeline (flat regions keep Split Radius output).
        h_max: NLM filter strength for narrow/edge regions (higher = more smoothing).
        h_color_base: NLM color filter strength for flat regions.
            Not used in V3 pipeline.
        h_color_max: NLM color filter strength for narrow/edge regions.
        narrow_threshold: Distance threshold for narrow region detection (pixels).
            Regions where distance transform < narrow_threshold are "narrow".
        edge_threshold: Not used in current implementation (reserved for future).
        template_window: NLM template window size.
        search_window: NLM search window size.
        n_clusters: Number of color clusters for region segmentation.
        min_cluster_area: Minimum cluster area to process (skip tiny clusters).
        blur_sigma: Gaussian blur sigma for narrow mask smoothing.
    """

    h_base: float = 10.0
    h_max: float = 15.0
    h_color_base: float = 10.0
    h_color_max: float = 15.0
    narrow_threshold: float = 8.0
    edge_threshold: float = 5.0
    template_window: int = 7
    search_window: int = 21
    n_clusters: int = 5
    min_cluster_area: int = 100
    blur_sigma: float = 2.0


def compute_narrow_region_mask(
    image_bgr: np.ndarray,
    config: SpatialAdaptiveNLMConfig,
) -> np.ndarray:
    """Compute narrow region weight map [0, 1].

    Uses K-means color clustering followed by distance transform
    on each cluster to identify regions where pixels are close
    to multiple cluster boundaries (i.e., narrow regions).

    Args:
        image_bgr: Input BGR image (H, W, 3) uint8.
        config: Configuration parameters.

    Returns:
        Weight map (H, W) float32 in [0, 1].
        1.0 = narrow region (apply strong NLM).
        0.0 = flat region (apply base NLM).
    """
    H, W = image_bgr.shape[:2]

    # Convert to LAB for better color clustering
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    n_clusters = config.n_clusters
    _, labels, _ = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    segmented = labels.reshape(H, W)

    # For each cluster, compute distance transform and identify narrow regions
    narrow_mask = np.zeros((H, W), dtype=np.float32)

    for cluster_id in range(n_clusters):
        cluster_mask = (segmented == cluster_id).astype(np.uint8)

        # Skip tiny clusters
        if cluster_mask.sum() < config.min_cluster_area:
            continue

        # Distance transform: distance to nearest boundary
        dist = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)

        # Narrow regions: distance < threshold
        # Weight: 1.0 at boundary, 0.0 at distance >= threshold
        narrow_contrib = np.clip(
            1.0 - dist / config.narrow_threshold, 0, 1
        ) * cluster_mask

        # Accumulate (max over all clusters)
        narrow_mask = np.maximum(narrow_mask, narrow_contrib)

    # Smooth the mask to avoid harsh transitions
    if config.blur_sigma > 0:
        narrow_mask = cv2.GaussianBlur(
            narrow_mask, (0, 0), config.blur_sigma
        )

    return narrow_mask.astype(np.float32)


def spatial_adaptive_nlm(
    image_bgr: np.ndarray,
    base_processed_bgr: Optional[np.ndarray] = None,
    config: Optional[SpatialAdaptiveNLMConfig] = None,
) -> np.ndarray:
    """V3 Stage 2: Spatial-adaptive NLM blending on top of base processing.

    For V3 pipeline, this function receives the Split Radius output as
    `base_processed_bgr` and applies strong NLM only to narrow regions,
    keeping the Split Radius result in flat regions.

    Matches experiment variant_r_base() from oee_vs_contour_combined.py:
    - Narrow regions (weight=1): Use strong NLM applied to base_processed
    - Flat regions (weight=0): Keep base_processed unchanged

    Args:
        image_bgr: Original input BGR image (H, W, 3) uint8.
            Used for narrow mask computation.
        base_processed_bgr: Output from Split Radius (Stage 1).
            If None, falls back to legacy behavior (applies NLM to image_bgr).
        config: Configuration parameters. If None, uses defaults.

    Returns:
        Processed BGR image (H, W, 3) uint8.

    Example:
        >>> # V3 pipeline usage (with Split Radius base)
        >>> from kp3d.modules.weave_removal import spatial_adaptive_nlm
        >>> split_out = split_radius_module.process_bgr(image_bgr)[0]
        >>> result = spatial_adaptive_nlm(image_bgr, split_out)

        >>> # Legacy usage (no Split Radius)
        >>> result = spatial_adaptive_nlm(image_bgr)
    """
    if config is None:
        config = SpatialAdaptiveNLMConfig()

    # Determine base image: use Split Radius output if provided, else original
    if base_processed_bgr is None:
        # Legacy mode: no Split Radius, apply NLM directly
        logger.debug("spatial_adaptive_nlm: legacy mode (no base_processed_bgr)")
        base_for_nlm = image_bgr
        base_for_blend = image_bgr
    else:
        # V3 mode: Split Radius already applied
        logger.debug("spatial_adaptive_nlm: V3 mode (Split Radius + NLM)")
        base_for_nlm = base_processed_bgr
        base_for_blend = base_processed_bgr

    logger.debug(
        f"spatial_adaptive_nlm: h_max={config.h_max}, "
        f"narrow_threshold={config.narrow_threshold}"
    )

    # Step 1: Compute narrow region mask from ORIGINAL image
    # This matches experiment: narrow mask computed on image_bgr
    narrow_mask = compute_narrow_region_mask(image_bgr, config)

    # Step 2: Apply strong NLM to base_for_nlm (Split Radius output or original)
    # Matches experiment: nlm_strong = cv2.fastNlMeansDenoisingColored(base, ...)
    nlm_strong = cv2.fastNlMeansDenoisingColored(
        base_for_nlm,
        None,
        config.h_max,
        config.h_color_max,
        config.template_window,
        config.search_window,
    )

    # Step 3: Blend based on narrow mask
    # narrow_mask = 0 -> use base_for_blend (flat regions, keep Split Radius)
    # narrow_mask = 1 -> use nlm_strong (narrow regions)
    w = narrow_mask[..., None]
    blended = (1.0 - w) * base_for_blend.astype(np.float64) + w * nlm_strong.astype(np.float64)
    result = np.clip(blended, 0, 255).astype(np.uint8)

    logger.debug(
        f"spatial_adaptive_nlm complete: narrow_mask coverage={narrow_mask.mean():.2%}"
    )

    return result
