"""Contour (ink line) enhancement for Korean traditional paintings.

Selectively restores and strengthens ink contours that may be weakened
during grid artifact removal, operating only on the LAB L-channel
to preserve colors (a, b channels unchanged).
"""

import cv2
import numpy as np
from loguru import logger

# Default parameters
DEFAULT_BOOST = 10.0
DEFAULT_BLOCK_SIZE = 15
DEFAULT_THRESH_C = 6.0
DEFAULT_BLUR_SIGMA = 3


def enhance_contours(
    original_bgr: np.ndarray,
    denoised_bgr: np.ndarray,
    boost: float = DEFAULT_BOOST,
    block_size: int = DEFAULT_BLOCK_SIZE,
    thresh_c: float = DEFAULT_THRESH_C,
    blur_sigma: int = DEFAULT_BLUR_SIGMA,
) -> np.ndarray:
    """Selectively enhance contour (ink) lines after grid removal.

    Restores ink lines weakened during spectral interpolation and optionally
    darkens them further by `boost`. Operates only on LAB L-channel,
    so painted colors (a, b) are fully preserved.

    Algorithm:
        1. Adaptive threshold on denoised L-channel -> binary ink detection
        2. Scharr gradient -> edge reinforcement
        3. Combined mask: max(ink_soft, gradient * 0.5)
        4. L darkening: L_enhanced = L_dn - mask * (max(L_diff, 0) + boost)
        5. a, b channels unchanged -> zero color impact

    Args:
        original_bgr: Original image BGR uint8 (H, W, 3).
        denoised_bgr: Grid-removed image BGR uint8 (H, W, 3).
        boost: Additional darkening amount beyond restoration (0=restore only).
        block_size: Adaptive threshold block size (must be odd).
        thresh_c: Adaptive threshold constant (lower = more area detected).
        blur_sigma: Contour mask smoothing kernel size (must be odd).

    Returns:
        Contour-enhanced image BGR uint8 (H, W, 3).

    Raises:
        ValueError: If input images are None, wrong shape, or mismatched sizes.
    """
    # Input validation
    if original_bgr is None or denoised_bgr is None:
        raise ValueError("Input images cannot be None")
    if original_bgr.ndim != 3 or original_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H,W,3), got shape {original_bgr.shape}")
    if denoised_bgr.ndim != 3 or denoised_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H,W,3), got shape {denoised_bgr.shape}")
    if original_bgr.shape != denoised_bgr.shape:
        raise ValueError(
            f"Image shapes must match: original {original_bgr.shape} vs "
            f"denoised {denoised_bgr.shape}"
        )

    logger.debug(
        f"Enhancing contours: boost={boost}, block_size={block_size}, "
        f"thresh_c={thresh_c}, image_shape={original_bgr.shape}"
    )

    # LAB conversion
    lab_dn = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    L_dn = lab_dn[:, :, 0]
    a_dn = lab_dn[:, :, 1]
    b_dn = lab_dn[:, :, 2]

    lab_orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    L_orig = lab_orig[:, :, 0]

    # Adaptive threshold: detect dark regions (ink lines) relative to neighbors
    adaptive_inv = cv2.adaptiveThreshold(
        L_dn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, thresh_c
    )
    ink_mask = cv2.GaussianBlur(
        adaptive_inv.astype(np.float32) / 255.0,
        (blur_sigma, blur_sigma), 0
    )

    # Scharr gradient: reinforce ink line boundaries
    gx = cv2.Scharr(L_dn, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(L_dn, cv2.CV_64F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_norm = grad / (grad.max() + 1e-8)

    # Combined mask: adaptive (ink interior) ∪ gradient (ink boundary)
    contour_mask = np.maximum(ink_mask, grad_norm.astype(np.float32) * 0.5)
    contour_mask = contour_mask / (contour_mask.max() + 1e-8)

    # L restoration: darken by amount lost during grid removal + boost
    L_diff = L_dn.astype(np.float64) - L_orig.astype(np.float64)
    L_enhanced = L_dn.astype(np.float64) - contour_mask * (
        np.maximum(L_diff, 0) + boost
    )
    L_enhanced = np.clip(L_enhanced, 0, 255).astype(np.uint8)

    # a, b unchanged -> zero color impact
    lab_result = cv2.merge([L_enhanced, a_dn, b_dn])
    result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

    logger.debug(f"Contour enhancement complete, mask coverage: {(contour_mask > 0.1).mean():.1%}")
    return result
