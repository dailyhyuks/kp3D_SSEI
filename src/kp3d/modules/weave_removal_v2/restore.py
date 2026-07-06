"""Stage 1 v2 복원 조립: 분해 -> C 직물 제거 -> L 정규화 -> 재합성 (스펙 §2, §4.4)."""
from dataclasses import dataclass

import numpy as np

from kp3d.modules.decomposition import decompose, recompose
from kp3d.modules.weave_removal_v2.line_layer import normalize_line_contrast
from kp3d.modules.weave_removal_v2.removal import WeaveRemovalV2Result, remove_weave


@dataclass
class RestorationResult:
    """계약 1->2 산출: R(restored) + noise_sigma."""
    restored: np.ndarray       # (H,W,3) uint8
    line_alpha: np.ndarray     # (H,W) float32, 정규화 후
    color_cleaned: np.ndarray  # (H,W,3) uint8
    weave: WeaveRemovalV2Result
    noise_sigma: float


def restore(image_bgr: np.ndarray) -> RestorationResult:
    """R = normalize(L) over remove_weave(C)."""
    img = np.asarray(image_bgr)
    dec = decompose(img)
    weave = remove_weave(dec.color_layer, noise_sigma=dec.noise_sigma)
    alpha = normalize_line_contrast(dec.line_alpha)
    restored = recompose(img, alpha, weave.cleaned)
    return RestorationResult(
        restored=restored,
        line_alpha=alpha,
        color_cleaned=weave.cleaned,
        weave=weave,
        noise_sigma=dec.noise_sigma,
    )
