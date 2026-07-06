"""자가 경쟁 게이트: 1/4 프록시에서 v2 vs v1 경합, 승자를 전체 해상도 적용 (스펙 §1.4)."""
from dataclasses import dataclass

import cv2
import numpy as np

from kp3d.modules.decomposition import decompose, estimate_noise_sigma
from kp3d.modules.weave_removal_v2.lattice import estimate_lattice
from kp3d.modules.weave_removal_v2.removal import weave_band_energy
from kp3d.modules.weave_removal_v2.restore import restore


@dataclass
class GateResult:
    """게이트 산출: 승자 경로의 R + noise_sigma (계약 1->2)."""
    restored: np.ndarray   # (H,W,3) uint8
    winner: str            # "v2" | "v1"
    quality_v2: float
    quality_v1: float
    noise_sigma: float


def _proxy_scale(image_shape: tuple[int, int], patch_size: int) -> float:
    """1/4 프록시(스펙 §1.4), 단 v1 patch_size 미만으로 줄지 않게 클램프."""
    h, w = image_shape[:2]
    return float(min(1.0, max(0.25, patch_size / min(h, w))))


def _run_v2(image_bgr: np.ndarray) -> np.ndarray:
    """경로 A: v2 복원 파이프라인."""
    return restore(image_bgr).restored


def _run_v1(image_bgr: np.ndarray) -> np.ndarray:
    """경로 B: v1 WeaveRemovalModule (torch 지연 임포트)."""
    from kp3d.modules.weave_removal import WeaveRemovalModule

    result_bgr, _confidence = WeaveRemovalModule(config=None).process_bgr(image_bgr)
    return np.asarray(result_bgr, dtype=np.uint8)


def _v1_patch_size() -> int:
    """v1 config에서 patch_size 유도 (torch 지연 임포트)."""
    from kp3d.modules.weave_removal import WeaveRemovalConfig

    return int(WeaveRemovalConfig().patch_size)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Scharr 그래디언트 크기."""
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.hypot(gx, gy)


def _preservation(orig_gray: np.ndarray, result_gray: np.ndarray,
                  line_mask: np.ndarray) -> float:
    """line_mask 한정 Scharr 그래디언트 Pearson 상관, [0,1] 클립. 마스크 비면 1.0."""
    mask = np.asarray(line_mask, dtype=bool)
    if not mask.any():
        return 1.0
    a = _gradient_magnitude(orig_gray)[mask]
    b = _gradient_magnitude(result_gray)[mask]
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(corr, 0.0, 1.0))


def _quality(proxy_gray: np.ndarray, result_bgr: np.ndarray,
             lattice, e_before: float, line_mask: np.ndarray) -> float:
    """Q = 직조 잔차 감쇠율 x 선 보존 상관 (무차원 곱)."""
    result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if e_before <= 0.0:
        reduction = 0.0
    else:
        e_after = weave_band_energy(result_gray, lattice)
        reduction = float(np.clip(1.0 - e_after / e_before, 0.0, 1.0))
    return reduction * _preservation(proxy_gray, result_gray, line_mask)


def self_competition_gate(image_bgr: np.ndarray) -> GateResult:
    """프록시 경합으로 v2/v1을 선택하고 승자를 전체 해상도에 적용."""
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"(H,W,3) 이미지가 필요합니다: {img.shape}")

    scale = _proxy_scale(img.shape[:2], _v1_patch_size())
    if scale < 1.0:
        proxy = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        proxy = img.copy()
    proxy_gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lattice = estimate_lattice(proxy_gray)
    e_before = weave_band_energy(proxy_gray, lattice)
    line_mask = decompose(proxy).line_mask

    q_v2 = _quality(proxy_gray, _run_v2(proxy), lattice, e_before, line_mask)
    q_v1 = _quality(proxy_gray, _run_v1(proxy), lattice, e_before, line_mask)

    winner = "v2" if q_v2 > q_v1 else "v1"  # 동률 -> v1 안전망 (스펙 §1.4)
    restored = _run_v2(img) if winner == "v2" else _run_v1(img)

    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return GateResult(
        restored=restored, winner=winner,
        quality_v2=q_v2, quality_v1=q_v1,
        noise_sigma=float(estimate_noise_sigma(full_gray)),
    )
