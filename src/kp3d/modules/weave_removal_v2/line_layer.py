"""선 레이어 대비 정규화: v1 contour boost 대체 (스펙 §2.3, P-adapt)."""
import numpy as np


def normalize_line_contrast(line_alpha: np.ndarray) -> np.ndarray:
    """선 픽셀(alpha>0) 분포의 [5,95] 백분위 아핀 스트레치.

    p5 -> 0, p95 -> 1로 사상 후 [0,1] 클립. 0 픽셀은 0 유지.
    퇴화(선 없음 또는 p95<=p5) 시 입력 복사본 반환.
    """
    alpha = np.asarray(line_alpha, dtype=np.float32)
    mask = alpha > 0.0
    if not mask.any():
        return alpha.copy()
    inside = alpha[mask]
    p5, p95 = np.percentile(inside, [5.0, 95.0])
    if p95 <= p5:
        return alpha.copy()
    out = np.zeros_like(alpha)
    out[mask] = np.clip((inside - p5) / (p95 - p5), 0.0, 1.0)
    return out
