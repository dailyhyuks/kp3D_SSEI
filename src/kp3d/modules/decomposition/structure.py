"""Rolling Guidance Filter 기반 구조 이미지 (Zhang et al., ECCV 2014).

RGF = 가우시안 초기화 후 joint bilateral filter 반복.
종료 기준은 P-adapt: 반복 간 변화량 중앙값 < noise_sigma (외부 상수 없음).
max_iters=10은 안전 상한 (문헌상 4회 내 수렴, 발산 방어용 — 튜닝 상수 아님).
"""
import cv2
import numpy as np

_MAX_ITERS = 10  # 안전 상한: RGF는 통상 4회 내 수렴 (Zhang et al. 2014)


def _derive_sigma_color(src: np.ndarray) -> float:
    """P-adapt: sigma_color를 그래디언트 크기 중앙값에서 유도."""
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0)  # ksize=3 (OpenCV default Sobel kernel)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1)  # ksize=3 (OpenCV default Sobel kernel)
    mag = np.hypot(gx, gy)
    nonzero = mag[mag > 0]
    if nonzero.size == 0:
        return 1.0  # 완전 균일 이미지: 임의 양수 (필터가 항등이 됨)
    return float(np.median(nonzero))


def compute_structure_image(
    image: np.ndarray, sigma_s: float, noise_sigma: float
) -> np.ndarray:
    """직조 주기(sigma_s) 이하 텍스처를 제거한 구조 이미지를 반환.

    Args:
        image: 2D(gray) 또는 3D(color) float32 배열.
        sigma_s: 공간 스케일 = 직조 주기 (estimate_weave_period에서 유도).
        noise_sigma: 종료 기준 (estimate_noise_sigma 출력).
    Returns:
        입력과 동일 shape의 float32 구조 이미지.
    """
    src = np.asarray(image, dtype=np.float32)
    if src.ndim not in (2, 3):
        raise ValueError("image must be 2D (gray) or 3D (color)")
    guide = cv2.GaussianBlur(src, (0, 0), sigma_s)  # RGF step 1: 소구조 완전 제거
    sigma_color = _derive_sigma_color(src)
    for _ in range(_MAX_ITERS):
        new_guide = cv2.ximgproc.jointBilateralFilter(
            guide, src, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_s
        )
        change = float(np.median(np.abs(new_guide - guide)))
        guide = new_guide
        if change < noise_sigma:
            break
    return guide
