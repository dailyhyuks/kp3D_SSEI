"""직물 제거 코어: 격자 유도 적응 notch + WOLA 재합성 + 잔차 루프 (스펙 §2)."""
from dataclasses import dataclass

import cv2
import numpy as np

from kp3d.modules.decomposition import estimate_noise_sigma
from kp3d.modules.weave_removal_v2.coherence import phase_coherence
from kp3d.modules.weave_removal_v2.lattice import (
    LatticeResult,
    estimate_lattice,
    predict_peak_freqs,
)
from kp3d.modules.weave_removal_v2.notch import fit_peak_gaussian, interpolate_notch

_MAX_ITERS = 10  # 안전 상한 (P-adapt 허용 상수 ②)


@dataclass
class WeaveRemovalV2Result:
    """Stage 1 v2 직물 제거 결과."""
    cleaned: np.ndarray        # (H,W,3) uint8
    lattice: LatticeResult
    iterations: int
    residual_energy: float     # 최종 직조 대역 RMS (그레이 레벨)
    noise_sigma: float


def derive_patch_size(lattice: LatticeResult, image_shape: tuple[int, int]) -> int:
    """패치 = 최대 격자 주기 × 8, 2의 거듭제곱 올림, 상한 min(H,W) 이하."""
    h, w = image_shape
    max_period = max(float(np.linalg.norm(b)) for b in lattice.basis)
    target = max_period * 8.0
    size = 1 << int(np.ceil(np.log2(target)))
    cap = 1 << int(np.floor(np.log2(min(h, w))))
    return int(min(size, cap))


def weave_band_energy(gray: np.ndarray, lattice: LatticeResult) -> float:
    """예측 직조 피크의 국소 바닥 초과 에너지 -> 공간 RMS 진폭.

    유도: 진폭 A 사인파의 반평면 FFT 피크 |F| = A*N/2, RMS = A/sqrt(2)
    -> 초과 E_k에 대해 RMS 합성 = sqrt(2 * sum(E_k^2)) / N.
    """
    if lattice.basis.shape[0] == 0:
        return 0.0
    g = np.asarray(gray, dtype=np.float64)
    h, w = g.shape
    n = h * w
    mag = np.abs(np.fft.fft2(g - g.mean()))
    max_period = max(float(np.linalg.norm(b)) for b in lattice.basis)
    radius = max(1, int(round(min(h, w) / (2.0 * max_period))))
    total = 0.0
    for fy, fx in predict_peak_freqs(lattice):
        by = int(round(fy * h)) % h
        bx = int(round(fx * w)) % w
        ys = np.arange(by - radius, by + radius + 1) % h
        xs = np.arange(bx - radius, bx + radius + 1) % w
        window = mag[np.ix_(ys, xs)]
        ring = np.concatenate(
            [window[0, :], window[-1, :], window[1:-1, 0], window[1:-1, -1]]
        )
        floor = float(np.median(ring))
        excess = max(0.0, float(mag[by, bx]) - floor)
        total += excess * excess
    return float(np.sqrt(2.0 * total) / n)


def _hann2d(size: int) -> np.ndarray:
    """주기적 Hann 창 (50% 겹침 COLA)."""
    n = np.arange(size)
    w1 = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)
    return np.outer(w1, w1)


def _filter_once(gray: np.ndarray, lattice: LatticeResult, patch_size: int) -> np.ndarray:
    """Hann WOLA 패치 순회로 결맞음 가중 notch 1회 적용."""
    g = np.asarray(gray, dtype=np.float64)
    h, w = g.shape
    s = patch_size
    stride = s // 2
    padded = np.pad(g, stride, mode="reflect")
    ph, pw = padded.shape
    window = _hann2d(s)
    origins = [
        (y0, x0)
        for y0 in range(0, ph - s + 1, stride)
        for x0 in range(0, pw - s + 1, stride)
    ]

    patch_ffts = np.stack(
        [np.fft.fft2(padded[y0:y0 + s, x0:x0 + s] * window) for y0, x0 in origins]
    )
    offsets = np.asarray(origins, dtype=np.float64)

    peak_freqs = predict_peak_freqs(lattice)
    peak_weights = []
    for freq in peak_freqs:
        r_coh, w_patch = phase_coherence(patch_ffts, offsets, freq, s)
        peak_weights.append(r_coh * w_patch)  # 전역 결맞음 x 패치 일치, 둘 다 [0,1]

    acc = np.zeros((ph, pw), dtype=np.float64)
    norm = np.zeros((ph, pw), dtype=np.float64)
    for idx, (y0, x0) in enumerate(origins):
        spec = patch_ffts[idx]
        for k, freq in enumerate(peak_freqs):
            weight = float(peak_weights[k][idx])
            if weight <= 0.0:
                continue
            peak = (int(round(freq[0] * s)) % s, int(round(freq[1] * s)) % s)
            sy, sx, amplitude, radius = fit_peak_gaussian(
                np.log1p(np.abs(spec)), peak
            )
            if amplitude <= 0.0:
                continue
            spec = interpolate_notch(spec, peak, (sy, sx), amplitude, radius, weight)
        cleaned = np.real(np.fft.ifft2(spec))
        acc[y0:y0 + s, x0:x0 + s] += cleaned * window
        norm[y0:y0 + s, x0:x0 + s] += window * window
    out = acc / np.maximum(norm, np.finfo(np.float64).tiny)
    return out[stride:stride + h, stride:stride + w]


def remove_weave(
    image_bgr: np.ndarray, noise_sigma: float | None = None
) -> WeaveRemovalV2Result:
    """직조 대역 잔차 < sigma_n까지 notch 반복 후 NLM 잔차 정리."""
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"(H,W,3) 이미지가 필요합니다: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma_n = float(estimate_noise_sigma(gray)) if noise_sigma is None else float(noise_sigma)

    lattice = estimate_lattice(gray)
    if lattice.basis.shape[0] == 0:
        return WeaveRemovalV2Result(
            cleaned=img.copy(), lattice=lattice, iterations=0,
            residual_energy=0.0, noise_sigma=sigma_n,
        )

    patch_size = derive_patch_size(lattice, gray.shape)
    channels = [img[:, :, c].astype(np.float64) for c in range(3)]
    residual = weave_band_energy(gray, lattice)
    iterations = 0
    while residual > sigma_n and iterations < _MAX_ITERS:
        channels = [_filter_once(ch, lattice, patch_size) for ch in channels]
        iterations += 1
        merged = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        work_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = weave_band_energy(work_gray, lattice)

    if iterations == 0:
        cleaned = img.copy()
    else:
        cleaned = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        h_nlm = float(np.hypot(sigma_n, residual))  # 독립 잔차의 RMS 합성
        if h_nlm > 0.0:
            cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, h_nlm, h_nlm)
    return WeaveRemovalV2Result(
        cleaned=cleaned, lattice=lattice, iterations=iterations,
        residual_energy=float(residual), noise_sigma=sigma_n,
    )
