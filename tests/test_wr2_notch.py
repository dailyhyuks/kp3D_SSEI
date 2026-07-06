"""적응 notch 보간 테스트."""
import numpy as np
import pytest

from kp3d.modules.weave_removal_v2.notch import (
    fit_peak_gaussian,
    interpolate_notch,
)


def _spectrum_with_peak(s: int = 64, amp: float = 12.0, period: float = 8.0):
    rng = np.random.default_rng(3)
    xx = np.arange(s, dtype=np.float64)[None, :]
    img = (rng.normal(0.0, 1.0, (s, s))
           + amp * np.cos(2.0 * np.pi * xx / period) * np.ones((s, 1)))
    return img, np.fft.fft2(img)


def test_fit_peak_gaussian_finds_narrow_peak():
    _, spec = _spectrum_with_peak()
    sy, sx, amplitude, radius = fit_peak_gaussian(np.log1p(np.abs(spec)), (0, 8))
    assert amplitude > 0.0
    assert radius >= 1
    assert 1.0 / np.sqrt(12.0) <= sy <= 4.0
    assert 1.0 / np.sqrt(12.0) <= sx <= 4.0


def test_interpolate_notch_suppresses_peak_and_preserves_rest():
    img, spec = _spectrum_with_peak()
    logm = np.log1p(np.abs(spec))
    sy, sx, amplitude, radius = fit_peak_gaussian(logm, (0, 8))
    out = interpolate_notch(spec, (0, 8), (sy, sx), amplitude, radius, 1.0)
    # 피크 크기가 배경 수준으로 감쇠
    assert np.abs(out[0, 8]) < 0.1 * np.abs(spec[0, 8])
    # 지지 창 밖 원거리 빈은 완전 불변
    assert np.abs(out[32, 32]) == pytest.approx(np.abs(spec[32, 32]))
    # 실수성 보존 (켤레 동시 처리)
    rec = np.fft.ifft2(out)
    assert float(np.max(np.abs(rec.imag))) < 1e-8
    # 사인파 에너지 대부분 제거
    assert rec.real.std() < img.std() * 0.5


def test_zero_weight_is_identity():
    _, spec = _spectrum_with_peak()
    logm = np.log1p(np.abs(spec))
    sy, sx, amplitude, radius = fit_peak_gaussian(logm, (0, 8))
    out = interpolate_notch(spec, (0, 8), (sy, sx), amplitude, radius, 0.0)
    assert np.array_equal(out, spec)


def test_flat_spectrum_fit_returns_zero_amplitude():
    logm = np.zeros((32, 32))
    sy, sx, amplitude, _ = fit_peak_gaussian(logm, (4, 4))
    assert amplitude == 0.0
    assert sy == pytest.approx(1.0 / np.sqrt(12.0))
    assert sx == pytest.approx(1.0 / np.sqrt(12.0))
