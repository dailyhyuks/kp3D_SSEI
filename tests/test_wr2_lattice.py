"""격자 추정 (2-기저-벡터) 테스트."""
import numpy as np

from kp3d.modules.weave_removal_v2.lattice import (
    LatticeResult,
    estimate_lattice,
    predict_peak_freqs,
)


def _grid(theta_deg: float, p1: float, p2: float, size: int = 192,
          amp: float = 10.0) -> np.ndarray:
    """회전각 theta의 2방향 격자 합성 이미지 (float64 2D)."""
    th = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    return (128.0
            + amp * np.cos(2.0 * np.pi * u / p1)
            + amp * np.cos(2.0 * np.pi * v / p2))


def test_axis_aligned_grid_recovers_periods():
    lat = estimate_lattice(_grid(0.0, 8.0, 12.0))
    assert isinstance(lat, LatticeResult)
    assert lat.basis.shape == (2, 2)
    norms = sorted(np.linalg.norm(lat.basis, axis=1))
    assert abs(norms[0] - 8.0) <= 1.0
    assert abs(norms[1] - 12.0) <= 1.0
    assert lat.strength > 0.2


def test_rotated_grid_recovers_periods():
    lat = estimate_lattice(_grid(15.0, 7.0, 9.0))
    assert lat.basis.shape == (2, 2)
    norms = sorted(np.linalg.norm(lat.basis, axis=1))
    assert abs(norms[0] - 7.0) <= 1.0
    assert abs(norms[1] - 9.0) <= 1.0
    # 두 기저는 비공선
    b1, b2 = lat.basis
    assert abs(b1[0] * b2[1] - b1[1] * b2[0]) > 1.0


def test_flat_image_yields_empty_lattice():
    lat = estimate_lattice(np.full((96, 96), 130.0))
    assert lat.basis.shape[0] == 0
    assert lat.strength == 0.0
    assert predict_peak_freqs(lat).shape == (0, 2)


def test_predict_contains_fundamentals_within_nyquist():
    lat = estimate_lattice(_grid(0.0, 8.0, 12.0))
    freqs = predict_peak_freqs(lat)
    assert freqs.shape[0] >= 2
    assert np.all(np.abs(freqs) <= 0.5 + 1e-12)
    # 기본 주파수 (0, 1/8), (1/12, 0)이 예측에 포함 (부호 대칭 감안해 절대값 비교)
    def _has(target):
        d = np.abs(np.abs(freqs) - np.abs(np.asarray(target))).sum(axis=1)
        return d.min() < 1e-6
    assert _has((0.0, 1.0 / 8.0))
    assert _has((1.0 / 12.0, 0.0))


def test_estimate_lattice_rejects_non_2d():
    import pytest
    with pytest.raises(ValueError):
        estimate_lattice(np.zeros((4, 4, 3)))
