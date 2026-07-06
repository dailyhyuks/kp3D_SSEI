"""line_layer.py 테스트: [5,95] 백분위 대비 정규화."""
import numpy as np

from kp3d.modules.weave_removal_v2 import normalize_line_contrast


def test_stretches_weak_alpha_to_full_range():
    rng = np.random.default_rng(0)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = rng.uniform(0.2, 0.6, (32, 32)).astype(np.float32)
    out = normalize_line_contrast(alpha)
    inside = out[alpha > 0]
    assert inside.max() > 0.9          # p95 부근이 1.0으로 스트레치
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.dtype == np.float32


def test_zero_pixels_stay_zero():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[10, 10] = 0.5
    alpha[20, 20] = 0.9
    out = normalize_line_contrast(alpha)
    assert np.all(out[alpha == 0] == 0.0)


def test_order_preserved():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[5, 5], alpha[6, 6], alpha[7, 7] = 0.3, 0.5, 0.7
    alpha[1:4, 1:20] = np.linspace(0.1, 0.9, 57).reshape(3, 19).astype(np.float32)
    out = normalize_line_contrast(alpha)
    assert out[5, 5] <= out[6, 6] <= out[7, 7]


def test_degenerate_constant_alpha_unchanged():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 0.4
    out = normalize_line_contrast(alpha)
    assert np.array_equal(out, alpha)


def test_all_zero_alpha_unchanged():
    alpha = np.zeros((16, 16), dtype=np.float32)
    out = normalize_line_contrast(alpha)
    assert np.array_equal(out, alpha)
