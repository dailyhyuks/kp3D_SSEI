"""Stage 0 분해 모듈 - 통계 측정 테스트."""
import numpy as np

from kp3d.modules.decomposition.statistics import estimate_noise_sigma, estimate_weave_period


def test_noise_sigma_on_gaussian_noise():
    """알려진 σ=5 가우시안 노이즈에서 ±20% 이내로 추정해야 한다."""
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 5.0, (256, 256))
    sigma = estimate_noise_sigma(img)
    assert 4.0 < sigma < 6.0


def test_noise_sigma_on_flat_image():
    """무노이즈 균일 이미지에서 0에 수렴해야 한다."""
    img = np.full((128, 128), 100.0)
    assert estimate_noise_sigma(img) < 0.01


def _synthetic_weave(px: float, py: float, noise: float = 2.0) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    img = 128 + 20 * np.sin(2 * np.pi * xx / px) + 20 * np.sin(2 * np.pi * yy / py)
    rng = np.random.default_rng(1)
    return img + rng.normal(0, noise, img.shape)


def test_weave_period_detection():
    """주기 8/12px 합성 직조에서 ±1px 이내로 검출해야 한다."""
    result = estimate_weave_period(_synthetic_weave(8.0, 12.0))
    assert abs(result.period_x - 8.0) <= 1.0
    assert abs(result.period_y - 12.0) <= 1.0
    assert result.strength_x > 0.5
    assert result.strength_y > 0.5


def test_weave_strength_low_on_pure_noise():
    """순수 노이즈에서는 피크 강도가 낮아야 한다 (주기성 없음 신호)."""
    rng = np.random.default_rng(2)
    img = rng.normal(128, 5, (256, 256))
    result = estimate_weave_period(img)
    assert result.strength_x < 0.3
    assert result.strength_y < 0.3
