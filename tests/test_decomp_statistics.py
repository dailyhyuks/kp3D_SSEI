"""Stage 0 분해 모듈 - 통계 측정 테스트."""
import numpy as np

from kp3d.modules.decomposition.statistics import estimate_noise_sigma


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
