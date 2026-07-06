"""restore.py 테스트: 분해 -> 직물 제거 -> 재합성 조립."""
import cv2
import numpy as np

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    restore,
    weave_band_energy,
)


def _weave_painting_with_lines(h: int = 256, w: int = 256, amp: float = 12.0) -> np.ndarray:
    """비주기 베이스(가우시안 블롭) + 축 정렬 직조(주기 8/12) 3채널 이미지."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = amp * np.cos(2 * np.pi * yy / 8.0) + amp * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def test_restore_reduces_weave_band_energy():
    img = _weave_painting_with_lines()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    e0 = weave_band_energy(gray0, lattice)
    result = restore(img)
    gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray1, lattice)
    assert e1 < e0
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8


def test_zero_alpha_pixels_copy_cleaned_color():
    img = _weave_painting_with_lines()
    result = restore(img)
    zero = result.line_alpha == 0.0
    assert zero.any()
    assert np.array_equal(result.restored[zero], result.color_cleaned[zero])


def test_noise_sigma_from_decomposition_is_propagated():
    img = _weave_painting_with_lines()
    result = restore(img)
    assert result.noise_sigma == result.weave.noise_sigma
    assert result.noise_sigma >= 0.0


def test_line_alpha_is_normalized_range():
    img = _weave_painting_with_lines()
    result = restore(img)
    assert result.line_alpha.dtype == np.float32
    assert result.line_alpha.min() >= 0.0
    assert result.line_alpha.max() <= 1.0
