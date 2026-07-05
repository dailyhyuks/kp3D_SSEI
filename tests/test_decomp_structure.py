"""RGF 구조 이미지 테스트: 미세 텍스처 제거 + 강한 에지 보존."""
import numpy as np
import pytest

from kp3d.modules.decomposition.structure import compute_structure_image


def _edge_plus_texture() -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    edge = np.where(xx < 128, 80.0, 180.0)
    texture = 15.0 * np.sin(2 * np.pi * xx / 6) * np.sin(2 * np.pi * yy / 6)
    return (edge + texture).astype(np.float32)


def test_rgf_suppresses_texture():
    """주기 6px 텍스처(진폭 15)를 75% 이상 억제해야 한다."""
    out = compute_structure_image(_edge_plus_texture(), sigma_s=6.0, noise_sigma=1.0)
    left_flat = out[64:192, 32:96]  # 에지에서 먼 균일 영역
    assert float(left_flat.std()) < 15.0 * 0.25


def test_rgf_preserves_strong_edge():
    """대비 100의 스텝 에지를 70% 이상 보존해야 한다."""
    out = compute_structure_image(_edge_plus_texture(), sigma_s=6.0, noise_sigma=1.0)
    contrast = float(out[:, 160:224].mean() - out[:, 32:96].mean())
    assert contrast > 100.0 * 0.7


def test_rgf_terminates_on_flat_image():
    """균일 이미지에서 첫 반복 후 즉시 수렴해야 한다 (무한 루프 방지)."""
    img = np.full((64, 64), 120.0, dtype=np.float32)
    out = compute_structure_image(img, sigma_s=4.0, noise_sigma=0.5)
    assert np.allclose(out, 120.0, atol=1.0)


def test_structure_image_rejects_invalid_ndim():
    """1D 입력은 ValueError를 발생시켜야 한다."""
    with pytest.raises(ValueError):
        compute_structure_image(np.zeros(16, dtype=np.float32), sigma_s=2.0, noise_sigma=1.0)
